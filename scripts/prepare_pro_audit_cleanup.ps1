param([ValidateSet('Inventory','Archive','Remove')][string]$Phase = 'Inventory')
$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')).TrimEnd('\')
if ($repo -ne 'D:\PythonProject\retina_rf_SNN') { throw 'Unexpected repository path' }
$audit = Join-Path $repo 'audit\cleanup_20260905'
$backup = Join-Path $repo '.local_archives\20260905-pre-audit-cleanup'
New-Item -ItemType Directory -Force -Path $audit,$backup | Out-Null
function Relative([string]$path) { $path.Substring($repo.Length + 1).Replace('\','/') }
function Resolve-Local([string]$path) {
    $full = [IO.Path]::GetFullPath((Join-Path $repo $path))
    if (-not $full.StartsWith($repo + '\', [StringComparison]::OrdinalIgnoreCase)) { throw "Outside repository: $path" }
    if ($full.StartsWith((Join-Path $repo '.git') + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Git internals are protected' }
    return $full
}
$primary = @(
 'output/architecture_conformance_20260831', 'output/audits',
 'output/synthetic_canonical_v1_shared_bc_noise_free_3seeds_20260830',
 'output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830',
 'output/real_data/schottdorf_canonical_v1_shared_bc_frozen_applications_20260830',
 'output/real_data/schottdorf_center_surround_ln_22cell_seed61001_20260830',
 'output/real_data/karamanlis_2024_population_rf_centers_v1',
 'output/real_data/karamanlis_2024_population_locality_graph_v1',
 'output/real_data/karamanlis_2024_population_rf_geometry_cell_gains_seed20260302',
 'output/real_data/karamanlis_2024_v1_independent_rf_validation_v1',
 'output/real_data/karamanlis_2024_v1_ac_perturbation_v3',
 '.omo/evidence/final_prediction_results', '.omo/evidence/compact_causal_cnn_baseline',
 '.omo/evidence/spatial_contrast_adapted', '.omo/evidence/spatial_contrast_baseline',
 '.omo/evidence/parametric_illusion_benchmark', '.omo/evidence/real_data_independent_seed_sanity',
 '.omo/evidence/schottdorf_lee_150hz_multispike_sanity',
 '.omo/evidence/schottdorf_lee_reset_preroll_sensitivity', '.omo/evidence/schottdorf_lee_frame_zero'
)
if ($Phase -eq 'Inventory') {
    if (Test-Path (Join-Path $backup 'recoverable-content.zip')) { throw 'Completed cleanup inventory is immutable; use a new dated cleanup for another task' }
    (& git -C $repo rev-parse HEAD) | Set-Content (Join-Path $audit 'git_head_before.txt')
    (& git -C $repo branch --show-current) | Set-Content (Join-Path $audit 'git_branch_before.txt')
    (& git -C $repo status --porcelain=v1 -uall 2>&1 | Out-String) | Set-Content (Join-Path $backup 'git_status_before.txt')
    (& git -C $repo diff --binary | Out-String) | Set-Content (Join-Path $backup 'tracked_changes_before.patch')
    $candidates = @(Get-ChildItem (Join-Path $repo 'output') -Directory | Where-Object Name -ne 'real_data')
    $candidates += @(Get-ChildItem (Join-Path $repo 'output/real_data') -Directory)
    $candidates += @(Get-ChildItem (Join-Path $repo '.omo/evidence') -Directory)
    $candidates += @(Get-ChildItem (Join-Path $repo 'runs') -Directory)
    $candidatePaths = @($candidates | ForEach-Object { Relative $_.FullName })
    $keep = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    foreach ($p in $primary) { [void]$keep.Add($p) }
    $sourceRoots = @('models','training','evaluation','data','baselines','scripts','tests','configs','loss','utils')
    $sourceFiles = @($sourceRoots | ForEach-Object { Get-ChildItem (Join-Path $repo $_) -File -Recurse -ErrorAction SilentlyContinue } |
        Where-Object { $_.Extension -in '.py','.yaml','.yml' -and $_.FullName -notmatch '__pycache__|data\\(real|isetbio|bsds|natural|long_micro)' })
    $queue = [Collections.Generic.Queue[IO.FileInfo]]::new()
    $seenFiles = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    foreach ($f in $sourceFiles) { $queue.Enqueue($f) }
    foreach ($p in $primary) {
        Get-ChildItem (Resolve-Local $p) -File -Recurse -ErrorAction Stop |
            Where-Object { $_.Extension -in '.py','.json','.md','.yaml','.yml' } |
            ForEach-Object { $queue.Enqueue($_) }
    }
    $dependencies = [Collections.Generic.List[object]]::new()
    while ($queue.Count -gt 0) {
        $file = $queue.Dequeue()
        if (-not $seenFiles.Add($file.FullName)) { continue }
        $text = [IO.File]::ReadAllText($file.FullName)
        foreach ($p in $candidatePaths) {
            if ($keep.Contains($p) -or $text.IndexOf(($p.Split('/')[-1]), [StringComparison]::Ordinal) -lt 0) { continue }
            [void]$keep.Add($p)
            $dependencies.Add([PSCustomObject]@{Path=$p;ReferencedBy=(Relative $file.FullName);Reason='Literal reference; retained conservatively'})
            Get-ChildItem (Resolve-Local $p) -File -Recurse -ErrorAction SilentlyContinue |
                Where-Object { $_.Extension -in '.py','.json','.md','.yaml','.yml' } |
                ForEach-Object { $queue.Enqueue($_) }
        }
    }
    $dependencies | Export-Csv (Join-Path $audit 'retained_dependencies.csv') -NoTypeInformation -Encoding utf8
    $primary | Set-Content (Join-Path $audit 'primary_bundles.txt') -Encoding utf8
    @($keep | Sort-Object) | Set-Content (Join-Path $audit 'retained_bundles.txt') -Encoding utf8
    $targets = [Collections.Generic.List[object]]::new()
    foreach ($candidate in $candidates) {
        $relative = Relative $candidate.FullName
        if ($keep.Contains($relative)) { continue }
        $errors = @()
        $files = @(Get-ChildItem -LiteralPath $candidate.FullName -Recurse -File -ErrorAction SilentlyContinue -ErrorVariable errors)
        if ($errors.Count) {
            Add-Content (Join-Path $audit 'inaccessible_retained.txt') $relative
            continue
        }
        foreach ($f in $files) {
            if ($f.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Reparse file: $($f.FullName)" }
            $targets.Add([PSCustomObject]@{Path=(Relative $f.FullName);Bytes=$f.Length;SHA256=(Get-FileHash $f.FullName -Algorithm SHA256).Hash;Category=$relative})
        }
    }
    foreach ($root in @('review_packages','report')) {
        Get-ChildItem (Resolve-Local $root) -Recurse -File -ErrorAction Stop | ForEach-Object {
            $targets.Add([PSCustomObject]@{Path=(Relative $_.FullName);Bytes=$_.Length;SHA256=(Get-FileHash $_.FullName -Algorithm SHA256).Hash;Category="historical-$root"})
        }
    }
    $targets | Export-Csv (Join-Path $audit 'archived_files.csv') -NoTypeInformation -Encoding utf8
    $snapshot = @($sourceFiles) + @(Get-Item (Join-Path $repo 'README.md'),(Join-Path $repo 'CURRENT_STATE.md'),(Join-Path $repo '.gitignore'))
    $snapshot | Sort-Object FullName -Unique | ForEach-Object {
        [PSCustomObject]@{Path=(Relative $_.FullName);Bytes=$_.Length;SHA256=(Get-FileHash $_.FullName -Algorithm SHA256).Hash}
    } | Export-Csv (Join-Path $audit 'source_before.csv') -NoTypeInformation -Encoding utf8
    [PSCustomObject]@{Phase=$Phase;CandidateFiles=$targets.Count;CandidateMB=[math]::Round(($targets|Measure-Object Bytes -Sum).Sum/1MB,2);RetainedBundles=$keep.Count;ReferenceFiles=$seenFiles.Count} | ConvertTo-Json
    exit
}
$files = @(Import-Csv (Join-Path $audit 'archived_files.csv'))
$snapshot = @(Import-Csv (Join-Path $audit 'source_before.csv'))
$zipPath = Join-Path $backup 'recoverable-content.zip'
Add-Type -AssemblyName System.IO.Compression.FileSystem
if ($Phase -eq 'Archive') {
    if (Test-Path $zipPath) { throw 'Backup already exists; do not overwrite' }
    $all = @($files + $snapshot | Sort-Object Path -Unique)
    $zip = [IO.Compression.ZipFile]::Open($zipPath,[IO.Compression.ZipArchiveMode]::Create)
    try {
        foreach ($f in $all) {
            $path = Resolve-Local $f.Path
            if ((Get-FileHash $path -Algorithm SHA256).Hash -ne $f.SHA256) { throw "Changed before backup: $($f.Path)" }
            [IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip,$path,$f.Path,[IO.Compression.CompressionLevel]::Optimal) | Out-Null
        }
    } finally { $zip.Dispose() }
    $zip = [IO.Compression.ZipFile]::OpenRead($zipPath)
    try {
        foreach ($f in $all) {
            $entry = $zip.GetEntry($f.Path)
            if ($null -eq $entry -or $entry.Length -ne [long]$f.Bytes) { throw "Backup length mismatch: $($f.Path)" }
            $stream = $entry.Open(); $sha = [Security.Cryptography.SHA256]::Create()
            try { $hash = [BitConverter]::ToString($sha.ComputeHash($stream)).Replace('-','') } finally { $stream.Dispose(); $sha.Dispose() }
            if ($hash -ne $f.SHA256) { throw "Backup hash mismatch: $($f.Path)" }
        }
    } finally { $zip.Dispose() }
    [PSCustomObject]@{Archive=(Relative $zipPath);SHA256=(Get-FileHash $zipPath -Algorithm SHA256).Hash;Bytes=(Get-Item $zipPath).Length;VerifiedEntries=$all.Count;Verified=$true} |
        ConvertTo-Json | Set-Content (Join-Path $audit 'backup_verification.json') -Encoding utf8
    Get-Content (Join-Path $audit 'backup_verification.json')
    exit
}
$verified = Get-Content (Join-Path $audit 'backup_verification.json') -Raw | ConvertFrom-Json
if (-not $verified.Verified -or (Get-FileHash $zipPath -Algorithm SHA256).Hash -ne $verified.SHA256) { throw 'Archive verification failed' }
foreach ($f in $files) {
    $path = Resolve-Local $f.Path
    if ((Get-FileHash $path -Algorithm SHA256).Hash -ne $f.SHA256) { throw "Changed before deletion: $($f.Path)" }
}
foreach ($f in $files) { Remove-Item -LiteralPath (Resolve-Local $f.Path) -Force }
[PSCustomObject]@{RemovedFiles=$files.Count;RemovedBytes=($files|Measure-Object Bytes -Sum).Sum;BackupBytes=$verified.Bytes;Recoverable=$true} |
    ConvertTo-Json | Set-Content (Join-Path $audit 'cleanup_result.json') -Encoding utf8
Get-Content (Join-Path $audit 'cleanup_result.json')
