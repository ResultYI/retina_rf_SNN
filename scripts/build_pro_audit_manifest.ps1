$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')).TrimEnd('\')
if ($repo -ne 'D:\PythonProject\retina_rf_SNN') { throw 'Unexpected repository path' }
$audit = Join-Path $repo 'audit/cleanup_20260905'
function Hash-Row([string]$relative, [string]$category) {
    $file = Get-Item -LiteralPath (Join-Path $repo $relative)
    [PSCustomObject]@{Path=$relative.Replace('\','/');Bytes=$file.Length;SHA256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash;Category=$category}
}
$paths = @(& git -C $repo -c core.quotepath=false ls-files --cached --others --exclude-standard) |
    Where-Object { Test-Path -LiteralPath (Join-Path $repo $_) -PathType Leaf } | Sort-Object -Unique
$publication = @($paths | Where-Object { $_ -notmatch '^audit/cleanup_20260905/(publish_manifest.csv|local_only_artifacts.csv|publication_verification.json|git_status_after.txt)$' } |
    ForEach-Object { Hash-Row $_ 'git-visible' })
$oversized = @($publication | Where-Object { [long]$_.Bytes -ge 100MB })
$private = @($publication | Where-Object { $_.Path -match '(^|/)(\.env[^/]*|id_rsa|id_ed25519)$|\.pem$|^\.local_archives/' })
if ($oversized.Count -or $private.Count) { throw 'Publication contains large or private-like files; inspect before proceeding' }
$publication | Export-Csv (Join-Path $audit 'publish_manifest.csv') -NoTypeInformation -Encoding utf8
$visible = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
foreach ($p in $paths) { [void]$visible.Add($p.Replace('\','/')) }
$roots = @(Get-Content (Join-Path $audit 'retained_bundles.txt')) + @('data/real')
$local = @($roots | ForEach-Object {
    Get-ChildItem -LiteralPath (Join-Path $repo $_) -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -notmatch '\\(__pycache__|\.git)\\' } |
        ForEach-Object {
            $relative = $_.FullName.Substring($repo.Length + 1).Replace('\','/')
            if (-not $visible.Contains($relative)) { Hash-Row $relative 'retained-local-only' }
        }
} | Sort-Object Path -Unique)
$local | Export-Csv (Join-Path $audit 'local_only_artifacts.csv') -NoTypeInformation -Encoding utf8
$source = @(Import-Csv (Join-Path $audit 'source_before.csv') | Where-Object { $_.Path -match '^(models|training|evaluation|data|baselines|tests|configs|loss|utils)/' })
$changed = @($source | Where-Object { -not (Test-Path -LiteralPath (Join-Path $repo $_.Path)) -or (Get-FileHash -LiteralPath (Join-Path $repo $_.Path) -Algorithm SHA256).Hash -ne $_.SHA256 })
if ($changed.Count) { throw 'Scientific source changed' }
$pred = Get-Content (Join-Path $repo '.omo/evidence/final_prediction_results/source_manifest.json') -Raw | ConvertFrom-Json
$failures = @($pred.source_sha256.PSObject.Properties | Where-Object {
    -not (Test-Path -LiteralPath $_.Name) -or (Get-FileHash -LiteralPath $_.Name -Algorithm SHA256).Hash -ne $_.Value
})
if ($failures.Count) { throw 'Prediction source missing or changed' }
$essential = @(
 '.omo/evidence/final_prediction_results/per_cell_nll.csv',
 '.omo/evidence/parametric_illusion_benchmark/aggregation_check/summary.md',
 '.omo/evidence/schottdorf_lee_frame_zero_resolution.md',
 '.omo/evidence/spatial_contrast_adapted/results.json',
 'output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/67_4/model-trained.pt',
 'output/real_data/schottdorf_center_surround_ln_22cell_seed61001_20260830/cells/67_4/ln-trained.pt',
 '.omo/evidence/compact_causal_cnn_baseline/cells/67_4/cnn-trained.pt'
)
foreach ($p in $essential) { if (-not $visible.Contains($p)) { throw "Required audit evidence is ignored: $p" } }
$checkpointCounts = @{
    Canonical=@($publication | Where-Object Path -like 'output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/*/model-trained.pt').Count
    LN=@($publication | Where-Object Path -like 'output/real_data/schottdorf_center_surround_ln_22cell_seed61001_20260830/cells/*/ln-trained.pt').Count
    CNN=@($publication | Where-Object Path -like '.omo/evidence/compact_causal_cnn_baseline/cells/*/cnn-trained.pt').Count
}
if (@($checkpointCounts.Values | Where-Object { $_ -ne 22 }).Count) { throw 'Final checkpoint publication coverage is incomplete' }
[PSCustomObject]@{
    GitHead=(& git -C $repo rev-parse HEAD);PublicationFiles=$publication.Count;
    PublicationBytes=($publication|Measure-Object Bytes -Sum).Sum;
    LargestFileBytes=($publication|Measure-Object Bytes -Maximum).Maximum;
    LocalOnlyFiles=$local.Count;LocalOnlyBytes=($local|Measure-Object Bytes -Sum).Sum;
    ScientificSourceFilesUnchanged=$source.Count;PredictionSourcesUnchanged=@($pred.source_sha256.PSObject.Properties).Count;
    EssentialPathsVisible=$true;FinalCheckpointCounts=$checkpointCounts;
    TrainingPerformed=$false;ModelSourcesModified=$false;Committed=$false;Pushed=$false
} | ConvertTo-Json | Set-Content (Join-Path $audit 'publication_verification.json') -Encoding utf8
Get-Content (Join-Path $audit 'publication_verification.json')
