from __future__ import annotations

from datetime import datetime, timezone
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'output/experiments/retipath_spatial_ei_temporal_confirmation'
SKIP = {'.git', '.codegraph', '.uv-cache', '__pycache__', '.pytest_cache', '.ruff_cache', 'node_modules', 'external'}
TEXT = {'.md', '.json', '.jsonl', '.csv', '.yaml', '.yml', '.toml', '.log', '.py', '.txt'}
SOURCE = re.compile(r'(?:lSS\d{5}-)?live-frames-(\d+)-(\d+)(?:-trial-\d+)?')
RANGES = re.compile(r'(?:live_range_s|live_frames_half_open|decoded_frames_half_open|time_range_s|temporal_range_s)["\s:=]+\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]', re.I)
MENTIONS = re.compile(r'[\[(]\s*(?:240\s*[,，]\s*300|36000\s*[,，]\s*45000|36751\s*[,，]\s*45751)\s*[\])]|\b(?:240|300)\s*[-–−]\s*(?:300|360)\b')


def scan(out: Path, supplement: bool = False) -> dict:
    if supplement:
        prior = json.loads((out/'provenance_scan.json').read_text(encoding='utf-8'))
        folders = [Path(ast.literal_eval(e[e.index("'"):])) for e in prior['errors']]
    else:
        out.mkdir(parents=True, exist_ok=False)
        folders = [ROOT]
    objects = []; markers = []; ranges = []; mentions = []; errors = []

    def inspect(name: str, data: bytes) -> None:
        value = data.decode('utf-8', errors='replace')
        ids = sorted(set((int(m[1]), int(m[2])) for m in SOURCE.finditer(value)))
        overlapping = [(a, b) for a, b in ids if a < 90000 and b >= 36000]
        objects.append({'path': name, 'bytes': len(data), 'max_source_frame': max((b for _, b in ids), default=-1),
                        'post240_source_ranges': overlapping})
        for match in RANGES.finditer(value):
            ranges.append({'path': name, 'literal': match[0], 'start': float(match[1]), 'stop': float(match[2])})
        snippets = [value[max(0, m.start()-100):m.end()+100] for m in MENTIONS.finditer(value)]
        if snippets:
            mentions.append({'path': name, 'snippets': list(dict.fromkeys(snippets))[:12]})
        if Path(name).name.upper() == 'TEST_CONSUMED.JSON':
            marker = json.loads(value)
            markers.append({'path': name, 'live_range_s': marker.get('live_range_s'),
                            'status': marker.get('status'), 'sha256': hashlib.sha256(data).hexdigest()})

    def ignored(relative: str) -> bool:
        return relative.startswith(('data/real/', out.relative_to(ROOT).as_posix()+'/', 'work/retipath_temporal_'))

    walking = (entry for root in folders for entry in os.walk(root, onerror=lambda e: errors.append(str(e))))
    for folder, dirs, files in walking:
        dirs[:] = [d for d in dirs if d not in SKIP and not ignored((Path(folder)/d).relative_to(ROOT).as_posix()+'/')]
        for filename in files:
            path = Path(folder)/filename; relative = path.relative_to(ROOT).as_posix()
            if ignored(relative):
                continue
            try:
                if path.suffix.lower() in TEXT:
                    inspect(relative, path.read_bytes())
                elif path.suffix.lower() == '.zip':
                    with zipfile.ZipFile(path) as archive:
                        for entry in archive.infolist():
                            name = entry.filename.replace('\\', '/')
                            if not entry.is_dir() and Path(name).suffix.lower() in TEXT and '/data/real/' not in '/'+name:
                                inspect(relative+'!'+name, archive.read(entry))
            except (OSError, zipfile.BadZipFile, json.JSONDecodeError) as error:
                errors.append(f'{relative}: {error}')
    result = {'scanned_utc': datetime.now(timezone.utc).isoformat(), 'objects': objects,
              'consumption_markers': markers, 'range_metadata': ranges, 'candidate_mentions': mentions,
              'post240_source_objects': [r for r in objects if r['post240_source_ranges']], 'errors': errors,
              'scope': 'Accessible repository text and archive text; excludes raw data, binary tensors/checkpoints, git history, caches and this new run.',
              'exclusions': sorted(SKIP), 'new_targets_read': 0}
    result['supplemental_roots'] = [str(p) for p in folders] if supplement else []
    with (out/('provenance_supplement.json' if supplement else 'provenance_scan.json')).open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    return {'objects': len(objects), 'markers': markers, 'post240_source_paths': [r['path'] for r in result['post240_source_objects']],
            'candidate_mention_paths': [r['path'] for r in mentions], 'errors': errors, 'out': str(out)}


if __name__ == '__main__':
    supplement = sys.argv[1:] == ['--supplement']
    destination = OUT if supplement or not OUT.exists() else OUT/('run_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    print(json.dumps(scan(destination, supplement), ensure_ascii=False, indent=2))
