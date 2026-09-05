import contextlib
import hashlib
import io
import json
from pathlib import Path
import runpy
import numpy as np

out = Path(__file__).resolve().parent
root = out.parents[1]
sources = [p for folder in ('models/mechanistic_retina', 'training/mechanistic_retina', 'evaluation/mechanistic_retina') for p in (root / folder).glob('*.py')]
before = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
stdout = io.StringIO()
with contextlib.redirect_stdout(stdout):
    probe = runpy.run_path(str(out / 'runtime_probe.py'))
observations = {'X': probe['x'].detach().numpy(), 'observed_counts': probe['h'].numpy()}
observations.update({f'normal__{k}': v.detach().numpy() for k, v in probe['base'].items()})
for name, clamps in probe['clamps'].items():
    _, tensors = probe['run'](frozenset(clamps))
    observations.update({f'{name}__{k}': v.detach().numpy() for k, v in tensors.items()})
np.savez_compressed(out / 'captured_tensors.npz', **observations)
(out / 'capture_run_results.json').write_text(stdout.getvalue(), encoding='utf-8')
after = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
manifest = {'source_sha256': before, 'all_sources_unchanged_during_capture': before == after, 'source_count': len(sources), 'tensor_array_count': len(observations), 'checkpoint_saved': False, 'training_run': False}
(out / 'source_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
with np.load(out / 'captured_tensors.npz') as archive:
    assert set(archive.files) == set(observations)
    assert all(np.array_equal(archive[key], value) for key, value in observations.items())
assert before == after
print(json.dumps({k:v for k,v in manifest.items() if k != 'source_sha256'}))
