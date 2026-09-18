from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import torch

from common import (ROOT, OUT, REGISTRY, PHASE2, SEEDS, SLOPES, OFFSETS, build, h1_scale,
                    stage_data, load_json, save_json, sha, tensor_sha, minibatches, compute_rms)
from interface_check import run as interface_check


def prepare() -> None:
    torch.set_num_threads(1)
    checks = interface_check()
    phase = load_json(PHASE2 / 'checkpoints/protocol.json')
    registry = load_json(REGISTRY)
    out = OUT
    if out.exists():
        out = out / ('run_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True, exist_ok=False)
    (out / 'checkpoints').mkdir()
    refs, rms_checks = {}, {}
    for cell, source in phase['cells'].items():
        ref = dict(source)
        assert sha(Path(ref['input_path'])) == ref['input_sha256']
        assert sha(Path(ref['metadata_path'])) == ref['metadata_sha256']
        metadata = torch.load(ref['metadata_path'], weights_only=True, mmap=True)
        ref['A'] = registry['cells'][cell]['RetiPath']
        ref['h1_rms'] = {}
        for stage in ('inner', 'refit'):
            data, _, mask = stage_data(ref, stage)
            for seed in SEEDS:
                assert tensor_sha(minibatches(len(data.cone_drive), seed)) == ref['schedule_hashes'][str(seed)][stage]
            initial = build(metadata, SEEDS[0], 'A')
            ref['h1_rms'][stage] = h1_scale(initial, data, mask)
            for seed in SEEDS:
                final_cp = torch.load(ref['A'][str(seed)]['path'], weights_only=True, mmap=True)
                cp = final_cp if stage == 'refit' else torch.load(Path(ref['A'][str(seed)]['path']).parent / 'inner.pt', weights_only=True, mmap=True)
                assert cp['schedule_sha256'] == ref['schedule_hashes'][str(seed)][stage]
                replay_initial = build(metadata, seed, 'A', rms=cp['rms'])
                assert all(torch.equal(v, cp['initial_state'][k]) for k, v in replay_initial.state_dict().items())
            rms_checks[cell + '/' + stage] = {'h1_rms': ref['h1_rms'][stage],
                                            'input_bins': int(mask.sum()), 'no_external_inputs': True}
        refs[cell] = ref
    sources = {str(p.relative_to(ROOT)): sha(p) for p in Path(__file__).parent.glob('*.py')}
    sources.update({path: sha(ROOT / path) for path in phase['source_hashes'] if path.startswith('models/')})
    for name in ('models/mechanistic_retina/retipath.py', 'configs/retipath_final.json',
                 'work/retipath_phase2_common.py', 'work/retipath_phase2_train.py',
                 'training/mechanistic_retina/losses.py', 'training/mechanistic_retina/optimizer.py',
                 'evaluation/mechanistic_retina/factorized_ln_split.py'):
        sources[name] = sha(ROOT / name)
    frozen = {'created_utc': datetime.now(timezone.utc).isoformat(),
              'branch': subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(),
              'HEAD': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'seeds': list(SEEDS), 'slopes': SLOPES, 'offsets': OFFSETS,
              'basis_parameters_fixed': True, 'cross_cell_gradient_updates': False,
              'initialization': 'fresh', 'maximum_updates': 3000, 'patience_updates': 400,
              'evaluation_interval': 25, 'lr': .003, 'batch_size': 4,
              'bootstrap_seed': 2026091399, 'bootstrap_samples': 10000,
              'radial_grid_deg': phase['radial_grid_deg'], 'evaluation_ranges': [[16, 20], [20, 60]],
              'A_registry': str(REGISTRY), 'A_registry_sha256': sha(REGISTRY),
              'cells': refs, 'source_hashes': sources, 'output': str(out)}
    template = Path(__file__).with_name('PROTOCOL_template.md').read_text(encoding='utf-8')
    (out / 'PROTOCOL.md').write_text(template + '\n```json\n' + json.dumps(frozen, ensure_ascii=False, indent=2) + '\n```\n', encoding='utf-8')
    frozen['protocol_sha256'] = sha(out / 'PROTOCOL.md')
    save_json(out / 'checkpoints/protocol.json', frozen)
    checks.update({'protocol_sha256': frozen['protocol_sha256'], 'experimental_protocol_frozen': True,
                   'H1_RMS': rms_checks, 'A_initialization_and_schedules_exact': True,
                   'training_complete': False, 'evaluation_complete': False})
    save_json(out / 'correctness.json', checks)
    print('FROZEN', out, '22 cells; 132 new selection fits + 132 fresh refits; 66 matched A refits reused', flush=True)


if __name__ == '__main__':
    prepare()
