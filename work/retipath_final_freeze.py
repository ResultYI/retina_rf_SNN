from __future__ import annotations

from dataclasses import asdict
import csv
from pathlib import Path
import subprocess

import numpy as np
import torch

from retipath_final_common import (
    ROOT, OUT, PHASE2, CONFIRM, POP, OLD_BENCH, APP_CODE, REG_CODE, APP_OLD, STIM_CODE,
    SEEDS, CONTROL, load_json, sha, tensor_sha, write_once, utc, application_imports,
)


def main() -> None:
    assert not (OUT / 'source_lock.json').exists()
    assert (OUT / 'MIGRATION_PROTOCOL.md').is_file()
    prior = load_json(CONFIRM / 'confirmatory_lock.json')
    assert load_json(CONFIRM / 'completion.json')['required_outputs_complete']
    frozen = dict(prior['frozen_sha256'])
    for path, digest in frozen.items():
        assert sha(ROOT / path) == digest, path
    def track(path: Path, expected: str | None = None) -> str:
        digest = sha(path)
        if expected is not None:
            assert digest == expected, path
        key = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)
        frozen[key] = digest
        return digest
    records = list(csv.DictReader((OLD_BENCH / 'per_cell_per_range.csv').open(encoding='utf-8')))
    base = {r['cell_id']: r for r in records if r['range'] == '[16,20)'}
    phase = load_json(PHASE2 / 'checkpoints/protocol.json')
    assert set(base) == set(phase['cells']) == set(prior['phase2_cohort'])
    registry = {}
    for cell in sorted(base):
        item = {'group': phase['cells'][cell]['group'], 'RetiPath': {}, CONTROL: {}}
        for name, tag in [('RetiPath', 'C'), (CONTROL, 'A')]:
            for seed in SEEDS:
                ref = prior['all_phase2_checkpoint_references'][cell][f'{seed}/{tag}/refit']
                track(ROOT / ref['path'], ref['sha256'])
                item[name][str(seed)] = ref
        for name in ('LN', 'CNN'):
            row = base[cell]
            path = ROOT / row[name + '_checkpoint']
            digest = track(path, row[name + '_checkpoint_sha256'])
            assert all(r[name + '_checkpoint_sha256'] == digest for r in records if r['cell_id'] == cell)
            item[name] = {'path': path.relative_to(ROOT).as_posix(), 'sha256': digest, 'seed': 61001}
        registry[cell] = item
    write_once(OUT / 'model_registry.json', {'model_name': 'RetiPath', 'architecture_id': 'retipath_spatial_conductance_v1',
        'entry_point': 'models.mechanistic_retina.retipath.RetiPath', 'seeds': list(SEEDS), 'cells': registry,
        'history_mapping_only': {'C': 'RetiPath', 'A': CONTROL, 'B': 'historical matched spatial-current control, not included'},
        'new_training': False, 'weights_renamed_or_copied': False})
    for name in ['MIGRATION_PROTOCOL.md', 'model_registry.json']:
        track(OUT / name)
    for directory, names in [
        (CONFIRM, ['confirmatory_lock.json', 'TEST_CONSUMED.json', 'completion.json', 'analysis_complete.json', 'rf_per_cell_seed.csv']),
        (PHASE2, ['rf_population.csv', 'prediction_population.csv', 'REPORT.md', 'checkpoints/protocol.json']),
        (OLD_BENCH, ['per_cell_per_range.csv', 'range_summary.csv', 'complexity_table.csv', 'REPORT.md']),
        (ROOT, ['models/mechanistic_retina/retipath.py', 'configs/retipath_final.json',
                'baselines/center_surround_ln.py', 'baselines/compact_causal_cnn.py',
                'work/retipath_final_common.py', 'work/retipath_final_freeze.py', 'work/retipath_final_prediction.py',
                'work/retipath_final_brightness.py', 'work/retipath_final_rf.py', 'tests/test_retipath_final.py']),
        (REG_CODE, ['frozen_registration.json', 'registration_stimuli.py', 'registered_responses.pt', 'comparison_metrics.py']),
        (APP_CODE, ['frozen_inputs.json', 'application_io.py', 'response_metrics.py', 'analyze_application.py', 'run_application.py']),
        (STIM_CODE, ['contract.py', 'protocol.json', 'test_contract.py']),
        (APP_OLD, ['retinotopic_registration_control/cell_registration.csv', 'retinotopic_registration_control/PROTOCOL.md']),
    ]:
        for name in names:
            track(directory / name)
    application_imports()
    from registration_stimuli import build_registered_bank, validate_all_centers
    from contract import TIME_MS, ONSET_MS, DURATION_MS, DT_MS
    from evaluation.mechanistic_retina.temporal_center_surround import _optional_box
    reg_lock = load_json(REG_CODE / 'frozen_registration.json')
    registration = APP_OLD / 'retinotopic_registration_control/cell_registration.csv'
    track(registration, reg_lock['registration_csv_sha256'])
    track(APP_OLD / 'retinotopic_registration_control/PROTOCOL.md', reg_lock['protocol_sha256'])
    for field in ('source_hashes', 'code_hashes'):
        for path, digest in reg_lock[field].items():
            track(ROOT / path, digest)
    original = torch.load(REG_CODE / 'registered_responses.pt', weights_only=True, map_location='cpu')
    geometry = validate_all_centers()
    assert [r.cell_id for r in geometry] == sorted(registry)
    patches = []
    pulse = _optional_box(TIME_MS, ONSET_MS, DURATION_MS, DT_MS)
    for i, row in enumerate(geometry):
        bank = build_registered_bank(row.center_x_deg, row.center_y_deg)
        assert torch.equal(bank.patches, original['stimulus_patches'][i])
        assert tuple(bank.names) == tuple(original['stimulus_names'])
        assert [asdict(p) for p in bank.comparisons] == original['comparisons']
        assert torch.equal(bank.history, original['observed_history'])
        assert torch.equal(bank.drive, pulse[None, :, None] * bank.patches.flatten(1)[:, None, :])
        patches.append(bank.patches)
        for seed in SEEDS:
            ref = registry[row.cell_id]['RetiPath'][str(seed)]
            cp = torch.load(ROOT / ref['path'], weights_only=True, map_location='cpu')
            expected = cp['cell_positions_degs'][0]
            assert tuple(expected.tolist()) == (row.center_x_deg, row.center_y_deg), row.cell_id
    with (OUT / 'registered_stimulus_bank.npz').open('xb') as stream:
        np.savez_compressed(stream, patches=torch.stack(patches).numpy(),
            history=bank.history.numpy(), pulse=pulse.numpy(),
            cell_ids=np.array(sorted(registry)), names=np.array(bank.names))
    track(OUT / 'registered_stimulus_bank.npz')
    old_input = ROOT / 'output/audits/macaque_aligned_heldout_pathway_evaluation_20260905'
    track(old_input / 'cnn_test_inputs.pt')
    for cell in sorted(registry):
        safe = cell.replace('#', '_')
        track(POP / 'inputs' / f'{safe}.pt')
        track(old_input / 'cells' / f'{safe}-test-predictions.pt')
        for seed in SEEDS:
            track(PHASE2 / 'checkpoints' / safe / str(seed) / 'analysis.json')
    for info in prior['cells']:
        safe = info['cell_id'].replace('#', '_')
        track(POP / 'confirmatory' / f'{safe}_split.pt')
        track(ROOT / '.omo/evidence/context-gain-confirmation-20260908' / f'{safe}.npz')
        track(CONFIRM / 'targets' / f'{safe}.npz')
        track(CONFIRM / 'targets' / f'{safe}.json')
        for seed in SEEDS:
            track(CONFIRM / 'results' / f'{safe}_{seed}.json')
    for p in [ROOT / '.omo/evidence/context-gain-confirmation-20260908/sampling_before_targets.npz',
              ROOT / 'output/experiments/context_gain_temporal_confirmation_20260908/confirmatory_lock.json',
              ROOT / 'output/experiments/bc_vs_rgc_temporal_confirmation_20260907/confirmatory_lock.json']:
        track(p)
    for p in [POP / 'TEST_CONSUMED.json',
              ROOT / 'output/experiments/context_gain_temporal_confirmation_20260908/TEST_CONSUMED.json',
              ROOT / 'output/experiments/bc_vs_rgc_temporal_confirmation_20260907/TEST_CONSUMED.json',
              old_input / 'TEST_CONSUMED.md']:
        track(p)
    write_once(OUT / 'source_lock.json', {'status': 'FROZEN_BEFORE_FINAL_MODEL_INFERENCE', 'frozen_utc': utc(),
        'branch': subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(),
        'HEAD': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'cells': sorted(registry), 'continuous_cells': prior['cells'], 'seeds': list(SEEDS),
        'frozen_sha256': frozen, 'checkpoint_count': 176, 'all_historical_phase2_checkpoints_also_frozen': 396,
        'registration': [asdict(r) for r in geometry], 'stimulus_comparisons': [asdict(p) for p in bank.comparisons],
        'bank_replay_bitwise_equal_for_all_22': True, 'centers_unchanged': True,
        'training_updates': 0, 'new_target_blocks': 0, 'forward_calls_at_lock': 0})
    print('MIGRATION LOCKED: formal architecture, 176 scoring checkpoints, original registered stimulus bank; no forwards yet.')


if __name__ == '__main__':
    main()
