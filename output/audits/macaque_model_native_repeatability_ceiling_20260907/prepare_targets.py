# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "numpy", "pyarrow", "opencv-python"]
# ///
# How to run: D:/anaconda/python.exe -B -u prepare_targets.py
from __future__ import annotations

import csv
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import re
import sys
from typing import Final

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

OUT: Final = Path(__file__).resolve().parent
ROOT: Final = OUT.parents[2]
R1: Final = ROOT / 'output/audits/macaque_retinal_reliability_noise_ceiling_20260907'
PRIOR: Final = ROOT / 'output/audits/macaque_aligned_heldout_pathway_evaluation_20260905'
sys.path.insert(0, str(ROOT))
from data.schottdorf_lee_2021 import SchottdorfAdapterConfig
from data.schottdorf_lee_catalog import mc_pc_recordings
from data.schottdorf_lee_multirecording import _bin_trial, load_schottdorf_cell, load_schottdorf_movie_drive
from data.schottdorf_lee_spikes import parse_recording_spike_trials


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def tensor_sha(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def main() -> None:
    assert not (OUT / 'protocol_hash.json').exists(), 'Do not overwrite frozen R1b outputs'
    torch.set_num_threads(2)
    prior = json.loads((R1 / 'evidence_manifest.json').read_text())
    inputs = prior['input_sha256'].copy()
    for name, digest in inputs.items():
        assert sha(ROOT / name) == digest, name
    for name, digest in prior['output_sha256'].items():
        assert sha(R1 / name) == digest, name
    for path in R1.iterdir():
        if path.is_file():
            inputs[path.relative_to(ROOT).as_posix()] = sha(path)
    for folder in ('data', 'models', 'baselines', 'training', 'evaluation'):
        for path in (ROOT / folder).rglob('*.py'):
            if '__pycache__' not in path.parts and '.git' not in path.parts:
                inputs[path.relative_to(ROOT).as_posix()] = sha(path)
    refs = prior['checkpoint_identity']
    aligned = {r['cell_id']: ROOT / r['path'] for r in refs if r['model'] == 'aligned'}
    alignment = ROOT / 'output/audits/macaque_fixed_alignment_experiment_20260905'
    align_inputs = json.loads((alignment / 'analysis-manifest.json').read_text())['inputs_sha256']
    for path in aligned.values():
        assert sha(path) == align_inputs[str(path.relative_to(ROOT))]
    with (R1 / 'REPEATED_RECORDING_INVENTORY.csv').open(newline='') as stream:
        cohort = list(csv.DictReader(stream))
    eligible = [r for r in cohort if r['usable_for_reliability'] == 'True']
    assert len(eligible) == 20
    assert {r['cell_id'] for r in cohort if r not in eligible} == {'67#4', '68#10'}
    for r in cohort:
        if r not in eligible:
            r['exclusion_reason'] = 'LOCAL_PUBLIC_RAW_REPEAT_FILE_UNAVAILABLE'
    source = json.loads((ROOT / 'output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/results.json').read_text())
    adapter = SchottdorfAdapterConfig(**source['adapter_config'])
    assert adapter.sequence_steps == 150 and adapter.warmup_steps == 30
    full = replace(adapter, validation_sequence_count=44)
    movie_path = ROOT / 'data/real/schottdorf_lee_2021_macaque/1x10_256.mpg'
    records = mc_pc_recordings(ROOT / 'data/real/schottdorf_lee_2021_repository/data')
    saved_paths = {r['cell_id']: [(ROOT / ref['path']).with_name('validation-predictions.pt') for ref in refs if ref['cell_id'] == r['cell_id']] + [PRIOR / 'cells' / (r['cell_id'].replace('#','_') + '-test-predictions.pt')] for r in eligible}
    for paths in saved_paths.values():
        for path in paths:
            assert path.is_file(), path
            inputs[path.relative_to(ROOT).as_posix()] = sha(path)
    with (OUT / 'protocol_hash.json').open('x') as stream:
        json.dump({'sha256': sha(OUT/'PROTOCOL.md'), 'frozen_utc': datetime.now(timezone.utc).isoformat(), 'before_reliability_and_inference': True, 'bootstrap_seed': 20260907}, stream, indent=2)
    with (OUT / 'input_lock.json').open('x') as stream:
        json.dump({'input_sha256': inputs, 'checkpoint_identity': refs, 'R1_preserved': True, 'cohort': cohort, 'adapter_config': source['adapter_config']}, stream, indent=2)
    partitions = [{'split': i+1, 'A': list(a), 'B': sorted(set(range(1,7))-set(a))} for i,a in enumerate(itertools.combinations(range(1,7),3)) if 1 in a]
    for i,row in enumerate(partitions): row['split'] = i+1
    assert len(partitions) == 10
    with (OUT/'split_half_partitions.json').open('x') as stream: json.dump(partitions,stream,indent=2)
    print('Input hashes verified; protocol frozen; decoding production movie',flush=True)
    movie = load_schottdorf_movie_drive(movie_path, full)
    old_movie = load_schottdorf_movie_drive(movie_path, adapter)
    assert np.array_equal(movie.sequences[:20], old_movie.sequences)
    rows = []
    all_counts = []
    payload_cells = {}
    pattern = re.compile(r'(lSS\d+)-live-frames-(\d+)-(\d+)-trial-(\d+)')
    raw_r1 = np.load(R1/'raw_repeated_spike_times.npz')
    for cell in eligible:
        cid, rid = cell['cell_id'], cell['repeated_recording_id']
        selected = tuple(r for r in records if r.cell_id == cid)
        repeat_record = next(r for r in selected if r.recording_id == rid)
        parsed = parse_recording_spike_trials(repeat_record)
        counts = np.stack([_bin_trial(t.numpy(),9000) for t in parsed.live_times_ms_by_trial])
        for index, times in enumerate(parsed.live_times_ms_by_trial):
            assert np.array_equal(times.numpy(), raw_r1[cid.replace('#','_')+f'_repeat_{index+1}_time_ms'])
        assert counts.shape == (6,9000) and counts.sum() == sum(len(t) for t in parsed.live_times_ms_by_trial)
        occupancy = (counts>0).astype(np.float32)
        datasets = [load_schottdorf_cell(selected,old_movie,adapter),load_schottdorf_cell(selected,movie,full)]
        observed_sources = set()
        for scope,data in zip(('production_20s','production_60s'),datasets,strict=True):
            for split in (data.train,data.validation):
                for i,sid in enumerate(split.source_image_ids):
                    match = pattern.fullmatch(sid)
                    assert match is not None
                    srid,begin,end,repeat = match.groups()
                    if srid != rid: continue
                    begin,end,repeat = int(begin),int(end)+1,int(repeat)-1
                    passed = np.array_equal(occupancy[repeat,begin:end],split.spike_events[i,:,0].numpy()) and np.array_equal(counts[repeat,begin:end],split.spike_counts[i,:,0].numpy()) and np.array_equal(split.cone_drive[i].numpy(),movie.sequences[begin//150]) and np.array_equal(split.valid_mask[i,:,0].numpy(),np.arange(150)>=30)
                    rows.append({'cell_id':cid,'recording_id':rid,'repeat':repeat+1,'start_bin':begin,'stop_bin':end,'reference':scope,'source_image_id':sid,'exact':passed,'reference_kind':'LIVE_UNCHANGED_PRODUCTION_ADAPTER'})
                    observed_sources.add(sid)
        assert len(observed_sources)==360
        for path in saved_paths[cid]:
            saved = torch.load(path,weights_only=True,map_location='cpu')
            for i,sid in enumerate(saved['source_image_ids']):
                match = pattern.fullmatch(sid)
                assert match is not None
                srid,begin,end,repeat=match.groups()
                if srid != rid: continue
                begin,end,repeat=int(begin),int(end)+1,int(repeat)-1
                passed=np.array_equal(occupancy[repeat,begin:end],saved['target'][i,:,0].numpy()) and np.array_equal(saved['valid_mask'][i,:,0].numpy(),np.arange(150)>=30)
                if 'spike_counts' in saved: passed=passed and np.array_equal(counts[repeat,begin:end],saved['spike_counts'][i,:,0].numpy())
                rows.append({'cell_id':cid,'recording_id':rid,'repeat':repeat+1,'start_bin':begin,'stop_bin':end,'reference':path.relative_to(ROOT).as_posix(),'source_image_id':sid,'exact':passed,'reference_kind':'EXISTING_SAVED_TARGET_TENSOR'})
        all_counts.append(counts)
        payload_cells[cid]={'group':cell['group'],'recording_id':rid,'counts':torch.from_numpy(counts),'events':torch.from_numpy(occupancy),'cone_positions_degs':movie.cone_positions_degs,'cell_types':datasets[0].cell_types,'polarities':datasets[0].polarities}
        print('Raw/production replay',cid,'exact=',all(r['exact'] for r in rows if r['cell_id']==cid),flush=True)
    with (OUT/'raw_to_target_exact_replay.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    passed=all(r['exact'] for r in rows)
    with (OUT/'preflight.json').open('x') as stream:
        json.dump({'status':'PASSED' if passed else 'STOP_RAW_TARGET_MISMATCH','all_passed':passed,'comparisons':len(rows),'cells':20,'repeats':120,'total_live_bins_per_repeat':9000,'scored_bins_per_repeat':7200,'total_repeat_scored_observations':864000,'stimulus_identity_exact':True,'original_20s_movie_exact':True,'frame_zero':751,'FRAME_ZERO_ACQUISITION_PROVENANCE':'UNKNOWN','training_saved_targets':'NOT_PERSISTED; exact current adapter replay, not historical tensor comparison','training':0,'parameter_fitting':0,'model_inference':0,'protocol_sha256':sha(OUT/'PROTOCOL.md')},stream,indent=2)
    assert passed, 'STOP: raw target mismatch'
    cube=np.stack(all_counts)
    common={'cell_id':np.repeat([r['cell_id'] for r in eligible],54000),'recording_id':np.repeat([r['repeated_recording_id'] for r in eligible],54000),'repeat':np.tile(np.repeat(np.arange(1,7),9000),20),'bin':np.tile(np.arange(9000),120),'time_s':np.tile(np.arange(9000)/150,120),'scored':np.tile(np.arange(9000)%150>=30,120)}
    metadata={b'artifact_class':b'NEW_DETERMINISTIC_DERIVED_ARTIFACT',b'protocol_sha256':sha(OUT/'PROTOCOL.md').encode()}
    for name,values in [('repeat_counts_150hz',cube),('repeat_occupancy_150hz',(cube>0).astype(np.int8))]:
        table=pa.table({**common,'value':values.reshape(-1)}).replace_schema_metadata(metadata)
        pq.write_table(table,OUT/(name+'.parquet'),compression='zstd')
    torch.save({'protocol_sha256':sha(OUT/'PROTOCOL.md'),'movie_sequences':torch.from_numpy(movie.sequences),'mask':torch.arange(9000)%150>=30,'cells':payload_cells},OUT/'inference_inputs.pt')
    with (OUT/'target_artifact_lock.json').open('x') as stream:
        json.dump({p.name:sha(p) for p in OUT.iterdir() if p.is_file()},stream,indent=2)
    print('EXACT REPLAY GATE PASSED',len(rows),'checks; saved 120 repeat arrays; no model inference',flush=True)


if __name__=='__main__': main()
