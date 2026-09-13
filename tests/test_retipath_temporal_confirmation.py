from io import StringIO
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'work'))
from retipath_temporal_common import block_counts, make_split, select_observations
from data.schottdorf_lee_multirecording import _bin_trial


def test_block_reader_matches_global_legacy_bin_boundaries():
    ticks = np.sort(np.concatenate((np.arange(2399000, 3000100, 67),
                                    [2400000, 2400066, 2400067, 2999999, 3000000])))
    payload = 'Video Start\t100\nNo\tTime\n'+''.join(f'{i}\t{t}\n' for i,t in enumerate(ticks))
    counts, provenance = block_counts(StringIO(payload), 240, 300)
    expected = _bin_trial(ticks.astype(np.float64)*.1, 45000)[36000:45000]
    np.testing.assert_array_equal(counts, expected)
    assert provenance['end_boundary_guard_seen'] and len(counts) == 9000


def test_split_and_frozen_two_stage_sampling():
    cones = np.zeros((60, 150, 289), dtype=np.float32)
    split = make_split(cones, np.zeros(9000, dtype=np.int64), 'lSS00000', 240)
    mask = np.zeros((60, 150), dtype=bool); mask[:, 45:] = True
    selected = select_observations(mask, split.source_image_ids)
    pairs = np.argwhere(mask)
    hundred = pairs[np.linspace(0, len(pairs)-1, 100).astype(np.int64)]
    np.testing.assert_array_equal(selected, hundred[np.linspace(0, 99, 20).astype(np.int64)])
    assert int(split.valid_mask.sum()) == 7200
    assert split.source_image_ids[0].endswith('036000-036149-trial-1')
    assert split.source_image_ids[-1].endswith('044850-044999-trial-1')
