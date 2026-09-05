from __future__ import annotations

import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_inference import load_cnn, load_ln  # noqa: E402


def main() -> None:
    ln, _ = load_ln("67#4")
    cnn, _ = load_cnn("67#4")
    assert float(ln.history_decay) == float(cnn.history_decay)
    print("PASS frozen LN/CNN history checkpoint adapters")


if __name__ == "__main__":
    main()
