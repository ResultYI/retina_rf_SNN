from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys

from run import OUT, CONDITIONS, SEEDS, UPDATES, sha, write_json, utc


def fit(job):
    condition, seed = job
    with (OUT / "curves" / f"{condition}_{seed}.log").open("x", encoding="utf-8") as stream:
        subprocess.run([sys.executable, "-u", str(Path(__file__).with_name("run.py")), "train",
                        "--condition", condition, "--seed", str(seed)], stdout=stream,
                       stderr=subprocess.STDOUT, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
    print("COMPLETE", condition, seed, flush=True)
    path = OUT / "checkpoints" / f"{condition}_{seed}_final.pt"
    return {"condition": condition, "seed": seed, "sha256": sha(path)}


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=12) as pool:
        completed = list(pool.map(fit, [(c, s) for s in SEEDS for c in CONDITIONS]))
    write_json(OUT / "training_complete.json", {"completed_utc": utc(), "fits": completed,
        "optimizer_updates_per_fit": UPDATES, "total_optimizer_updates": 12 * UPDATES,
        "protocol_sha256": sha(OUT / "protocol.json")})
