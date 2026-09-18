from concurrent.futures import ThreadPoolExecutor
import subprocess
import sys

from run import OUT, CONDITIONS, SEEDS, sha, write_json, utc


def fit(job):
    condition, seed = job
    path = OUT / "curves" / f"{condition}_{seed}.log"
    with path.open("x", encoding="utf-8") as stream:
        subprocess.run([sys.executable, "-u", str(__file__).replace("launch.py", "run.py"),
                        "train", "--condition", condition, "--seed", str(seed)],
                       stdout=stream, stderr=subprocess.STDOUT, check=True,
                       creationflags=subprocess.CREATE_NO_WINDOW)
    print("COMPLETE", condition, seed, flush=True)
    checkpoint = OUT / "checkpoints" / f"{condition}_{seed}_final.pt"
    return {"condition": condition, "seed": seed, "sha256": sha(checkpoint)}


if __name__ == "__main__":
    jobs = [(condition, seed) for seed in SEEDS for condition in CONDITIONS]
    with ThreadPoolExecutor(max_workers=5) as pool:
        completed = list(pool.map(fit, jobs))
    write_json(OUT / "training_complete.json", {"completed_utc": utc(), "fits": completed,
        "optimizer_updates_per_fit": 3000, "total_optimizer_updates": 45000,
        "protocol_sha256": sha(OUT / "protocol.json")})
