"""Manual acceptance campaign of quip-cuda-msa-descent on the 100 archived ice-charge models.

Each model runs as one `--solve` job, one at a time, on physical GPU 1
(CUDA_VISIBLE_DEVICES=1, so the process sees it as device 0). Each job is
wrapped in the bench-evidence telemetry script, because GPU 1 throttles about
45 s into load. The job matches the archived research job: 64 reads, 65,536
sweeps, one sweep per beta, the production default beta range, and the
research seed protocol_seed(nonce) (qpu-f17/generate.py:42-44).

The script appends one JSON line per model to OUT. A rerun skips the nonces
that OUT already holds. Fields: nonce, kernel_s and descent_s (from the
binary's "msa descent" log line), wall_s, raw_best, greedy_best, loop_best,
final_best, qpu80_pp, margin (final_best - qpu80_pp, at or below 0 meets the
target), and rescore_ok (the host energy of the best returned state equals
its reported energy). Energies are integer milli without an offset.

The model files and the targets file are only read. This is a manual run of
two to three hours, never part of CI. Run from the repository root:

    cargo build --release --bin quip-cuda-msa-descent
    /usr/bin/python3 -I -B tools/msa_descent_campaign.py --out .tmp/cu7/campaign/ice-charge.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

TARGETS = Path("/home/carback1/quip-data/regimes/new-pow/targets-pp-100.json")
TELEMETRY = Path("/zfspool/quip-mining/bench-evidence/cuda-sa-limitations/telemetry-run.sh")
BINARY = Path("target/release/quip-cuda-msa-descent")
READS = 64
SWEEPS = 65_536
ANSI = re.compile(r"\x1b\[[0-9;]*m")
FIELD = re.compile(r"(\w+)=(\"[^\"]*\"|\S+)")


def protocol_seed(nonce: str) -> int:
    digest = hashlib.sha256(b"suite-v1" + bytes.fromhex(nonce)).digest()
    return int.from_bytes(digest[:8], "little")


def load_model(path: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in ("h_milli", "edges", "j_milli")}


def energy(spins: np.ndarray, model: dict[str, np.ndarray]) -> int:
    s = spins.astype(np.int64)
    e = model["edges"]
    return int((s[e[:, 0]] * s[e[:, 1]]) @ model["j_milli"] + s @ model["h_milli"])


def problem(nonce: str, model: dict[str, np.ndarray]) -> dict:
    return {
        "h": (model["h_milli"] / 1000.0).tolist(),
        "j": (model["j_milli"] / 1000.0).tolist(),
        "edges": model["edges"].tolist(),
        "num_reads": READS,
        "num_sweeps": SWEEPS,
        "sweeps_per_beta": 1,
        "beta_range": None,
        "seed": protocol_seed(nonce),
    }


def descent_fields(log: str) -> dict[str, str]:
    for line in reversed(ANSI.sub("", log).splitlines()):
        if "msa descent" in line:
            return {k: v.strip('"') for k, v in FIELD.findall(line)}
    raise RuntimeError("the binary wrote no msa descent log line")


def run_one(row: dict, binary: Path, scratch: Path) -> dict:
    nonce = row["nonce"]
    model = load_model(row["model"])
    job = scratch / f"{nonce}.json"
    job.write_text(json.dumps(problem(nonce, model)))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="1", TELEMETRY_GPU="1", NO_COLOR="1")
    command = [str(TELEMETRY), f"cu7-{nonce[:12]}", "sh", "-c",
               f'exec "{binary}" --device 0 --solve --log-level info < "{job}"']
    started = time.perf_counter()
    done = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
    wall_s = time.perf_counter() - started
    if done.returncode != 0:
        raise RuntimeError(f"{nonce[:12]}: exit {done.returncode}: {done.stderr[-2000:]}")
    solutions = json.loads(done.stdout)
    fields = descent_fields(done.stderr)
    best = min(solutions, key=lambda s: s["energy_milli"])
    final_best = int(best["energy_milli"])
    rescore_ok = energy(np.asarray(best["spins"], dtype=np.int8), model) == final_best
    loop_best = fields.get("loop_best", "None")
    loop_value = re.search(r"-?\d+", loop_best)
    return {
        "nonce": nonce,
        "kernel_s": float(fields["kernel_s"]),
        "descent_s": float(fields["descent_s"]),
        "wall_s": wall_s,
        "raw_best": int(fields["raw_best"]),
        "greedy_best": int(fields["greedy_best"]),
        "loop_best": int(loop_value.group()) if loop_value else None,
        "final_best": final_best,
        "qpu80_pp": int(row["qpu80_pp"]),
        "margin": final_best - int(row["qpu80_pp"]),
        "rescore_ok": rescore_ok and int(fields["final_best"]) == final_best,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="JSONL output, appended")
    parser.add_argument("--binary", type=Path, default=BINARY)
    parser.add_argument("--limit", type=int, default=None, help="run at most this many models")
    args = parser.parse_args()
    rows = json.loads(TARGETS.read_text())["ice-charge"]["rows"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    if args.out.exists():
        seen = {json.loads(line)["nonce"] for line in args.out.read_text().splitlines() if line}
    todo = [r for r in rows if r["nonce"] not in seen][: args.limit]
    binary = args.binary.resolve()
    with tempfile.TemporaryDirectory(dir=args.out.parent) as scratch:
        for k, row in enumerate(todo, 1):
            record = run_one(row, binary, Path(scratch))
            with args.out.open("a") as out:
                out.write(json.dumps(record) + "\n")
            print(f"{k}/{len(todo)} {record['nonce'][:12]} final {record['final_best']} "
                  f"margin {record['margin']} wall {record['wall_s']:.1f}s", flush=True)
    records = [json.loads(line) for line in args.out.read_text().splitlines() if line]
    met = sum(r["margin"] <= 0 for r in records)
    print(f"{met} of {len(records)} models at or below qpu80_pp; "
          f"rescore failures {sum(not r['rescore_ok'] for r in records)}")


if __name__ == "__main__":
    main()
