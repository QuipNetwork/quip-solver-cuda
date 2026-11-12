"""Record the research descent results that tests/msa_descent.rs checks.

The tool runs three steps:

1. `cargo run --example msa_descent_reference -- export DATA` regenerates each
   fixture instance (quip/charged-ice@1, ice-charge recipe, z(12, 4)) and its
   start reads into DATA.
2. This process runs the research code on each instance: natural-move greedy
   descent on every read, then `nat_loop` on the lowest raw read with zero
   budget 0 and the nonce seed, as campaign_greedy.py:243-273 does.
3. `cargo run --example msa_descent_reference -- record DATA` hashes the
   results into tests/fixtures/msa_descent_parity.json.

The research files are copied into WORK/research first and imported from
there. The research tree is only read. descent.py imports dimod,
dwave.samplers and scipy.sparse, which /usr/bin/python3 does not have. The
descent path uses none of dimod or dwave, and uses scipy.sparse only to build
CSR index arrays, so this tool installs small stand-ins for those modules.

Run from the repository root with:
    /usr/bin/python3 -I -B tools/msa_descent_reference.py --work .tmp/cu7/ref
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import types
from pathlib import Path

import numpy as np

RESEARCH = Path("/home/carback1/quip-data/regimes/new-pow")
FILES = {
    "descent.py": RESEARCH / "gpu/scripts/descent.py",
    "greedy.c": RESEARCH / "gpu/scripts/greedy.c",
    "libgreedy.so": RESEARCH / "gpu/scripts/libgreedy.so",
    "prep.py": RESEARCH / "gpu/scripts/prep.py",
    "prep_groups.py": RESEARCH / "gpu/scripts/prep_groups.py",
    "rescore.py": RESEARCH / "rescore.py",
}


class CsrMatrix:
    """The part of scipy.sparse.csr_matrix that descent.GreedyC and MoveSet read.

    Built from (data, (rows, cols)) in canonical form: entries sorted by row
    then column, duplicates summed, explicit zeros kept, as scipy does.
    """

    def __init__(self, arg, shape=None, dtype=None):
        data, (rows, cols) = arg
        data = np.asarray(data, dtype=dtype or np.asarray(data).dtype)
        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)
        self.shape = tuple(shape)
        self._coo = (data, rows, cols)
        n_rows, n_cols = self.shape
        if len(rows):
            keys = rows * n_cols + cols
            order = np.argsort(keys, kind="stable")
            keys, data = keys[order], data[order]
            unique, start = np.unique(keys, return_index=True)
            data = np.add.reduceat(data, start)
            rows, cols = unique // n_cols, unique % n_cols
        self.data = data
        self.indices = cols.astype(np.int32)
        self.indptr = np.searchsorted(rows, np.arange(n_rows + 1)).astype(np.int32)

    @property
    def T(self):  # noqa: N802  (scipy name)
        data, rows, cols = self._coo
        return CsrMatrix((data, (cols, rows)), shape=(self.shape[1], self.shape[0]))

    def tocsr(self):
        return self

    def __matmul__(self, other):
        raise NotImplementedError("the descent path never multiplies")


def install_stand_ins() -> None:
    scipy = types.ModuleType("scipy")
    sparse = types.ModuleType("scipy.sparse")
    sparse.csr_matrix = CsrMatrix
    scipy.sparse = sparse
    dimod = types.ModuleType("dimod")
    dimod.BinaryQuadraticModel = object
    dimod.SPIN = "SPIN"
    dwave = types.ModuleType("dwave")
    samplers = types.ModuleType("dwave.samplers")
    samplers.SimulatedAnnealingSampler = object
    samplers.SteepestDescentSolver = object
    dwave.samplers = samplers
    sys.modules.update({"scipy": scipy, "scipy.sparse": sparse, "dimod": dimod, "dwave": dwave,
                        "dwave.samplers": samplers})


def copy_research(dest: Path) -> dict[str, str]:
    dest.mkdir(parents=True, exist_ok=True)
    digests = {}
    for name, src in FILES.items():
        shutil.copyfile(src, dest / name)
        digests[name] = hashlib.sha256((dest / name).read_bytes()).hexdigest()
    return digests


def import_research(dest: Path):
    """Import the copies. prep, prep_groups and rescore load first, so the
    research sys.path edits in descent.py find them in sys.modules."""
    install_stand_ins()
    sys.path.insert(0, str(dest))
    import prep  # noqa: F401, PLC0415
    import prep_groups  # noqa: F401, PLC0415
    import rescore  # noqa: F401, PLC0415

    import descent  # noqa: PLC0415

    for name in ("prep", "prep_groups", "rescore", "descent"):
        origin = Path(sys.modules[name].__file__).resolve()
        if origin.parent != dest.resolve():
            raise RuntimeError(f"{name} imported from {origin}, not the copy")
    return descent


def run_one(D, at: Path) -> dict:
    spec = json.loads((at / "model.json").read_text())
    model = {
        "h_milli": np.asarray(spec["h_milli"], dtype=np.int64),
        "edges": np.asarray(spec["edges"], dtype=np.int64),
        "j_milli": np.asarray(spec["j_milli"], dtype=np.int64),
        "tetrahedra": np.asarray(spec["tetrahedra"], dtype=np.int64),
    }
    n = len(model["h_milli"])
    spins = np.fromfile(at / "reads.bin", dtype=np.int8).reshape(spec["reads"], n)
    ising = D.Ising(model)
    moves, counts = D.natural_moves(ising)
    gnat = D.GreedyC(ising, D.MoveSet(ising, moves))
    loop = D.LoopC(ising, gnat)
    raw = ising.energies(spins)
    nat, _ = gnat.descend(spins)
    i = int(raw.argmin())
    loop_s, rounds = D.nat_loop(ising, gnat, loop, spins[i:i + 1], 0, int(spec["loop_seed"]))
    nat_e = ising.energies(nat)
    loop_e = int(ising.energies(loop_s)[0])
    np.ascontiguousarray(nat, dtype=np.int8).tofile(at / "greedy.bin")
    np.ascontiguousarray(loop_s, dtype=np.int8).tofile(at / "loop.bin")
    result = {
        "greedy_energies": [int(e) for e in nat_e],
        "raw_best": int(raw.min()),
        "greedy_best": int(nat_e.min()),
        "loop_from": i,
        "loop_energy": loop_e,
        "loop_rounds": int(rounds),
        "moves": counts,
        "tiles": loop.counts,
        "tiled_spins": len(loop.tiled),
    }
    (at / "result.json").write_text(json.dumps(result))
    return result


def cargo(step: str, data: Path) -> None:
    subprocess.run(["cargo", "run", "--release", "--example", "msa_descent_reference", "--", step,
                    str(data)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work", type=Path, required=True, help="scratch directory, never the research tree")
    args = parser.parse_args()
    work = args.work.resolve()
    if RESEARCH.resolve() in (work, *work.parents):
        parser.exit(2, "the work directory must be outside the research tree\n")
    digests = copy_research(work / "research")
    print(json.dumps({"research_sha256": digests}, indent=1))
    data = work / "data"
    cargo("export", data)
    D = import_research(work / "research")
    for at in sorted(p for p in data.iterdir() if p.is_dir()):
        result = run_one(D, at)
        summary = {k: result[k] for k in ("raw_best", "greedy_best", "loop_from", "loop_energy", "loop_rounds",
                                          "tiles", "tiled_spins", "moves")}
        print(at.name, json.dumps(summary))
    cargo("record", data)


if __name__ == "__main__":
    main()
