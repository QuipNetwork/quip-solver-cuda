#!/bin/bash
# usage: INSTANCE_DIR=... [SEARCH_CONFIG=...] refine_ab_run.sh ARM BINARY SECONDS PAR -> results/ARM_<salt>_<seed>.json with final energy, info and the returned spins
ARM=$1; BIN=$2; SECS=${3:-8}; PAR=${4:-6}; H=${INSTANCE_DIR:?set INSTANCE_DIR (contains win/<salt>.inst.json and out2/<salt>.json)}; C=${SEARCH_CONFIG:-configs/example.json}
mkdir -p results
for s in ${SALTS:?set SALTS="69021 69147 ..."}; do for seed in ${SEEDS:-11 22 33 44 55 66 77 88}; do echo "$s $seed"; done; done | \
xargs -P $PAR -L 1 bash -c 'f=results/'"$ARM"'_$0_$1.json; [ -s $f ] || '"$BIN"' --search-config '"$C"' --refine-test '"$H"'/win/$0.inst.json '"$H"'/out2/$0.json --refine-seconds '"$SECS"' --refine-seed $1 2>/dev/null | python3 -c "import sys,json;d=json.loads(sys.stdin.readline());print(json.dumps({\"final\":d[\"final\"],\"info\":d[\"info\"],\"spins\":d[\"refinement\"][\"spins\"]}))" > $f'
