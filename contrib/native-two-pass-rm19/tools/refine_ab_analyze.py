#!/usr/bin/env python3
"""Paired comparison of two refiner builds from tools/refine_ab_run.sh results.
usage: refine_ab_analyze.py BASE_ARM CAND_ARM   (run inside the directory that holds results/, start.json and chain.json)
 start.json: {salt: start energy_milli}   chain.json: {salt: energy_milli of the on-chain winner's state}"""
import json,glob,statistics as S,random,sys
base,cand=sys.argv[1],sys.argv[2]
st={int(k):v for k,v in json.load(open('start.json')).items()}; ch={int(k):v for k,v in json.load(open('chain.json')).items()}
def load(arm):
    d={}
    for f in glob.glob(f'results/{arm}_*.json'):
        p=f.split('/')[-1][:-5].split('_'); d[(int(p[1]),int(p[2]))]=json.load(open(f))
    return d
b,c=load(base),load(cand); keys=sorted(set(b)&set(c)); g=lambda a,k:(st[k[0]]-a[k]['final'])/1000.0
gb=[g(b,k) for k in keys]; gc=[g(c,k) for k in keys]; d=[y-x for x,y in zip(gb,gc)]
w=sum(1 for v in d if v>1e-9); l=sum(1 for v in d if v<-1e-9)
hit=lambda a:sum(1 for k in keys if a[k]['final']<=ch[k[0]])
salts=sorted(set(k[0] for k in keys)); per={s:S.mean([d[i] for i,k in enumerate(keys) if k[0]==s]) for s in salts}
random.seed(1); boots=sorted(S.mean([per[random.choice(salts)] for _ in salts]) for _ in range(4000))
print(f'pairs {len(keys)}  improved {sum(x>1e-9 for x in gb)}->{sum(x>1e-9 for x in gc)}  mean gain {S.mean(gb):.3f}->{S.mean(gc):.3f} ({S.mean(d):+.3f}, 95% CI over salts {boots[100]:+.3f}..{boots[3899]:+.3f})')
print(f'candidate deeper/shallower/tie {w}/{l}/{len(d)-w-l}; chain hits {hit(b)}->{hit(c)}')
