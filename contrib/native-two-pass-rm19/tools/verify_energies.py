# Independently recompute the energy of every returned spin vector from the saved instance and compare with the claimed final energy.
import json,glob,sys,numpy as np
import os
H=os.environ['INSTANCE_DIR'].rstrip('/')+'/win/'
inst={}
def load(s):
    if s not in inst:
        i=json.load(open(f'{H}{s}.inst.json')); e=np.array(i['edges']); inst[s]=(e[:,0],e[:,1],np.array(i['j_milli'],dtype=np.int64),np.array(i['h_milli'],dtype=np.int64),i['offset_milli'])
    return inst[s]
bad=0;n=0
for arm in sys.argv[1:]:
    for f in sorted(glob.glob(f'results/{arm}_*.json')):
        p=f.split('/')[-1][:-5].split('_'); s=int(p[1]); d=json.load(open(f)); sp=np.array(d['spins'],dtype=np.int64)
        u,v,j,h,off=load(s); E=int((j*sp[u]*sp[v]).sum()+(h*sp).sum()+off); n+=1
        if E!=d['final'] or set(np.unique(sp))-{-1,1}: bad+=1; print('MISMATCH',f,E,d['final'])
    print(arm,'checked',n,'mismatches',bad)
