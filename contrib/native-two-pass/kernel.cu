#ifndef QUIP_TEMPERATURE
#define QUIP_TEMPERATURE 45.f
#endif
#ifndef QUIP_MINSUM_REPLICAS
#define QUIP_MINSUM_REPLICAS 42
#endif
#ifndef QUIP_DAMPING_BASE
#define QUIP_DAMPING_BASE .1f
#endif
#ifndef QUIP_DAMPING_STEP
#define QUIP_DAMPING_STEP .1f
#endif
#ifndef QUIP_PENALTY_BASE
#define QUIP_PENALTY_BASE 250.f
#endif
#ifndef QUIP_PENALTY_STEP
#define QUIP_PENALTY_STEP 125.f
#endif
#ifndef QUIP_SEEDED_RAMP_SCALE
#define QUIP_SEEDED_RAMP_SCALE 1.f
#endif
#ifndef QUIP_RAMP_BASE
#define QUIP_RAMP_BASE 3000.f
#endif
#ifndef QUIP_RAMP_STEP
#define QUIP_RAMP_STEP 1000.f
#endif
#ifndef QUIP_RHO_CAP
#define QUIP_RHO_CAP .95f
#endif
#ifdef QUIP_RAMP_POWER
__device__ __forceinline__ float native_rho(float t,float ramp){return fminf(QUIP_RHO_CAP,powf(fmaxf(t,0.f)/ramp,QUIP_RAMP_POWER));}
#else
__device__ __forceinline__ float native_rho(float t,float ramp){return fminf(QUIP_RHO_CAP,t/ramp);}
#endif
__device__ __forceinline__ float native_reinforcement_ramp(int replica,int initial_mode,float rb,float rs){return initial_mode?(3000.f+1000.f*(replica%6))*QUIP_SEEDED_RAMP_SCALE:rb+rs*(replica%6);}
// Local diagnostic: explicit physical-spin native-loop factors, all penalties soft.
// Exact small-group descent ported from the hybrid readout. One warp owns one
// physical read; groups are sequential, so weak bonds cannot create update races.
extern "C" __global__ void polish_physical_groups(const int*row,const int*neighbors,const int*weights,const int*fields,const int*groups,const int*starts,int count,int n,int passes,signed char*states){
 int lane=threadIdx.x;extern __shared__ signed char s[];
 __shared__ long long outer[4],inner[16];__shared__ int changed,mask_now,nodes[4],take;
 for(int u=lane;u<n;u+=32)s[u]=states[blockIdx.x*n+u];__syncwarp();
 for(int pass=0;pass<passes;pass++){
  if(lane==0)changed=0;__syncwarp();
  for(int g=0;g<count;g++){
   int k=starts[g+1]-starts[g];
   if(lane<k)nodes[lane]=groups[starts[g]+lane];
   if(lane<16)inner[lane]=0;__syncwarp();
   if(lane<k){int u=nodes[lane];long long value=fields[u];for(int e=row[u];e<row[u+1];e++){int v=neighbors[e],q=0;while(q<k&&nodes[q]!=v)q++;if(q==k)value+=(long long)weights[e]*s[v];else if(lane<q)inner[4*lane+q]+=weights[e];}outer[lane]=value;}
   __syncwarp();
   if(lane==0){int m=0;for(int p=0;p<k;p++)m|=(s[nodes[p]]>0)<<p;mask_now=m;}__syncwarp();
   long long value=0;
   if(lane<(1<<k)){for(int p=0;p<k;p++){int a=(lane>>p&1)?1:-1;value+=outer[p]*a;for(int q=p+1;q<k;q++)value+=inner[4*p+q]*a*((lane>>q&1)?1:-1);}}
   else value=0x3fffffffffffffffLL;
   long long old=__shfl_sync(0xffffffff,value,mask_now);int choice=lane;
   for(int step=16;step;step>>=1){long long other=__shfl_down_sync(0xffffffff,value,step);int at=__shfl_down_sync(0xffffffff,choice,step);if(lane+step<32&&(other<value||(other==value&&at<choice))){value=other;choice=at;}}
   if(lane==0){take=value<old?choice:-1;if(take>=0)changed=1;}__syncwarp();
   if(lane<k&&take>=0)s[nodes[lane]]=(take>>lane&1)?1:-1;__syncwarp();
  }
  if(!changed)break;
 }
 for(int u=lane;u<n;u+=32)states[blockIdx.x*n+u]=s[u];
}
__device__ unsigned long long next_rng(unsigned long long&x){x^=x<<13;x^=x>>7;x^=x<<17;return x;}
// A gauge-invariant fingerprint of the current logical spin pattern. The
// original energy-reduction scratch is reused; no extra shared allocation.
// Integer associativity is exact within the verified original-H bounds.
// All 512 lanes participate; two barriers replace the nine-level block tree.
__device__ __forceinline__ void native_energy_reduce(int value,int tid,int*sums){
#if QUIP_FAST_INTEGER_REDUCTION
 int lane=tid&31;for(int offset=16;offset;offset>>=1){int other=__shfl_down_sync(0xffffffff,value,offset);if(lane+offset<32)value+=other;}
 if(lane==0)sums[tid>>5]=value;__syncthreads();
 if(tid<32){int total=tid<16?sums[tid]:0;for(int offset=16;offset;offset>>=1){int other=__shfl_down_sync(0xffffffff,total,offset);if(lane+offset<32)total+=other;}if(tid==0)sums[0]=total;}
 __syncthreads();
#else
 sums[tid]=value;__syncthreads();for(int step=256;step;step>>=1){if(tid<step)sums[tid]+=sums[tid+step];__syncthreads();}
#endif
}
extern "C" __global__ void native_block_sum_probe(const int*input,int*output){__shared__ int sums[512];int tid=threadIdx.x;native_energy_reduce(input[blockIdx.x*512+tid],tid,sums);if(tid==0)output[blockIdx.x]=sums[0];}
__device__ __forceinline__ unsigned record_mix(unsigned x){x+=0x9e3779b9u;x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;return x^(x>>16);}
__device__ __forceinline__ int record_reduce(unsigned key,int tid,int*sums){sums[tid]=int(key);__syncthreads();for(int step=256;step;step>>=1){if(tid<step)sums[tid]^=sums[tid+step];__syncthreads();}return unsigned(sums[0])&31u;}
__device__ __forceinline__ int record_cell(const float*h,int n,int tid,int*sums){unsigned key=0;bool first=h[0]>0;for(int u=tid;u<n;u+=512)if((h[u]>0)!=first)key^=record_mix(unsigned(u));return record_reduce(key,tid,sums);}
extern "C" __global__ void native_record_cell_probe(const float*h,int n,int*out){__shared__ int sums[512];int cell=record_cell(h+blockIdx.x*n,n,threadIdx.x,sums);if(threadIdx.x==0)out[blockIdx.x]=cell;}
extern "C" __global__ void native_charge_cell_probe(const int*q,int n,int*out){__shared__ int sums[512];unsigned key=0;for(int t=threadIdx.x;t<n;t+=512)if(q[blockIdx.x*n+t]!=0)key^=record_mix(unsigned(t));int cell=record_reduce(key,threadIdx.x,sums);if(threadIdx.x==0)out[blockIdx.x]=cell;}
__device__ __forceinline__ int record_strong_reduce(int local,int tid,int*sums){sums[tid]=local;__syncthreads();for(int step=256;step;step>>=1){if(tid<step)sums[tid]+=sums[tid+step];__syncthreads();}return sums[0]<32?sums[0]:-1;}
extern "C" __global__ void native_strong_cost_probe(const int*q2,int n,int*out){__shared__ int sums[512];int local=0;for(int t=threadIdx.x;t<n;t+=512)local+=q2[blockIdx.x*n+t]/4;int cell=record_strong_reduce(local,threadIdx.x,sums);if(threadIdx.x==0)out[blockIdx.x]=cell;}
__device__ __forceinline__ float focusing(float cavity,float rho){
 const float temperature=QUIP_TEMPERATURE;rho=fminf(QUIP_RHO_CAP,fmaxf(0.f,rho));if(rho==0.f)return 0.f;
 float coupling=sqrtf(rho),bound=temperature*atanhf(coupling);
 if(!isfinite(cavity))return isnan(cavity)?0.f:copysignf(bound,cavity);
 return temperature*atanhf(coupling*tanhf(atanhf(coupling*tanhf(cavity/temperature))/(1.f-rho)));
}
__device__ void joint_message(const int*q2,const int*fixed,int k,const float*cavity,float weight,float*out){
 float pos[8],neg[8];
 for(int u=0;u<k;u++){pos[u]=1.e30f;neg[u]=1.e30f;}
 for(int mask=0;mask<(1<<k);mask++){
  float linear=0.f;for(int u=0;u<k;u++)linear+=cavity[u]*((mask>>u&1)?1.f:-1.f);
  float cost=weight*q2[mask]+fixed[mask]+linear;
  for(int u=0;u<k;u++){
   float value=cost-cavity[u]*((mask>>u&1)?1.f:-1.f);
   if(mask>>u&1)pos[u]=fminf(pos[u],value);else neg[u]=fminf(neg[u],value);
  }
 }
 for(int u=0;u<k;u++)out[u]=.5f*(pos[u]-neg[u]);
}

// One whole warp evaluates independent patterns of one joint region. The
// floating additions inside each pattern keep the serial order unchanged.
__device__ void warp_joint_message(const int*q2,const int*fixed,int k,const float*cavity,float weight,float*out){
 int lane=threadIdx.x&31;float pos[8],neg[8];
 for(int u=0;u<k;u++){pos[u]=1.e30f;neg[u]=1.e30f;}
 for(int mask=lane;mask<(1<<k);mask+=32){
  float linear=0.f;for(int u=0;u<k;u++)linear+=cavity[u]*((mask>>u&1)?1.f:-1.f);
  float cost=weight*q2[mask]+fixed[mask]+linear;
  for(int u=0;u<k;u++){float v=cost-cavity[u]*((mask>>u&1)?1.f:-1.f);if(mask>>u&1)pos[u]=fminf(pos[u],v);else neg[u]=fminf(neg[u],v);}
 }
 for(int u=0;u<k;u++){
  float p=pos[u],n=neg[u];for(int offset=16;offset;offset>>=1){p=fminf(p,__shfl_down_sync(0xffffffff,p,offset));n=fminf(n,__shfl_down_sync(0xffffffff,n,offset));}
  if(lane==0)out[u]=.5f*(p-n);
 }
}
extern "C" __global__ void region_message_probe(const int*q2,const int*fixed,int k,const float*cavity,float weight,float*out){
 warp_joint_message(q2,fixed,k,cavity,weight,out);
}

__device__ __forceinline__ void junction_cost_ab(const int*q2,const int*fixed,const int*fixed_b,float beta,const float*cavity,
 const int*dirs,int degree,const int*offsets,const int*maps,const float*incoming,float weight,float*cost);
__device__ __forceinline__ void junction_cost(const int*q2,const int*fixed,const float*cavity,
 const int*dirs,int degree,const int*offsets,const int*maps,const float*incoming,float weight,float*cost){
 junction_cost_ab(q2,fixed,fixed,0.f,cavity,dirs,degree,offsets,maps,incoming,weight,cost);
}
__device__ __forceinline__ void junction_cost_ab(const int*q2,const int*fixed,const int*fixed_b,float beta,const float*cavity,
 const int*dirs,int degree,const int*offsets,const int*maps,const float*incoming,float weight,float*cost){
 #pragma unroll
 for(int mask=0;mask<16;mask++){
  float linear=0.f;
  #pragma unroll
  for(int u=0;u<4;u++)linear+=cavity[u]*((mask>>u&1)?1.f:-1.f);
  float fx=beta>0.f?float(fixed[mask])+beta*float(fixed_b[mask]-fixed[mask]):float(fixed[mask]);
  float e=weight*q2[mask]+fx+linear;
  for(int k=0;k<degree;k++)e+=incoming[offsets[dirs[k]^1]+maps[k*16+mask]];
  cost[mask]=e;
 }
}
__device__ __forceinline__ void separator_minimum(const float*cost,const float*incoming,const int*map,int size,float*out){
 int mp[16];
 #pragma unroll
 for(int mask=0;mask<16;mask++)mp[mask]=map[mask];
 #pragma unroll
 for(int v=0;v<8;v++){
  if(v<size){float m=1.e30f;
   #pragma unroll
   for(int mask=0;mask<16;mask++)if(mp[mask]==v)m=fminf(m,cost[mask]-incoming[v]);
   out[v]=m;}
 }
 float normal=out[0];for(int v=1;v<size;v++)normal=fminf(normal,out[v]);
 for(int v=0;v<size;v++)out[v]-=normal;
}
extern "C" __global__ void junction_message_probe(const int*q2,const int*fixed,const float*cavity,
 const int*dirs,int degree,const int*offsets,const int*maps,const float*incoming,float weight,float*out){
 if(threadIdx.x)return;
 float cost[16];junction_cost(q2,fixed,cavity,dirs,degree,offsets,maps,incoming,weight,cost);
 for(int u=0;u<4;u++){
  float pos=1.e30f,neg=1.e30f;
  for(int mask=0;mask<16;mask++){float v=cost[mask]-cavity[u]*((mask>>u&1)?1.f:-1.f);if(mask>>u&1)pos=fminf(pos,v);else neg=fminf(neg,v);}
  out[u]=.5f*(pos-neg);
 }
 int at=4;
 for(int k=0;k<degree;k++){int d=dirs[k],size=offsets[d+1]-offsets[d];float target[8];separator_minimum(cost,incoming+offsets[d^1],maps+16*k,size,target);for(int v=0;v<size;v++)out[at++]=target[v];}
}
// Finite-temperature counterparts, sharing the focusing replica temperature.
// Stable conditional log-sum-exp; no physical coefficients or scores change.
__device__ float native_thermal_pair(float j,float cavity,float temperature){
 float a=fabsf((cavity+j)/temperature),b=fabsf((cavity-j)/temperature);
 return -.5f*temperature*((a-b)+log1pf(expf(-2.f*a))-log1pf(expf(-2.f*b)));
}
__device__ float native_thermal_single(const float*cost,int target,float cavity,float temperature){
 float minimum[2]={1.e30f,1.e30f},z[2]={0.f,0.f};
 for(int mask=0;mask<16;mask++){int s=(mask>>target)&1;minimum[s]=fminf(minimum[s],cost[mask]);}
 for(int mask=0;mask<16;mask++){int s=(mask>>target)&1;z[s]+=expf(-(cost[mask]-minimum[s])/temperature);}
 return .5f*(minimum[1]-minimum[0]-temperature*logf(z[1]/z[0]))-cavity;
}
__device__ void native_thermal_separator(const float*cost,const float*incoming,const int*map,int size,float temperature,float*out){
 float z[8];for(int v=0;v<size;v++){out[v]=1.e30f;z[v]=0.f;}
 for(int mask=0;mask<16;mask++){int v=map[mask];out[v]=fminf(out[v],cost[mask]-incoming[v]);}
 for(int mask=0;mask<16;mask++){int v=map[mask];z[v]+=expf(-(cost[mask]-incoming[v]-out[v])/temperature);}
 float minimum=1.e30f;for(int v=0;v<size;v++){out[v]-=temperature*logf(z[v]);minimum=fminf(minimum,out[v]);}
 for(int v=0;v<size;v++)out[v]-=minimum;
}
extern "C" __global__ void native_thermal_probe(const float*cost,const float*cavity,const float*incoming,const int*map,int size,float j,float pair_cavity,float temperature,float*out){
 if(threadIdx.x)return;for(int u=0;u<4;u++)out[u]=native_thermal_single(cost,u,cavity[u],temperature);
 out[4]=native_thermal_pair(j,pair_cavity,temperature);native_thermal_separator(cost,incoming,map,size,temperature,out+5);
}
__device__ __forceinline__ float native_pair_biased(float j,float own,float neighbor,float cavity){return own-(j>0?1.f:-1.f)*fminf(fabsf(j),fmaxf(-fabsf(j),cavity+neighbor));}
// MPLP factor-block redistribution (min-sum convention). The BP message
// excludes this variable's own cavity; divide the full min-marginal by arity.
__device__ __forceinline__ float native_mplp_message(float bp,float cavity,int arity){return (bp-(arity-1)*cavity)/arity;}
extern "C" __global__ void native_mplp_probe(const int*q2,const int*fixed,const float*cavity,const float*weights,const float*pair_j,int count,float*out){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;const float*c=cavity+4*i;float cost[16];
 junction_cost(q2+16*i,fixed+16*i,c,nullptr,0,nullptr,nullptr,nullptr,weights[i],cost);
 for(int u=0;u<4;u++){float pos=1.e30f,neg=1.e30f;for(int mask=0;mask<16;mask++){float v=cost[mask]-c[u]*((mask>>u&1)?1.f:-1.f);if(mask>>u&1)pos=fminf(pos,v);else neg=fminf(neg,v);}out[6*i+u]=native_mplp_message(.5f*(pos-neg),c[u],4);}
 out[6*i+4]=native_mplp_message(native_pair_biased(pair_j[i],0.f,0.f,c[1]),c[0],2);
 out[6*i+5]=native_mplp_message(native_pair_biased(pair_j[i],0.f,0.f,c[0]),c[1],2);
}
// Fractional pair update: h includes rho-weighted incoming messages;
// subtract the unweighted reverse message and scale the pair potential by rho.
// This local variant is not a certified global tree-reweighted bound.
__device__ __forceinline__ float native_fractional_pair(float j,float own,float neighbor,float h,float old,float appearance,bool thermal,float temperature){
 float cavity=h-old/appearance+neighbor/appearance,scaled=j/appearance;
 float message=thermal?native_thermal_pair(scaled,cavity,temperature):-(scaled>0?1.f:-1.f)*fminf(fabsf(scaled),fmaxf(-fabsf(scaled),cavity));
 return own+appearance*message;
}
extern "C" __global__ void native_fractional_pair_probe(const float*input,int count,float*out){int u=blockIdx.x*blockDim.x+threadIdx.x;if(u<count){const float*x=input+8*u;out[u]=native_fractional_pair(x[0],x[1],x[2],x[3],x[4],x[5],x[6]!=0.f,x[7]);}}
extern "C" __global__ void native_pair_bias_probe(const float*input,int n,float*out){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){const float*x=input+5*i;out[2*i]=native_pair_biased(x[0],x[1],x[2],x[4]);out[2*i+1]=native_pair_biased(x[0],x[2],x[1],x[3]);}}
// Seed fields affect initialization only; the original physical score is unchanged.
__device__ __forceinline__ float native_seed_initial(int replica,int bias,float noise){return (replica%3==2?0.f:bias*(replica%2?1.f:10.f))+noise;}
extern "C" __global__ void native_seed_initial_probe(const int*bias,const float*noise,int n,float*out){int u=blockIdx.x*blockDim.x+threadIdx.x;if(u<n)out[u]=native_seed_initial(u,bias[u],noise[u]);}
__device__ __forceinline__ int native_sweep_color(int step,int colors,int iter,int policy){return (policy&1)&&(iter&1)?colors-1-step:step;}
__device__ __forceinline__ float native_region_weight(int replica,int iter,int policy){int stage=(policy&2)?(iter*7/6000):0;return QUIP_PENALTY_BASE+QUIP_PENALTY_STEP*((replica+stage)%7);}
// Interleave complete conflict-free colour groups. Every pair and native
// colour is still updated exactly once per iteration; no cross-type overlap.
__device__ __forceinline__ int native_message_stages(int colors,int policy){return (policy&8)&&colors>0?colors:1;}
__device__ __forceinline__ int2 native_message_bounds(int stage,int ec,int colors,int policy,bool pairs){
 bool mixed=(policy&8)&&colors>0;if(pairs)return mixed?make_int2(stage*ec/colors,(stage+1)*ec/colors):make_int2(0,ec);
 return mixed?make_int2(stage,stage+1):make_int2(0,colors);
}
extern "C" __global__ void native_message_schedule_probe(int*out){
 if(threadIdx.x)return;int code=blockIdx.x,ec=code%64;code/=64;int colors=code%33;code/=33;int policy=(code%2?8:0)+(code/2%2?1:0),iter=code/4;int at=0;
 for(int stage=0;stage<native_message_stages(colors,policy);stage++){
  int2 e=native_message_bounds(stage,ec,colors,policy,true),q=native_message_bounds(stage,ec,colors,policy,false);
  for(int i=e.x;i<e.y;i++)out[blockIdx.x*96+at++]=native_sweep_color(i,ec,iter,policy);
  for(int i=q.x;i<q.y;i++)out[blockIdx.x*96+at++]=-1-native_sweep_color(i,colors,iter,policy);
 }
 for(;at<96;at++)out[blockIdx.x*96+at]=-999;
}
extern "C" __global__ void native_penalty_cycle_probe(float*out){int iter=blockIdx.x%6000,policy=blockIdx.x/6000?2:0,r=threadIdx.x;out[blockIdx.x*128+r]=native_region_weight(r,iter,policy);}
extern "C" __global__ void native_sweep_color_probe(int*out){int colors=blockIdx.x%64+1,iter=(blockIdx.x/64)%8,policy=blockIdx.x/512,step=threadIdx.x;out[blockIdx.x*64+step]=step<colors?native_sweep_color(step,colors,iter,policy):-1;}
__device__ __forceinline__ void native_seed_strong_defects(const int*eu,const int*ev,const int*ej,int edges,const float*h,int*mask,int tid,int stride){
 for(int e=tid;e<edges;e+=stride){int u=eu[e],v=ev[e],j=ej[e];int su=h[u]>0.f?-1:1,sv=h[v]>0.f?-1:1;if(u!=v&&abs(j)>=1000&&j*su*sv>0){atomicOr(mask+u,1);atomicOr(mask+v,1);}}
}
extern "C" __global__ void native_strong_defect_seed_probe(const int*eu,const int*ev,const int*ej,int edges,const float*h,int n,int*mask){native_seed_strong_defects(eu,ev,ej,edges,h+blockIdx.x*n,mask+blockIdx.x*n,threadIdx.x,blockDim.x);}
// Revisit a selected decision without altering its incoming factor-message sum.
__device__ __forceinline__ void native_backtrack_field(float&h,float&b,int strength){float cavity=h-b;float amplitude=fabsf(cavity)+float(strength);b=h>0.f?-amplitude:amplitude;h=cavity+b;}
extern "C" __global__ void native_backtrack_field_probe(const float*input,int count,int strength,float*out){int u=blockIdx.x*blockDim.x+threadIdx.x;if(u<count){float h=input[2*u],b=input[2*u+1];native_backtrack_field(h,b,strength);out[2*u]=h;out[2*u+1]=b;}}
__device__ __forceinline__ float fsel4(float4 a,int i){return i==0?a.x:(i==1?a.y:(i==2?a.z:a.w));}

// Thread-count generic copies of the reductions used by loop_factor_bp (QUIP_NT threads per block;
// integer / xor reductions are exact in any order, so QUIP_NT=256 gives the same values as 512).
#ifndef QUIP_NT
#define QUIP_NT 512
#endif
__device__ __forceinline__ void native_energy_reduce_nt(int value,int tid,int*sums){
#if QUIP_FAST_INTEGER_REDUCTION
 int lane=tid&31;for(int offset=16;offset;offset>>=1){int other=__shfl_down_sync(0xffffffff,value,offset);if(lane+offset<32)value+=other;}
 if(lane==0)sums[tid>>5]=value;__syncthreads();
 if(tid<32){int total=tid<(QUIP_NT/32)?sums[tid]:0;for(int offset=16;offset;offset>>=1){int other=__shfl_down_sync(0xffffffff,total,offset);if(lane+offset<32)total+=other;}if(tid==0)sums[0]=total;}
 __syncthreads();
#else
 sums[tid]=value;__syncthreads();for(int step=QUIP_NT/2;step;step>>=1){if(tid<step)sums[tid]+=sums[tid+step];__syncthreads();}
#endif
}
__device__ __forceinline__ int record_reduce_nt(unsigned key,int tid,int*sums){sums[tid]=int(key);__syncthreads();for(int step=QUIP_NT/2;step;step>>=1){if(tid<step)sums[tid]^=sums[tid+step];__syncthreads();}return unsigned(sums[0])&31u;}
__device__ __forceinline__ int record_cell_nt(const float*h,int n,int tid,int*sums){unsigned key=0;bool first=h[0]>0;for(int u=tid;u<n;u+=QUIP_NT)if((h[u]>0)!=first)key^=record_mix(unsigned(u));return record_reduce_nt(key,tid,sums);}
__device__ __forceinline__ int record_strong_reduce_nt(int local,int tid,int*sums){sums[tid]=local;__syncthreads();for(int step=QUIP_NT/2;step;step>>=1){if(tid<step)sums[tid]+=sums[tid+step];__syncthreads();}return sums[0]<32?sums[0]:-1;}
extern "C" __global__ __launch_bounds__(QUIP_NT,512/QUIP_NT) void loop_factor_bp(
 const int*eu,const int*ev,const int*ej,const int*eo,int ec,int edges,
 const int*rn,const int*roff,const int*taboff,const int*q2,const int*fixed,
 const int*rc,int colors,int regions,int slots,const int*fields,int n,int constant,
 int start,int count,float*gh,float*gb,float2*em,float*rm,int*best,signed char*states,
 const unsigned long long*seed_ptr, const int*physical, const int*jo,const int*jdirs,const int*joffset,const int*jmaps,int jtables,float*jm,int reinforcement_epoch,int release_mode,int kick_strength,int*unpin_mask,int guided_mode,int*edge_penalties,int*last_best,int thermal_messages,int record_slots,int*record_energies,signed char*record_states,int initial_mode,const int*initial_bias,int edge_unary_mode,const int*edge_unary,int record_descriptor,int sweep_policy,const int*working_ej,const int*working_fixed,const int*working_fields,float ramp_base_arg,float ramp_step_arg,float*adapt_state,int weak_sets,int weak_stride){
 int r=blockIdx.x,tid=threadIdx.x;extern __shared__ float smem[];const bool mplp=(sweep_policy&16)!=0;
 const int wk=weak_sets>1?(blockIdx.x%weak_sets):0;working_ej+=wk*edges;working_fixed+=wk*weak_stride;working_fields+=wk*n;
 const bool wsched=weak_sets<0;const int*wej_b=working_ej+edges;const int*wfx_b=working_fixed+weak_stride;
 const bool seed_strong_defects=release_mode==5;const bool external_backtrack=release_mode==7;const bool fixed_boundary=release_mode==8;if(seed_strong_defects||external_backtrack)release_mode=2;if(fixed_boundary)release_mode=1;
 if(colors>32)return; // Offline input guard; current prepared models use3/4.
 __shared__ int split[32];
 if(tid<colors){int t=rc[tid];while(t<rc[tid+1]&&roff[t+1]-roff[t]>4)t++;split[tid]=t;}
 float*h=smem,*b=h+n;int*sums=(int*)(b+n);
 unsigned long long rng=*seed_ptr^(0x9e3779b97f4a7c15ULL*(r+1))^(tid+1);
 if(start==0){
  if(guided_mode){for(int e=tid;e<edges;e+=QUIP_NT)edge_penalties[r*edges+e]=0;if(tid==0)last_best[r]=2147483647;}
#if QUIP_NT==512
  for(int u=tid;u<n;u+=512){float noise=((next_rng(rng)>>40)*(1.f/16777216.f)-.5f);b[u]=initial_mode==3&&unpin_mask[r*n+u]==0?float(initial_bias[r*n+u]):initial_mode?native_seed_initial(r,initial_bias[(initial_mode>=2?r*n:0)+u],noise):noise;h[u]=working_fields[u]+b[u];}
#else
  for(int vt=tid;vt<512;vt+=QUIP_NT){unsigned long long rng=*seed_ptr^(0x9e3779b97f4a7c15ULL*(r+1))^(vt+1);for(int u=vt;u<n;u+=512){float noise=((next_rng(rng)>>40)*(1.f/16777216.f)-.5f);b[u]=initial_mode==3&&unpin_mask[r*n+u]==0?float(initial_bias[r*n+u]):initial_mode?native_seed_initial(r,initial_bias[(initial_mode>=2?r*n:0)+u],noise):noise;h[u]=working_fields[u]+b[u];}}
#endif
  for(int e=tid;e<edges;e+=QUIP_NT)em[r*edges+e]=make_float2(0,0);
  for(int e=tid;e<slots;e+=QUIP_NT)rm[r*slots+e]=0.f;
  for(int e=tid;e<jtables;e+=QUIP_NT)jm[r*jtables+e]=0.f;
 }else for(int u=tid;u<n;u+=QUIP_NT){h[u]=gh[r*n+u];b[u]=gb[r*n+u];}
 if(start==0&&tid==0){adapt_state[r*3]=0.f;adapt_state[r*3+1]=0.f;adapt_state[r*3+2]=0.f;}
 __syncthreads();float damping=QUIP_DAMPING_BASE+QUIP_DAMPING_STEP*(r%6),weight=native_region_weight(r,start,sweep_policy);
#ifndef QUIP_NO_FAST_SWEEPS
 // Fast sweeps: same arithmetic in the same order as the generic loops below, with wide loads,
 // register-resident 4-entry junction tables and the next edge colour prefetched while waiting at the barrier.
 const bool fast_edges=!guided_mode&&!edge_unary_mode&&!(sweep_policy&(4|8|16))&&!(thermal_messages&&r>=QUIP_MINSUM_REPLICAS);
 __shared__ int fast_reg_ok;
 if(tid==0)fast_reg_ok=(slots==4*regions)&&(jtables%4==0)&&!((((unsigned long long)rm)|((unsigned long long)rn)|((unsigned long long)physical)|((unsigned long long)q2)|((unsigned long long)working_fixed)|((unsigned long long)wfx_b)|((unsigned long long)jm)|((unsigned long long)jmaps))&15ULL);
 __syncthreads();
 for(int t=tid;t<=regions;t+=QUIP_NT)if(roff[t]!=4*t||taboff[t]!=16*t)fast_reg_ok=0;
 __syncthreads();
 const bool fast_regions=fast_reg_ok&&!mplp&&!(thermal_messages&&r>=QUIP_MINSUM_REPLICAS);
#endif
 for(int iter=start;iter<start+count;iter++){
  if(sweep_policy&2)weight=native_region_weight(r,iter,sweep_policy);
  float wbeta=wsched?fmaxf(0.f,1.f-float(iter)/ramp_base_arg):0.f;
  float damping_now=damping;float weight_now=weight;
#ifdef QUIP_DAMP_DECAY
  damping_now=damping_now*(1.f-QUIP_DAMP_DECAY*(float(iter)/6000.f));
#endif
#ifdef QUIP_WEIGHT_GROWTH
  weight_now=weight*(1.f+QUIP_WEIGHT_GROWTH*(float(iter)/6000.f));
#endif
  float temp_now=QUIP_TEMPERATURE;
#ifdef QUIP_TEMP_ANNEAL
  if(initial_mode==0){float frac=fminf(1.f,float(iter)/ramp_base_arg);temp_now=QUIP_TEMPERATURE*(1.f+QUIP_TEMP_ANNEAL*(1.f-frac));}
#endif
  if(reinforcement_epoch>0&&iter==reinforcement_epoch){
   if(release_mode>=2&&!external_backtrack){
    for(int u=tid;u<n;u+=QUIP_NT)unpin_mask[r*n+u]=0;__syncthreads();
    for(int t=tid;t<regions;t+=QUIP_NT){int mask=0;for(int u=roff[t];u<roff[t+1];u++)mask|=(h[rn[u]]<=0)<<(u-roff[t]);if(q2[taboff[t]+mask]!=0){for(int u=roff[t];u<roff[t+1];u++)atomicOr(unpin_mask+r*n+rn[u],1);}}__syncthreads();
    if(seed_strong_defects)native_seed_strong_defects(eu,ev,ej,edges,h,unpin_mask+r*n,tid,QUIP_NT);__syncthreads();
    for(int e=tid;e<edges;e+=QUIP_NT)if(abs(ej[e])>=1000){int u=eu[e],v=ev[e];if(atomicOr(unpin_mask+r*n+u,0)&1)atomicOr(unpin_mask+r*n+v,2);if(atomicOr(unpin_mask+r*n+v,0)&1)atomicOr(unpin_mask+r*n+u,2);}__syncthreads();
   }
   for(int u=tid;u<n;u+=QUIP_NT)if(release_mode<2||unpin_mask[r*n+u]!=0){if(external_backtrack)native_backtrack_field(h[u],b[u],kick_strength);else{h[u]-=b[u];b[u]=0.f;}}__syncthreads();
   if(release_mode>=3){
    for(int c=0;c<ec;c++){for(int e=eo[c]+tid;e<eo[c+1];e+=QUIP_NT){int u=eu[e],v=ev[e];if(unpin_mask[r*n+u]||unpin_mask[r*n+v]){float2 old=em[r*edges+e];h[u]-=old.x;h[v]-=old.y;em[r*edges+e]=make_float2(0.f,0.f);}}__syncthreads();}
    for(int c=0;c<colors;c++){for(int t=rc[c]+tid;t<rc[c+1];t+=QUIP_NT){int at=roff[t];bool clear=false;for(int u=at;u<roff[t+1];u++)clear|=unpin_mask[r*n+rn[u]]!=0;if(clear){for(int u=at;u<roff[t+1];u++){if(r<QUIP_MINSUM_REPLICAS||physical[u])h[rn[u]]-=rm[r*slots+u];rm[r*slots+u]=0.f;}if(r>=QUIP_MINSUM_REPLICAS)for(int k=jo[t];k<jo[t+1];k++){int d=jdirs[k];for(int v=joffset[d];v<joffset[d+1];v++)jm[r*jtables+v]=0.f;}}}__syncthreads();}
    if(release_mode==4){for(int u=tid;u<n;u+=QUIP_NT){gh[r*n+u]=h[u];gb[r*n+u]=b[u];}return;}
   }
  }
  for(int stage=0;stage<native_message_stages(colors,sweep_policy);stage++){
  int2 pair_bounds=native_message_bounds(stage,ec,colors,sweep_policy,true),region_bounds=native_message_bounds(stage,ec,colors,sweep_policy,false);
#ifndef QUIP_NO_FAST_SWEEPS
  if(fast_edges){
   int fU[3],fV[3];float fJ[3];float2 fO[3];int cb=0,cn=0;
#define FE_LOAD() do{_Pragma("unroll") for(int k=0;k<3;k++){int e=cb+tid+QUIP_NT*k;if(e<cn){fU[k]=__ldg(eu+e);fV[k]=__ldg(ev+e);int wj=__ldg(working_ej+e);fJ[k]=wbeta>0.f?float(wj)+wbeta*float(__ldg(wej_b+e)-wj):float(wj);fO[k]=em[r*edges+e];}}}while(0)
   if(pair_bounds.x<pair_bounds.y){int c=native_sweep_color(pair_bounds.x,ec,iter,sweep_policy);cb=eo[c];cn=eo[c+1];FE_LOAD();}
   for(int sweep=pair_bounds.x;sweep<pair_bounds.y;sweep++){
    #pragma unroll
    for(int k=0;k<3;k++){int e=cb+tid+QUIP_NT*k;if(e<cn){
     int u=fU[k],v=fV[k];float j=fJ[k];float2 old=fO[k];float hu=h[u]-old.x,hv=h[v]-old.y;
     float mu=-(j>0?1.f:-1.f)*fminf(fabsf(j),fmaxf(-fabsf(j),hv));
     float mv=-(j>0?1.f:-1.f)*fminf(fabsf(j),fmaxf(-fabsf(j),hu));
     float a=old.x+damping_now*(mu-old.x),bb=old.y+damping_now*(mv-old.y);
     h[u]+=a-old.x;h[v]+=bb-old.y;em[r*edges+e]=make_float2(a,bb);}}
    for(int e=cb+tid+3*QUIP_NT;e<cn;e+=QUIP_NT){
     int u=eu[e],v=ev[e];float j=wbeta>0.f?float(working_ej[e])+wbeta*float(wej_b[e]-working_ej[e]):float(working_ej[e]);float2 old=em[r*edges+e];float hu=h[u]-old.x,hv=h[v]-old.y;
     float mu=-(j>0?1.f:-1.f)*fminf(fabsf(j),fmaxf(-fabsf(j),hv));
     float mv=-(j>0?1.f:-1.f)*fminf(fabsf(j),fmaxf(-fabsf(j),hu));
     float a=old.x+damping_now*(mu-old.x),bb=old.y+damping_now*(mv-old.y);
     h[u]+=a-old.x;h[v]+=bb-old.y;em[r*edges+e]=make_float2(a,bb);}
    if(sweep+1<pair_bounds.y){int c=native_sweep_color(sweep+1,ec,iter,sweep_policy);cb=eo[c];cn=eo[c+1];FE_LOAD();}
    __syncthreads();
   }
#undef FE_LOAD
  }else
#endif
  for(int sweep=pair_bounds.x;sweep<pair_bounds.y;sweep++){int c=native_sweep_color(sweep,ec,iter,sweep_policy);
   for(int e=eo[c]+tid;e<eo[c+1];e+=QUIP_NT){
    int u=eu[e],v=ev[e];float j=wbeta>0.f?float(working_ej[e])+wbeta*float(wej_b[e]-working_ej[e]):float(working_ej[e]);if(guided_mode&&abs(ej[e])<1000)j*=1.f+edge_penalties[r*edges+e];float2 old=em[r*edges+e];float hu=h[u]-old.x,hv=h[v]-old.y;
    float mu=!mplp&&thermal_messages&&r>=QUIP_MINSUM_REPLICAS?native_thermal_pair(j,hv,temp_now):-(j>0?1.f:-1.f)*fminf(fabsf(j),fmaxf(-fabsf(j),hv));
    float mv=!mplp&&thermal_messages&&r>=QUIP_MINSUM_REPLICAS?native_thermal_pair(j,hu,temp_now):-(j>0?1.f:-1.f)*fminf(fabsf(j),fmaxf(-fabsf(j),hu));
    if(edge_unary_mode){float au=edge_unary[2*e],av=edge_unary[2*e+1];mu=!mplp&&thermal_messages&&r>=QUIP_MINSUM_REPLICAS?au+native_thermal_pair(j,hv+av,temp_now):native_pair_biased(j,au,av,hv);mv=!mplp&&thermal_messages&&r>=QUIP_MINSUM_REPLICAS?av+native_thermal_pair(j,hu+au,temp_now):native_pair_biased(j,av,au,hu);}
    if((sweep_policy&4)&&abs(ej[e])<1000){float au=edge_unary_mode?edge_unary[2*e]:0.f,av=edge_unary_mode?edge_unary[2*e+1]:0.f;const float appearance=.5f;
     mu=native_fractional_pair(j,au,av,h[u],old.x,appearance,thermal_messages&&r>=QUIP_MINSUM_REPLICAS,temp_now);
     mv=native_fractional_pair(j,av,au,h[v],old.y,appearance,thermal_messages&&r>=QUIP_MINSUM_REPLICAS,temp_now);
    }
    if(mplp){mu=native_mplp_message(mu,hu,2);mv=native_mplp_message(mv,hv,2);}
    float a=old.x+damping_now*(mu-old.x),bb=old.y+damping_now*(mv-old.y);
    h[u]+=a-old.x;h[v]+=bb-old.y;em[r*edges+e]=make_float2(a,bb);
   }__syncthreads();
  }
  for(int sweep=region_bounds.x;sweep<region_bounds.y;sweep++){int c=native_sweep_color(sweep,colors,iter,sweep_policy);
   for(int t=rc[c]+tid;t<rc[c+1];t+=QUIP_NT){
#ifndef QUIP_NO_FAST_SWEEPS
    if(fast_regions){
     const bool uj=!mplp&&r>=QUIP_MINSUM_REPLICAS;const int at=4*t;
     const int degree=uj?jo[t+1]-jo[t]:0;const int j0=uj?jo[t]:0;
     const float4 o4=*(const float4*)(rm+r*slots+at);
     const int4 n4=__ldg((const int4*)(rn+at));
     const int4 p4=uj?__ldg((const int4*)(physical+at)):make_int4(1,1,1,1);
     const float fo[4]={o4.x,o4.y,o4.z,o4.w};const int fnn[4]={n4.x,n4.y,n4.z,n4.w};const int fpp[4]={p4.x,p4.y,p4.z,p4.w};
     float cav[4];
     #pragma unroll
     for(int u=0;u<4;u++)cav[u]=(!uj||fpp[u])?h[fnn[u]]-fo[u]:0.f;
     int qv[16],fv[16];
     #pragma unroll
     for(int i=0;i<4;i++){int4 a=__ldg((const int4*)(q2+16*t)+i);qv[4*i]=a.x;qv[4*i+1]=a.y;qv[4*i+2]=a.z;qv[4*i+3]=a.w;int4 b2=__ldg((const int4*)(working_fixed+16*t)+i);fv[4*i]=b2.x;fv[4*i+1]=b2.y;fv[4*i+2]=b2.z;fv[4*i+3]=b2.w;}
     float cost[16];
     #pragma unroll
     for(int mask=0;mask<16;mask++){
      float linear=0.f;
      #pragma unroll
      for(int u=0;u<4;u++)linear+=cav[u]*((mask>>u&1)?1.f:-1.f);
      float fx=wbeta>0.f?float(fv[mask])+wbeta*float(__ldg(wfx_b+16*t+mask)-fv[mask]):float(fv[mask]);
      cost[mask]=weight_now*qv[mask]+fx+linear;
     }
     for(int k=0;k<degree;k++){
      const int kk=j0+k;const int d=__ldg(jdirs+kk);const int bi=__ldg(joffset+(d^1));const int si=__ldg(joffset+(d^1)+1)-bi;
      const float*inc=jm+r*jtables+bi;const int*mp=jmaps+16*kk;
      if(si==4&&!(bi&3)){
       const float4 in4=*(const float4*)inc;
       #pragma unroll
       for(int i=0;i<4;i++){int4 m=__ldg((const int4*)mp+i);cost[4*i]+=fsel4(in4,m.x);cost[4*i+1]+=fsel4(in4,m.y);cost[4*i+2]+=fsel4(in4,m.z);cost[4*i+3]+=fsel4(in4,m.w);}
      }else{
       #pragma unroll
       for(int mask=0;mask<16;mask++)cost[mask]+=inc[__ldg(mp+mask)];
      }
     }
     #pragma unroll
     for(int u=0;u<4;u++)if(!uj||fpp[u]){
      float pos=1.e30f,neg=1.e30f;
      #pragma unroll
      for(int mask=0;mask<16;mask++){float v=cost[mask]-cav[u]*((mask>>u&1)?1.f:-1.f);if(mask>>u&1)pos=fminf(pos,v);else neg=fminf(neg,v);}
      float message=.5f*(pos-neg);float next=fo[u]+damping_now*(message-fo[u]);h[fnn[u]]+=next-fo[u];rm[r*slots+at+u]=next;
     }
     for(int k=0;k<degree;k++){
      const int kk=j0+k;const int d=__ldg(jdirs+kk);const int bo=__ldg(joffset+d),size=__ldg(joffset+d+1)-bo;const int bi=__ldg(joffset+(d^1));
      const float*inc=jm+r*jtables+bi;const int*mp=jmaps+16*kk;
      if(size==4&&!(bo&3)&&!(bi&3)){
       const float4 in4=*(const float4*)inc;const float inv[4]={in4.x,in4.y,in4.z,in4.w};
       int mv[16];
       #pragma unroll
       for(int i=0;i<4;i++){int4 m=__ldg((const int4*)mp+i);mv[4*i]=m.x;mv[4*i+1]=m.y;mv[4*i+2]=m.z;mv[4*i+3]=m.w;}
       float o[4];
       #pragma unroll
       for(int v=0;v<4;v++){float m=1.e30f;
        #pragma unroll
        for(int mask=0;mask<16;mask++)if(mv[mask]==v)m=fminf(m,cost[mask]-inv[v]);
        o[v]=m;}
       float normal=o[0];normal=fminf(normal,o[1]);normal=fminf(normal,o[2]);normal=fminf(normal,o[3]);
       #pragma unroll
       for(int v=0;v<4;v++)o[v]-=normal;
       float4*jp=(float4*)(jm+r*jtables+bo);float4 ol=*jp;
       ol.x=ol.x+damping_now*(o[0]-ol.x);ol.y=ol.y+damping_now*(o[1]-ol.y);ol.z=ol.z+damping_now*(o[2]-ol.z);ol.w=ol.w+damping_now*(o[3]-ol.w);
       *jp=ol;
      }else{
       float target[8];separator_minimum(cost,inc,mp,size,target);
       for(int v=0;v<size;v++){int idx=r*jtables+bo+v;float old=jm[idx];jm[idx]=old+damping_now*(target[v]-old);}
      }
     }
     continue;
    }
#endif
    bool use_joint=!mplp&&r>=QUIP_MINSUM_REPLICAS;int at=roff[t],degree=use_joint?jo[t+1]-jo[t]:0;float cavity[4],old[4],cost[16];
    for(int u=0;u<4;u++){old[u]=rm[r*slots+at+u];cavity[u]=(!use_joint||physical[at+u])?h[rn[at+u]]-old[u]:0.f;}
    junction_cost_ab(q2+taboff[t],working_fixed+taboff[t],wfx_b+taboff[t],wbeta,cavity,jdirs+jo[t],degree,joffset,jmaps+16*jo[t],jm+r*jtables,weight_now,cost);
    // All outputs use the SAME pre-update incoming tables and cavities.
    for(int u=0;u<4;u++)if(!use_joint||physical[at+u]){
     float pos=1.e30f,neg=1.e30f;
     for(int mask=0;mask<16;mask++){float v=cost[mask]-cavity[u]*((mask>>u&1)?1.f:-1.f);if(mask>>u&1)pos=fminf(pos,v);else neg=fminf(neg,v);}
     float message=thermal_messages&&use_joint?native_thermal_single(cost,u,cavity[u],temp_now):.5f*(pos-neg);if(mplp)message=native_mplp_message(message,cavity[u],4);float next=old[u]+damping_now*(message-old[u]);h[rn[at+u]]+=next-old[u];rm[r*slots+at+u]=next;
    }
    for(int k=jo[t];use_joint&&k<jo[t+1];k++){
     int d=jdirs[k],size=joffset[d+1]-joffset[d];float target[8];
     if(thermal_messages)native_thermal_separator(cost,jm+r*jtables+joffset[d^1],jmaps+16*k,size,temp_now,target);else separator_minimum(cost,jm+r*jtables+joffset[d^1],jmaps+16*k,size,target);
     for(int v=0;v<size;v++){int idx=r*jtables+joffset[d]+v;float old=jm[idx];jm[idx]=old+damping_now*(target[v]-old);}
    }
   }__syncthreads();
  }

  } // complete one interleaved group (or the unmodified full sweep)

  int value=0,strong_units=0;unsigned charge_key=0;
  for(int u=tid;u<n;u+=QUIP_NT)value+=fields[u]*(h[u]>0?-1:1);
  for(int e=tid;e<edges;e+=QUIP_NT){int su=h[eu[e]]>0?-1:1,sv=h[ev[e]]>0?-1:1;value+=ej[e]*su*sv;if(edge_unary_mode)value+=edge_unary[2*e]*su+edge_unary[2*e+1]*sv;}
  for(int t=tid;t<regions;t+=QUIP_NT){int mask=0;for(int u=roff[t];u<roff[t+1];u++)mask|=(h[rn[u]]<=0)<<(u-roff[t]);int at=taboff[t]+mask;value+=500*q2[at]+fixed[at];if(record_slots==64){if(record_descriptor==1)strong_units+=q2[at]/4;else if(q2[at]!=0)charge_key^=record_mix(unsigned(t));}}
  native_energy_reduce_nt(value,tid,sums);
  __shared__ int accepted,current_energy;if(tid==0){int energy=constant+sums[0];current_energy=energy;accepted=energy<best[r];if(accepted)best[r]=energy;}__syncthreads();
  if(accepted)for(int u=tid;u<n;u+=QUIP_NT)states[r*n+u]=h[u]>0?-1:1;
  __syncthreads();
  if(accepted&&record_slots>=32){int cell=record_cell_nt(h,n,tid,sums);int slot=r*record_slots+cell;if(tid==0)record_energies[slot]=best[r];for(int u=tid;u<n;u+=QUIP_NT)record_states[slot*n+u]=h[u]>0?-1:1;__syncthreads();}
  // Separate neutral-state archive: never alter strict record cells or live messages.
  if(record_slots==96&&!accepted&&current_energy==best[r]){int cell=record_cell_nt(h,n,tid,sums);int slot=r*record_slots+64+cell;__shared__ int save_neutral;if(tid==0){save_neutral=current_energy<record_energies[slot];if(save_neutral)record_energies[slot]=current_energy;}__syncthreads();if(save_neutral)for(int u=tid;u<n;u+=QUIP_NT)record_states[slot*n+u]=h[u]>0?-1:1;__syncthreads();}
  if(record_slots==64){int extra=record_descriptor==1?record_strong_reduce_nt(strong_units,tid,sums):record_reduce_nt(charge_key,tid,sums);int cell=32+extra;int slot=r*record_slots+cell;__shared__ int save_charge;if(tid==0){save_charge=extra>=0&&current_energy<record_energies[slot];if(save_charge)record_energies[slot]=current_energy;}__syncthreads();if(save_charge)for(int u=tid;u<n;u+=QUIP_NT)record_states[slot*n+u]=h[u]>0?-1:1;__syncthreads();}


  if(guided_mode&&(iter+1)%100==0){
   __shared__ int stalled;if(tid==0){stalled=best[r]>=last_best[r];last_best[r]=best[r];}__syncthreads();
   if(stalled){int utility=0;for(int e=tid;e<edges;e+=QUIP_NT){int j=ej[e];if(abs(j)<1000&&j*int(states[r*n+eu[e]])*int(states[r*n+ev[e]])>0){float value=float(abs(j))/(1.f+edge_penalties[r*edges+e]);utility=max(utility,__float_as_int(value));}}sums[tid]=utility;__syncthreads();for(int step=QUIP_NT/2;step;step>>=1){if(tid<step)sums[tid]=max(sums[tid],sums[tid+step]);__syncthreads();}int maximum=sums[0];for(int e=tid;e<edges;e+=QUIP_NT){int j=ej[e];if(abs(j)<1000&&j*int(states[r*n+eu[e]])*int(states[r*n+ev[e]])>0){float value=float(abs(j))/(1.f+edge_penalties[r*edges+e]);if(maximum>0&&__float_as_int(value)==maximum)edge_penalties[r*edges+e]++;}}__syncthreads();}
  }
  for(int u=tid;u<n;u+=QUIP_NT){int epoch=(release_mode>=2&&unpin_mask[r*n+u]==0)?0:reinforcement_epoch;float tphase=float(iter-epoch);
#ifdef QUIP_ADAPT
  if(initial_mode==0&&release_mode<2)tphase=adapt_state[r*3];
#endif
  float rho=native_rho(tphase,native_reinforcement_ramp(r,initial_mode,ramp_base_arg,ramp_step_arg));float next=fixed_boundary&&unpin_mask[r*n+u]==0?b[u]:r<QUIP_MINSUM_REPLICAS?fminf(20000.f,fmaxf(-20000.f,rho*h[u])):focusing(h[u]-b[u],rho);h[u]+=next-b[u];b[u]=next;}
  __syncthreads();
#ifdef QUIP_ADAPT
  if(tid==0&&initial_mode==0&&release_mode<2){float E=float(current_energy);float prev=adapt_state[r*3+1];float ema=adapt_state[r*3+2];float dE=(iter==0)?0.f:fabsf(E-prev);ema=0.9f*ema+0.1f*dE;float rate=1.f/(1.f+ema/QUIP_ADAPT);adapt_state[r*3]+=rate;adapt_state[r*3+1]=E;adapt_state[r*3+2]=ema;}
  __syncthreads();
#endif
 }
 for(int u=tid;u<n;u+=QUIP_NT){gh[r*n+u]=h[u];gb[r*n+u]=b[u];}
}

// Experimental Cosm (Collective Switched Motion), arXiv:2605.30355v4.
// One CUDA block per independent circular-state trajectory. Within a block,
// each conflict-free edge color is processed in parallel; color barriers
// enforce sequential switching. The optimization has no shared state with MSA.
// The original integer Ising energy is always evaluated on the host.
// NVRTC compiles without host libc header search paths.
// CUDA built-in short and unsigned short are 16-bit types.

__device__ __forceinline__ unsigned int cosm_rng(unsigned int &state) {
    state ^= state << 13; state ^= state >> 17; state ^= state << 5;
    return state;
}
__device__ __forceinline__ int cosm_sign_wrap(unsigned short a, unsigned short b) {
    // Signed modular difference in [-180, 180) degrees (Q16 circle).
    const short delta = (short)(unsigned short)((int)a - (int)b);
    return delta >= 0 ? 1 : -1;
}

extern "C" __global__ void ice_cosm_collective(
    const int* __restrict__ edge_u,
    const int* __restrict__ edge_v,
    const int* __restrict__ edge_j,
    const int* __restrict__ color_starts,
    const int* __restrict__ field_h,
    const int* __restrict__ initial_spins,
    unsigned short* __restrict__ phases,
    const int nodes,
    const int colors,
    const int sweeps,
    const int initial_alpha_q16,
    const int dwt_interval,
    const int dwt_multiplier_milli,
    const unsigned int master_seed,
    const int cold_replicas
) {
    const int replica = (int)blockIdx.x;
    const int tid = (int)threadIdx.x;
    const int base = replica * nodes;
    unsigned int rng = master_seed ^ (0x9e3779b9u * (replica + 1u));
    rng ^= rng >> 16; rng *= 0x7feb352du; rng ^= rng >> 15;
    rng *= 0x846ca68bu; rng ^= rng >> 16;
    if (!rng) rng = 1u;

    for (int i = tid; i < nodes; i += (int)blockDim.x) {
        unsigned int r = rng ^ ((unsigned int)(i + 1) * 0x85ebca6bu);
        if (!r) r = 1u;
        r = cosm_rng(r);
        unsigned short phase;
        if (replica < cold_replicas || initial_spins[base + i] == 0) {
            phase = (unsigned short)r;
        } else {
            // Initialized near an MSA spin, with independent +/-45 degree jitter.
            int angle = (initial_spins[base + i] > 0 ? 0 : 32768);
            angle += (int)(r & 16383u) - 8192;
            phase = (unsigned short)angle;
        }
        phases[base + i] = phase;
    }
    __syncthreads();

    for (int sweep = 0; sweep < sweeps; ++sweep) {
        // 14-degree linear schedule, 0 on last sweep; constant within sweep.
        const int alpha = (int)(((long long)initial_alpha_q16 * (sweeps - sweep - 1)) /
                                (sweeps > 1 ? sweeps - 1 : 1));
        for (int c = 0; c < colors; ++c) {
            const int begin = color_starts[c];
            const int end = color_starts[c + 1];
            for (int e = begin + tid; e < end; e += (int)blockDim.x) {
                const int a = edge_u[e], b = edge_v[e];
                const unsigned short xa = phases[base + a];
                const unsigned short xb = phases[base + b];
                // Paper's Hamiltonian is -J s_i s_j; ICE uses +j s_i s_j.
                // Thus J_paper = -j_ICE and dx_i = +alpha*j_ICE*sign(dx).
                const int sign = cosm_sign_wrap(xa, xb);
                const int delta = (alpha * edge_j[e]) / 1000;
                phases[base + a] = (unsigned short)((int)xa + sign * delta);
                phases[base + b] = (unsigned short)((int)xb - sign * delta);
            }
            // Without this barrier the graph's edge colors are NOT sequential.
            __syncthreads();
        }
        // External fields are equivalent to couplings to a fixed phase-zero
        // anchor. Each field acts on a distinct vertex, so this is conflict-free.
        for (int i = tid; i < nodes; i += (int)blockDim.x) {
            if (field_h[i]) {
                const unsigned short xi = phases[base + i];
                const int delta = (alpha * field_h[i]) / 1000;
                phases[base + i] = (unsigned short)((int)xi +
                    cosm_sign_wrap(xi, (unsigned short)0) * delta);
            }
        }
        __syncthreads();

        // Dual Window Twist (DWT), two opposed 90-degree windows.
        if (dwt_interval > 0 && (sweep + 1) % dwt_interval == 0) {
            // Same deterministic reference per block; no thread race/RNG state.
            unsigned int r = rng ^ (unsigned int)(sweep * 0x27d4eb2du + 0x1234567u);
            if (!r) r = 1u;
            const unsigned short theta = (unsigned short)cosm_rng(r);
            const int twist = (alpha * dwt_multiplier_milli) / 1000;
            for (int i = tid; i < nodes; i += (int)blockDim.x) {
                const unsigned short x = phases[base + i];
                const int d = (int)(short)(unsigned short)((int)x - (int)theta);
                // Circular distance within +/-45 degrees of theta OR theta+180.
                if ((d > -8192 && d < 8192) || d <= -24576 || d >= 24576) {
                    phases[base + i] = (unsigned short)((int)x + twist);
                }
            }
            __syncthreads();
        }
    }
}

// Symplectic-Euler ballistic/discrete simulated bifurcation, original +J H.
// One block owns a whole trajectory. Force and coordinate phases are separate.
extern "C" __global__ void native_bifurcation(const int*row,const int*col,const int*j,const int*field,const int*diagonal,int n,long long constant,int count,float dt,float coupling,int discrete,const float*initial_x,const float*initial_y,long long*best,signed char*states,float*final_x,float*final_y,int*faults){
 int r=blockIdx.x,tid=threadIdx.x,stride=blockDim.x;extern __shared__ float memory[];float*x=memory,*y=x+n;long long*sums=(long long*)(y+n);__shared__ int accepted,faulty;
 if(tid==0)faulty=0;for(int u=tid;u<n;u+=stride){x[u]=initial_x[r*n+u];y[u]=initial_y[r*n+u];}__syncthreads();
 for(int iter=0;iter<=count;iter++){
  long long energy=0;for(int u=tid;u<n;u+=stride){int su=x[u]>=0.f?1:-1;energy+=2LL*field[u]*su;for(int at=row[u];at<row[u+1];at++)energy+=(long long)j[at]*su*(x[col[at]]>=0.f?1:-1);}
  sums[tid]=energy;__syncthreads();for(int step=stride/2;step;step>>=1){if(tid<step)sums[tid]+=sums[tid+step];__syncthreads();}
  if(tid==0){long long energy=constant+sums[0]/2;accepted=energy<best[r];if(accepted)best[r]=energy;}__syncthreads();
  if(accepted)for(int u=tid;u<n;u+=stride)states[r*n+u]=x[u]>=0.f?1:-1;__syncthreads();if(iter==count)break;
  float pump=count>1?float(iter)/float(count-1):1.f;
  for(int u=tid;u<n;u+=stride){float force=field[u]+diagonal[u]*(discrete?(x[u]>=0.f?1.f:-1.f):x[u]);for(int at=row[u];at<row[u+1];at++){float value=x[col[at]];force+=j[at]*(discrete?(value>=0.f?1.f:-1.f):value);}y[u]+=dt*(-(1.f-pump)*x[u]-coupling*force);if(!isfinite(y[u]))atomicOr(&faulty,1);}
  __syncthreads();
  for(int u=tid;u<n;u+=stride){float next=x[u]+dt*y[u];if(!isfinite(next))atomicOr(&faulty,1);if(fabsf(next)>1.f){next=copysignf(1.f,next);y[u]=0.f;}x[u]=next;}__syncthreads();if(faulty){if(tid==0)faults[r]=1;return;}
 }
 for(int u=tid;u<n;u+=stride){final_x[r*n+u]=x[u];final_y[r*n+u]=y[u];}
}
// NATIVE_CLUSTER_EXTENSION
// Selective reinforcement keeps each variable's incoming-message cavity.
__device__ __forceinline__ void native_core_apply(float&h,float&b,float proposal,int is_native,int selective){float next=selective&&!is_native?0.f:proposal;h+=next-b;b=next;}
extern "C" __global__ void native_core_reinforcement_probe(const float*input,const int*mask,int count,int selective,float*out){int u=blockIdx.x*blockDim.x+threadIdx.x;if(u<count){float h=input[3*u],b=input[3*u+1];native_core_apply(h,b,input[3*u+2],mask[u],selective);out[2*u]=h;out[2*u+1]=b;}}
// Region tables of arity 5..8 use one warp. Each lane owns at most eight
// assignments. Every outgoing table is derived from the same incoming costs.
__device__ __forceinline__ float native_cluster_warp_min(float x){for(int d=16;d;d>>=1)x=fminf(x,__shfl_xor_sync(0xffffffff,x,d));return x;}
__device__ __forceinline__ float native_cluster_warp_sum(float x){for(int d=16;d;d>>=1)x+=__shfl_xor_sync(0xffffffff,x,d);return x;}
__device__ __noinline__ void native_cluster_region_thread(const int*rn,const int*roff,const int*taboff,const int*q2,const int*working_fixed,const int*physical,const int*jo,const int*jdirs,const int*joffset,const int*jmaps,const int*jmap_offsets,int t,int r,int slots,int jtables,float weight,float damping,bool use_joint,bool mplp,int thermal_messages,float*h,float*rm,float*jm){
    int at=roff[t],degree=use_joint?jo[t+1]-jo[t]:0;float cavity[4],old[4],cost[16];
    for(int u=0;u<4;u++){old[u]=rm[r*slots+at+u];cavity[u]=(!use_joint||physical[at+u])?h[rn[at+u]]-old[u]:0.f;}
    for(int mask=0;mask<16;mask++){float linear=0.f;for(int u=0;u<4;u++)linear+=cavity[u]*((mask>>u&1)?1.f:-1.f);float e=weight*q2[taboff[t]+mask]+working_fixed[taboff[t]+mask]+linear;for(int k=jo[t];use_joint&&k<jo[t+1];k++)e+=jm[r*jtables+joffset[jdirs[k]^1]+jmaps[jmap_offsets[k]+mask]];cost[mask]=e;}
    // Finish all outgoing singleton values before writing fields or messages.
    float next_messages[4];
    for(int u=0;u<4;u++)if(!use_joint||physical[at+u]){
     float pos=1.e30f,neg=1.e30f;
     for(int mask=0;mask<16;mask++){float v=cost[mask]-cavity[u]*((mask>>u&1)?1.f:-1.f);if(mask>>u&1)pos=fminf(pos,v);else neg=fminf(neg,v);}
     float message=.5f*(pos-neg);
     if(thermal_messages&&use_joint){float zp=0.f,zn=0.f;for(int mask=0;mask<16;mask++){float v=cost[mask]-cavity[u]*((mask>>u&1)?1.f:-1.f);if(mask>>u&1)zp+=expf(-(v-pos)/QUIP_TEMPERATURE);else zn+=expf(-(v-neg)/QUIP_TEMPERATURE);}message-=.5f*QUIP_TEMPERATURE*logf(zp/zn);}
     if(mplp)message=native_mplp_message(message,cavity[u],4);next_messages[u]=old[u]+damping*(message-old[u]);
    }
    for(int u=0;u<4;u++)if(!use_joint||physical[at+u]){h[rn[at+u]]+=next_messages[u]-old[u];rm[r*slots+at+u]=next_messages[u];}
    for(int k=jo[t];use_joint&&k<jo[t+1];k++){
     int d=jdirs[k],size=joffset[d+1]-joffset[d];float target[8];
     if(thermal_messages)native_thermal_separator(cost,jm+r*jtables+joffset[d^1],jmaps+jmap_offsets[k],size,QUIP_TEMPERATURE,target);else separator_minimum(cost,jm+r*jtables+joffset[d^1],jmaps+jmap_offsets[k],size,target);
     for(int v=0;v<size;v++){int idx=r*jtables+joffset[d]+v;float old=jm[idx];jm[idx]=old+damping*(target[v]-old);}
    }

}
__device__ __noinline__ float native_residual_region_thread(const int*rn,const int*roff,const int*taboff,const int*q2,const int*working_fixed,const int*physical,const int*jo,const int*jdirs,const int*joffset,const int*jmaps,const int*jmap_offsets,int t,int r,int slots,int jtables,float weight,float damping,bool use_joint,bool mplp,int thermal_messages,float*h,float*rm,float*jm){
    float residual=0.f;int at=roff[t],degree=use_joint?jo[t+1]-jo[t]:0;float cavity[4],old[4],cost[16];
    for(int u=0;u<4;u++){old[u]=rm[r*slots+at+u];cavity[u]=(!use_joint||physical[at+u])?h[rn[at+u]]-old[u]:0.f;}
    for(int mask=0;mask<16;mask++){float linear=0.f;for(int u=0;u<4;u++)linear+=cavity[u]*((mask>>u&1)?1.f:-1.f);float e=weight*q2[taboff[t]+mask]+working_fixed[taboff[t]+mask]+linear;for(int k=jo[t];use_joint&&k<jo[t+1];k++)e+=jm[r*jtables+joffset[jdirs[k]^1]+jmaps[jmap_offsets[k]+mask]];cost[mask]=e;}
    // Finish all outgoing singleton values before writing fields or messages.
    float next_messages[4];
    for(int u=0;u<4;u++)if(!use_joint||physical[at+u]){
     float pos=1.e30f,neg=1.e30f;
     for(int mask=0;mask<16;mask++){float v=cost[mask]-cavity[u]*((mask>>u&1)?1.f:-1.f);if(mask>>u&1)pos=fminf(pos,v);else neg=fminf(neg,v);}
     float message=.5f*(pos-neg);
     if(thermal_messages&&use_joint){float zp=0.f,zn=0.f;for(int mask=0;mask<16;mask++){float v=cost[mask]-cavity[u]*((mask>>u&1)?1.f:-1.f);if(mask>>u&1)zp+=expf(-(v-pos)/QUIP_TEMPERATURE);else zn+=expf(-(v-neg)/QUIP_TEMPERATURE);}message-=.5f*QUIP_TEMPERATURE*logf(zp/zn);}
     if(mplp)message=native_mplp_message(message,cavity[u],4);next_messages[u]=old[u]+damping*(message-old[u]);
    }
    for(int u=0;u<4;u++)if(!use_joint||physical[at+u]){residual=fmaxf(residual,fabsf(next_messages[u]-old[u]));h[rn[at+u]]+=next_messages[u]-old[u];rm[r*slots+at+u]=next_messages[u];}
    for(int k=jo[t];use_joint&&k<jo[t+1];k++){
     int d=jdirs[k],size=joffset[d+1]-joffset[d];float target[8];
     if(thermal_messages)native_thermal_separator(cost,jm+r*jtables+joffset[d^1],jmaps+jmap_offsets[k],size,QUIP_TEMPERATURE,target);else separator_minimum(cost,jm+r*jtables+joffset[d^1],jmaps+jmap_offsets[k],size,target);
     for(int v=0;v<size;v++){int idx=r*jtables+joffset[d]+v;float old=jm[idx];jm[idx]=old+damping*(target[v]-old);residual=fmaxf(residual,fabsf(jm[idx]-old));}
    }
 return residual;
}

// Lagged observed residuals change only the order, never visit counts.
// Each replica orders its native colours independently, once per sweep.
__device__ void native_residual_sort(const float*scores,int*order,int count){
 for(int i=0;i<count;i++)order[i]=i;
 for(int i=1;i<count;i++){int v=order[i],j=i;while(j>0&&(scores[v]>scores[order[j-1]]||(scores[v]==scores[order[j-1]]&&v<order[j-1]))){order[j]=order[j-1];j--;}order[j]=v;}
}
extern "C" __global__ void native_residual_sort_probe(const float*scores,int*order,int count){if(threadIdx.x==0)native_residual_sort(scores+blockIdx.x*count,order+blockIdx.x*count,count);}

template<int K> __device__ __noinline__ void native_cluster_region_warp(const int*rn,const int*roff,const int*taboff,const int*q2,const int*working_fixed,const int*physical,const int*jo,const int*jdirs,const int*joffset,const int*jmaps,const int*jmap_offsets,int t,int r,int slots,int jtables,float weight,float damping,bool use_joint,bool mplp,int thermal_messages,float*h,float*rm,float*jm){
 constexpr int WORDS=(1<<K)/32;int lane=threadIdx.x&31,at=roff[t];float cavity[K],old[K],cost[WORDS];
 #pragma unroll
 for(int u=0;u<K;u++){old[u]=rm[r*slots+at+u];cavity[u]=(!use_joint||physical[at+u])?h[rn[at+u]]-old[u]:0.f;}
 #pragma unroll
 for(int p=0;p<WORDS;p++){int mask=lane+32*p;float linear=0.f;
  #pragma unroll
  for(int u=0;u<K;u++)linear+=cavity[u]*((mask>>u&1)?1.f:-1.f);
  float value=weight*q2[taboff[t]+mask]+working_fixed[taboff[t]+mask]+linear;
  for(int k=jo[t];use_joint&&k<jo[t+1];k++)value+=jm[r*jtables+joffset[jdirs[k]^1]+jmaps[jmap_offsets[k]+mask]];
  cost[p]=value;
 }
 __syncwarp();
 #pragma unroll
 for(int u=0;u<K;u++)if(!use_joint||physical[at+u]){
  float pos=1.e30f,neg=1.e30f;
  #pragma unroll
  for(int p=0;p<WORDS;p++){int mask=lane+32*p;float value=cost[p]-cavity[u]*((mask>>u&1)?1.f:-1.f);if(mask>>u&1)pos=fminf(pos,value);else neg=fminf(neg,value);}
  pos=native_cluster_warp_min(pos);neg=native_cluster_warp_min(neg);float message=.5f*(pos-neg);
  if(thermal_messages&&use_joint){float zp=0.f,zn=0.f;
   #pragma unroll
   for(int p=0;p<WORDS;p++){int mask=lane+32*p;float value=cost[p]-cavity[u]*((mask>>u&1)?1.f:-1.f);if(mask>>u&1)zp+=expf(-(value-pos)/QUIP_TEMPERATURE);else zn+=expf(-(value-neg)/QUIP_TEMPERATURE);}
   zp=native_cluster_warp_sum(zp);zn=native_cluster_warp_sum(zn);message-=.5f*QUIP_TEMPERATURE*logf(zp/zn);
  }
  if(mplp)message=native_mplp_message(message,cavity[u],K);
  if(lane==0){float next=old[u]+damping*(message-old[u]);h[rn[at+u]]+=next-old[u];rm[r*slots+at+u]=next;}
 }
 for(int k=jo[t];use_joint&&k<jo[t+1];k++){
  int d=jdirs[k],size=joffset[d+1]-joffset[d];const int*map=jmaps+jmap_offsets[k];float target[8];float normal=1.e30f;
  for(int v=0;v<size;v++){float minimum=1.e30f;float incoming=jm[r*jtables+joffset[d^1]+v];
   #pragma unroll
   for(int p=0;p<WORDS;p++){int mask=lane+32*p;if(map[mask]==v)minimum=fminf(minimum,cost[p]-incoming);}
   minimum=native_cluster_warp_min(minimum);
   if(thermal_messages){float z=0.f;
    #pragma unroll
    for(int p=0;p<WORDS;p++){int mask=lane+32*p;if(map[mask]==v)z+=expf(-(cost[p]-incoming-minimum)/QUIP_TEMPERATURE);}
    z=native_cluster_warp_sum(z);minimum-=QUIP_TEMPERATURE*logf(z);
   }
   target[v]=minimum;normal=fminf(normal,minimum);
  }
  if(lane==0)for(int v=0;v<size;v++){int idx=r*jtables+joffset[d]+v;float old=jm[idx];jm[idx]=old+damping*(target[v]-normal-old);}
 }
 __syncwarp();
}
extern "C" __global__ __launch_bounds__(512,1) void loop_native_cluster_bp(
 const int*eu,const int*ev,const int*ej,const int*eo,int ec,int edges,
 const int*rn,const int*roff,const int*taboff,const int*q2,const int*fixed,
 const int*rc,int colors,int regions,int slots,const int*fields,int n,int constant,
 int start,int count,float*gh,float*gb,float2*em,float*rm,int*best,signed char*states,
 const unsigned long long*seed_ptr, const int*physical, const int*jo,const int*jdirs,const int*joffset,const int*jmaps,int jtables,float*jm,int reinforcement_epoch,int release_mode,int kick_strength,int*unpin_mask,int guided_mode,int*edge_penalties,int*last_best,int thermal_messages,int record_slots,int*record_energies,signed char*record_states,int initial_mode,const int*initial_bias,int edge_unary_mode,const int*edge_unary,int record_descriptor,int sweep_policy,const int*working_ej,const int*working_fixed,const int*working_fields,const int*jmap_offsets,const int*native_focus_mask){
 int r=blockIdx.x,tid=threadIdx.x;extern __shared__ float smem[];const bool mplp=(sweep_policy&16)!=0;
 const bool seed_strong_defects=release_mode==5;const bool external_backtrack=release_mode==7;const bool fixed_boundary=release_mode==8;if(seed_strong_defects||external_backtrack)release_mode=2;if(fixed_boundary)release_mode=1;
 if(colors>32)return; // Offline input guard; current prepared models use3/4.
 __shared__ int split[32],residual_order[32];__shared__ float residual_scores[32];
 const bool residual_schedule=(sweep_policy&64)!=0;
 if(tid<colors){residual_order[tid]=tid;residual_scores[tid]=0.f;}
 if(tid<colors){int t=rc[tid];while(t<rc[tid+1]&&roff[t+1]-roff[t]>4)t++;split[tid]=t;}
 float*h=smem,*b=h+n;int*sums=(int*)(b+n);
 unsigned long long rng=*seed_ptr^(0x9e3779b97f4a7c15ULL*(r+1))^(tid+1);
 if(start==0){
  if(guided_mode){for(int e=tid;e<edges;e+=512)edge_penalties[r*edges+e]=0;if(tid==0)last_best[r]=2147483647;}
  for(int u=tid;u<n;u+=512){float noise=((next_rng(rng)>>40)*(1.f/16777216.f)-.5f);b[u]=initial_mode==3&&unpin_mask[r*n+u]==0?float(initial_bias[r*n+u]):initial_mode?native_seed_initial(r,initial_bias[(initial_mode>=2?r*n:0)+u],noise):noise;h[u]=working_fields[u]+b[u];}
  for(int e=tid;e<edges;e+=512)em[r*edges+e]=make_float2(0,0);
  for(int e=tid;e<slots;e+=512)rm[r*slots+e]=0.f;
  for(int e=tid;e<jtables;e+=512)jm[r*jtables+e]=0.f;
 }else for(int u=tid;u<n;u+=512){h[u]=gh[r*n+u];b[u]=gb[r*n+u];}
 __syncthreads();float damping=QUIP_DAMPING_BASE+QUIP_DAMPING_STEP*(r%6),weight=native_region_weight(r,start,sweep_policy);
 for(int iter=start;iter<start+count;iter++){
  if(residual_schedule){if(tid==0)native_residual_sort(residual_scores,residual_order,colors);__syncthreads();if(tid<colors)residual_scores[tid]=0.f;__syncthreads();}
  if(sweep_policy&2)weight=native_region_weight(r,iter,sweep_policy);
  if(reinforcement_epoch>0&&iter==reinforcement_epoch){
   if(release_mode>=2&&!external_backtrack){
    for(int u=tid;u<n;u+=512)unpin_mask[r*n+u]=0;__syncthreads();
    for(int t=tid;t<regions;t+=512){int mask=0;for(int u=roff[t];u<roff[t+1];u++)mask|=(h[rn[u]]<=0)<<(u-roff[t]);if(q2[taboff[t]+mask]!=0){for(int u=roff[t];u<roff[t+1];u++)atomicOr(unpin_mask+r*n+rn[u],1);}}__syncthreads();
    if(seed_strong_defects)native_seed_strong_defects(eu,ev,ej,edges,h,unpin_mask+r*n,tid,512);__syncthreads();
    for(int e=tid;e<edges;e+=512)if(abs(ej[e])>=1000){int u=eu[e],v=ev[e];if(atomicOr(unpin_mask+r*n+u,0)&1)atomicOr(unpin_mask+r*n+v,2);if(atomicOr(unpin_mask+r*n+v,0)&1)atomicOr(unpin_mask+r*n+u,2);}__syncthreads();
   }
   for(int u=tid;u<n;u+=512)if(release_mode<2||unpin_mask[r*n+u]!=0){if(external_backtrack)native_backtrack_field(h[u],b[u],kick_strength);else{h[u]-=b[u];b[u]=0.f;}}__syncthreads();
   if(release_mode>=3){
    for(int c=0;c<ec;c++){for(int e=eo[c]+tid;e<eo[c+1];e+=512){int u=eu[e],v=ev[e];if(unpin_mask[r*n+u]||unpin_mask[r*n+v]){float2 old=em[r*edges+e];h[u]-=old.x;h[v]-=old.y;em[r*edges+e]=make_float2(0.f,0.f);}}__syncthreads();}
    for(int c=0;c<colors;c++){for(int t=rc[c]+tid;t<rc[c+1];t+=512){int at=roff[t];bool clear=false;for(int u=at;u<roff[t+1];u++)clear|=unpin_mask[r*n+rn[u]]!=0;if(clear){for(int u=at;u<roff[t+1];u++){if(r<QUIP_MINSUM_REPLICAS||physical[u])h[rn[u]]-=rm[r*slots+u];rm[r*slots+u]=0.f;}if(r>=QUIP_MINSUM_REPLICAS)for(int k=jo[t];k<jo[t+1];k++){int d=jdirs[k];for(int v=joffset[d];v<joffset[d+1];v++)jm[r*jtables+v]=0.f;}}}__syncthreads();}
    if(release_mode==4){for(int u=tid;u<n;u+=512){gh[r*n+u]=h[u];gb[r*n+u]=b[u];}return;}
   }
  }
  for(int stage=0;stage<native_message_stages(colors,sweep_policy);stage++){
  int2 pair_bounds=native_message_bounds(stage,ec,colors,sweep_policy,true),region_bounds=native_message_bounds(stage,ec,colors,sweep_policy,false);
  for(int sweep=pair_bounds.x;sweep<pair_bounds.y;sweep++){int c=native_sweep_color(sweep,ec,iter,sweep_policy);
   for(int e=eo[c]+tid;e<eo[c+1];e+=512){
    int u=eu[e],v=ev[e];float j=working_ej[e];if(guided_mode&&abs(ej[e])<1000)j*=1.f+edge_penalties[r*edges+e];float2 old=em[r*edges+e];float hu=h[u]-old.x,hv=h[v]-old.y;
    float mu=!mplp&&thermal_messages&&r>=QUIP_MINSUM_REPLICAS?native_thermal_pair(j,hv,QUIP_TEMPERATURE):-(j>0?1.f:-1.f)*fminf(fabsf(j),fmaxf(-fabsf(j),hv));
    float mv=!mplp&&thermal_messages&&r>=QUIP_MINSUM_REPLICAS?native_thermal_pair(j,hu,QUIP_TEMPERATURE):-(j>0?1.f:-1.f)*fminf(fabsf(j),fmaxf(-fabsf(j),hu));
    if(edge_unary_mode){float au=edge_unary[2*e],av=edge_unary[2*e+1];mu=!mplp&&thermal_messages&&r>=QUIP_MINSUM_REPLICAS?au+native_thermal_pair(j,hv+av,QUIP_TEMPERATURE):native_pair_biased(j,au,av,hv);mv=!mplp&&thermal_messages&&r>=QUIP_MINSUM_REPLICAS?av+native_thermal_pair(j,hu+au,QUIP_TEMPERATURE):native_pair_biased(j,av,au,hu);}
    if((sweep_policy&4)&&abs(ej[e])<1000){float au=edge_unary_mode?edge_unary[2*e]:0.f,av=edge_unary_mode?edge_unary[2*e+1]:0.f;const float appearance=.5f;
     mu=native_fractional_pair(j,au,av,h[u],old.x,appearance,thermal_messages&&r>=QUIP_MINSUM_REPLICAS,QUIP_TEMPERATURE);
     mv=native_fractional_pair(j,av,au,h[v],old.y,appearance,thermal_messages&&r>=QUIP_MINSUM_REPLICAS,QUIP_TEMPERATURE);
    }
    if(mplp){mu=native_mplp_message(mu,hu,2);mv=native_mplp_message(mv,hv,2);}
    float a=old.x+damping*(mu-old.x),bb=old.y+damping*(mv-old.y);
    h[u]+=a-old.x;h[v]+=bb-old.y;em[r*edges+e]=make_float2(a,bb);
   }__syncthreads();
  }
  for(int sweep=region_bounds.x;sweep<region_bounds.y;sweep++){int c=residual_schedule?residual_order[sweep]:native_sweep_color(sweep,colors,iter,sweep_policy);
   bool use_joint=!mplp&&r>=QUIP_MINSUM_REPLICAS;
   for(int t=rc[c]+tid/32;t<split[c];t+=16){int arity=roff[t+1]-roff[t];switch(arity){
    case 5:native_cluster_region_warp<5>(rn,roff,taboff,q2,working_fixed,physical,jo,jdirs,joffset,jmaps,jmap_offsets,t,r,slots,jtables,weight,damping,use_joint,mplp,thermal_messages,h,rm,jm);break;
    case 6:native_cluster_region_warp<6>(rn,roff,taboff,q2,working_fixed,physical,jo,jdirs,joffset,jmaps,jmap_offsets,t,r,slots,jtables,weight,damping,use_joint,mplp,thermal_messages,h,rm,jm);break;
    case 7:native_cluster_region_warp<7>(rn,roff,taboff,q2,working_fixed,physical,jo,jdirs,joffset,jmaps,jmap_offsets,t,r,slots,jtables,weight,damping,use_joint,mplp,thermal_messages,h,rm,jm);break;
    case 8:native_cluster_region_warp<8>(rn,roff,taboff,q2,working_fixed,physical,jo,jdirs,joffset,jmaps,jmap_offsets,t,r,slots,jtables,weight,damping,use_joint,mplp,thermal_messages,h,rm,jm);break;
    case 9:native_cluster_region_warp<9>(rn,roff,taboff,q2,working_fixed,physical,jo,jdirs,joffset,jmaps,jmap_offsets,t,r,slots,jtables,weight,damping,use_joint,mplp,thermal_messages,h,rm,jm);break;
   }}
   if(residual_schedule){float maximum=0.f;
    for(int t=split[c]+tid;t<rc[c+1];t+=512)maximum=fmaxf(maximum,native_residual_region_thread(rn,roff,taboff,q2,working_fixed,physical,jo,jdirs,joffset,jmaps,jmap_offsets,t,r,slots,jtables,weight,damping,use_joint,mplp,thermal_messages,h,rm,jm));
    for(int d=16;d;d>>=1)maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,d));
    if((tid&31)==0)atomicMax((int*)(residual_scores+c),__float_as_int(maximum));
   }else{
   for(int t=split[c]+tid;t<rc[c+1];t+=512)native_cluster_region_thread(rn,roff,taboff,q2,working_fixed,physical,jo,jdirs,joffset,jmaps,jmap_offsets,t,r,slots,jtables,weight,damping,use_joint,mplp,thermal_messages,h,rm,jm);
   }

   __syncthreads();
  }

  } // complete one interleaved group (or the unmodified full sweep)

  int value=0,strong_units=0;unsigned charge_key=0;
  for(int u=tid;u<n;u+=512)value+=fields[u]*(h[u]>0?-1:1);
  for(int e=tid;e<edges;e+=512){int su=h[eu[e]]>0?-1:1,sv=h[ev[e]]>0?-1:1;value+=ej[e]*su*sv;if(edge_unary_mode)value+=edge_unary[2*e]*su+edge_unary[2*e+1]*sv;}
  for(int t=tid;t<regions;t+=512){int mask=0;for(int u=roff[t];u<roff[t+1];u++)mask|=(h[rn[u]]<=0)<<(u-roff[t]);int at=taboff[t]+mask;value+=500*q2[at]+fixed[at];if(record_slots==64){if(record_descriptor==1)strong_units+=q2[at]/4;else if(q2[at]!=0)charge_key^=record_mix(unsigned(t));}}
  native_energy_reduce(value,tid,sums);
  __shared__ int accepted,current_energy;if(tid==0){int energy=constant+sums[0];current_energy=energy;accepted=energy<best[r];if(accepted)best[r]=energy;}__syncthreads();
  if(accepted)for(int u=tid;u<n;u+=512)states[r*n+u]=h[u]>0?-1:1;
  __syncthreads();
  if(accepted&&record_slots>=32){int cell=record_cell(h,n,tid,sums);int slot=r*record_slots+cell;if(tid==0)record_energies[slot]=best[r];for(int u=tid;u<n;u+=512)record_states[slot*n+u]=h[u]>0?-1:1;__syncthreads();}
  // Separate neutral-state archive: never alter strict record cells or live messages.
  if(record_slots==96&&!accepted&&current_energy==best[r]){int cell=record_cell(h,n,tid,sums);int slot=r*record_slots+64+cell;__shared__ int save_neutral;if(tid==0){save_neutral=current_energy<record_energies[slot];if(save_neutral)record_energies[slot]=current_energy;}__syncthreads();if(save_neutral)for(int u=tid;u<n;u+=512)record_states[slot*n+u]=h[u]>0?-1:1;__syncthreads();}
  if(record_slots==64){int extra=record_descriptor==1?record_strong_reduce(strong_units,tid,sums):record_reduce(charge_key,tid,sums);int cell=32+extra;int slot=r*record_slots+cell;__shared__ int save_charge;if(tid==0){save_charge=extra>=0&&current_energy<record_energies[slot];if(save_charge)record_energies[slot]=current_energy;}__syncthreads();if(save_charge)for(int u=tid;u<n;u+=512)record_states[slot*n+u]=h[u]>0?-1:1;__syncthreads();}


  if(guided_mode&&(iter+1)%100==0){
   __shared__ int stalled;if(tid==0){stalled=best[r]>=last_best[r];last_best[r]=best[r];}__syncthreads();
   if(stalled){int utility=0;for(int e=tid;e<edges;e+=512){int j=ej[e];if(abs(j)<1000&&j*int(states[r*n+eu[e]])*int(states[r*n+ev[e]])>0){float value=float(abs(j))/(1.f+edge_penalties[r*edges+e]);utility=max(utility,__float_as_int(value));}}sums[tid]=utility;__syncthreads();for(int step=256;step;step>>=1){if(tid<step)sums[tid]=max(sums[tid],sums[tid+step]);__syncthreads();}int maximum=sums[0];for(int e=tid;e<edges;e+=512){int j=ej[e];if(abs(j)<1000&&j*int(states[r*n+eu[e]])*int(states[r*n+ev[e]])>0){float value=float(abs(j))/(1.f+edge_penalties[r*edges+e]);if(maximum>0&&__float_as_int(value)==maximum)edge_penalties[r*edges+e]++;}}__syncthreads();}
  }
  for(int u=tid;u<n;u+=512){int epoch=(release_mode>=2&&unpin_mask[r*n+u]==0)?0:reinforcement_epoch;float rho=native_rho(float(iter-epoch),native_reinforcement_ramp(r,initial_mode,QUIP_RAMP_BASE,QUIP_RAMP_STEP));float next=(sweep_policy&32)&&!native_focus_mask[u]?0.f:fixed_boundary&&unpin_mask[r*n+u]==0?b[u]:r<QUIP_MINSUM_REPLICAS?fminf(20000.f,fmaxf(-20000.f,rho*h[u])):focusing(h[u]-b[u],rho);native_core_apply(h[u],b[u],next,native_focus_mask[(sweep_policy&32)?u:0],sweep_policy&32);}
  __syncthreads();
 }
 for(int u=tid;u<n;u+=512){gh[r*n+u]=h[u];gb[r*n+u]=b[u];}
}

// Executes the actual update routines for independent CPU/GPU oracle checks.
extern "C" __global__ void native_cluster_message_probe(const int*rn,const int*roff,const int*taboff,const int*q2,const int*working_fixed,const int*physical,const int*jo,const int*jdirs,const int*joffset,const int*jmaps,const int*jmap_offsets,int arity,int jtables,float weight,float damping,int joint,int thermal,float*h,float*rm,float*jm){
 int r=blockIdx.x,t=0,slots=arity;bool use_joint=joint!=0,mplp=false;int thermal_messages=thermal;h+=r*arity;
 switch(arity){
 case 4:if(threadIdx.x==0)native_cluster_region_thread(rn,roff,taboff,q2,working_fixed,physical,jo,jdirs,joffset,jmaps,jmap_offsets,t,r,slots,jtables,weight,damping,use_joint,mplp,thermal_messages,h,rm,jm);break;
 case 5:native_cluster_region_warp<5>(rn,roff,taboff,q2,working_fixed,physical,jo,jdirs,joffset,jmaps,jmap_offsets,t,r,slots,jtables,weight,damping,use_joint,mplp,thermal_messages,h,rm,jm);break;
 case 6:native_cluster_region_warp<6>(rn,roff,taboff,q2,working_fixed,physical,jo,jdirs,joffset,jmaps,jmap_offsets,t,r,slots,jtables,weight,damping,use_joint,mplp,thermal_messages,h,rm,jm);break;
 case 7:native_cluster_region_warp<7>(rn,roff,taboff,q2,working_fixed,physical,jo,jdirs,joffset,jmaps,jmap_offsets,t,r,slots,jtables,weight,damping,use_joint,mplp,thermal_messages,h,rm,jm);break;
 case 8:native_cluster_region_warp<8>(rn,roff,taboff,q2,working_fixed,physical,jo,jdirs,joffset,jmaps,jmap_offsets,t,r,slots,jtables,weight,damping,use_joint,mplp,thermal_messages,h,rm,jm);break;
 case 9:native_cluster_region_warp<9>(rn,roff,taboff,q2,working_fixed,physical,jo,jdirs,joffset,jmaps,jmap_offsets,t,r,slots,jtables,weight,damping,use_joint,mplp,thermal_messages,h,rm,jm);break;

 }
}

// Called only between completed persistent-kernel chunks on one CUDA stream.
// Read old cohort messages; write a separate buffer before updating any h.
extern "C" __global__ void native_cohort_cavity(const float*h,const float*b,const float*old_bias,float*next_bias,int n,int replicas,int width,int iteration,int initial_mode,int control){
 int at=blockIdx.x*blockDim.x+threadIdx.x,groups=(replicas+width-1)/width;
 if(at>=groups*n)return;int u=at%n,first=(at/n)*width,last=min(first+width,replicas);
 if(control){for(int r=first;r<last;r++)next_bias[r*n+u]=0.f;return;}
 float incoming[4],q[4];const float temperature=QUIP_TEMPERATURE;
 for(int r=first;r<last;r++){int k=r-first;float rho=fminf(.95f,fmaxf(0.f,float(iteration)/native_reinforcement_ramp(r,initial_mode,QUIP_RAMP_BASE,QUIP_RAMP_STEP)));q[k]=sqrtf(rho);float cavity=h[r*n+u]-b[r*n+u]-old_bias[r*n+u];incoming[k]=temperature*atanhf(q[k]*tanhf(cavity/temperature));}
 for(int r=first;r<last;r++){float cavity=0.f;for(int j=first;j<last;j++)if(j!=r)cavity+=incoming[j-first];next_bias[r*n+u]=temperature*atanhf(q[r-first]*tanhf(cavity/temperature));}
}
extern "C" __global__ void native_cohort_apply(float*h,float*old_bias,const float*next_bias,int count){
 int u=blockIdx.x*blockDim.x+threadIdx.x;if(u>=count)return;float change=next_bias[u]-old_bias[u];if(change!=0.f)h[u]+=change;old_bias[u]=next_bias[u];
}

// Four-variable actual message update plus its observed max residual.
extern "C" __global__ void native_residual_message_probe(const int*rn,const int*roff,const int*taboff,const int*q2,const int*working_fixed,const int*physical,const int*jo,const int*jdirs,const int*joffset,const int*jmaps,const int*jmap_offsets,int arity,int jtables,float weight,float damping,int joint,int thermal,float*h,float*rm,float*jm,float*residual){
 if(threadIdx.x==0)residual[blockIdx.x]=native_residual_region_thread(rn,roff,taboff,q2,working_fixed,physical,jo,jdirs,joffset,jmaps,jmap_offsets,0,blockIdx.x,arity,jtables,weight,damping,joint!=0,false,thermal,h+blockIdx.x*arity,rm,jm);
}

// CSCREEN_CACHED_POLISH_EXTENSION
// Same ordered absolute-mask decisions as polish_physical_groups. Cache local
// fields; update neighbours with exact integer additions after accepted moves.
extern "C" __global__ void cscreen_polish_cached(const int*row,const int*neighbors,const int*weights,const int*fields,const int*groups,const int*starts,const int*internal,int count,int n,int passes,signed char*states){
 int lane=threadIdx.x;extern __shared__ int storage[];int*local=storage;signed char*s=(signed char*)(local+n);
 __shared__ int nodes[4],mask_now,take,changed;
 for(int u=lane;u<n;u+=32)s[u]=states[blockIdx.x*n+u];__syncwarp();
 for(int u=lane;u<n;u+=32){long long field=fields[u];for(int at=row[u];at<row[u+1];at++)field+=(long long)weights[at]*s[neighbors[at]];local[u]=(int)field;}__syncwarp();
 for(int pass=0;pass<passes;pass++){
  if(lane==0)changed=0;__syncwarp();
  for(int g=0;g<count;g++){
   int k=starts[g+1]-starts[g];if(lane<k)nodes[lane]=groups[starts[g]+lane];__syncwarp();
   if(lane==0){int code=0;for(int p=0;p<k;p++)code|=(s[nodes[p]]>0)<<p;mask_now=code;}__syncwarp();
   long long delta=0x3fffffffffffffffLL;
   if(lane<(1<<k)){delta=0;int flips=lane^mask_now;for(int p=0;p<k;p++)if(flips>>p&1){int u=nodes[p];delta-=2LL*s[u]*local[u];for(int q=p+1;q<k;q++)if(flips>>q&1)delta+=4LL*internal[16*g+4*p+q]*s[u]*s[nodes[q]];}}
   int choice=lane;for(int step=16;step;step>>=1){long long other=__shfl_down_sync(0xffffffff,delta,step);int at=__shfl_down_sync(0xffffffff,choice,step);if(lane+step<32&&(other<delta||(other==delta&&at<choice))){delta=other;choice=at;}}
   if(lane==0){take=delta<0?choice:-1;if(take>=0)changed=1;}__syncwarp();
   if(take>=0){int flips=take^mask_now;for(int p=0;p<k;p++)if(flips>>p&1){int u=nodes[p],old=s[u];for(int at=row[u]+lane;at<row[u+1];at+=32)atomicAdd(local+neighbors[at],-2*weights[at]*old);}__syncwarp();if(lane<k)s[nodes[lane]]=(take>>lane&1)?1:-1;__syncwarp();}
  }
  if(!changed)break;
 }
 for(int u=lane;u<n;u+=32)states[blockIdx.x*n+u]=s[u];
}
