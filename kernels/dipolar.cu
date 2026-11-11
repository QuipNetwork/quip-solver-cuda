// D-static oracle search. CPU-13 ec84be8, dipolar_search.rs:804-919,
// 1032-1159, 1235-1243, 1404-1468 define the move and scoring rules.
// One warp owns one independent replica. No inter-replica state or atomics.
#define MAX_LINKS 24
#define MAX_VERTICES 16

__device__ unsigned long long next_random(unsigned long long &state) {
    unsigned long long z = (state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

extern "C" __global__ void quip_dipolar(
    int n, int vertices, int proposals, int max_move,
    const long long *h, const long long *matrix, const int *topology,
    const signed char *starts, unsigned long long seed, double hot, double cold,
    signed char *finals, long long *final_scores,
    signed char *terminal, long long *terminal_scores,
    signed char *raw, long long *raw_scores, int *moves, int *lengths, int *accepted
) {
    const int lane = threadIdx.x;
    const int replica = blockIdx.x;
    const int *tail = topology, *head = tail+n, *position = head+n;
    const int *pinned = position+n, *free_vertex = pinned+n;
    __shared__ signed char spin[MAX_LINKS], best[MAX_LINKS];
    __shared__ long long field[MAX_LINKS], parts[32], score, best_score, delta;
    __shared__ int flip[MAX_LINKS], path[MAX_LINKS], path_len, accept;
    __shared__ unsigned long long rng;
    if (lane < n) {
        spin[lane] = starts[replica*n+lane];
        best[lane] = spin[lane];
    }
    if (lane == 0) {
        // Same replica seed derivation as CPU-13 read_seed, different stream.
        rng = seed + (replica+1ULL)*0x9e3779b97f4a7c15ULL;
        unsigned long long z = rng;
        z = (z ^ (z >> 30))*0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27))*0x94d049bb133111ebULL;
        rng = z ^ (z >> 31);
    }
    __syncthreads();
    long long local = 0;
    if (lane < n) {
        long long f = h[lane];
        local = h[lane]*spin[lane];
        for (int j=0; j<n; ++j) {
            f += matrix[lane*n+j]*spin[j];
            if (j>lane) local += matrix[lane*n+j]*spin[lane]*spin[j];
        }
        field[lane] = f;
    }
    parts[lane] = local;
    __syncthreads();
    if (lane == 0) {
        score = 0;
        for (int i=0;i<32;++i) score += parts[i];
        best_score = score;
    }
    __syncthreads();
    for (int t=0; t<proposals; ++t) {
        if (lane<n) flip[lane]=0;
        __syncthreads();
        if (lane==0) {
            int seen[MAX_VERTICES];
            for (int v=0;v<vertices;++v) seen[v]=-1;
            int at = next_random(rng)%vertices;
            seen[at]=0;
            int first_free=free_vertex[at] ? 0 : -1;
            path_len=0;
            const int limit = max_move<n ? max_move : n;
            // CPU-13: stop at the first cycle or pair of free vertices.
            for (int step=0;step<limit;++step) {
                int out[4], to[4], count=0;
                for (int e=0;e<n;++e) {
                    int a = spin[position[e]]==1 ? tail[e] : head[e];
                    int b = spin[position[e]]==1 ? head[e] : tail[e];
                    if (a==at && !pinned[e]) {out[count]=e;to[count++]=b;}
                }
                if (!count) break;
                int chosen=next_random(rng)%count;
                path[step]=out[chosen];
                at=to[chosen];
                int begin=-1;
                if (seen[at]>=0) begin=seen[at];
                else {
                    seen[at]=step+1;
                    if (free_vertex[at]) {
                        if (first_free>=0) begin=first_free;
                        else first_free=step+1;
                    }
                }
                if (begin>=0) {
                    path_len=step+1-begin;
                    for (int i=0;i<path_len;++i) path[i]=path[i+begin];
                    break;
                }
            }
            for (int i=0;i<path_len;++i) flip[position[path[i]]]=1;
        }
        __syncthreads();
        local=0;
        if (lane<n && flip[lane]) {
            local=-2LL*spin[lane]*field[lane];
            // Internal flipped pairs cancel, CPU-13:1431-1437.
            for (int j=lane+1;j<n;++j)
                if (flip[j]) local+=4LL*matrix[lane*n+j]*spin[lane]*spin[j];
        }
        parts[lane]=local;
        __syncthreads();
        const int slot=replica*proposals+t;
        if (lane==0) {
            delta=0;
            for (int i=0;i<32;++i) delta+=parts[i];
            double beta=hot*pow(cold/hot, double(t)/double(proposals));
            // Exactly representable 53 random bits, in [0,1).
            double uniform=double(next_random(rng)>>11)*0x1.0p-53;
            accept=path_len && (delta<=0 || uniform<exp(-beta*double(delta)));
            raw_scores[slot]=score+delta;
            lengths[slot]=path_len;
            accepted[slot]=accept;
            for (int k=0;k<MAX_LINKS;++k) moves[slot*MAX_LINKS+k]=k<path_len ? path[k] : -1;
        }
        if (lane<n) raw[slot*n+lane]=flip[lane] ? -spin[lane] : spin[lane];
        __syncthreads();
        // Each lane owns a field. Read old spins before any lane flips.
        if (lane<n && accept) {
            long long f=field[lane];
            for (int j=0;j<n;++j)
                if (flip[j]) f-=2LL*matrix[lane*n+j]*spin[j];
            field[lane]=f;
        }
        __syncthreads();
        if (lane<n && accept && flip[lane]) spin[lane]=-spin[lane];
        if (lane==0 && accept) score+=delta;
        __syncthreads();
        if (score<best_score && lane<n) best[lane]=spin[lane];
        __syncthreads();
        if (lane==0 && score<best_score) best_score=score;
        __syncthreads();
    }
    if (lane<n) {
        finals[replica*n+lane]=best[lane];
        terminal[replica*n+lane]=spin[lane];
    }
    if (lane==0) {
        final_scores[replica]=best_score;
        terminal_scores[replica]=score;
    }
}
