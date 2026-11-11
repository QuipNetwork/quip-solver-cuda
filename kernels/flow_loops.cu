// CPU-12 ec84be8a, src/families/flow_loops.rs:472-530: signed charge,
// per-pin admission, balanced boundary flux, and exact direct rescoring.
// One independent block per start. No atomics, floating point, or RNG.

#define THREADS 128
#define MAX_PATH 64
#define FREE 127
typedef long long ScoreInt;

struct Shared {
    int path[MAX_PATH], length, keep, failed, changed;
    ScoreInt new_gap[THREADS], old_gap[THREADS];
    ScoreInt new_obj[THREADS], old_obj[THREADS], flux[THREADS];
    int pin_bad[THREADS], defect[THREADS];
    ScoreInt best_delta[THREADS];
    int best_index[THREADS];
};

__device__ int magnitude(int a) { return a < 0 ? -a : a; }
__device__ bool in_path(const Shared &w, int e) {
    for (int k = 0; k < w.length; ++k) if (w.path[k] == e) return true;
    return false;
}
__device__ int other(const int *ends, int e, int v) {
    return ends[2 * e] == v ? ends[2 * e + 1] : ends[2 * e];
}
__device__ ScoreInt reversal(const signed char *spins, const signed char *targets,
                             const ScoreInt *costs, int e) {
    return spins[e] == targets[e] ? costs[e] : -costs[e];
}
__device__ int proposed_charge(const int *inc, const signed char *spins, int v, const Shared &w) {
    int charge = 0;
    for (int k = 0; k < 4; ++k) {
        int signed_e = inc[v * 4 + k];
        if (signed_e == 0) continue;
        int e = magnitude(signed_e) - 1;
        int s = in_path(w, e) ? -spins[e] : spins[e];
        charge += (signed_e > 0 ? 1 : -1) * s;
    }
    return charge;
}

// Batched full direct scores for both orientations. The old score and
// charges must agree with the running state before a proposal is considered.
// A proposal may repair gaps but cannot make any individual pin worse.
__device__ void try_move(int n, int vertices, const int *inc, const int *labels,
                        const ScoreInt *costs, const signed char *targets,
                        signed char *spins, int *charges, ScoreInt *scores,
                        int *status, Shared &w) {
    int t = threadIdx.x;
    ScoreInt ng = 0, og = 0, no = 0, oo = 0, flux = 0;
    int bad = 0, defect = 0;
    for (int v = t; v < vertices; v += THREADS) {
        int before = 0;
        for (int k = 0; k < 4; ++k) {
            int e = inc[v * 4 + k];
            if (e) before += (e > 0 ? 1 : -1) * spins[magnitude(e) - 1];
        }
        defect |= before != charges[v];
        int after = proposed_charge(inc, spins, v, w);
        flux += after - before;
        if (labels[v] != FREE) {
            ScoreInt a = after - labels[v], b = before - labels[v];
            ng += a * a;
            og += b * b;
            bad |= a * a > b * b;
        }
    }
    for (int e = t; e < n; e += THREADS) {
        int s = spins[e];
        oo += s != targets[e] ? costs[e] : 0;
        if (in_path(w, e)) s = -s;
        no += s != targets[e] ? costs[e] : 0;
    }
    w.new_gap[t] = ng; w.old_gap[t] = og;
    w.new_obj[t] = no; w.old_obj[t] = oo; w.flux[t] = flux;
    w.pin_bad[t] = bad; w.defect[t] = defect;
    __syncthreads();
    for (int d = THREADS / 2; d; d /= 2) {
        if (t < d) {
            w.new_gap[t] += w.new_gap[t + d]; w.old_gap[t] += w.old_gap[t + d];
            w.new_obj[t] += w.new_obj[t + d]; w.old_obj[t] += w.old_obj[t + d];
            w.flux[t] += w.flux[t + d];
            w.pin_bad[t] |= w.pin_bad[t + d]; w.defect[t] |= w.defect[t + d];
        }
        __syncthreads();
    }
    if (t == 0) {
        bool consistent = !w.defect[0] && scores[0] == w.old_gap[0] && scores[1] == w.old_obj[0];
        if (!consistent || w.flux[0] != 0) *status = 1;
        w.keep = consistent && w.flux[0] == 0 && !w.pin_bad[0] && w.length > 0 &&
            (w.new_gap[0] < scores[0] || (w.new_gap[0] == scores[0] && w.new_obj[0] < scores[1]));
    }
    __syncthreads();
    // Compute new charges while every thread still reads the old spins.
    if (w.keep) {
        for (int v = t; v < vertices; v += THREADS) charges[v] = proposed_charge(inc, spins, v, w);
    }
    __syncthreads();
    if (t == 0 && w.keep) {
        for (int k = 0; k < w.length; ++k) spins[w.path[k]] = -spins[w.path[k]];
        scores[0] = w.new_gap[0]; scores[1] = w.new_obj[0];
        w.changed = 1;
    }
    __syncthreads();
}

// CPU-12 flow_loops.rs:731-773. Directed BFS in incidence order, length
// at most 64, from a short pin to an excess pin or the free boundary.
__device__ void repair_path(int root, int vertices, const int *ends, const int *inc,
                           const int *labels, const signed char *spins, const int *charges,
                           int *scratch, Shared &w) {
    w.length = 0;
    if (labels[root] == FREE || charges[root] - labels[root] > -2) return;
    int *reach = scratch, *depth = scratch + vertices, *queue = scratch + 2 * vertices;
    for (int v = 0; v < vertices; ++v) reach[v] = -1;
    int tail = 0, head = 1, found = -1;
    queue[0] = root; depth[root] = 0; reach[root] = -2;
    while (tail < head && found < 0) {
        int u = queue[tail++];
        if (depth[u] >= MAX_PATH) continue;
        for (int k = 0; k < 4; ++k) {
            int se = inc[4 * u + k];
            if (!se) continue;
            int e = magnitude(se) - 1;
            if ((se > 0 ? 1 : -1) * spins[e] != -1) continue;
            int v = other(ends, e, u);
            if (reach[v] != -1) continue;
            reach[v] = e; depth[v] = depth[u] + 1;
            if (labels[v] == FREE || charges[v] - labels[v] >= 2) { found = v; break; }
            queue[head++] = v;
        }
    }
    if (found < 0) return;
    int at = found;
    while (at != root) {
        int e = reach[at];
        w.path[w.length++] = e;
        at = other(ends, e, at);
    }
}

// CPU-12 flow_loops.rs:600-665, with first-incidence ties and no random
// visitation. Never repeat a link, and undo a walk that exceeds 64 links.
__device__ void worm_path(int first, const int *ends, const int *inc, const int *labels,
                         const signed char *spins, const signed char *targets,
                         const ScoreInt *costs, Shared &w) {
    w.path[0] = first; w.length = 1;
    int at = spins[first] == 1 ? ends[2 * first + 1] : ends[2 * first];
    while (labels[at] != FREE && proposed_charge(inc, spins, at, w) < labels[at]) {
        if (w.length == MAX_PATH) { w.length = 0; return; }
        int pick = -1;
        ScoreInt best = 0;
        for (int k = 0; k < 4; ++k) {
            int se = inc[4 * at + k];
            if (!se) continue;
            int e = magnitude(se) - 1;
            if (in_path(w, e) || (se > 0 ? 1 : -1) * spins[e] != -1) continue;
            ScoreInt delta = reversal(spins, targets, costs, e);
            if (pick < 0 || delta < best) { pick = e; best = delta; }
        }
        if (pick < 0) { w.length = 0; return; }
        w.path[w.length++] = pick;
        at = other(ends, pick, at);
    }
}

extern "C" __global__ void quip_flow_loops(
    int n, int vertices, int cycle_count, int passes,
    const int *ends, const int *inc, const int *labels,
    const ScoreInt *costs, const signed char *targets, const int *cycles,
    signed char *all_spins, int *all_charges, ScoreInt *all_scores,
    int *all_scratch, int *all_status) {
    unsigned long long replica = blockIdx.x;
    signed char *spins = all_spins + replica * n;
    int *charges = all_charges + replica * vertices;
    ScoreInt *scores = all_scores + 2 * replica;
    int *scratch = all_scratch + replica * vertices * 3;
    int *status = all_status + replica;
    int t = threadIdx.x;
    __shared__ Shared w;
    for (int pass = 0; pass < passes; ++pass) {
        if (t == 0) w.changed = 0;
        __syncthreads();
        for (int v = 0; v < vertices; ++v) {
            if (t == 0) repair_path(v, vertices, ends, inc, labels, spins, charges, scratch, w);
            __syncthreads();
            if (w.length) try_move(n, vertices, inc, labels, costs, targets, spins, charges, scores, status, w);
        }
        // CPU-12 flow_loops.rs:561-593: directed short-cycle steepest
        // descent. Evaluate candidates concurrently, first cycle on ties.
        while (true) {
            ScoreInt best = 0; int best_i = -1;
            for (int c = t; c < cycle_count; c += THREADS) {
                const int *row = cycles + 5 * c;
                int sense = 0; bool directed = true; ScoreInt delta = 0;
                for (int k = 1; k <= row[0]; ++k) {
                    int se = row[k], e = magnitude(se) - 1;
                    int s = spins[e] * (se > 0 ? 1 : -1);
                    if (k == 1) sense = s;
                    directed &= s == sense;
                    delta += reversal(spins, targets, costs, e);
                }
                if (directed && delta < best) { best = delta; best_i = c; }
            }
            w.best_delta[t] = best; w.best_index[t] = best_i;
            __syncthreads();
            for (int d = THREADS / 2; d; d /= 2) {
                if (t < d && (w.best_delta[t + d] < w.best_delta[t] ||
                    (w.best_delta[t + d] == w.best_delta[t] && w.best_index[t + d] >= 0 &&
                     (w.best_index[t] < 0 || w.best_index[t + d] < w.best_index[t])))) {
                    w.best_delta[t] = w.best_delta[t + d]; w.best_index[t] = w.best_index[t + d];
                }
                __syncthreads();
            }
            if (w.best_index[0] < 0) break;
            if (t == 0) {
                const int *row = cycles + 5 * w.best_index[0];
                w.length = row[0];
                for (int k = 0; k < w.length; ++k) w.path[k] = magnitude(row[k + 1]) - 1;
            }
            __syncthreads();
            try_move(n, vertices, inc, labels, costs, targets, spins, charges, scores, status, w);
            if (!w.keep || *status) break;
        }
        for (int e = 0; e < n; ++e) {
            if (t == 0) worm_path(e, ends, inc, labels, spins, targets, costs, w);
            __syncthreads();
            if (w.length) try_move(n, vertices, inc, labels, costs, targets, spins, charges, scores, status, w);
        }
        if (!w.changed || *status) break;
    }
}
