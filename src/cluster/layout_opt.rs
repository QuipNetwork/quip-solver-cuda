//! Shared-memory bank-conflict aware node order for the cluster kernel.
//!
//! The kernel reads `s[v]` (one byte per node) and `parent[v]` (one word per node) for the neighbours `v` of the 32 nodes of a
//! warp. Nsight Compute measured 32% of all shared wavefronts as excess (about 3.4 wavefronts per request where one would do).
//! The nodes of one colour class are interchangeable, so their order inside the class is free. This module searches an order
//! that spreads the neighbours of each warp over distinct banks. The graph is the same for every instance of a family, so the
//! order is computed once and stored.

const WARP: usize = 32;

pub(crate) struct Problem {
    n: usize,
    /// Neighbour slots of every original node, in the order the ELL builder lists them.
    rows: Vec<Vec<u32>>,
    /// Colour class offsets in device order.
    off: Vec<usize>,
}

impl Problem {
    pub(crate) fn new(n: usize, edges: &[(usize, usize)], color: &[usize]) -> Self {
        let mut rows = vec![Vec::new(); n];
        for reverse in [false, true] {
            for &(u, v) in edges {
                let (u, v) = if reverse { (v, u) } else { (u, v) };
                rows[u].push(v as u32);
            }
        }
        let classes = color.iter().max().map_or(0, |m| m + 1);
        let mut off = vec![0usize; classes + 1];
        for &c in color {
            off[c + 1] += 1;
        }
        for c in 1..off.len() {
            off[c] += off[c - 1];
        }
        Self { n, rows, off }
    }

    /// Device position of every original node for the order `perm` (device position to original node).
    fn inverse(&self, perm: &[usize]) -> Vec<usize> {
        let mut inv = vec![0usize; self.n];
        for (k, &o) in perm.iter().enumerate() {
            inv[o] = k;
        }
        inv
    }

    /// Cost of one warp group starting at device position `start` with `len` lanes: for every neighbour slot, the largest number
    /// of distinct shared words that fall in one bank, once for the byte array (4 nodes per word) and once for the word array.
    fn warp_cost(&self, perm: &[usize], inv: &[usize], start: usize, len: usize) -> u32 {
        let width = (0..len).map(|l| self.rows[perm[start + l]].len()).max().unwrap_or(0);
        let mut total = 0u32;
        let mut bytes = [[usize::MAX; WARP]; WARP];
        let mut words = [[usize::MAX; WARP]; WARP];
        for e in 0..width {
            let mut bs = [0u8; WARP];
            let mut ws = [0u8; WARP];
            for l in 0..len {
                let row = &self.rows[perm[start + l]];
                if e >= row.len() {
                    continue;
                }
                let v = inv[row[e] as usize];
                let (wb, bb) = (v / 4, (v / 4) % WARP);
                let seen = &mut bytes[bb][..bs[bb] as usize];
                if !seen.contains(&wb) {
                    bytes[bb][bs[bb] as usize] = wb;
                    bs[bb] += 1;
                }
                let bw = v % WARP;
                let seen = &mut words[bw][..ws[bw] as usize];
                if !seen.contains(&v) {
                    words[bw][ws[bw] as usize] = v;
                    ws[bw] += 1;
                }
            }
            total += u32::from(*bs.iter().max().unwrap_or(&0)) + u32::from(*ws.iter().max().unwrap_or(&0));
        }
        total
    }

    /// The warp groups of the colour classes: `(start, len)` per group.
    fn groups(&self) -> Vec<(usize, usize)> {
        let mut g = Vec::new();
        for c in 0..self.off.len() - 1 {
            let (mut s, e) = (self.off[c], self.off[c + 1]);
            while s < e {
                let len = WARP.min(e - s);
                g.push((s, len));
                s += len;
            }
        }
        g
    }

    /// Total cost of an order and the same cost with every slot conflict-free (the ideal), for reporting.
    pub(crate) fn total(&self, perm: &[usize]) -> (u64, u64) {
        let inv = self.inverse(perm);
        let (mut cost, mut ideal) = (0u64, 0u64);
        for (s, l) in self.groups() {
            cost += u64::from(self.warp_cost(perm, &inv, s, l));
            let width = (0..l).map(|k| self.rows[perm[s + k]].len()).max().unwrap_or(0);
            ideal += 2 * width as u64;
        }
        (cost, ideal)
    }

    /// Search an order by swapping nodes inside colour classes and keeping swaps that do not raise the cost.
    pub(crate) fn optimize(&self, mut perm: Vec<usize>, moves: u64, seed: u64) -> Vec<usize> {
        let groups = self.groups();
        let mut gid = vec![0usize; self.n];
        for (i, &(s, l)) in groups.iter().enumerate() {
            for p in s..s + l {
                gid[p] = i;
            }
        }
        let mut inv = self.inverse(&perm);
        let mut wc: Vec<u32> = groups.iter().map(|&(s, l)| self.warp_cost(&perm, &inv, s, l)).collect();
        let mut rng = seed | 1;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let classes = self.off.len() - 1;
        let mut touched: Vec<usize> = Vec::new();
        for it in 0..moves {
            let c = (next() % classes as u64) as usize;
            let size = self.off[c + 1] - self.off[c];
            if size < 2 {
                continue;
            }
            let a = self.off[c] + (next() % size as u64) as usize;
            let b = self.off[c] + (next() % size as u64) as usize;
            if a == b {
                continue;
            }
            // Groups whose cost can change: those of the two positions and those holding a neighbour of either node.
            touched.clear();
            touched.push(gid[a]);
            touched.push(gid[b]);
            for &x in &[perm[a], perm[b]] {
                for &v in &self.rows[x] {
                    touched.push(gid[inv[v as usize]]);
                }
            }
            touched.sort_unstable();
            touched.dedup();
            let before: u32 = touched.iter().map(|&g| wc[g]).sum();
            perm.swap(a, b);
            inv[perm[a]] = a;
            inv[perm[b]] = b;
            let new: Vec<u32> = touched.iter().map(|&g| self.warp_cost(&perm, &inv, groups[g].0, groups[g].1)).collect();
            let after: u32 = new.iter().sum();
            // Slowly cooling acceptance of small increases.
            let t0: f64 = std::env::var("LAYOUT_T").ok().and_then(|v| v.parse().ok()).unwrap_or(0.25);
            let temp = t0 * (1.0 - it as f64 / moves as f64);
            let accept = after <= before || ((next() >> 11) as f64 / (1u64 << 53) as f64) < (-(f64::from(after - before)) / temp.max(1e-3)).exp();
            if accept {
                for (g, &w) in touched.iter().zip(&new) {
                    wc[*g] = w;
                }
            } else {
                perm.swap(a, b);
                inv[perm[a]] = a;
                inv[perm[b]] = b;
            }
        }
        perm
    }
}
