//! A lean CPU generator of the couplings of `quip/twisted-ice-dimer@1`.
//!
//! It reproduces only what the structural screen needs (the coupling of every edge) and skips what the canonical
//! `generate_instance` spends most of its time on: re-finding the candidate tiles and spine sides for every nonce, building
//! and validating an `Instance` (digests, sorting, metadata round trip). The result is the same `j` vector, edge for edge, as
//! the canonical instance; `lean_matches_canonical` checks that. It is also the reference for the CUDA port.

use quip_solver_core::ising_generators::family::BaseTopology;
use quip_solver_core::ising_generators::ice::{candidates, select, sides, Candidates, Lattice, Sides, PAIR_MILLI, STRONG_MILLI};
use quip_solver_core::ising_generators::rng::{shuffle, stream_seed, uniform, ChaCha8Rng};
use std::num::NonZeroU32;

pub(crate) struct LeanGen {
    cands: Candidates,
    sides: Sides,
    /// Sorted edges of the topology, as the canonical instance lists them.
    pub(crate) edges: Vec<(u32, u32)>,
    /// Edge positions of the six edges of every candidate tile (parallel to `cands.tiles`).
    tile_edges: Vec<[u32; 6]>,
    nodes: usize,
    weak: i32,
    pairs: usize,
    desc_digest: [u8; 32],
    order_digest: [u8; 32],
    gt: Option<GpuTables>,
}

const LEFT: &[u8] = b"left";
const RIGHT: &[u8] = b"right";
const GAUGE: &[u8] = b"gauge";
const WEAK: &[u8] = b"weak";
const DIMER: &[u8] = b"dimer";

impl LeanGen {
    pub(crate) fn new(base: &BaseTopology, desc_digest: [u8; 32], m: u16, t: u8, weak: i32, pairs: usize) -> Option<Self> {
        let lattice = Lattice::Zephyr { m, t };
        let cands = candidates(lattice, base).ok()?;
        let sides = sides(lattice, base, &cands).ok()?;
        let mut edges: Vec<(u32, u32)> = base.edges().iter().map(|&(u, v)| (u.min(v), u.max(v))).collect();
        edges.sort_unstable();
        let mut tile_edges = Vec::with_capacity(cands.tiles.len());
        for tile in &cands.tiles {
            let [a, b, c, d] = tile.nodes;
            let mut e = [0u32; 6];
            for (k, pair) in [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)].iter().enumerate() {
                e[k] = u32::try_from(edges.binary_search(pair).ok()?).ok()?;
            }
            tile_edges.push(e);
        }
        let mut this = Self { cands, sides, edges, tile_edges, nodes: base.node_ids().len(), weak, pairs, desc_digest, order_digest: *base.order_digest(), gt: None };
        this.gt = this.build_gpu_tables();
        Some(this)
    }

    fn rng(&self, nonce: &[u8; 32], label: &[u8]) -> ChaCha8Rng {
        ChaCha8Rng::from_seed(stream_seed(nonce, &self.desc_digest, &self.order_digest, label).expect("static label"))
    }

    /// The coupling of every edge (milli units), in the order of `self.edges`.
    pub(crate) fn couplings(&self, nonce: &[u8; 32]) -> Option<Vec<i32>> {
        let mut left: Vec<u32> = (0..u32::try_from(self.sides.left.len()).ok()?).collect();
        shuffle(&mut self.rng(nonce, LEFT), &mut left).ok()?;
        let mut right: Vec<u32> = (0..u32::try_from(self.sides.right.len()).ok()?).collect();
        shuffle(&mut self.rng(nonce, RIGHT), &mut right).ok()?;
        let tiles = select(&self.cands, &self.sides, &left, &right).ok()?;
        let two = NonZeroU32::new(2)?;
        let mut gauge = self.rng(nonce, GAUGE);
        let mut strong: Vec<i32> = vec![0; self.edges.len()];
        let mut is_strong = vec![false; self.edges.len()];
        for tile in &tiles {
            let g: [i32; 4] = [(); 4].map(|()| if uniform(&mut gauge, two) == 0 { 1 } else { -1 });
            let at = self.cands.tiles.binary_search(tile).ok()?;
            let [a, b, c, d] = g;
            let pairs = [a * b, a * c, a * d, b * c, b * d, c * d];
            for (k, &e) in self.tile_edges[at].iter().enumerate() {
                strong[e as usize] = STRONG_MILLI * pairs[k];
                is_strong[e as usize] = true;
            }
        }
        let mut weak = self.rng(nonce, WEAK);
        let mut j: Vec<i32> = Vec::with_capacity(self.edges.len());
        for (at, &fixed) in is_strong.iter().enumerate() {
            j.push(if fixed {
                strong[at]
            } else if uniform(&mut weak, two) == 0 {
                self.weak
            } else {
                -self.weak
            });
        }
        let mut order: Vec<u32> = (0..self.edges.len()).filter(|&at| !is_strong[at]).map(|at| at as u32).collect();
        shuffle(&mut self.rng(nonce, DIMER), &mut order).ok()?;
        let mut used = vec![false; self.nodes];
        let mut taken = 0usize;
        for at in order {
            if taken >= self.pairs {
                break;
            }
            let (u, v) = self.edges[at as usize];
            if used[u as usize] || used[v as usize] {
                continue;
            }
            used[u as usize] = true;
            used[v as usize] = true;
            j[at as usize] = PAIR_MILLI;
            taken += 1;
        }
        (taken == self.pairs).then_some(j)
    }
}

/// The triangles of the topology as edge-position triples, for the structural features of a coupling vector.
pub(crate) struct Triangles {
    tri: Vec<[u32; 3]>,
    edge_count: usize,
}

impl Triangles {
    pub(crate) fn new(n: usize, edges: &[(u32, u32)]) -> Self {
        // Adjacency with the edge position of each neighbour, sorted by neighbour id.
        let mut adj: Vec<Vec<(u32, u32)>> = vec![Vec::new(); n];
        for (at, &(a, b)) in edges.iter().enumerate() {
            adj[a as usize].push((b, at as u32));
            adj[b as usize].push((a, at as u32));
        }
        for row in &mut adj {
            row.sort_unstable_by_key(|x| x.0);
        }
        let mut tri = Vec::new();
        for (at, &(a, b)) in edges.iter().enumerate() {
            let (a, b) = if a < b { (a as usize, b) } else { (b as usize, a) };
            let (ra, rb) = (&adj[a], &adj[b as usize]);
            let (mut i, mut k) = (0usize, 0usize);
            while i < ra.len() && k < rb.len() {
                let (na, ea) = ra[i];
                let (nb, eb) = rb[k];
                if na < nb {
                    i += 1;
                } else if na > nb {
                    k += 1;
                } else {
                    if na > b {
                        tri.push([at as u32, ea, eb]);
                    }
                    i += 1;
                    k += 1;
                }
            }
        }
        Self { tri, edge_count: edges.len() }
    }

    pub(crate) fn len(&self) -> usize {
        self.tri.len()
    }

    /// The nine structural features of `crate::cluster::sampler::lease_run::features_raw`, from the triangle table.
    pub(crate) fn features(&self, js: &[i32]) -> [f64; 9] {
        // Two flags per edge: strong (|j| >= 500) in bit 0, non-positive (j <= 0) in bit 1.
        let flag: Vec<u8> = js.iter().map(|&j| u8::from(j.abs() >= 500) | (u8::from(j <= 0) << 1)).collect();
        let mut cnt = [0u32; 8];
        for t in &self.tri {
            let (x, y, z) = (flag[t[0] as usize], flag[t[1] as usize], flag[t[2] as usize]);
            let ns = usize::from((x & 1) + (y & 1) + (z & 1));
            let nonpos = ((x >> 1) + (y >> 1) + (z >> 1)) as usize;
            cnt[ns * 2 + usize::from(nonpos % 2 == 0)] += 1;
        }
        let weak_pos = js.iter().filter(|&&j| j != 0 && j.abs() < 500 && j > 0).count();
        let mf = self.edge_count.max(1) as f64;
        let mut f = [0f64; 9];
        for i in 0..8 {
            f[i] = f64::from(cnt[i]) / mf;
        }
        f[8] = weak_pos as f64 / mf;
        f
    }
}

/// The fixed structure the GPU generator needs, in flat arrays.
pub(crate) struct GpuTables {
    pub(crate) left: usize,
    pub(crate) right: usize,
    pub(crate) cands: usize,
    pub(crate) edges: usize,
    pub(crate) nodes: usize,
    pub(crate) sel_expected: usize,
    pub(crate) weak: i32,
    pub(crate) pairs: usize,
    pub(crate) cand_edges: Vec<u32>,
    pub(crate) adj_off: Vec<i32>,
    pub(crate) adj_r: Vec<u16>,
    pub(crate) adj_c: Vec<u32>,
    pub(crate) eu: Vec<u16>,
    pub(crate) ev: Vec<u16>,
}

impl LeanGen {
    /// The ChaCha keys of the five streams of `nonce` (left, right, gauge, weak, dimer), eight words each.
    pub(crate) fn key_words(&self, nonce: &[u8; 32]) -> [u32; 40] {
        let mut out = [0u32; 40];
        for (k, label) in [LEFT, RIGHT, GAUGE, WEAK, DIMER].iter().enumerate() {
            let seed = stream_seed(nonce, &self.desc_digest, &self.order_digest, label).expect("static label");
            let words = quip_solver_core::ising_generators::rng::key_words(&seed);
            out[8 * k..8 * k + 8].copy_from_slice(&words);
        }
        out
    }

    pub(crate) fn gpu_tables(&self) -> Option<&GpuTables> {
        self.gt.as_ref()
    }

    fn build_gpu_tables(&self) -> Option<GpuTables> {
        // Left spine -> (right spine, candidate tile), as `select` pairs them.
        let mut adj: Vec<Vec<(u16, u32)>> = vec![Vec::new(); self.sides.left.len()];
        for (c, spine) in self.cands.spines.iter().enumerate() {
            let [e, f] = (*spine)?;
            let straight = (self.sides.left.binary_search(&e), self.sides.right.binary_search(&f));
            let swapped = (self.sides.left.binary_search(&f), self.sides.right.binary_search(&e));
            let (Ok(l), Ok(r)) = (if straight.0.is_ok() { straight } else { swapped }) else { return None };
            adj[l].push((u16::try_from(r).ok()?, u32::try_from(c).ok()?));
        }
        let mut adj_off = vec![0i32];
        let (mut adj_r, mut adj_c) = (Vec::new(), Vec::new());
        for list in &adj {
            for &(r, c) in list {
                adj_r.push(r);
                adj_c.push(c);
            }
            adj_off.push(i32::try_from(adj_r.len()).ok()?);
        }
        let cand_edges: Vec<u32> = self.tile_edges.iter().flatten().copied().collect();
        // Size of the maximum matching, from one canonical selection.
        let mut left: Vec<u32> = (0..u32::try_from(self.sides.left.len()).ok()?).collect();
        let mut right: Vec<u32> = (0..u32::try_from(self.sides.right.len()).ok()?).collect();
        let nonce = [7u8; 32];
        shuffle(&mut self.rng(&nonce, LEFT), &mut left).ok()?;
        shuffle(&mut self.rng(&nonce, RIGHT), &mut right).ok()?;
        let sel_expected = select(&self.cands, &self.sides, &left, &right).ok()?.len();
        Some(GpuTables {
            left: self.sides.left.len(),
            right: self.sides.right.len(),
            cands: self.cands.tiles.len(),
            edges: self.edges.len(),
            nodes: self.nodes,
            sel_expected,
            weak: self.weak,
            pairs: self.pairs,
            cand_edges,
            adj_off,
            adj_r,
            adj_c,
            eu: self.edges.iter().map(|e| e.0 as u16).collect(),
            ev: self.edges.iter().map(|e| e.1 as u16).collect(),
        })
    }
}

impl Triangles {
    /// The triangle table as u16 edge-position triples.
    pub(crate) fn flat_u16(&self) -> Vec<u16> {
        self.tri.iter().flat_map(|t| t.iter().map(|&x| x as u16)).collect()
    }
}

/// Reusable buffers for [`LeanGen::fast_flags`]: nothing is allocated per salt.
pub(crate) struct FastScratch {
    left: Vec<u16>,
    pa: Vec<u16>,
    rank: Vec<u16>,
    matched: Vec<i16>,
    stamp: Vec<u16>,
    sel: Vec<u32>,
    used: Vec<u32>,
    /// Per edge: bit 0 strong, bit 1 non-positive.
    pub(crate) flags: Vec<u8>,
}

impl FastScratch {
    pub(crate) fn new(gt: &GpuTables) -> Self {
        Self {
            left: vec![0; gt.left],
            pa: vec![0; gt.edges.max(gt.right)],
            rank: vec![0; gt.right],
            matched: vec![-1; gt.right],
            stamp: vec![0; gt.right],
            sel: vec![0; (gt.cands + 31) / 32],
            used: vec![0; (gt.nodes + 31) / 32],
            flags: vec![0; gt.edges],
        }
    }
}

impl LeanGen {
    /// The coupling classes of the instance of `nonce`, without allocation (same construction as `couplings`, which it matches
    /// bit for bit, and as the CUDA kernel `gen_s0.cu`). Fills `sc.flags`.
    pub(crate) fn fast_flags(&self, nonce: &[u8; 32], sc: &mut FastScratch) -> bool {
        let Some(gt) = self.gt.as_ref() else { return false };
        let two = NonZeroU32::MIN.saturating_add(1);
        for (i, v) in sc.left.iter_mut().enumerate() {
            *v = i as u16;
        }
        if shuffle(&mut self.rng(nonce, LEFT), &mut sc.left).is_err() {
            return false;
        }
        {
            let order = &mut sc.pa[..gt.right];
            for (i, v) in order.iter_mut().enumerate() {
                *v = i as u16;
            }
            if shuffle(&mut self.rng(nonce, RIGHT), order).is_err() {
                return false;
            }
            for (p, &r) in order.iter().enumerate() {
                sc.rank[r as usize] = p as u16;
            }
        }
        sc.matched.iter_mut().for_each(|m| *m = -1);
        sc.stamp.iter_mut().for_each(|m| *m = 0);
        sc.sel.iter_mut().for_each(|m| *m = 0);
        // Maximum matching, augmenting paths over the right spines in rank order.
        let sorted = |left: usize, rank: &[u16], rs: &mut [u16; 4], cs: &mut [u32; 4]| -> usize {
            let b = gt.adj_off[left] as usize;
            let n = gt.adj_off[left + 1] as usize - b;
            for k in 0..n {
                let (r, c) = (gt.adj_r[b + k], gt.adj_c[b + k]);
                let mut p = k;
                while p > 0 && rank[rs[p - 1] as usize] > rank[r as usize] {
                    rs[p] = rs[p - 1];
                    cs[p] = cs[p - 1];
                    p -= 1;
                }
                rs[p] = r;
                cs[p] = c;
            }
            n
        };
        let mut sl = [0u16; 256];
        let mut sp = [0u8; 256];
        for p in 0..gt.left {
            let id = (p + 1) as u16;
            let mut depth = 1usize;
            sl[0] = sc.left[p];
            sp[0] = 0;
            while depth > 0 {
                let left = sl[depth - 1] as usize;
                let position = sp[depth - 1] as usize;
                let (mut rs, mut cs) = ([0u16; 4], [0u32; 4]);
                let n = sorted(left, &sc.rank, &mut rs, &mut cs);
                if position >= n {
                    depth -= 1;
                    continue;
                }
                sp[depth - 1] = (position + 1) as u8;
                let right = rs[position] as usize;
                if sc.stamp[right] == id {
                    continue;
                }
                sc.stamp[right] = id;
                let owner = sc.matched[right];
                if owner >= 0 {
                    if depth >= 256 {
                        return false;
                    }
                    sl[depth] = owner as u16;
                    sp[depth] = 0;
                    depth += 1;
                    continue;
                }
                for f in 0..depth {
                    let (mut rr, mut cc) = ([0u16; 4], [0u32; 4]);
                    sorted(sl[f] as usize, &sc.rank, &mut rr, &mut cc);
                    sc.matched[rr[sp[f] as usize - 1] as usize] = sl[f] as i16;
                }
                break;
            }
        }
        let mut selected = 0usize;
        for r in 0..gt.right {
            let l = sc.matched[r];
            if l < 0 {
                continue;
            }
            let b = gt.adj_off[l as usize] as usize;
            let n = gt.adj_off[l as usize + 1] as usize - b;
            for k in 0..n {
                if gt.adj_r[b + k] as usize == r {
                    let c = gt.adj_c[b + k] as usize;
                    sc.sel[c >> 5] |= 1 << (c & 31);
                    selected += 1;
                    break;
                }
            }
        }
        if selected != gt.sel_expected {
            return false;
        }
        sc.flags.iter_mut().for_each(|f| *f = 0);
        let mut gauge = self.rng(nonce, GAUGE);
        for c in 0..gt.cands {
            if (sc.sel[c >> 5] >> (c & 31)) & 1 == 0 {
                continue;
            }
            let mut g = [1i32; 4];
            for v in &mut g {
                *v = if uniform(&mut gauge, two) == 0 { 1 } else { -1 };
            }
            let ce = &gt.cand_edges[c * 6..c * 6 + 6];
            let prod = [g[0] * g[1], g[0] * g[2], g[0] * g[3], g[1] * g[2], g[1] * g[3], g[2] * g[3]];
            for k in 0..6 {
                sc.flags[ce[k] as usize] = 1 | if prod[k] < 0 { 2 } else { 0 };
            }
        }
        let mut weak = self.rng(nonce, WEAK);
        let mut m = 0usize;
        for e in 0..gt.edges {
            if sc.flags[e] & 1 != 0 {
                continue;
            }
            sc.flags[e] = if uniform(&mut weak, two) == 0 { 0 } else { 2 };
            sc.pa[m] = e as u16;
            m += 1;
        }
        if shuffle(&mut self.rng(nonce, DIMER), &mut sc.pa[..m]).is_err() {
            return false;
        }
        sc.used.iter_mut().for_each(|u| *u = 0);
        let mut taken = 0usize;
        for k in 0..m {
            if taken >= gt.pairs {
                break;
            }
            let e = sc.pa[k] as usize;
            let (u, v) = (gt.eu[e] as usize, gt.ev[e] as usize);
            if (sc.used[u >> 5] >> (u & 31)) & 1 != 0 || (sc.used[v >> 5] >> (v & 31)) & 1 != 0 {
                continue;
            }
            sc.used[u >> 5] |= 1 << (u & 31);
            sc.used[v >> 5] |= 1 << (v & 31);
            sc.flags[e] = 3;
            taken += 1;
        }
        taken == gt.pairs
    }
}

impl Triangles {
    /// The nine structural features from per-edge flags (bit 0 strong, bit 1 non-positive).
    pub(crate) fn features_flags(&self, flags: &[u8]) -> [f64; 9] {
        let mut cnt = [0u32; 8];
        for t in &self.tri {
            let (x, y, z) = (flags[t[0] as usize], flags[t[1] as usize], flags[t[2] as usize]);
            let ns = usize::from((x & 1) + (y & 1) + (z & 1));
            let nonpos = (((x >> 1) & 1) + ((y >> 1) & 1) + ((z >> 1) & 1)) as usize;
            cnt[ns * 2 + usize::from(nonpos % 2 == 0)] += 1;
        }
        let weak_pos = flags.iter().filter(|&&f| f & 3 == 0).count();
        let mf = self.edge_count.max(1) as f64;
        let mut f = [0f64; 9];
        for i in 0..8 {
            f[i] = f64::from(cnt[i]) / mf;
        }
        f[8] = weak_pos as f64 / mf;
        f
    }
}
