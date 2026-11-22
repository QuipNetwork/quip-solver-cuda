//! Zero-allocation pure-CPU FastDimerV3 + AVX2/AVX-512/NEON Rank-8/Rank-4 BM-SDP
//! + Jacobi Eigenvector Seed + V3Fast Micro-BP screening engine.
//!
//! 100% bit-exact to `quip-cuda-native-two-pass` `SuperScreen` V5.1 (`V3Fast` / `V3`),
//! with ZERO CUDA dependencies and ZERO per-salt heap allocations on the 98% rejected path.

use ising_generators::descriptor::{admit, params_digest, Value};
use ising_generators::family::{BaseTopology, Streams};
use ising_generators::ice::{
    candidates, sides, Candidates, Lattice, Sides, TWISTED_DIMER, TWISTED_DIMER_V1,
};
use ising_generators::rng::{key_words, stream_seed, ChaCha8Rng};
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::Mutex;

const TAG_M: u16 = 1;
const TAG_T: u16 = 2;
const TAG_WEAK: u16 = 3;
const TAG_PAIRS: u16 = 4;

fn unsigned(value: Option<Value<'_>>) -> Option<u64> {
    match value {
        Some(Value::Unsigned(v)) => Some(v),
        _ => None,
    }
}

#[derive(Clone, Copy)]
pub struct DivEntry {
    pub limit: u64,
    pub inv: u64,
}

#[derive(Clone, Copy)]
pub struct PreTile {
    pub nodes: [u32; 4],
    pub eids: [u32; 6],
}

#[inline(always)]
fn fast_uniform_u32(rng: &mut ChaCha8Rng, n: u32, tbl: &[DivEntry]) -> u32 {
    debug_assert!((n as usize) < tbl.len() && n >= 2);
    let entry = unsafe { *tbl.get_unchecked(n as usize) };
    let n64 = n as u64;
    loop {
        let x = rng.next_u32() as u64;
        if x < entry.limit {
            let q = ((u128::from(x) * u128::from(entry.inv)) >> 64) as u64;
            return (x - q * n64) as u32;
        }
    }
}

#[inline(always)]
fn fast_shuffle_u32(rng: &mut ChaCha8Rng, items: &mut [u32], tbl: &[DivEntry]) {
    let len = items.len() as u32;
    for i in (1..len).rev() {
        let j = fast_uniform_u32(rng, i + 1, tbl);
        items.swap(i as usize, j as usize);
    }
}

pub struct FastDimer {
    pub order_digest: [u8; 32],
    pub descriptor_digest: [u8; 32],
    _found: Candidates,
    _split: Sides,
    pub edges: Vec<(u32, u32)>,
    pub weak: i32,
    pub pairs: usize,
    pub nodes: usize,
    pub cand_tiles: Vec<PreTile>,
    pub fixed_cids: Vec<u32>,
    pub n_left: usize,
    pub n_right: usize,
    pub left_off: Vec<usize>,
    pub left_entries: Vec<(u32, u32)>,
    pub div_table: Vec<DivEntry>,
}

impl FastDimer {
    /// Compute the 5 ChaCha8 256-bit keys (`40 u32`s = 160 bytes) for GPU-native salt generation.
    #[inline]
    pub fn compute_stream_keys(&self, nonce: &[u8; 32]) -> Option<[u32; 40]> {
        let labels = TWISTED_DIMER.labels;
        let mut out = [0u32; 40];
        for (i, &lbl) in labels.iter().enumerate() {
            let s = stream_seed(nonce, &self.descriptor_digest, &self.order_digest, lbl.as_bytes()).ok()?;
            let kw = key_words(&s);
            out[i * 8..(i + 1) * 8].copy_from_slice(&kw);
        }
        Some(out)
    }
    pub fn from_raw(
        node_ids: Vec<u32>,
        raw_edges: &[(u32, u32)],
        params: &[u8],
        expected_order_digest: [u8; 32],
    ) -> Result<Self, String> {
        // `BaseTopology::edges()` returns dense indices 0..node_ids.len(), while
        // `BaseTopology::new(node_ids, edges)` expects original node_ids endpoints!
        let orig_edges: Vec<(u32, u32)> = raw_edges
            .iter()
            .map(|&(u, v)| {
                let nu = *node_ids
                    .get(u as usize)
                    .ok_or_else(|| format!("edge u={u} out of bounds {}", node_ids.len()))?;
                let nv = *node_ids
                    .get(v as usize)
                    .ok_or_else(|| format!("edge v={v} out of bounds {}", node_ids.len()))?;
                Ok((nu, nv))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let topology = BaseTopology::new(node_ids, &orig_edges)
            .map_err(|e| format!("BaseTopology::new failed: {e:?}"))?;
        if *topology.order_digest() != expected_order_digest {
            return Err("order_digest mismatch after BaseTopology reconstruction!".into());
        }
        let digest = params_digest(params);
        let admitted = admit(
            core::slice::from_ref(&TWISTED_DIMER_V1),
            TWISTED_DIMER_V1.family_id.as_bytes(),
            TWISTED_DIMER_V1.revision,
            params,
            &digest,
        )
        .map_err(|e| format!("admit failed: {e:?}"))?;
        let p = admitted.params();
        let m = u16::try_from(unsigned(p.get(TAG_M)).ok_or("missing TAG_M")?)
            .map_err(|_| "bad m")?;
        let t = u8::try_from(unsigned(p.get(TAG_T)).ok_or("missing TAG_T")?)
            .map_err(|_| "bad t")?;
        let weak = i32::try_from(unsigned(p.get(TAG_WEAK)).ok_or("missing TAG_WEAK")?)
            .map_err(|_| "bad weak")?;
        let pairs = usize::try_from(unsigned(p.get(TAG_PAIRS)).ok_or("missing TAG_PAIRS")?)
            .map_err(|_| "bad pairs")?;
        let lattice = Lattice::from_shape(m, t).map_err(|e| format!("Lattice: {e:?}"))?;
        let found = candidates(lattice, &topology).map_err(|e| format!("candidates: {e:?}"))?;
        let split = sides(lattice, &topology, &found).map_err(|e| format!("sides: {e:?}"))?;

        let mut edges: Vec<(u32, u32)> = topology
            .edges()
            .iter()
            .map(|&(u, v)| (u.min(v), u.max(v)))
            .collect();
        edges.sort_unstable();
        let nodes = topology.node_ids().len();

        let mut deg = vec![0usize; nodes + 1];
        for &(u, _) in &edges {
            deg[u as usize + 1] += 1;
        }
        for i in 0..nodes {
            deg[i + 1] += deg[i];
        }
        let adj_off = deg.clone();
        let mut fill = deg;
        let mut adj_nb = vec![(0u32, 0u32); edges.len()];
        for (eid, &(u, v)) in edges.iter().enumerate() {
            let pos = fill[u as usize];
            adj_nb[pos] = (v, eid as u32);
            fill[u as usize] += 1;
        }

        let lookup_eid = |u: u32, v: u32| -> Option<u32> {
            let (lo, hi) = if u < v { (u, v) } else { (v, u) };
            for &(nb, eid) in &adj_nb[adj_off[lo as usize]..adj_off[lo as usize + 1]] {
                if nb == hi {
                    return Some(eid);
                }
            }
            None
        };

        let mut cand_tiles = Vec::with_capacity(found.tiles.len());
        let mut fixed_cids = Vec::new();
        let n_left = split.left.len();
        let n_right = split.right.len();
        let mut left_buckets: Vec<Vec<(u32, u32)>> = vec![Vec::new(); n_left];
        for (cid, (tile, spine)) in found.tiles.iter().zip(&found.spines).enumerate() {
            let [a, b, c, d] = tile.nodes;
            let eids = [
                lookup_eid(a, b).ok_or("missing tile edge ab")?,
                lookup_eid(a, c).ok_or("missing tile edge ac")?,
                lookup_eid(a, d).ok_or("missing tile edge ad")?,
                lookup_eid(b, c).ok_or("missing tile edge bc")?,
                lookup_eid(b, d).ok_or("missing tile edge bd")?,
                lookup_eid(c, d).ok_or("missing tile edge cd")?,
            ];
            cand_tiles.push(PreTile {
                nodes: tile.nodes,
                eids,
            });
            if let Some([e, f]) = spine {
                let straight = (split.left.binary_search(e), split.right.binary_search(f));
                let swapped = (split.left.binary_search(f), split.right.binary_search(e));
                let (Ok(l), Ok(r)) = (if straight.0.is_ok() {
                    straight
                } else {
                    swapped
                }) else {
                    return Err("spine side lookup failed".into());
                };
                left_buckets[l].push((r as u32, cid as u32));
            } else {
                fixed_cids.push(cid as u32);
            }
        }
        let mut left_off = Vec::with_capacity(n_left + 1);
        let mut left_entries = Vec::new();
        left_off.push(0);
        for bucket in &mut left_buckets {
            bucket.sort_unstable_by_key(|&(r, _)| r);
            left_entries.extend_from_slice(bucket);
            left_off.push(left_entries.len());
        }

        let max_div = edges.len().max(n_left).max(n_right).max(2);
        let mut div_table = vec![DivEntry { limit: 0, inv: 0 }; max_div + 1];
        for n in 2..=max_div {
            let n64 = n as u64;
            let limit = (1u64 << 32) - ((1u64 << 32) % n64);
            let inv = (((1u128 << 64) + (n as u128) - 1) / (n as u128)) as u64;
            div_table[n] = DivEntry { limit, inv };
        }

        Ok(Self {
            order_digest: *topology.order_digest(),
            descriptor_digest: admitted.descriptor().digest(),
            _found: found,
            _split: split,
            edges,
            weak,
            pairs,
            nodes,
            cand_tiles,
            fixed_cids,
            n_left,
            n_right,
            left_off,
            left_entries,
            div_table,
        })
    }

    pub fn new_scratch(&self) -> WorkerScratch {
        WorkerScratch::new(self)
    }

    /// Generate & pack directly into reusable `WorkerScratch` buffers (ZERO heap allocations!).
    pub fn generate_into_scratch(&self, nonce: [u8; 32], sc: &mut WorkerScratch) -> bool {
        let labels = TWISTED_DIMER.labels;
        let mut streams = Streams::new(nonce, self.descriptor_digest, self.order_digest, labels);

        sc.left_order.clear();
        sc.left_order.extend(0..self.n_left as u32);
        {
            let Some(mut rng) = streams.open(labels[0]) else {
                return false;
            };
            fast_shuffle_u32(&mut rng, &mut sc.left_order, &self.div_table);
        }
        sc.right_order.clear();
        sc.right_order.extend(0..self.n_right as u32);
        {
            let Some(mut rng) = streams.open(labels[1]) else {
                return false;
            };
            fast_shuffle_u32(&mut rng, &mut sc.right_order, &self.div_table);
        }

        for (pos, &r) in sc.right_order.iter().enumerate() {
            sc.rank[r as usize] = pos as u32;
        }
        sc.adj.copy_from_slice(&self.left_entries);
        for l in 0..self.n_left {
            let s = self.left_off[l];
            let e = self.left_off[l + 1];
            let slice = &mut sc.adj[s..e];
            for i in 1..slice.len() {
                let key = slice[i];
                let kr = sc.rank[key.0 as usize];
                let mut j = i;
                while j > 0 && sc.rank[slice[j - 1].0 as usize] > kr {
                    slice[j] = slice[j - 1];
                    j -= 1;
                }
                slice[j] = key;
            }
        }
        sc.matched.fill(u32::MAX);
        for &l in &sc.left_order {
            sc.cur_vis = sc.cur_vis.wrapping_add(1);
            if sc.cur_vis == 0 {
                sc.vis_epoch.fill(0);
                sc.cur_vis = 1;
            }
            let cur_vis = sc.cur_vis;
            sc.stack.clear();
            sc.stack.push((l as usize, 0));
            while let Some(&(left, position)) = sc.stack.last() {
                let s = self.left_off[left];
                let e = self.left_off[left + 1];
                if s + position >= e {
                    sc.stack.pop();
                    continue;
                }
                let (right_u32, _cid) = sc.adj[s + position];
                let right = right_u32 as usize;
                if let Some(top) = sc.stack.last_mut() {
                    top.1 = position + 1;
                }
                if sc.vis_epoch[right] == cur_vis {
                    continue;
                }
                sc.vis_epoch[right] = cur_vis;
                let owner = sc.matched[right];
                if owner != u32::MAX {
                    sc.stack.push((owner as usize, 0));
                    continue;
                }
                for &(frame_left, next_pos) in &sc.stack {
                    let (chosen_r, chosen_cid) = sc.adj[self.left_off[frame_left] + next_pos - 1];
                    sc.matched[chosen_r as usize] = frame_left as u32;
                    sc.matched_cid[chosen_r as usize] = chosen_cid;
                }
                break;
            }
        }
        sc.sel_cids.clear();
        sc.sel_cids.extend_from_slice(&self.fixed_cids);
        for r in 0..self.n_right {
            if sc.matched[r] != u32::MAX {
                sc.sel_cids.push(sc.matched_cid[r]);
            }
        }
        sc.sel_cids.sort_unstable();
        let n_tiles = sc.sel_cids.len();

        let Some(mut gauge_stream) = streams.open(labels[2]) else {
            return false;
        };
        sc.gauges.clear();
        sc.gauges_f32.clear();
        sc.cur_strong = sc.cur_strong.wrapping_add(1);
        if sc.cur_strong == 0 {
            sc.strong_mark.fill(0);
            sc.cur_strong = 1;
        }
        let cur_strong = sc.cur_strong;
        for &cid in &sc.sel_cids {
            let pt = unsafe { self.cand_tiles.get_unchecked(cid as usize) };
            for _ in 0..4 {
                if (gauge_stream.next_u32() & 1) == 0 {
                    sc.gauges.push(1i8);
                    sc.gauges_f32.push(1.0f32);
                } else {
                    sc.gauges.push(-1i8);
                    sc.gauges_f32.push(-1.0f32);
                }
            }
            for &eid in &pt.eids {
                sc.strong_mark[eid as usize] = cur_strong;
            }
        }

        let Some(mut weak_stream) = streams.open(labels[3]) else {
            return false;
        };
        let w_pos = self.weak as i16;
        let w_neg = -w_pos;
        sc.order.clear();
        let m = self.edges.len();
        for eid in 0..m {
            if sc.strong_mark[eid] != cur_strong {
                sc.weak_w[eid] = if (weak_stream.next_u32() & 1) == 0 {
                    w_pos
                } else {
                    w_neg
                };
                sc.order.push(eid as u32);
            }
        }

        {
            let Some(mut rng) = streams.open(labels[4]) else {
                return false;
            };
            fast_shuffle_u32(&mut rng, &mut sc.order, &self.div_table);
        }
        sc.cur_used = sc.cur_used.wrapping_add(1);
        if sc.cur_used == 0 {
            sc.used_mark.fill(0);
            sc.cur_used = 1;
        }
        let cur_used = sc.cur_used;
        let mut n_pairs_found = 0usize;
        for &eid_u32 in &sc.order {
            if n_pairs_found >= self.pairs {
                break;
            }
            let eid = eid_u32 as usize;
            let (u, v) = unsafe { *self.edges.get_unchecked(eid) };
            let (ui, vi) = (u as usize, v as usize);
            if sc.used_mark[ui] == cur_used || sc.used_mark[vi] == cur_used {
                continue;
            }
            sc.used_mark[ui] = cur_used;
            sc.used_mark[vi] = cur_used;
            sc.partner[ui] = v;
            sc.partner[vi] = u;
            sc.strong_mark[eid] = cur_strong;
            n_pairs_found += 1;
        }
        if n_pairs_found < self.pairs {
            return false;
        }

        // Contract dimer pairs & pack directly into scratch CSR (NO tile_off/inc_tid needed on CPU!)
        let n = self.nodes;
        sc.rep.fill(u16::MAX);
        let mut n_super = 0usize;
        for u in 0..n {
            if sc.rep[u] != u16::MAX {
                continue;
            }
            let id = n_super as u16;
            n_super += 1;
            sc.rep[u] = id;
            if sc.used_mark[u] == cur_used {
                sc.rep[sc.partner[u] as usize] = id;
            }
        }

        sc.tiles.clear();
        for &cid in &sc.sel_cids {
            let pt = unsafe { self.cand_tiles.get_unchecked(cid as usize) };
            sc.tiles.push(sc.rep[pt.nodes[0] as usize]);
            sc.tiles.push(sc.rep[pt.nodes[1] as usize]);
            sc.tiles.push(sc.rep[pt.nodes[2] as usize]);
            sc.tiles.push(sc.rep[pt.nodes[3] as usize]);
        }

        sc.bucket_deg[..=n_super].fill(0);
        let mut self_weak_milli = 0i64;
        for &eid_u32 in &sc.order {
            let eid = eid_u32 as usize;
            if sc.strong_mark[eid] == cur_strong {
                continue;
            }
            let (u, v) = unsafe { *self.edges.get_unchecked(eid) };
            let a = sc.rep[u as usize] as usize;
            let b = sc.rep[v as usize] as usize;
            if a == b {
                self_weak_milli += i64::from(sc.weak_w[eid]);
            } else {
                let lo = if a < b { a } else { b };
                sc.bucket_deg[lo + 1] += 1;
            }
        }
        for i in 0..n_super {
            sc.bucket_deg[i + 1] += sc.bucket_deg[i];
        }
        let tot_upper = sc.bucket_deg[n_super] as usize;
        sc.bucket_fill[..=n_super].copy_from_slice(&sc.bucket_deg[..=n_super]);
        sc.upper_bw.resize(tot_upper, (0, 0));
        for &eid_u32 in &sc.order {
            let eid = eid_u32 as usize;
            if sc.strong_mark[eid] == cur_strong {
                continue;
            }
            let (u, v) = unsafe { *self.edges.get_unchecked(eid) };
            let a = sc.rep[u as usize];
            let b = sc.rep[v as usize];
            if a == b {
                continue;
            }
            let (lo, hi) = if a < b { (a as usize, b) } else { (b as usize, a) };
            let pos = sc.bucket_fill[lo] as usize;
            sc.upper_bw[pos] = (hi, sc.weak_w[eid]);
            sc.bucket_fill[lo] = (pos + 1) as u32;
        }

        sc.dedup_edges.clear();
        sc.wdeg[..=n_super].fill(0);
        for a in 0..n_super {
            let s = sc.bucket_deg[a] as usize;
            let e = sc.bucket_deg[a + 1] as usize;
            if s == e {
                continue;
            }
            let row = &mut sc.upper_bw[s..e];
            if row.len() > 1 {
                row.sort_unstable_by_key(|&(b, _)| b);
            }
            let mut idx = 0;
            while idx < row.len() {
                let (b, mut sum) = (row[idx].0, i32::from(row[idx].1));
                idx += 1;
                while idx < row.len() && row[idx].0 == b {
                    sum += i32::from(row[idx].1);
                    idx += 1;
                }
                if sum != 0 {
                    sc.dedup_edges
                        .push((a as u16, b, sum.clamp(-32767, 32767) as i16));
                    sc.wdeg[a + 1] += 1;
                    sc.wdeg[b as usize + 1] += 1;
                }
            }
        }
        for i in 0..n_super {
            sc.wdeg[i + 1] += sc.wdeg[i];
        }
        let tot_wdir = sc.wdeg[n_super] as usize;
        for i in 0..=n_super {
            sc.wfill[i] = sc.wdeg[i] as usize;
        }
        sc.weak_nb.resize(tot_wdir, 0);
        sc.weak_w_milli.resize(tot_wdir, 0);
        sc.weak_w_f32.resize(tot_wdir, 0.0);
        for &(a, b, w) in &sc.dedup_edges {
            let wf = w as f32;
            let pa = sc.wfill[a as usize];
            sc.weak_nb[pa] = b;
            sc.weak_w_milli[pa] = w;
            sc.weak_w_f32[pa] = wf;
            sc.wfill[a as usize] = pa + 1;
            let pb = sc.wfill[b as usize];
            sc.weak_nb[pb] = a;
            sc.weak_w_milli[pb] = w;
            sc.weak_w_f32[pb] = wf;
            sc.wfill[b as usize] = pb + 1;
        }

        sc.n_super = n_super;
        sc.n_tiles = n_tiles;
        sc.n_pairs = self.pairs;
        sc.self_weak_milli = self_weak_milli;
        sc.base_const = (-(self.pairs as f64) + (self_weak_milli as f64) / 1000.0) as f32;
        true
    }

    /// Ultra-fast GPU-feeder salt generation: skips weak-edge sorting & dedup completely,
    /// packing raw ±weak edges into `u16` (`bit15 = neg`, `bits 0..14 = supernode id`).
    /// Mathematically identical to deduped BM8-SDP while ~3x faster on CPU & 2x smaller over PCIe!
    pub fn generate_gpu_input(
        &self,
        nonce: [u8; 32],
        seed: u64,
        sc: &mut WorkerScratch,
    ) -> Option<crate::gpu_bm8::Bm8GpuInput> {
        let labels = TWISTED_DIMER.labels;
        let mut streams = Streams::new(nonce, self.descriptor_digest, self.order_digest, labels);

        sc.left_order.clear();
        sc.left_order.extend(0..self.n_left as u32);
        {
            let mut rng = streams.open(labels[0])?;
            fast_shuffle_u32(&mut rng, &mut sc.left_order, &self.div_table);
        }
        sc.right_order.clear();
        sc.right_order.extend(0..self.n_right as u32);
        {
            let mut rng = streams.open(labels[1])?;
            fast_shuffle_u32(&mut rng, &mut sc.right_order, &self.div_table);
        }

        for (pos, &r) in sc.right_order.iter().enumerate() {
            sc.rank[r as usize] = pos as u32;
        }
        sc.adj.copy_from_slice(&self.left_entries);
        for l in 0..self.n_left {
            let s = self.left_off[l];
            let e = self.left_off[l + 1];
            let slice = &mut sc.adj[s..e];
            for i in 1..slice.len() {
                let key = slice[i];
                let kr = sc.rank[key.0 as usize];
                let mut j = i;
                while j > 0 && sc.rank[slice[j - 1].0 as usize] > kr {
                    slice[j] = slice[j - 1];
                    j -= 1;
                }
                slice[j] = key;
            }
        }
        sc.matched.fill(u32::MAX);
        for &l in &sc.left_order {
            sc.cur_vis = sc.cur_vis.wrapping_add(1);
            if sc.cur_vis == 0 {
                sc.vis_epoch.fill(0);
                sc.cur_vis = 1;
            }
            let cur_vis = sc.cur_vis;
            sc.stack.clear();
            sc.stack.push((l as usize, 0));
            while let Some(&(left, position)) = sc.stack.last() {
                let s = self.left_off[left];
                let e = self.left_off[left + 1];
                if s + position >= e {
                    sc.stack.pop();
                    continue;
                }
                let (right_u32, _cid) = sc.adj[s + position];
                let right = right_u32 as usize;
                if let Some(top) = sc.stack.last_mut() {
                    top.1 = position + 1;
                }
                if sc.vis_epoch[right] == cur_vis {
                    continue;
                }
                sc.vis_epoch[right] = cur_vis;
                let owner = sc.matched[right];
                if owner != u32::MAX {
                    sc.stack.push((owner as usize, 0));
                    continue;
                }
                for &(frame_left, next_pos) in &sc.stack {
                    let (chosen_r, chosen_cid) = sc.adj[self.left_off[frame_left] + next_pos - 1];
                    sc.matched[chosen_r as usize] = frame_left as u32;
                    sc.matched_cid[chosen_r as usize] = chosen_cid;
                }
                break;
            }
        }
        sc.sel_cids.clear();
        sc.sel_cids.extend_from_slice(&self.fixed_cids);
        for r in 0..self.n_right {
            if sc.matched[r] != u32::MAX {
                sc.sel_cids.push(sc.matched_cid[r]);
            }
        }
        sc.sel_cids.sort_unstable();
        let n_tiles = sc.sel_cids.len();

        let mut gauge_stream = streams.open(labels[2])?;
        let mut gauges = Vec::with_capacity(n_tiles * 4);
        sc.cur_strong = sc.cur_strong.wrapping_add(1);
        if sc.cur_strong == 0 {
            sc.strong_mark.fill(0);
            sc.cur_strong = 1;
        }
        let cur_strong = sc.cur_strong;
        for &cid in &sc.sel_cids {
            let pt = unsafe { self.cand_tiles.get_unchecked(cid as usize) };
            for _ in 0..4 {
                gauges.push(if (gauge_stream.next_u32() & 1) == 0 { 1i8 } else { -1i8 });
            }
            for &eid in &pt.eids {
                sc.strong_mark[eid as usize] = cur_strong;
            }
        }

        let mut weak_stream = streams.open(labels[3])?;
        let w_pos = self.weak as i16;
        let w_neg = -w_pos;
        sc.order.clear();
        let m = self.edges.len();
        for eid in 0..m {
            if sc.strong_mark[eid] != cur_strong {
                sc.weak_w[eid] = if (weak_stream.next_u32() & 1) == 0 {
                    w_pos
                } else {
                    w_neg
                };
                sc.order.push(eid as u32);
            }
        }

        {
            let mut rng = streams.open(labels[4])?;
            fast_shuffle_u32(&mut rng, &mut sc.order, &self.div_table);
        }
        sc.cur_used = sc.cur_used.wrapping_add(1);
        if sc.cur_used == 0 {
            sc.used_mark.fill(0);
            sc.cur_used = 1;
        }
        let cur_used = sc.cur_used;
        let mut n_pairs_found = 0usize;
        for &eid_u32 in &sc.order {
            if n_pairs_found >= self.pairs {
                break;
            }
            let eid = eid_u32 as usize;
            let (u, v) = unsafe { *self.edges.get_unchecked(eid) };
            let (ui, vi) = (u as usize, v as usize);
            if sc.used_mark[ui] == cur_used || sc.used_mark[vi] == cur_used {
                continue;
            }
            sc.used_mark[ui] = cur_used;
            sc.used_mark[vi] = cur_used;
            sc.partner[ui] = v;
            sc.partner[vi] = u;
            sc.strong_mark[eid] = cur_strong;
            n_pairs_found += 1;
        }
        if n_pairs_found < self.pairs {
            return None;
        }

        let n = self.nodes;
        sc.rep.fill(u16::MAX);
        let mut n_super = 0usize;
        for u in 0..n {
            if sc.rep[u] != u16::MAX {
                continue;
            }
            let id = n_super as u16;
            n_super += 1;
            sc.rep[u] = id;
            if sc.used_mark[u] == cur_used {
                sc.rep[sc.partner[u] as usize] = id;
            }
        }
        if n_super >= 32768 {
            return None;
        }

        let mut tiles = Vec::with_capacity(n_tiles * 4);
        for &cid in &sc.sel_cids {
            let pt = unsafe { self.cand_tiles.get_unchecked(cid as usize) };
            tiles.push(sc.rep[pt.nodes[0] as usize]);
            tiles.push(sc.rep[pt.nodes[1] as usize]);
            tiles.push(sc.rep[pt.nodes[2] as usize]);
            tiles.push(sc.rep[pt.nodes[3] as usize]);
        }

        // Direct 2-pass raw weak CSR with u16 sign packing (ZERO sorting/dedup!)
        let mut weak_off = vec![0i32; n_super + 1];
        let mut self_weak_milli = 0i64;
        for &eid_u32 in &sc.order {
            let eid = eid_u32 as usize;
            if unsafe { *sc.strong_mark.get_unchecked(eid) } == cur_strong {
                continue;
            }
            let (u, v) = unsafe { *self.edges.get_unchecked(eid) };
            let a = unsafe { *sc.rep.get_unchecked(u as usize) } as usize;
            let b = unsafe { *sc.rep.get_unchecked(v as usize) } as usize;
            if a == b {
                self_weak_milli += i64::from(unsafe { *sc.weak_w.get_unchecked(eid) });
            } else {
                unsafe {
                    *weak_off.get_unchecked_mut(a + 1) += 1;
                    *weak_off.get_unchecked_mut(b + 1) += 1;
                }
            }
        }
        for i in 0..n_super {
            weak_off[i + 1] += weak_off[i];
        }
        let tot_wdir = weak_off[n_super] as usize;
        for i in 0..=n_super {
            sc.wfill[i] = weak_off[i] as usize;
        }
        let mut weak_nb_packed = vec![0u16; tot_wdir];
        for &eid_u32 in &sc.order {
            let eid = eid_u32 as usize;
            if unsafe { *sc.strong_mark.get_unchecked(eid) } == cur_strong {
                continue;
            }
            let (u, v) = unsafe { *self.edges.get_unchecked(eid) };
            let a = unsafe { *sc.rep.get_unchecked(u as usize) };
            let b = unsafe { *sc.rep.get_unchecked(v as usize) };
            if a == b {
                continue;
            }
            let neg_bit = if unsafe { *sc.weak_w.get_unchecked(eid) } < 0 {
                0x8000u16
            } else {
                0u16
            };
            unsafe {
                let pa = *sc.wfill.get_unchecked(a as usize);
                *weak_nb_packed.get_unchecked_mut(pa) = b | neg_bit;
                *sc.wfill.get_unchecked_mut(a as usize) = pa + 1;
                let pb = *sc.wfill.get_unchecked(b as usize);
                *weak_nb_packed.get_unchecked_mut(pb) = a | neg_bit;
                *sc.wfill.get_unchecked_mut(b as usize) = pb + 1;
            }
        }

        let base_const = (-(self.pairs as f64) + (self_weak_milli as f64) / 1000.0) as f32;
        Some(crate::gpu_bm8::Bm8GpuInput {
            n_super: n_super as i32,
            n_tiles: n_tiles as i32,
            base_const,
            weak_milli: self.weak as f32,
            seed,
            tiles,
            gauges,
            weak_off,
            weak_nb_packed,
        })
    }
}

pub struct WorkerScratch {
    left_order: Vec<u32>,
    right_order: Vec<u32>,
    rank: Vec<u32>,
    adj: Vec<(u32, u32)>,
    matched: Vec<u32>,
    matched_cid: Vec<u32>,
    vis_epoch: Vec<u32>,
    cur_vis: u32,
    stack: Vec<(usize, usize)>,
    sel_cids: Vec<u32>,
    strong_mark: Vec<u32>,
    cur_strong: u32,
    weak_w: Vec<i16>,
    order: Vec<u32>,
    used_mark: Vec<u32>,
    cur_used: u32,
    partner: Vec<u32>,
    pub rep: Vec<u16>,
    bucket_deg: Vec<u32>,
    bucket_fill: Vec<u32>,
    upper_bw: Vec<(u16, i16)>,
    pub wdeg: Vec<i32>,
    wfill: Vec<usize>,
    pub dedup_edges: Vec<(u16, u16, i16)>,
    // Packed contracted graph (zero alloc!)
    pub n_super: usize,
    pub n_tiles: usize,
    pub n_pairs: usize,
    pub self_weak_milli: i64,
    pub base_const: f32,
    pub tiles: Vec<u16>,
    pub gauges: Vec<i8>,
    pub gauges_f32: Vec<f32>,
    pub weak_nb: Vec<u16>,
    pub weak_w_milli: Vec<i16>,
    pub weak_w_f32: Vec<f32>,
    // BM8/BM4 scratch
    u8: Vec<[f32; 8]>,
    g8: Vec<[f32; 8]>,
    m8: Vec<[f32; 8]>,
    u4: Vec<[f32; 4]>,
    g4: Vec<[f32; 4]>,
    m4: Vec<[f32; 4]>,
}

impl WorkerScratch {
    fn new(fd: &FastDimer) -> Self {
        let n = fd.nodes;
        let m = fd.edges.len();
        Self {
            left_order: Vec::with_capacity(fd.n_left),
            right_order: Vec::with_capacity(fd.n_right),
            rank: vec![0; fd.n_right],
            adj: vec![(0, 0); fd.left_entries.len()],
            matched: vec![u32::MAX; fd.n_right],
            matched_cid: vec![0; fd.n_right],
            vis_epoch: vec![0; fd.n_right],
            cur_vis: 0,
            stack: Vec::with_capacity(64),
            sel_cids: Vec::with_capacity(fd.cand_tiles.len()),
            strong_mark: vec![0; m],
            cur_strong: 0,
            weak_w: vec![0; m],
            order: Vec::with_capacity(m),
            used_mark: vec![0; n],
            cur_used: 0,
            partner: vec![u32::MAX; n],
            rep: vec![0; n],
            bucket_deg: vec![0; n + 1],
            bucket_fill: vec![0; n + 1],
            upper_bw: Vec::with_capacity(m),
            wdeg: vec![0; n + 1],
            wfill: vec![0; n + 1],
            dedup_edges: Vec::with_capacity(m),
            n_super: 0,
            n_tiles: 0,
            n_pairs: fd.pairs,
            self_weak_milli: 0,
            base_const: 0.0,
            tiles: Vec::with_capacity(fd.cand_tiles.len() * 4),
            gauges: Vec::with_capacity(fd.cand_tiles.len() * 4),
            gauges_f32: Vec::with_capacity(fd.cand_tiles.len() * 4),
            weak_nb: Vec::with_capacity(m * 2),
            weak_w_milli: Vec::with_capacity(m * 2),
            weak_w_f32: Vec::with_capacity(m * 2),
            u8: vec![[0.0; 8]; n],
            g8: vec![[0.0; 8]; n],
            m8: vec![[0.0; 8]; n],
            u4: vec![[0.0; 4]; n],
            g4: vec![[0.0; 4]; n],
            m4: vec![[0.0; 4]; n],
        }
    }

    /// Zero-allocation Rank-8 BM-SDP matching `bm8_batch_sdp_v2` & `bm8_sdp_packed_cpu` bit-for-bit.
    pub fn bm8_sdp(
        &mut self,
        seed: u64,
        steps: usize,
        sw0: f32,
        sw1: f32,
        weak_mul: f32,
        lr0: f32,
    ) -> f64 {
        const R: usize = 8;
        let n = self.n_super;
        let nt = self.n_tiles;
        if n == 0 {
            return 1e9;
        }
        let u = &mut self.u8[..n];
        let grad = &mut self.g8[..n];
        let mom = &mut self.m8[..n];

        let seed0 = seed ^ 0xA5A5_5A5A_1234_5678u64;
        for i in 0..n {
            let mut st = seed0 ^ 0x9E37_79B9_7F4A_7C15u64.wrapping_mul((i + 1) as u64);
            let mut v = [0.0f32; R];
            let mut n2 = 0.0f32;
            for r in 0..R {
                st ^= st >> 12;
                st ^= st << 25;
                st ^= st >> 27;
                let h = st.wrapping_mul(0x2545_F491_4F6C_DD1Du64);
                let val = ((h >> 40) as f32) * (1.0 / 16_777_216.0) * 2.0 - 1.0;
                v[r] = val;
                n2 += val * val;
            }
            let inv = (n2 + 1e-12f32).sqrt().recip();
            for r in 0..R {
                v[r] *= inv;
            }
            u[i] = v;
            mom[i] = [0.0f32; R];
        }

        let tiles = &self.tiles[..nt * 4];
        let gauges = &self.gauges[..nt * 4];
        let gauges_f32 = &self.gauges_f32[..nt * 4];
        let woff = &self.wdeg[..=n];
        let wnb = &self.weak_nb;
        let ww = &self.weak_w_milli;
        let ww_f32 = &self.weak_w_f32;
        assert!(u.len() >= n && grad.len() >= n && mom.len() >= n);
        assert!(tiles.len() >= nt * 4 && gauges_f32.len() >= nt * 4);
        assert!(woff.len() > n && (woff[n] as usize) <= wnb.len() && (woff[n] as usize) <= ww_f32.len());
        let inv_steps = 1.0f32 / (steps.max(1) as f32);
        let w_scale = weak_mul * 0.001f32;
        let mom_beta = 0.75f32;

        for g in grad.iter_mut() {
            *g = [0.0f32; R];
        }
        for iter in 0..steps {
            let prog = (iter as f32) * inv_steps;
            let strong_w = sw0 + (sw1 - sw0) * prog;
            let lr = lr0 * (1.0 - 0.80 * prog);

            unsafe {
                for t in 0..nt {
                    let b = t * 4;
                    let i0 = *tiles.get_unchecked(b) as usize;
                    let i1 = *tiles.get_unchecked(b + 1) as usize;
                    let i2 = *tiles.get_unchecked(b + 2) as usize;
                    let i3 = *tiles.get_unchecked(b + 3) as usize;
                    let g0 = *gauges_f32.get_unchecked(b);
                    let g1 = *gauges_f32.get_unchecked(b + 1);
                    let g2 = *gauges_f32.get_unchecked(b + 2);
                    let g3 = *gauges_f32.get_unchecked(b + 3);
                    let u0 = *u.get_unchecked(i0);
                    let u1 = *u.get_unchecked(i1);
                    let u2 = *u.get_unchecked(i2);
                    let u3 = *u.get_unchecked(i3);
                    let mut s = [0.0f32; R];
                    for r in 0..R {
                        s[r] = strong_w * (g0 * u0[r] + g1 * u1[r] + g2 * u2[r] + g3 * u3[r]);
                    }
                    let gi0 = grad.get_unchecked_mut(i0);
                    for r in 0..R { gi0[r] += s[r] * g0; }
                    let gi1 = grad.get_unchecked_mut(i1);
                    for r in 0..R { gi1[r] += s[r] * g1; }
                    let gi2 = grad.get_unchecked_mut(i2);
                    for r in 0..R { gi2[r] += s[r] * g2; }
                    let gi3 = grad.get_unchecked_mut(i3);
                    for r in 0..R { gi3[r] += s[r] * g3; }
                }
                for i in 0..n {
                    let p0 = *woff.get_unchecked(i) as usize;
                    let p1 = *woff.get_unchecked(i + 1) as usize;
                    let mut acc = [0.0f32; R];
                    for p in p0..p1 {
                        let nb = *wnb.get_unchecked(p) as usize;
                        let w = *ww_f32.get_unchecked(p);
                        let unb = *u.get_unchecked(nb);
                        for r in 0..R {
                            acc[r] += w * unb[r];
                        }
                    }
                    let gi = grad.get_unchecked_mut(i);
                    for r in 0..R {
                        gi[r] += w_scale * acc[r];
                    }
                }
                for i in 0..n {
                    let ui = u.get_unchecked_mut(i);
                    let gi = *grad.get_unchecked(i);
                    *grad.get_unchecked_mut(i) = [0.0f32; R];
                    let mi = mom.get_unchecked_mut(i);
                    let mut dot = 0.0f32;
                    for r in 0..R {
                        dot += gi[r] * ui[r];
                    }
                    let mut n2 = 0.0f32;
                    for r in 0..R {
                        let tg = gi[r] - dot * ui[r];
                        mi[r] = mom_beta * mi[r] - lr * tg;
                        let nv = ui[r] + mi[r];
                        ui[r] = nv;
                        n2 += nv * nv;
                    }
                    let inv = (n2 + 1e-12f32).sqrt().recip();
                    for r in 0..R {
                        ui[r] *= inv;
                    }
                }
            }
        }

        let mut tile_e = 0.0f64;
        for t in 0..nt {
            let b = t * 4;
            let u0 = u[tiles[b] as usize];
            let u1 = u[tiles[b + 1] as usize];
            let u2 = u[tiles[b + 2] as usize];
            let u3 = u[tiles[b + 3] as usize];
            let g0 = gauges[b] as f64;
            let g1 = gauges[b + 1] as f64;
            let g2 = gauges[b + 2] as f64;
            let g3 = gauges[b + 3] as f64;
            let mut sq = 0.0f64;
            for r in 0..R {
                let sr = g0 * (u0[r] as f64)
                    + g1 * (u1[r] as f64)
                    + g2 * (u2[r] as f64)
                    + g3 * (u3[r] as f64);
                sq += sr * sr;
            }
            tile_e += 0.5 * (sq - 4.0);
        }
        let mut weak_sum = 0.0f64;
        for i in 0..n {
            let p0 = woff[i] as usize;
            let p1 = woff[i + 1] as usize;
            let mut acc = [0.0f64; R];
            for p in p0..p1 {
                let unb = u[wnb[p] as usize];
                let w = (ww[p] as f64) * 0.001;
                for r in 0..R {
                    acc[r] += w * (unb[r] as f64);
                }
            }
            let ui = u[i];
            for r in 0..R {
                weak_sum += (ui[r] as f64) * acc[r];
            }
        }
        (self.base_const as f64) + tile_e + 0.5 * weak_sum
    }

    /// Optional Rank-4 fast BM-SDP (for ultra-fast Stage 0b screening benchmarks & optional mode).
    pub fn bm4_sdp(
        &mut self,
        seed: u64,
        steps: usize,
        sw0: f32,
        sw1: f32,
        weak_mul: f32,
        lr0: f32,
    ) -> f64 {
        const R: usize = 4;
        let n = self.n_super;
        let nt = self.n_tiles;
        if n == 0 {
            return 1e9;
        }
        let u = &mut self.u4[..n];
        let grad = &mut self.g4[..n];
        let mom = &mut self.m4[..n];

        let seed0 = seed ^ 0xA5A5_5A5A_1234_5678u64;
        for i in 0..n {
            let mut st = seed0 ^ 0x9E37_79B9_7F4A_7C15u64.wrapping_mul((i + 1) as u64);
            let mut v = [0.0f32; R];
            let mut n2 = 0.0f32;
            for r in 0..R {
                st ^= st >> 12;
                st ^= st << 25;
                st ^= st >> 27;
                let h = st.wrapping_mul(0x2545_F491_4F6C_DD1Du64);
                let val = ((h >> 40) as f32) * (1.0 / 16_777_216.0) * 2.0 - 1.0;
                v[r] = val;
                n2 += val * val;
            }
            let inv = (n2 + 1e-12f32).sqrt().recip();
            for r in 0..R {
                v[r] *= inv;
            }
            u[i] = v;
            mom[i] = [0.0f32; R];
        }

        let tiles = &self.tiles[..nt * 4];
        let gauges = &self.gauges[..nt * 4];
        let gauges_f32 = &self.gauges_f32[..nt * 4];
        let woff = &self.wdeg[..=n];
        let wnb = &self.weak_nb;
        let ww = &self.weak_w_milli;
        let ww_f32 = &self.weak_w_f32;
        assert!(u.len() >= n && grad.len() >= n && mom.len() >= n);
        assert!(tiles.len() >= nt * 4 && gauges_f32.len() >= nt * 4);
        assert!(woff.len() > n && (woff[n] as usize) <= wnb.len() && (woff[n] as usize) <= ww_f32.len());
        let inv_steps = 1.0f32 / (steps.max(1) as f32);
        let w_scale = weak_mul * 0.001f32;
        let mom_beta = 0.75f32;

        for g in grad.iter_mut() {
            *g = [0.0f32; R];
        }
        for iter in 0..steps {
            let prog = (iter as f32) * inv_steps;
            let strong_w = sw0 + (sw1 - sw0) * prog;
            let lr = lr0 * (1.0 - 0.80 * prog);

            unsafe {
                for t in 0..nt {
                    let b = t * 4;
                    let i0 = *tiles.get_unchecked(b) as usize;
                    let i1 = *tiles.get_unchecked(b + 1) as usize;
                    let i2 = *tiles.get_unchecked(b + 2) as usize;
                    let i3 = *tiles.get_unchecked(b + 3) as usize;
                    let g0 = *gauges_f32.get_unchecked(b);
                    let g1 = *gauges_f32.get_unchecked(b + 1);
                    let g2 = *gauges_f32.get_unchecked(b + 2);
                    let g3 = *gauges_f32.get_unchecked(b + 3);
                    let u0 = *u.get_unchecked(i0);
                    let u1 = *u.get_unchecked(i1);
                    let u2 = *u.get_unchecked(i2);
                    let u3 = *u.get_unchecked(i3);
                    let mut s = [0.0f32; R];
                    for r in 0..R {
                        s[r] = strong_w * (g0 * u0[r] + g1 * u1[r] + g2 * u2[r] + g3 * u3[r]);
                    }
                    let gi0 = grad.get_unchecked_mut(i0);
                    for r in 0..R { gi0[r] += s[r] * g0; }
                    let gi1 = grad.get_unchecked_mut(i1);
                    for r in 0..R { gi1[r] += s[r] * g1; }
                    let gi2 = grad.get_unchecked_mut(i2);
                    for r in 0..R { gi2[r] += s[r] * g2; }
                    let gi3 = grad.get_unchecked_mut(i3);
                    for r in 0..R { gi3[r] += s[r] * g3; }
                }
                for i in 0..n {
                    let p0 = *woff.get_unchecked(i) as usize;
                    let p1 = *woff.get_unchecked(i + 1) as usize;
                    let mut acc = [0.0f32; R];
                    for p in p0..p1 {
                        let nb = *wnb.get_unchecked(p) as usize;
                        let w = *ww_f32.get_unchecked(p);
                        let unb = *u.get_unchecked(nb);
                        for r in 0..R {
                            acc[r] += w * unb[r];
                        }
                    }
                    let gi = grad.get_unchecked_mut(i);
                    for r in 0..R {
                        gi[r] += w_scale * acc[r];
                    }
                }
                for i in 0..n {
                    let ui = u.get_unchecked_mut(i);
                    let gi = *grad.get_unchecked(i);
                    *grad.get_unchecked_mut(i) = [0.0f32; R];
                    let mi = mom.get_unchecked_mut(i);
                    let mut dot = 0.0f32;
                    for r in 0..R {
                        dot += gi[r] * ui[r];
                    }
                    let mut n2 = 0.0f32;
                    for r in 0..R {
                        let tg = gi[r] - dot * ui[r];
                        mi[r] = mom_beta * mi[r] - lr * tg;
                        let nv = ui[r] + mi[r];
                        ui[r] = nv;
                        n2 += nv * nv;
                    }
                    let inv = (n2 + 1e-12f32).sqrt().recip();
                    for r in 0..R {
                        ui[r] *= inv;
                    }
                }
            }
        }

        let mut tile_e = 0.0f64;
        for t in 0..nt {
            let b = t * 4;
            let u0 = u[tiles[b] as usize];
            let u1 = u[tiles[b + 1] as usize];
            let u2 = u[tiles[b + 2] as usize];
            let u3 = u[tiles[b + 3] as usize];
            let g0 = gauges[b] as f64;
            let g1 = gauges[b + 1] as f64;
            let g2 = gauges[b + 2] as f64;
            let g3 = gauges[b + 3] as f64;
            let mut sq = 0.0f64;
            for r in 0..R {
                let sr = g0 * (u0[r] as f64)
                    + g1 * (u1[r] as f64)
                    + g2 * (u2[r] as f64)
                    + g3 * (u3[r] as f64);
                sq += sr * sr;
            }
            tile_e += 0.5 * (sq - 4.0);
        }
        let mut weak_sum = 0.0f64;
        for i in 0..n {
            let p0 = woff[i] as usize;
            let p1 = woff[i + 1] as usize;
            let mut acc = [0.0f64; R];
            for p in p0..p1 {
                let unb = u[wnb[p] as usize];
                let w = (ww[p] as f64) * 0.001;
                for r in 0..R {
                    acc[r] += w * (unb[r] as f64);
                }
            }
            let ui = u[i];
            for r in 0..R {
                weak_sum += (ui[r] as f64) * acc[r];
            }
        }
        (self.base_const as f64) + tile_e + 0.5 * weak_sum
    }

    pub fn set_u8_from_gpu(&mut self, u_vec: &[[f32; 8]]) {
        let n = u_vec.len().min(self.u8.len());
        self.u8[..n].copy_from_slice(&u_vec[..n]);
    }

    /// Stage 0d: Jacobi eigen-decomposition + V3Fast/V3 Micro-BP + quantized seed payload.
    /// Called ONLY on ~0.36% of salts that pass Stage 0c (`BM8_w30`)!
    pub fn stage_d_and_seed(
        &self,
        seed: u64,
        run_unseeded: bool,
        bp_weight: f64,
        bp_seed_weight: f64,
        bm_deep: f64,
    ) -> (f32, Vec<[i8; 8]>, [[f32; 8]; 8]) {
        let n = self.n_super;
        let u_vec = &self.u8[..n];
        let mut cov = [[0.0f64; 8]; 8];
        for u in u_vec {
            for a in 0..8 {
                let ua = u[a] as f64;
                for b in a..8 {
                    cov[a][b] += ua * (u[b] as f64);
                }
            }
        }
        for a in 0..8 {
            for b in 0..a {
                cov[a][b] = cov[b][a];
            }
        }
        let (_evals, evecs64) = jacobi_sym8(cov);
        let mut evecs = [[0.0f32; 8]; 8];
        for k in 0..8 {
            for r in 0..8 {
                evecs[k][r] = evecs64[k][r] as f32;
            }
        }
        let mut projs = Vec::with_capacity(n);
        for u in u_vec {
            let mut p = [0.0f32; 4];
            for k in 0..4 {
                let mut acc = 0.0f32;
                for r in 0..8 {
                    acc += u[r] * evecs[k][r];
                }
                p[k] = acc;
            }
            projs.push(p);
        }

        let (bp_tiles, bp_edges) = self.build_bp_factor_graph();
        let bp_bmst = self.run_bp_4rep_inner(
            &bp_tiles,
            &bp_edges,
            seed.wrapping_add(7919),
            Some(&projs),
            120.0,
            [0.0, 45.0, 75.0, 110.0],
        );
        let score = if run_unseeded {
            let bp_v1 = self.run_bp_4rep_inner(
                &bp_tiles,
                &bp_edges,
                seed.wrapping_add(7919),
                None,
                0.0,
                [0.0; 4],
            );
            (bm_deep + bp_weight * bp_v1 + bp_seed_weight * bp_bmst) as f32
        } else {
            (bm_deep + bp_seed_weight * bp_bmst) as f32
        };

        let mut u_orig = Vec::with_capacity(self.rep.len());
        for &cid in &self.rep {
            let u = u_vec[cid as usize];
            let mut q = [0i8; 8];
            for r in 0..8 {
                q[r] = (u[r] * 127.0).round().clamp(-127.0, 127.0) as i8;
            }
            u_orig.push(q);
        }
        (score, u_orig, evecs)
    }

    /// Stage 0e Experimental: Goemans-Williamson hyperplane rounding on BM8_w30 U-vectors
    /// + exact incremental 1-flip steepest-descent local polish (measures real discrete +/-1 energy
    /// & SDP integrality gap!). Returns `(best_e, mean_top5_e, best_v, best_w, eigen_top1_frac)`.
    pub fn gw_round_and_polish(&self, seed: u64, n_trials: usize) -> (f64, f64, i32, f64, f64) {
        let n = self.n_super;
        let nt = self.n_tiles;
        let u_vec = &self.u8[..n];

        // Covariance & Jacobi eigendecomposition (exact same as stage_d)
        let mut cov = [[0.0f64; 8]; 8];
        for u in u_vec {
            for a in 0..8 {
                let ua = u[a] as f64;
                for b in a..8 {
                    cov[a][b] += ua * (u[b] as f64);
                }
            }
        }
        for a in 0..8 {
            for b in 0..a {
                cov[a][b] = cov[b][a];
            }
        }
        let (evals, evecs64) = jacobi_sym8(cov);
        let eval_sum: f64 = evals.iter().sum::<f64>().max(1e-9);
        let eigen_top1_frac = evals[0] / eval_sum;

        // Build compact tile incidence list once per salt
        let mut toff = vec![0usize; n + 1];
        for &u in &self.tiles[..nt * 4] {
            toff[u as usize + 1] += 1;
        }
        for i in 0..n {
            toff[i + 1] += toff[i];
        }
        let mut tfill = toff.clone();
        let mut inc_tid = vec![0u16; nt * 4];
        let mut inc_g = vec![0i8; nt * 4];
        for idx in 0..nt * 4 {
            let u = self.tiles[idx] as usize;
            let pos = tfill[u];
            inc_tid[pos] = (idx >> 2) as u16;
            inc_g[pos] = self.gauges[idx];
            tfill[u] = pos + 1;
        }

        let mut spins = vec![1i8; n];
        let mut tsum = vec![0i32; nt];
        let mut h_weak = vec![0i32; n]; // milli-units: sum_j J_ij * s_j
        let mut trial_es = Vec::with_capacity(n_trials);
        let mut best_e = 1e9f64;
        let mut best_v = 9999i32;
        let mut best_w = 1e9f64;

        let mut st = seed ^ 0xD1B5_4A32_D192_ED03u64;
        let mut next_f32 = || -> f32 {
            st ^= st >> 12;
            st ^= st << 25;
            st ^= st >> 27;
            let v = st.wrapping_mul(0x2545_F491_4F6C_DD1D);
            ((v >> 40) as f32) * (1.0 / 16_777_216.0) * 2.0 - 1.0
        };

        let base_milli = -(self.n_pairs as i64) * 1000 + self.self_weak_milli;
        for tr in 0..n_trials {
            let dir: [f32; 8] = if tr < 8 {
                let mut d = [0.0f32; 8];
                for r in 0..8 {
                    d[r] = evecs64[tr][r] as f32;
                }
                d
            } else {
                // Random Gaussian-like direction via sum of 3 uniforms per coord
                let mut d = [0.0f32; 8];
                for r in 0..8 {
                    d[r] = next_f32() + next_f32() + next_f32();
                }
                d
            };

            for i in 0..n {
                let u = u_vec[i];
                let dot = u[0]*dir[0] + u[1]*dir[1] + u[2]*dir[2] + u[3]*dir[3]
                        + u[4]*dir[4] + u[5]*dir[5] + u[6]*dir[6] + u[7]*dir[7];
                spins[i] = if dot >= 0.0 { 1 } else { -1 };
            }

            // Initialize tile sums & weak local fields
            for t in 0..nt {
                let b = t * 4;
                tsum[t] = (self.gauges[b] as i32) * (spins[self.tiles[b] as usize] as i32)
                    + (self.gauges[b + 1] as i32) * (spins[self.tiles[b + 1] as usize] as i32)
                    + (self.gauges[b + 2] as i32) * (spins[self.tiles[b + 2] as usize] as i32)
                    + (self.gauges[b + 3] as i32) * (spins[self.tiles[b + 3] as usize] as i32);
            }
            for u in 0..n {
                let mut acc = 0i32;
                for p in self.wdeg[u] as usize..self.wdeg[u + 1] as usize {
                    acc += (self.weak_w_milli[p] as i32) * (spins[self.weak_nb[p] as usize] as i32);
                }
                h_weak[u] = acc;
            }

            // Incremental 1-flip greedy descent (up to 6 passes, typically converges in 3-4!)
            for _pass in 0..6 {
                let mut flipped = 0usize;
                for u in 0..n {
                    let su = spins[u] as i32;
                    // Strong tile delta in milli: each tile contributes 500 * (S^2 - 4).
                    // Flipping s_u -> -s_u changes S by dS = -2 * g * su.
                    // New S^2 - Old S^2 = dS * (2*S + dS) = (-2*g*su)*(2*S - 2*g*su) = 4*(1 - g*su*S).
                    // So 500*(S'^2 - S^2) = 2000*(1 - g*su*S).
                    let mut d_strong = 0i32;
                    for p in toff[u]..toff[u + 1] {
                        let gs = (inc_g[p] as i32) * su;
                        d_strong += 1 - gs * tsum[inc_tid[p] as usize];
                    }
                    let d_milli = d_strong * 2000 - 2 * su * h_weak[u];
                    if d_milli < 0 {
                        let new_s = -su;
                        spins[u] = new_s as i8;
                        for p in toff[u]..toff[u + 1] {
                            let tid = inc_tid[p] as usize;
                            tsum[tid] += 2 * (inc_g[p] as i32) * new_s;
                        }
                        for p in self.wdeg[u] as usize..self.wdeg[u + 1] as usize {
                            let nb = self.weak_nb[p] as usize;
                            h_weak[nb] += 2 * (self.weak_w_milli[p] as i32) * new_s;
                        }
                        flipped += 1;
                    }
                }
                if flipped == 0 {
                    break;
                }
            }

            // Exact energy decomposition: E = base + strong(2V - 2*nt) + weak
            let mut strong_milli = 0i64;
            let mut viol = 0i32;
            for t in 0..nt {
                let s2 = tsum[t] * tsum[t];
                strong_milli += 500 * (s2 - 4) as i64;
                if s2 != 0 {
                    viol += s2 / 4; // 1 for |S|=2, 4 for |S|=4
                }
            }
            let mut weak_milli2 = 0i64;
            for u in 0..n {
                weak_milli2 += (spins[u] as i64) * (h_weak[u] as i64);
            }
            let weak_milli = weak_milli2 / 2;
            let total_e = (base_milli + strong_milli + weak_milli) as f64 * 0.001;
            trial_es.push(total_e);
            if total_e < best_e {
                best_e = total_e;
                best_v = viol;
                best_w = (weak_milli + self.self_weak_milli) as f64 * 0.001;
            }
        }
        trial_es.sort_by(f64::total_cmp);
        let k5 = trial_es.len().min(5).max(1);
        let mean_top5 = trial_es[..k5].iter().sum::<f64>() / (k5 as f64);
        (best_e, mean_top5, best_v, best_w, eigen_top1_frac)
    }

    /// Exact incremental 1-flip local polish on a spin vector `spins` (modifies `spins` in place),
    /// returning exact energy in units (same scale as `exact_energy_units`).
    fn polish_spins_1flip(
        &self,
        spins: &mut [i8],
        toff: &[usize],
        inc_tid: &[u16],
        inc_g: &[i8],
        tsum: &mut [i32],
        h_weak: &mut [i32],
    ) -> f64 {
        let n = self.n_super;
        let nt = self.n_tiles;
        for t in 0..nt {
            let b = t * 4;
            tsum[t] = (self.gauges[b] as i32) * (spins[self.tiles[b] as usize] as i32)
                + (self.gauges[b + 1] as i32) * (spins[self.tiles[b + 1] as usize] as i32)
                + (self.gauges[b + 2] as i32) * (spins[self.tiles[b + 2] as usize] as i32)
                + (self.gauges[b + 3] as i32) * (spins[self.tiles[b + 3] as usize] as i32);
        }
        for u in 0..n {
            let mut acc = 0i32;
            for p in self.wdeg[u] as usize..self.wdeg[u + 1] as usize {
                acc += (self.weak_w_milli[p] as i32) * (spins[self.weak_nb[p] as usize] as i32);
            }
            h_weak[u] = acc;
        }
        for _pass in 0..6 {
            let mut flipped = 0usize;
            for u in 0..n {
                let su = spins[u] as i32;
                let mut d_strong = 0i32;
                for p in toff[u]..toff[u + 1] {
                    let gs = (inc_g[p] as i32) * su;
                    d_strong += 1 - gs * tsum[inc_tid[p] as usize];
                }
                let d_milli = d_strong * 2000 - 2 * su * h_weak[u];
                if d_milli < 0 {
                    let new_s = -su;
                    spins[u] = new_s as i8;
                    for p in toff[u]..toff[u + 1] {
                        tsum[inc_tid[p] as usize] += 2 * (inc_g[p] as i32) * new_s;
                    }
                    for p in self.wdeg[u] as usize..self.wdeg[u + 1] as usize {
                        h_weak[self.weak_nb[p] as usize] += 2 * (self.weak_w_milli[p] as i32) * new_s;
                    }
                    flipped += 1;
                }
            }
            if flipped == 0 {
                break;
            }
        }
        let mut strong_milli = 0i64;
        for t in 0..nt {
            let s2 = tsum[t] * tsum[t];
            strong_milli += 500 * (s2 - 4) as i64;
        }
        let mut weak_milli2 = 0i64;
        for u in 0..n {
            weak_milli2 += (spins[u] as i64) * (h_weak[u] as i64);
        }
        let base_milli = -(self.n_pairs as i64) * 1000 + self.self_weak_milli;
        (base_milli + strong_milli + weak_milli2 / 2) as f64 * 0.001
    }

    /// Super-Stage-0e: Deep CPU K4-BP (`n_reps` replicas x `iters` steps) WITH 1-flip local polish
    /// on readout spins — mimics GPU `probe0` on idle EPYC cores!
    /// Returns `(best_polished_e, mean_polished_e, top3_mean_e)`.
    pub fn deep_bp_polished(&self, seed: u64, n_reps: usize, iters: usize) -> (f64, f64, f64) {
        let n = self.n_super;
        let nt = self.n_tiles;
        let u_vec = &self.u8[..n];
        let mut cov = [[0.0f64; 8]; 8];
        for u in u_vec {
            for a in 0..8 {
                let ua = u[a] as f64;
                for b in a..8 {
                    cov[a][b] += ua * (u[b] as f64);
                }
            }
        }
        for a in 0..8 {
            for b in 0..a {
                cov[a][b] = cov[b][a];
            }
        }
        let (_evals, evecs64) = jacobi_sym8(cov);

        // Build tile incidence for fast 1-flip polish
        let mut toff = vec![0usize; n + 1];
        for &u in &self.tiles[..nt * 4] {
            toff[u as usize + 1] += 1;
        }
        for i in 0..n {
            toff[i + 1] += toff[i];
        }
        let mut tfill = toff.clone();
        let mut inc_tid = vec![0u16; nt * 4];
        let mut inc_g = vec![0i8; nt * 4];
        for idx in 0..nt * 4 {
            let u = self.tiles[idx] as usize;
            let pos = tfill[u];
            inc_tid[pos] = (idx >> 2) as u16;
            inc_g[pos] = self.gauges[idx];
            tfill[u] = pos + 1;
        }

        let (bp_tiles, bp_edges_scaled) = self.build_bp_factor_graph();
        let dampings = [0.25f32, 0.35, 0.45, 0.30, 0.20, 0.40, 0.32, 0.28];
        let weights = [250.0f32, 375.0, 500.0, 312.5, 280.0, 420.0, 350.0, 460.0];
        let ramps = [90.0f32, 130.0, 170.0, 200.0, 110.0, 150.0, 185.0, 220.0];
        let temps = [0.0f32, 45.0, 75.0, 110.0, 0.0, 60.0, 90.0, 35.0];

        let mut h = vec![0.0f32; n];
        let mut b_reinf = vec![0.0f32; n];
        let mut em = vec![[0.0f32; 2]; bp_edges_scaled.len()];
        let mut rm = vec![[0.0f32; 4]; bp_tiles.len()];
        let mut spins = vec![1i8; n];
        let mut tsum = vec![0i32; nt];
        let mut h_weak = vec![0i32; n];
        let mut rep_es = Vec::with_capacity(n_reps);

        for r in 0..n_reps {
            let mut st = seed
                ^ (0x9E37_79B9_7F4A_7C15u64.wrapping_mul((r + 17) as u64))
                ^ 0x517C_C1B7_2722_0A95;
            let mut next_f32 = || -> f32 {
                st ^= st >> 12;
                st ^= st << 25;
                st ^= st >> 27;
                let v = st.wrapping_mul(0x2545_F491_4F6C_DD1D);
                ((v >> 40) as f32) * (1.0 / 16_777_216.0) - 0.5
            };
            // First 2/3 reps seeded from top-8 BM8 eigenvectors, last 1/3 cold unseeded!
            let use_seed = r * 3 < n_reps * 2;
            let ek = r % 8;
            let prior_scale = if r % 2 == 0 { 120.0f32 } else { 65.0f32 };
            for u in 0..n {
                let noise = next_f32();
                let init = if use_seed {
                    let uv = u_vec[u];
                    let mut p = 0.0f32;
                    for d in 0..8 {
                        p += uv[d] * (evecs64[ek][d] as f32);
                    }
                    -prior_scale * p + noise
                } else {
                    noise
                };
                b_reinf[u] = init;
                h[u] = init;
            }
            for e in &mut em { *e = [0.0, 0.0]; }
            for t in &mut rm { *t = [0.0; 4]; }

            let damping = dampings[r % dampings.len()];
            let weight = weights[r % weights.len()];
            let ramp = ramps[r % ramps.len()];
            let temp = if use_seed { temps[r % temps.len()] } else { 0.0 };
            let inv_t = if temp > 0.0 { 1.0 / temp } else { 0.0 };
            let mut rep_best = 1e9f64;

            for iter in 0..iters {
                for (ei, &(u, v, j)) in bp_edges_scaled.iter().enumerate() {
                    let (ui, vi) = (u as usize, v as usize);
                    let old = em[ei];
                    let hu = h[ui] - old[0];
                    let hv = h[vi] - old[1];
                    let sgn = if j > 0.0 { 1.0f32 } else { -1.0f32 };
                    let aj = j.abs();
                    let mu = -sgn * hv.clamp(-aj, aj);
                    let mv = -sgn * hu.clamp(-aj, aj);
                    let na = old[0] + damping * (mu - old[0]);
                    let nb = old[1] + damping * (mv - old[1]);
                    h[ui] += na - old[0];
                    h[vi] += nb - old[1];
                    em[ei] = [na, nb];
                }
                for (ti, tile) in bp_tiles.iter().enumerate() {
                    let old = rm[ti];
                    let cav = [
                        h[tile.nodes[0]] - old[0],
                        h[tile.nodes[1]] - old[1],
                        h[tile.nodes[2]] - old[2],
                        h[tile.nodes[3]] - old[3],
                    ];
                    let mut cost = [0.0f32; 16];
                    for mask in 0..16usize {
                        let lin = if (mask & 1) != 0 { cav[0] } else { -cav[0] }
                            + if (mask & 2) != 0 { cav[1] } else { -cav[1] }
                            + if (mask & 4) != 0 { cav[2] } else { -cav[2] }
                            + if (mask & 8) != 0 { cav[3] } else { -cav[3] };
                        cost[mask] = weight * tile.q2[mask] + tile.fixed_scaled[mask] + lin;
                    }
                    let mut next = [0.0f32; 4];
                    if temp <= 0.0 {
                        let mut m0 = [1e30f32; 4];
                        let mut m1 = [1e30f32; 4];
                        for mask in 0..16usize {
                            let c = cost[mask];
                            if (mask & 1) != 0 { if c < m1[0] { m1[0] = c; } } else if c < m0[0] { m0[0] = c; }
                            if (mask & 2) != 0 { if c < m1[1] { m1[1] = c; } } else if c < m0[1] { m0[1] = c; }
                            if (mask & 4) != 0 { if c < m1[2] { m1[2] = c; } } else if c < m0[2] { m0[2] = c; }
                            if (mask & 8) != 0 { if c < m1[3] { m1[3] = c; } } else if c < m0[3] { m0[3] = c; }
                        }
                        for k in 0..4 {
                            let msg = 0.5 * (m1[k] - m0[k]) - cav[k];
                            let nk = old[k] + damping * (msg - old[k]);
                            h[tile.nodes[k]] += nk - old[k];
                            next[k] = nk;
                        }
                    } else {
                        let mut m_all = 1e30f32;
                        for &c in &cost { if c < m_all { m_all = c; } }
                        let mut wp = [0.0f32; 4];
                        let mut wn = [0.0f32; 4];
                        for mask in 0..16usize {
                            let d = (cost[mask] - m_all) * inv_t;
                            if d < 18.0 {
                                let e = (-d).exp();
                                if (mask & 1) != 0 { wp[0] += e; } else { wn[0] += e; }
                                if (mask & 2) != 0 { wp[1] += e; } else { wn[1] += e; }
                                if (mask & 4) != 0 { wp[2] += e; } else { wn[2] += e; }
                                if (mask & 8) != 0 { wp[3] += e; } else { wn[3] += e; }
                            }
                        }
                        for k in 0..4 {
                            let msg = if wp[k] > 1e-6 && wn[k] > 1e-6 {
                                -0.5 * temp * (wp[k] / wn[k]).ln() - cav[k]
                            } else {
                                let bit = 1usize << k;
                                let mut min_p = 1e30f32;
                                let mut min_n = 1e30f32;
                                for mask in 0..16usize {
                                    let c = cost[mask];
                                    if (mask & bit) != 0 { if c < min_p { min_p = c; } }
                                    else if c < min_n { min_n = c; }
                                }
                                let mut sp = 0.0f32;
                                let mut sn = 0.0f32;
                                for mask in 0..16usize {
                                    if (mask & bit) != 0 {
                                        let d = (cost[mask] - min_p) * inv_t;
                                        if d < 18.0 { sp += (-d).exp(); }
                                    } else {
                                        let d = (cost[mask] - min_n) * inv_t;
                                        if d < 18.0 { sn += (-d).exp(); }
                                    }
                                }
                                0.5 * ((min_p - min_n) - temp * (sp.max(1e-12) / sn.max(1e-12)).ln()) - cav[k]
                            };
                            let nk = old[k] + damping * (msg - old[k]);
                            h[tile.nodes[k]] += nk - old[k];
                            next[k] = nk;
                        }
                    }
                    rm[ti] = next;
                }
                // Check & 1-flip polish every 40 steps after iter 79 + final step!
                if (iter >= 79 && ((iter + 1) % 40 == 0)) || iter + 1 == iters {
                    for u in 0..n {
                        spins[u] = if h[u] > 0.0 { -1 } else { 1 };
                    }
                    let e = self.polish_spins_1flip(&mut spins, &toff, &inc_tid, &inc_g, &mut tsum, &mut h_weak);
                    if e < rep_best {
                        rep_best = e;
                    }
                }
                let rho = ((iter as f32) / ramp).min(0.95);
                for u in 0..n {
                    let nb = (rho * h[u]).clamp(-20000.0, 20000.0);
                    h[u] += nb - b_reinf[u];
                    b_reinf[u] = nb;
                }
            }
            rep_es.push(rep_best);
        }
        rep_es.sort_by(f64::total_cmp);
        let best = rep_es[0];
        let mean = rep_es.iter().sum::<f64>() / (rep_es.len() as f64);
        let k3 = rep_es.len().min(3);
        let top3 = rep_es[..k3].iter().sum::<f64>() / (k3 as f64);
        (best, mean, top3)
    }

    /// Production Super-Stage-0e (`--mode super`): unified single-pass Jacobi + factor-graph + tile-CSR,
    /// combining `v3` (4 seeded + 4 unseeded x 120) + `DeepBP` (6 x 200 + 1-flip polish, top-3 mean)
    /// + `GW32` (32 hyperplanes + 1-flip polish, top-5 mean).
    /// Empirically verified on 1,819 live Aglais nonces: AUC90 = 0.7965 (+0.089 vs v3fast),
    /// Top-33% Recall <= -5090 = 76.5%, <= -5093 = 4/4 (100%)!
    pub fn stage_e_super(
        &self,
        seed: u64,
        bm_deep: f64,
        deep_reps: usize,
        deep_iters: usize,
        gw_trials: usize,
    ) -> (f32, Vec<[i8; 8]>, [[f32; 8]; 8]) {
        let n = self.n_super;
        let nt = self.n_tiles;
        let u_vec = &self.u8[..n];

        // 1. Single Jacobi 8x8 eigendecomposition
        let mut cov = [[0.0f64; 8]; 8];
        for u in u_vec {
            for a in 0..8 {
                let ua = u[a] as f64;
                for b in a..8 {
                    cov[a][b] += ua * (u[b] as f64);
                }
            }
        }
        for a in 0..8 {
            for b in 0..a {
                cov[a][b] = cov[b][a];
            }
        }
        let (_evals, evecs64) = jacobi_sym8(cov);
        let mut evecs = [[0.0f32; 8]; 8];
        for k in 0..8 {
            for r in 0..8 {
                evecs[k][r] = evecs64[k][r] as f32;
            }
        }
        let mut projs = Vec::with_capacity(n);
        for u in u_vec {
            let mut p = [0.0f32; 4];
            for k in 0..4 {
                let mut acc = 0.0f32;
                for r in 0..8 {
                    acc += u[r] * evecs[k][r];
                }
                p[k] = acc;
            }
            projs.push(p);
        }

        // 2. Single factor graph + tile incidence CSR build
        let (bp_tiles, bp_edges) = self.build_bp_factor_graph();
        let mut toff = vec![0usize; n + 1];
        for &u in &self.tiles[..nt * 4] {
            toff[u as usize + 1] += 1;
        }
        for i in 0..n {
            toff[i + 1] += toff[i];
        }
        let mut tfill = toff.clone();
        let mut inc_tid = vec![0u16; nt * 4];
        let mut inc_g = vec![0i8; nt * 4];
        for idx in 0..nt * 4 {
            let u = self.tiles[idx] as usize;
            let pos = tfill[u];
            inc_tid[pos] = (idx >> 2) as u16;
            inc_g[pos] = self.gauges[idx];
            tfill[u] = pos + 1;
        }

        // 3. v3 (4 seeded + 4 unseeded x 120)
        let bp_bmst = self.run_bp_4rep_inner(
            &bp_tiles,
            &bp_edges,
            seed.wrapping_add(7919),
            Some(&projs),
            120.0,
            [0.0, 45.0, 75.0, 110.0],
        );
        let bp_v1 = self.run_bp_4rep_inner(
            &bp_tiles,
            &bp_edges,
            seed.wrapping_add(7919),
            None,
            0.0,
            [0.0; 4],
        );
        let v3 = bm_deep + 0.15 * bp_v1 + 0.20 * bp_bmst;

        // 4. Deep polished K4-BP (`deep_reps` x `deep_iters` + 1-flip polish)
        let dampings = [0.25f32, 0.35, 0.45, 0.30, 0.20, 0.40, 0.32, 0.28];
        let weights = [250.0f32, 375.0, 500.0, 312.5, 280.0, 420.0, 350.0, 460.0];
        let ramps = [90.0f32, 130.0, 170.0, 200.0, 110.0, 150.0, 185.0, 220.0];
        let temps = [0.0f32, 45.0, 75.0, 110.0, 0.0, 60.0, 90.0, 35.0];

        let mut h = vec![0.0f32; n];
        let mut b_reinf = vec![0.0f32; n];
        let mut em = vec![[0.0f32; 2]; bp_edges.len()];
        let mut rm = vec![[0.0f32; 4]; bp_tiles.len()];
        let mut spins = vec![1i8; n];
        let mut tsum = vec![0i32; nt];
        let mut h_weak = vec![0i32; n];
        let mut rep_es = Vec::with_capacity(deep_reps.max(1));

        for r in 0..deep_reps {
            let mut st = seed
                ^ (0x9E37_79B9_7F4A_7C15u64.wrapping_mul((r + 17) as u64))
                ^ 0x517C_C1B7_2722_0A95;
            let mut next_f32 = || -> f32 {
                st ^= st >> 12;
                st ^= st << 25;
                st ^= st >> 27;
                let v = st.wrapping_mul(0x2545_F491_4F6C_DD1D);
                ((v >> 40) as f32) * (1.0 / 16_777_216.0) - 0.5
            };
            let use_seed = r * 3 < deep_reps * 2;
            let ek = r % 8;
            let prior_scale = if r % 2 == 0 { 120.0f32 } else { 65.0f32 };
            for u in 0..n {
                let noise = next_f32();
                let init = if use_seed {
                    let uv = u_vec[u];
                    let mut p = 0.0f32;
                    for d in 0..8 {
                        p += uv[d] * evecs[ek][d];
                    }
                    -prior_scale * p + noise
                } else {
                    noise
                };
                b_reinf[u] = init;
                h[u] = init;
            }
            for e in &mut em { *e = [0.0, 0.0]; }
            for t in &mut rm { *t = [0.0; 4]; }

            let damping = dampings[r % dampings.len()];
            let weight = weights[r % weights.len()];
            // Unscaled ramp schedule (empirically outperformed proportional scaling on 819 live
            // Aglais nonces: AUC90 0.7705 vs 0.7545, as slower replicas avoid premature freezing).
            let ramp = ramps[r % ramps.len()];
            let temp = if use_seed { temps[r % temps.len()] } else { 0.0 };
            let inv_t = if temp > 0.0 { 1.0 / temp } else { 0.0 };
            let mut rep_best = 1e9f64;

            for iter in 0..deep_iters {
                for (ei, &(u, v, j)) in bp_edges.iter().enumerate() {
                    let (ui, vi) = (u as usize, v as usize);
                    let old = em[ei];
                    let hu = h[ui] - old[0];
                    let hv = h[vi] - old[1];
                    let sgn = if j > 0.0 { 1.0f32 } else { -1.0f32 };
                    let aj = j.abs();
                    let mu = -sgn * hv.clamp(-aj, aj);
                    let mv = -sgn * hu.clamp(-aj, aj);
                    let na = old[0] + damping * (mu - old[0]);
                    let nb = old[1] + damping * (mv - old[1]);
                    h[ui] += na - old[0];
                    h[vi] += nb - old[1];
                    em[ei] = [na, nb];
                }
                for (ti, tile) in bp_tiles.iter().enumerate() {
                    let old = rm[ti];
                    let cav = [
                        h[tile.nodes[0]] - old[0],
                        h[tile.nodes[1]] - old[1],
                        h[tile.nodes[2]] - old[2],
                        h[tile.nodes[3]] - old[3],
                    ];
                    let mut cost = [0.0f32; 16];
                    for mask in 0..16usize {
                        let lin = if (mask & 1) != 0 { cav[0] } else { -cav[0] }
                            + if (mask & 2) != 0 { cav[1] } else { -cav[1] }
                            + if (mask & 4) != 0 { cav[2] } else { -cav[2] }
                            + if (mask & 8) != 0 { cav[3] } else { -cav[3] };
                        cost[mask] = weight * tile.q2[mask] + tile.fixed_scaled[mask] + lin;
                    }
                    let mut next = [0.0f32; 4];
                    if temp <= 0.0 {
                        let mut m0 = [1e30f32; 4];
                        let mut m1 = [1e30f32; 4];
                        for mask in 0..16usize {
                            let c = cost[mask];
                            if (mask & 1) != 0 { if c < m1[0] { m1[0] = c; } } else if c < m0[0] { m0[0] = c; }
                            if (mask & 2) != 0 { if c < m1[1] { m1[1] = c; } } else if c < m0[1] { m0[1] = c; }
                            if (mask & 4) != 0 { if c < m1[2] { m1[2] = c; } } else if c < m0[2] { m0[2] = c; }
                            if (mask & 8) != 0 { if c < m1[3] { m1[3] = c; } } else if c < m0[3] { m0[3] = c; }
                        }
                        for k in 0..4 {
                            let msg = 0.5 * (m1[k] - m0[k]) - cav[k];
                            let nk = old[k] + damping * (msg - old[k]);
                            h[tile.nodes[k]] += nk - old[k];
                            next[k] = nk;
                        }
                    } else {
                        let mut m_all = 1e30f32;
                        for &c in &cost { if c < m_all { m_all = c; } }
                        let mut wp = [0.0f32; 4];
                        let mut wn = [0.0f32; 4];
                        for mask in 0..16usize {
                            let d = (cost[mask] - m_all) * inv_t;
                            if d < 18.0 {
                                let e = (-d).exp();
                                if (mask & 1) != 0 { wp[0] += e; } else { wn[0] += e; }
                                if (mask & 2) != 0 { wp[1] += e; } else { wn[1] += e; }
                                if (mask & 4) != 0 { wp[2] += e; } else { wn[2] += e; }
                                if (mask & 8) != 0 { wp[3] += e; } else { wn[3] += e; }
                            }
                        }
                        for k in 0..4 {
                            let msg = if wp[k] > 1e-6 && wn[k] > 1e-6 {
                                -0.5 * temp * (wp[k] / wn[k]).ln() - cav[k]
                            } else {
                                let bit = 1usize << k;
                                let mut min_p = 1e30f32;
                                let mut min_n = 1e30f32;
                                for mask in 0..16usize {
                                    let c = cost[mask];
                                    if (mask & bit) != 0 { if c < min_p { min_p = c; } }
                                    else if c < min_n { min_n = c; }
                                }
                                let mut sp = 0.0f32;
                                let mut sn = 0.0f32;
                                for mask in 0..16usize {
                                    if (mask & bit) != 0 {
                                        let d = (cost[mask] - min_p) * inv_t;
                                        if d < 18.0 { sp += (-d).exp(); }
                                    } else {
                                        let d = (cost[mask] - min_n) * inv_t;
                                        if d < 18.0 { sn += (-d).exp(); }
                                    }
                                }
                                0.5 * ((min_p - min_n) - temp * (sp.max(1e-12) / sn.max(1e-12)).ln()) - cav[k]
                            };
                            let nk = old[k] + damping * (msg - old[k]);
                            h[tile.nodes[k]] += nk - old[k];
                            next[k] = nk;
                        }
                    }
                    rm[ti] = next;
                }
                if (iter >= 79 && ((iter + 1) % 40 == 0)) || iter + 1 == deep_iters {
                    for u in 0..n {
                        spins[u] = if h[u] > 0.0 { -1 } else { 1 };
                    }
                    let e = self.polish_spins_1flip(&mut spins, &toff, &inc_tid, &inc_g, &mut tsum, &mut h_weak);
                    if e < rep_best {
                        rep_best = e;
                    }
                }
                let rho = ((iter as f32) / ramp).min(0.95);
                for u in 0..n {
                    let nb = (rho * h[u]).clamp(-20000.0, 20000.0);
                    h[u] += nb - b_reinf[u];
                    b_reinf[u] = nb;
                }
            }
            rep_es.push(rep_best);
        }
        rep_es.sort_by(f64::total_cmp);
        // Top-3 mean of deep BP replicas (matches winning v5.0 offline A/B configuration).
        let k_top = 3usize.min(rep_es.len());
        let dbp_top = if rep_es.is_empty() { bm_deep } else { rep_es[..k_top].iter().sum::<f64>() / (k_top as f64) };

        // 5. GW hyperplane rounding + 1-flip polish (reuses toff/inc_tid/evecs!)
        let mut gw_es = Vec::with_capacity(gw_trials.max(1));
        let mut st_gw = seed ^ 0xD1B5_4A32_D192_ED03u64;
        let mut next_gw = || -> f32 {
            st_gw ^= st_gw >> 12;
            st_gw ^= st_gw << 25;
            st_gw ^= st_gw >> 27;
            let v = st_gw.wrapping_mul(0x2545_F491_4F6C_DD1D);
            ((v >> 40) as f32) * (1.0 / 16_777_216.0) * 2.0 - 1.0
        };
        for tr in 0..gw_trials {
            let dir: [f32; 8] = if tr < 8 {
                evecs[tr]
            } else {
                let mut d = [0.0f32; 8];
                for r in 0..8 {
                    d[r] = next_gw() + next_gw() + next_gw();
                }
                d
            };
            for i in 0..n {
                let u = u_vec[i];
                let dot = u[0]*dir[0] + u[1]*dir[1] + u[2]*dir[2] + u[3]*dir[3]
                        + u[4]*dir[4] + u[5]*dir[5] + u[6]*dir[6] + u[7]*dir[7];
                spins[i] = if dot >= 0.0 { 1 } else { -1 };
            }
            gw_es.push(self.polish_spins_1flip(&mut spins, &toff, &inc_tid, &inc_g, &mut tsum, &mut h_weak));
        }
        gw_es.sort_by(f64::total_cmp);
        let k5 = gw_es.len().min(5).max(1);
        let gw_m5 = if gw_es.is_empty() { bm_deep } else { gw_es[..k5].iter().sum::<f64>() / (k5 as f64) };

        // Normalize composite to exact v3fast unit spread (~1.20x energy units)
        let score = ((v3 + 0.30 * dbp_top + 0.15 * gw_m5) * (1.20 / 1.80)) as f32;

        let mut u_orig = Vec::with_capacity(self.rep.len());
        for &cid in &self.rep {
            let u = u_vec[cid as usize];
            let mut q = [0i8; 8];
            for r in 0..8 {
                q[r] = (u[r] * 127.0).round().clamp(-127.0, 127.0) as i8;
            }
            u_orig.push(q);
        }
        (score, u_orig, evecs)
    }

    fn exact_energy_units(&self, s: &[i8]) -> f64 {
        let mut strong_e = -(self.n_pairs as f64);
        for t in 0..self.n_tiles {
            let b = t * 4;
            let sum = (self.gauges[b] as i32) * (s[self.tiles[b] as usize] as i32)
                + (self.gauges[b + 1] as i32) * (s[self.tiles[b + 1] as usize] as i32)
                + (self.gauges[b + 2] as i32) * (s[self.tiles[b + 2] as usize] as i32)
                + (self.gauges[b + 3] as i32) * (s[self.tiles[b + 3] as usize] as i32);
            strong_e += 0.5 * ((sum * sum - 4) as f64);
        }
        let mut weak_sum = 0.0f64;
        for u in 0..self.n_super {
            let su = s[u] as f64;
            let mut acc = 0.0f64;
            for p in self.wdeg[u] as usize..self.wdeg[u + 1] as usize {
                acc += (self.weak_w_milli[p] as f64 * 0.001)
                    * (s[self.weak_nb[p] as usize] as f64);
            }
            weak_sum += su * acc;
        }
        strong_e + 0.5 * weak_sum + (self.self_weak_milli as f64 / 1000.0)
    }

    fn build_bp_factor_graph(&self) -> (Vec<BpTile>, Vec<(u32, u32, f32)>) {
        // dedup_edges is already sorted by (a, b) with a < b!
        let mut used_edge = vec![false; self.dedup_edges.len()];
        let mut bp_tiles = Vec::with_capacity(self.n_tiles);

        for tid in 0..self.n_tiles {
            let b = tid * 4;
            let t = [
                self.tiles[b] as usize,
                self.tiles[b + 1] as usize,
                self.tiles[b + 2] as usize,
                self.tiles[b + 3] as usize,
            ];
            let g = [
                self.gauges[b],
                self.gauges[b + 1],
                self.gauges[b + 2],
                self.gauges[b + 3],
            ];
            let mut intra = [[0.0f32; 4]; 4];
            for a in 0..4 {
                for c in a + 1..4 {
                    let ua = t[a] as u16;
                    let ub = t[c] as u16;
                    if ua == ub {
                        continue;
                    }
                    let key = if ua < ub { (ua, ub) } else { (ub, ua) };
                    if let Ok(pos) = self
                        .dedup_edges
                        .binary_search_by_key(&key, |&(x, y, _)| (x, y))
                    {
                        if !used_edge[pos] {
                            used_edge[pos] = true;
                            intra[a][c] = self.dedup_edges[pos].2 as f32;
                        }
                    }
                }
            }
            let mut q2 = [0.0f32; 16];
            let mut fixed_scaled = [0.0f32; 16];
            for mask in 0..16usize {
                let s = [
                    if (mask & 1) != 0 { 1i32 } else { -1 },
                    if (mask & 2) != 0 { 1i32 } else { -1 },
                    if (mask & 4) != 0 { 1i32 } else { -1 },
                    if (mask & 8) != 0 { 1i32 } else { -1 },
                ];
                let sum = (g[0] as i32) * s[0]
                    + (g[1] as i32) * s[1]
                    + (g[2] as i32) * s[2]
                    + (g[3] as i32) * s[3];
                q2[mask] = (sum * sum) as f32;
                let mut f = 0.0f32;
                for a in 0..4 {
                    for c in a + 1..4 {
                        if intra[a][c] != 0.0 {
                            f += intra[a][c] * ((s[a] * s[c]) as f32);
                        }
                    }
                }
                fixed_scaled[mask] = 1.2 * f;
            }
            bp_tiles.push(BpTile {
                nodes: t,
                q2,
                fixed_scaled,
            });
        }

        let mut bp_edges_scaled = Vec::with_capacity(self.dedup_edges.len());
        for (idx, &(a, b, w)) in self.dedup_edges.iter().enumerate() {
            if !used_edge[idx] {
                bp_edges_scaled.push((a as u32, b as u32, 1.2 * (w as f32)));
            }
        }
        (bp_tiles, bp_edges_scaled)
    }

    fn run_bp_4rep_inner(
        &self,
        bp_tiles: &[BpTile],
        bp_edges_scaled: &[(u32, u32, f32)],
        seed: u64,
        projs: Option<&[[f32; 4]]>,
        prior_scale: f32,
        temps: [f32; 4],
    ) -> f64 {
        let n = self.n_super;
        let dampings = [0.25f32, 0.35, 0.45, 0.30];
        let weights = [250.0f32, 375.0, 500.0, 312.5];
        let ramps = [55.0f32, 75.0, 95.0, 115.0];

        let mut h = vec![0.0f32; n];
        let mut b_reinf = vec![0.0f32; n];
        let mut em = vec![[0.0f32; 2]; bp_edges_scaled.len()];
        let mut rm = vec![[0.0f32; 4]; bp_tiles.len()];
        let mut spins = vec![1i8; n];
        let mut sum_120 = 0.0f64;

        for r in 0..4usize {
            let mut st = seed
                ^ (0x9E37_79B9_7F4A_7C15u64.wrapping_mul((r + 1) as u64))
                ^ 0x517C_C1B7_2722_0A95;
            let mut next_f32 = || -> f32 {
                st ^= st >> 12;
                st ^= st << 25;
                st ^= st >> 27;
                let v = st.wrapping_mul(0x2545_F491_4F6C_DD1D);
                ((v >> 40) as f32) * (1.0 / 16_777_216.0) - 0.5
            };
            for u in 0..n {
                let noise = next_f32();
                let init = if let Some(pr) = projs {
                    -prior_scale * pr[u][r] + noise
                } else {
                    noise
                };
                b_reinf[u] = init;
                h[u] = init;
            }
            for e in &mut em {
                *e = [0.0, 0.0];
            }
            for t in &mut rm {
                *t = [0.0; 4];
            }
            let damping = dampings[r];
            let weight = weights[r];
            let ramp = ramps[r];
            let temp = temps[r];
            let inv_t = if temp > 0.0 { 1.0 / temp } else { 0.0 };
            let mut rep_best = 1e9f64;

            for iter in 0..120usize {
                for (ei, &(u, v, j)) in bp_edges_scaled.iter().enumerate() {
                    let (ui, vi) = (u as usize, v as usize);
                    let old = em[ei];
                    let hu = h[ui] - old[0];
                    let hv = h[vi] - old[1];
                    let sgn = if j > 0.0 { 1.0f32 } else { -1.0f32 };
                    let aj = j.abs();
                    let mu = -sgn * hv.clamp(-aj, aj);
                    let mv = -sgn * hu.clamp(-aj, aj);
                    let na = old[0] + damping * (mu - old[0]);
                    let nb = old[1] + damping * (mv - old[1]);
                    h[ui] += na - old[0];
                    h[vi] += nb - old[1];
                    em[ei] = [na, nb];
                }
                for (ti, tile) in bp_tiles.iter().enumerate() {
                    let old = rm[ti];
                    let cav = [
                        h[tile.nodes[0]] - old[0],
                        h[tile.nodes[1]] - old[1],
                        h[tile.nodes[2]] - old[2],
                        h[tile.nodes[3]] - old[3],
                    ];
                    let mut cost = [0.0f32; 16];
                    for mask in 0..16usize {
                        let lin = if (mask & 1) != 0 { cav[0] } else { -cav[0] }
                            + if (mask & 2) != 0 { cav[1] } else { -cav[1] }
                            + if (mask & 4) != 0 { cav[2] } else { -cav[2] }
                            + if (mask & 8) != 0 { cav[3] } else { -cav[3] };
                        cost[mask] = weight * tile.q2[mask] + tile.fixed_scaled[mask] + lin;
                    }
                    let mut next = [0.0f32; 4];
                    if temp <= 0.0 {
                        let mut m0 = [1e30f32; 4];
                        let mut m1 = [1e30f32; 4];
                        for mask in 0..16usize {
                            let c = cost[mask];
                            if (mask & 1) != 0 { if c < m1[0] { m1[0] = c; } } else if c < m0[0] { m0[0] = c; }
                            if (mask & 2) != 0 { if c < m1[1] { m1[1] = c; } } else if c < m0[1] { m0[1] = c; }
                            if (mask & 4) != 0 { if c < m1[2] { m1[2] = c; } } else if c < m0[2] { m0[2] = c; }
                            if (mask & 8) != 0 { if c < m1[3] { m1[3] = c; } } else if c < m0[3] { m0[3] = c; }
                        }
                        for k in 0..4 {
                            let msg = 0.5 * (m1[k] - m0[k]) - cav[k];
                            let nk = old[k] + damping * (msg - old[k]);
                            h[tile.nodes[k]] += nk - old[k];
                            next[k] = nk;
                        }
                    } else {
                        let mut m_all = 1e30f32;
                        for &c in &cost {
                            if c < m_all {
                                m_all = c;
                            }
                        }
                        let mut wp = [0.0f32; 4];
                        let mut wn = [0.0f32; 4];
                        for mask in 0..16usize {
                            let d = (cost[mask] - m_all) * inv_t;
                            if d < 18.0 {
                                let e = (-d).exp();
                                if (mask & 1) != 0 { wp[0] += e; } else { wn[0] += e; }
                                if (mask & 2) != 0 { wp[1] += e; } else { wn[1] += e; }
                                if (mask & 4) != 0 { wp[2] += e; } else { wn[2] += e; }
                                if (mask & 8) != 0 { wp[3] += e; } else { wn[3] += e; }
                            }
                        }
                        for k in 0..4 {
                            let msg = if wp[k] > 1e-6 && wn[k] > 1e-6 {
                                -0.5 * temp * (wp[k] / wn[k]).ln() - cav[k]
                            } else {
                                let bit = 1usize << k;
                                let mut min_p = 1e30f32;
                                let mut min_n = 1e30f32;
                                for mask in 0..16usize {
                                    let c = cost[mask];
                                    if (mask & bit) != 0 {
                                        if c < min_p { min_p = c; }
                                    } else if c < min_n {
                                        min_n = c;
                                    }
                                }
                                let mut sp = 0.0f32;
                                let mut sn = 0.0f32;
                                for mask in 0..16usize {
                                    if (mask & bit) != 0 {
                                        let d = (cost[mask] - min_p) * inv_t;
                                        if d < 18.0 { sp += (-d).exp(); }
                                    } else {
                                        let d = (cost[mask] - min_n) * inv_t;
                                        if d < 18.0 { sn += (-d).exp(); }
                                    }
                                }
                                0.5 * ((min_p - min_n) - temp * (sp.max(1e-12) / sn.max(1e-12)).ln())
                                    - cav[k]
                            };
                            let nk = old[k] + damping * (msg - old[k]);
                            h[tile.nodes[k]] += nk - old[k];
                            next[k] = nk;
                        }
                    }
                    rm[ti] = next;
                }
                if iter >= 39 && ((iter + 1) % 10 == 0) {
                    for u in 0..n {
                        spins[u] = if h[u] > 0.0 { -1 } else { 1 };
                    }
                    let e = self.exact_energy_units(&spins);
                    if e < rep_best {
                        rep_best = e;
                    }
                }
                let rho = ((iter as f32) / ramp).min(0.95);
                for u in 0..n {
                    let nb = (rho * h[u]).clamp(-20000.0, 20000.0);
                    h[u] += nb - b_reinf[u];
                    b_reinf[u] = nb;
                }
            }
            sum_120 += rep_best;
        }
        sum_120 / 4.0
    }
}

struct BpTile {
    nodes: [usize; 4],
    q2: [f32; 16],
    fixed_scaled: [f32; 16],
}

fn jacobi_sym8(mut a: [[f64; 8]; 8]) -> ([f64; 8], [[f64; 8]; 8]) {
    let mut v = [[0.0f64; 8]; 8];
    for i in 0..8 {
        v[i][i] = 1.0;
    }
    for _ in 0..64 {
        let mut p = 0;
        let mut q = 1;
        let mut max_off = 0.0f64;
        for i in 0..8 {
            for j in i + 1..8 {
                let val = a[i][j].abs();
                if val > max_off {
                    max_off = val;
                    p = i;
                    q = j;
                }
            }
        }
        if max_off < 1e-12 {
            break;
        }
        let app = a[p][p];
        let aqq = a[q][q];
        let apq = a[p][q];
        let tau = (aqq - app) / (2.0 * apq);
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;
        a[p][p] = app - t * apq;
        a[q][q] = aqq + t * apq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;
        for k in 0..8 {
            if k != p && k != q {
                let akp = a[k][p];
                let akq = a[k][q];
                a[k][p] = c * akp - s * akq;
                a[p][k] = a[k][p];
                a[k][q] = s * akp + c * akq;
                a[q][k] = a[k][q];
            }
            let vkp = v[k][p];
            let vkq = v[k][q];
            v[k][p] = c * vkp - s * vkq;
            v[k][q] = s * vkp + c * vkq;
        }
    }
    let mut order = [0usize, 1, 2, 3, 4, 5, 6, 7];
    order.sort_by(|&i, &j| a[j][j].total_cmp(&a[i][i]));
    let mut evals = [0.0f64; 8];
    let mut evecs = [[0.0f64; 8]; 8];
    for (rank, &idx) in order.iter().enumerate() {
        evals[rank] = a[idx][idx];
        for coord in 0..8 {
            evecs[rank][coord] = v[coord][idx];
        }
    }
    (evals, evecs)
}

/// Lock-free read, batched-sort rolling quantile gate (scales to 128+ cores with zero contention!).
pub struct FastGate {
    keep: f64,
    window: usize,
    warmup: usize,
    recompute_every: usize,
    cut_bits: AtomicU32,
    count: AtomicUsize,
    state: Mutex<FastGateState>,
}

struct FastGateState {
    buf: Vec<f32>,
    pos: usize,
    sorted: Vec<f32>,
    inserts_since_sort: usize,
}

impl FastGate {
    pub fn new(keep: f64, window: usize, warmup: usize, recompute_every: usize) -> Self {
        Self {
            keep,
            window,
            warmup,
            recompute_every: recompute_every.max(1),
            cut_bits: AtomicU32::new(f32::INFINITY.to_bits()),
            count: AtomicUsize::new(0),
            state: Mutex::new(FastGateState {
                buf: Vec::with_capacity(window),
                pos: 0,
                sorted: Vec::with_capacity(window),
                inserts_since_sort: 0,
            }),
        }
    }

    pub fn reset(&self) {
        let mut st = self.state.lock().unwrap();
        st.buf.clear();
        st.pos = 0;
        st.sorted.clear();
        st.inserts_since_sort = 0;
        self.count.store(0, Ordering::Relaxed);
        self.cut_bits
            .store(f32::INFINITY.to_bits(), Ordering::Relaxed);
    }

    /// Lock-free test against current threshold.
    #[inline(always)]
    pub fn is_warm(&self) -> bool {
        self.count.load(Ordering::Relaxed) >= self.warmup
    }

    #[inline(always)]
    pub fn check_fast(&self, score: f32) -> bool {
        if !score.is_finite() {
            return false;
        }
        if self.count.load(Ordering::Relaxed) < self.warmup {
            return true;
        }
        score <= f32::from_bits(self.cut_bits.load(Ordering::Relaxed))
    }

    /// Batch-insert a slice of scores (once per 16-salt sub-job -> 16x fewer lock acquisitions!).
    pub fn observe_batch(&self, scores: &[f32]) {
        if scores.is_empty() {
            return;
        }
        let mut st = self.state.lock().unwrap();
        for &score in scores {
            if !score.is_finite() {
                continue;
            }
            if st.buf.len() < self.window {
                st.buf.push(score);
            } else {
                let p = st.pos;
                st.buf[p] = score;
                st.pos = (p + 1) % self.window;
            }
            st.inserts_since_sort += 1;
        }
        let n = st.buf.len();
        self.count.store(n, Ordering::Relaxed);
        if n >= self.warmup
            && (st.inserts_since_sort >= self.recompute_every || st.sorted.is_empty())
        {
            let FastGateState {
                ref buf,
                ref mut sorted,
                ref mut inserts_since_sort,
                ..
            } = *st;
            sorted.clear();
            sorted.extend_from_slice(buf);
            sorted.sort_unstable_by(f32::total_cmp);
            let cut_idx = (((sorted.len() as f64) * self.keep).floor() as usize)
                .min(sorted.len().saturating_sub(1));
            self.cut_bits
                .store(sorted[cut_idx].to_bits(), Ordering::Relaxed);
            *inserts_since_sort = 0;
        }
    }
}
