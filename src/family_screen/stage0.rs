//! Stage 0 for `quip/twisted-ice-dimer@1`: the "lean" linear model of
//! quip-build/agentA (`screen_weights.json`, `lean_tri_pairs_Pquads`), ported
//! from agentA's `gen/src/main.rs` (`features`, `build_topo_q`).
//!
//! Every edge is typed from the instance's own tile list:
//! - S: one of the six edges of a selected tile;
//! - P: not a tile edge, and J = -1000 (a dimer pair);
//! - W: everything else (the weak +-62 couplings).
//!
//! The tile list comes from the generator (`FastDimer::couplings`) or, on the
//! canonical path, from the instance metadata (`local_gauges`), exactly as
//! agentA does. J alone cannot type a -1000 edge, because a tile edge under
//! its tile's gauge is also +-1000.
//!
//! Features: pair edges by the tile membership of their ends (p00 .. p22);
//! triangles and 4-cycles by their (P, S) edge counts, all and frustrated
//! (an odd number of J > 0 edges). The model reads only 4-cycles holding a
//! pair edge, so only those are counted. The score is
//! `bias + sum w * (x - mean) / sd`; lower is predicted deeper.

use std::sync::{Arc, Mutex, PoisonError};

#[derive(Clone, Copy)]
enum Feature {
    /// Pair category 0..6: p00, p01, p02, p11, p12, p22.
    Pair(usize),
    /// Triangle composition `np * 5 + ns`; `true` = frustrated count.
    Tri(usize, bool),
    /// 4-cycle composition `np * 5 + ns`; `true` = frustrated count.
    Quad(usize, bool),
}

/// (feature, mean, sd, weight), from `screen_weights.json`.
#[rustfmt::skip]
#[allow(clippy::unreadable_literal, clippy::excessive_precision)]
const LEAN: [(Feature, f64, f64, f64); 32] = [
    (Feature::Pair(0), 43.797863973175474, 5.211858692877697, -0.0372787149051312), // p00
    (Feature::Pair(1), 113.76093885830194, 8.12555014847171, -0.08374414029922582), // p01
    (Feature::Pair(2), 28.355280043051703, 4.778828605528716, -0.012400878834888482), // p02
    (Feature::Pair(3), 74.14958397151965, 7.3238160819247256, -0.05780948751432105), // p11
    (Feature::Pair(4), 339.0505443556733, 13.874951267976874, 0.02044223365748958), // p12
    (Feature::Pair(5), 600.885788798278, 11.937954310820095, 0.08994600842515237), // p22
    (Feature::Tri(0, false), 2943.1513019000704, 28.081090178220514, 0.07081783611005421), // tri_W3P0S0_n
    (Feature::Tri(0, true), 1471.5398021277476, 30.42861630921806, 0.03366225981626178), // tri_W3P0S0_f
    (Feature::Tri(1, false), 18945.952125677857, 26.651369368771984, 0.03941624699729029), // tri_W2P0S1_n
    (Feature::Tri(1, true), 9472.863869685805, 70.06326741750118, 0.08129371070048483), // tri_W2P0S1_f
    (Feature::Tri(2, false), 2900.3968414952187, 11.433466024838962, 0.033831427364505874), // tri_W1P0S2_n
    (Feature::Tri(2, true), 1450.1341019166287, 27.490020902075553, 0.0706284215608746), // tri_W1P0S2_f
    (Feature::Tri(5, false), 389.45280870969077, 27.64992395982353, 0.026377844068531477), // tri_W2P1S0_n
    (Feature::Tri(5, true), 194.6796580701246, 16.97943760071955, 0.08270696288893374), // tri_W2P1S0_f
    (Feature::Tri(6, false), 1600.6993418056877, 25.410768139834317, -0.14211656781337068), // tri_W1P1S1_n
    (Feature::Tri(6, true), 800.464420250859, 23.700586293794583, 0.3050897556881421), // tri_W1P1S1_f
    (Feature::Tri(7, false), 118.34758041147494, 10.404898527210808, -0.05228342237948826), // tri_W0P1S2_n
    (Feature::Tri(7, true), 59.13882104565964, 7.542019733748637, 0.17928567585264973), // tri_W0P1S2_f
    (Feature::Quad(5, false), 41348.10502131887, 482.19322675219206, -0.16365656409855345), // quad_W3P1S0_n
    (Feature::Quad(5, true), 20673.445688620275, 261.0040661206277, 0.13848957768453635), // quad_W3P1S0_f
    (Feature::Quad(6, false), 47085.834292337626, 506.3877022347934, -0.22542004669936958), // quad_W2P1S1_n
    (Feature::Quad(6, true), 23542.919692014737, 274.9535223759191, 0.3314956313633421), // quad_W2P1S1_f
    (Feature::Quad(7, false), 13530.441962992094, 202.75547526417418, -0.23137519690574898), // quad_W1P1S2_n
    (Feature::Quad(7, true), 6765.232748271723, 120.25338908801061, 0.3225933713331564), // quad_W1P1S2_f
    (Feature::Quad(8, false), 627.5539802127747, 41.40753863111315, -0.0608082113839482), // quad_W0P1S3_n
    (Feature::Quad(8, true), 313.8963240468601, 30.58367358064216, 0.10760454055174373), // quad_W0P1S3_f
    (Feature::Quad(10, false), 883.4291095748645, 30.06907562089457, -0.05744226551146338), // quad_W2P2S0_n
    (Feature::Quad(10, true), 441.69087635054024, 21.090954877876502, 0.08080570831311594), // quad_W2P2S0_f
    (Feature::Quad(11, false), 644.5222295814877, 24.13288739873911, -0.08045936497116118), // quad_W1P2S1_n
    (Feature::Quad(11, true), 322.26619613362584, 17.49966702710768, 0.13045668886043052), // quad_W1P2S1_f
    (Feature::Quad(12, false), 134.50993500848617, 10.976514508051055, -0.05483913245648599), // quad_W0P2S2_n
    (Feature::Quad(12, true), 67.22784286128244, 7.981212552881767, 0.1094751246081622), // quad_W0P2S2_f
];
#[allow(clippy::unreadable_literal)]
const BIAS: f64 = 0.002898131483262118;

/// Composition index `np * 5 + ns`, summed per edge code: W adds 0, P 5, S 1.
const STEP: [usize; 6] = [0, 0, 5, 5, 1, 1];

const W: u8 = 0;
const P: u8 = 1;
const S: u8 = 2;

/// The fixed structure of a topology: sorted adjacency with edge ids, its
/// triangles and its 4-cycles (agentA's `build_topo_q`), as edge ids.
pub(crate) struct Topo {
    edges: Arc<[(usize, usize)]>,
    n: usize,
    adj_start: Vec<usize>,
    adj: Vec<(u32, u32)>,
    tris: Vec<[u32; 3]>,
    quads: Vec<[u32; 4]>,
    /// For each edge, the 4-cycles through it (CSR): the lean model reads
    /// only 4-cycles that hold a pair edge, so they are found from the pairs.
    quad_start: Vec<u32>,
    quad_list: Vec<u32>,
}

impl Topo {
    #[allow(clippy::cast_possible_truncation)] // Device-sized graphs: ids fit u32.
    pub(crate) fn new(n: usize, edges: Arc<[(usize, usize)]>) -> Self {
        let mut deg = vec![0usize; n];
        for &(u, v) in edges.iter() {
            deg[u] += 1;
            deg[v] += 1;
        }
        let mut adj_start = vec![0usize; n + 1];
        for i in 0..n {
            adj_start[i + 1] = adj_start[i] + deg[i];
        }
        let mut fill = adj_start.clone();
        let mut adj = vec![(0u32, 0u32); adj_start[n]];
        for (e, &(u, v)) in edges.iter().enumerate() {
            adj[fill[u]] = (v as u32, e as u32);
            fill[u] += 1;
            adj[fill[v]] = (u as u32, e as u32);
            fill[v] += 1;
        }
        for i in 0..n {
            adj[adj_start[i]..adj_start[i + 1]].sort_unstable();
        }
        let mut topo = Self {
            edges,
            n,
            adj_start,
            adj,
            tris: Vec::new(),
            quads: Vec::new(),
            quad_start: Vec::new(),
            quad_list: Vec::new(),
        };
        let mut tris = Vec::new();
        for (e, &(u, v)) in topo.edges.iter().enumerate() {
            for &(w, euw) in topo.neighbours(u) {
                if w as usize <= v {
                    continue;
                }
                if let Some(evw) = topo.edge_id(v, w as usize) {
                    tris.push([e as u32, euw, evw]);
                }
            }
        }
        // 4-cycles a-b-c-d-a with a the smallest node and b < d.
        let mut quads = Vec::new();
        for a in 0..n {
            for &(b, eab) in topo.neighbours(a) {
                if b as usize <= a {
                    continue;
                }
                for &(c, ebc) in topo.neighbours(b as usize) {
                    if c as usize <= a || c == b {
                        continue;
                    }
                    for &(d, ead) in topo.neighbours(a) {
                        if d <= b || d == c {
                            continue;
                        }
                        if let Some(ecd) = topo.edge_id(c as usize, d as usize) {
                            quads.push([eab, ebc, ecd, ead]);
                        }
                    }
                }
            }
        }
        let mut quad_start = vec![0u32; topo.edges.len() + 1];
        for q in &quads {
            for &e in q {
                quad_start[e as usize + 1] += 1;
            }
        }
        for e in 0..topo.edges.len() {
            quad_start[e + 1] += quad_start[e];
        }
        let mut fill = quad_start.clone();
        let mut quad_list = vec![0u32; quad_start[topo.edges.len()] as usize];
        for (k, q) in quads.iter().enumerate() {
            for &e in q {
                quad_list[fill[e as usize] as usize] = k as u32;
                fill[e as usize] += 1;
            }
        }
        topo.tris = tris;
        topo.quads = quads;
        topo.quad_start = quad_start;
        topo.quad_list = quad_list;
        topo
    }

    fn neighbours(&self, a: usize) -> &[(u32, u32)] {
        &self.adj[self.adj_start[a]..self.adj_start[a + 1]]
    }

    #[allow(clippy::cast_possible_truncation)]
    fn edge_id(&self, a: usize, b: usize) -> Option<u32> {
        let row = self.neighbours(a);
        row.binary_search_by_key(&(b as u32), |x| x.0)
            .ok()
            .map(|i| row[i].1)
    }

    /// The structure for `edges`, built once and shared while the edges stay
    /// the same.
    pub(crate) fn shared(n: usize, edges: &Arc<[(usize, usize)]>) -> Arc<Self> {
        static CACHE: Mutex<Option<Arc<Topo>>> = Mutex::new(None);
        let mut cached = CACHE.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(topo) = cached.as_ref() {
            if topo.n == n && (Arc::ptr_eq(&topo.edges, edges) || *topo.edges == **edges) {
                return Arc::clone(topo);
            }
        }
        let topo = Arc::new(Self::new(n, Arc::clone(edges)));
        *cached = Some(Arc::clone(&topo));
        topo
    }
}

/// The lean score of one instance. `j` is aligned with the topology's edges,
/// `tiles` holds the four node positions of each selected tile. Lower is
/// predicted deeper. `None` when a tile names an edge the topology lacks.
#[allow(clippy::many_single_char_names)] // Edge codes a..d, as in agentA.
pub(crate) fn lean_score(topo: &Topo, j: &[i32], tiles: &[[u32; 4]]) -> Option<f32> {
    if j.len() != topo.edges.len() {
        return None;
    }
    let mut membership = vec![0u8; topo.n];
    let mut ty = vec![W; j.len()];
    for &[a, b, c, d] in tiles {
        for p in [a, b, c, d] {
            *membership.get_mut(p as usize)? += 1;
        }
        for (x, y) in [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)] {
            ty[topo.edge_id(x as usize, y as usize)? as usize] = S;
        }
    }
    let mut pairs = [0u32; 6];
    for (k, &jj) in j.iter().enumerate() {
        if ty[k] != S && jj == -1000 {
            ty[k] = P;
            let (u, v) = topo.edges[k];
            let (mu, mv) = (membership[u], membership[v]);
            let category = match (mu.min(mv), mu.max(mv)) {
                (0, 0) => 0,
                (0, 1) => 1,
                (0, 2) => 2,
                (1, 1) => 3,
                (1, 2) => 4,
                _ => 5,
            };
            pairs[category] += 1;
        }
    }
    // Per edge: type in bits 1..2, J > 0 in bit 0.
    let code: Vec<u8> = ty
        .iter()
        .zip(j)
        .map(|(&t, &jj)| (t << 1) | u8::from(jj > 0))
        .collect();
    let mut tri = [[0u32; 2]; 25];
    for t in &topo.tris {
        let (a, b, c) = (
            code[t[0] as usize],
            code[t[1] as usize],
            code[t[2] as usize],
        );
        let at = STEP[a as usize] + STEP[b as usize] + STEP[c as usize];
        tri[at][usize::from((a ^ b ^ c) & 1)] += 1;
    }
    // 4-cycles with at least one pair edge, each counted once: from its
    // lowest-numbered pair edge.
    let mut quad = [[0u32; 2]; 25];
    for (e, &c) in code.iter().enumerate() {
        if c >> 1 != P {
            continue;
        }
        let through = &topo.quad_list[topo.quad_start[e] as usize..topo.quad_start[e + 1] as usize];
        for &k in through {
            let q = topo.quads[k as usize];
            if q.iter()
                .any(|&o| (o as usize) < e && code[o as usize] >> 1 == P)
            {
                continue;
            }
            let (a, b, c, d) = (
                code[q[0] as usize],
                code[q[1] as usize],
                code[q[2] as usize],
                code[q[3] as usize],
            );
            let at = STEP[a as usize] + STEP[b as usize] + STEP[c as usize] + STEP[d as usize];
            quad[at][usize::from((a ^ b ^ c ^ d) & 1)] += 1;
        }
    }
    let mut score = BIAS;
    for (feature, mean, sd, weight) in LEAN {
        let x = match feature {
            Feature::Pair(k) => pairs[k],
            // [0] unfrustrated, [1] frustrated; "_n" features are the total.
            Feature::Tri(k, true) => tri[k][1],
            Feature::Tri(k, false) => tri[k][0] + tri[k][1],
            Feature::Quad(k, true) => quad[k][1],
            Feature::Quad(k, false) => quad[k][0] + quad[k][1],
        };
        score += weight * (f64::from(x) - mean) / sd;
    }
    #[allow(clippy::cast_possible_truncation)]
    Some(score as f32)
}
