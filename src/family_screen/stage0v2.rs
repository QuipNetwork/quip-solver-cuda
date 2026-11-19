//! Stage 0, model 2, for `quip/twisted-ice-dimer@1` (agent H, Otsuka labels).
//!
//! Edges are typed from the instance's own tile list, as in model 1:
//! S = one of the six edges of a selected tile, P = any other edge with
//! J = -1000 (a dimer pair), W = everything else (the weak +-62 layer).
//!
//! Every pair is contracted into one super-node. The features are the short
//! cycles of that contracted multigraph, by composition and frustration:
//! - 2-cycles: two parallel edges between the same two super-nodes;
//! - triangles and 4-cycles.
//!
//! A cycle that enters a pair at one end and leaves at the other runs
//! through its pair edge; `np` counts those traversals, `nw` the weak edges
//! of the contracted cycle (the rest are S). A cycle is frustrated when an
//! odd number of its edges has J > 0 (pair edges are -1000, never positive).
//! In original-graph terms the classes are every triangle and 4-cycle plus
//! the 5- to 8-cycles that close through pairs. Pair categories by tile
//! membership of the two ends (p00 .. p22) complete the vector.
//!
//! The score is `BIAS + sum WEIGHT[i] * x[i]`, a ridge fit of the GPU Otsuka
//! (x512) best energy in units of its standard deviation; lower is predicted
//! deeper. Self-contained: no crate imports beyond `std`.

use std::sync::{Arc, Mutex, PoisonError};

/// Pair categories.
pub const PAIRS: usize = 6;
/// 2-cycles: np in 1..=2, nw in 0..=2, (all, frustrated).
pub const C2: usize = 2 * 3 * 2;
/// Triangles: np in 0..=3, nw in 0..=3.
pub const CT: usize = 4 * 4 * 2;
/// 4-cycles: np in 0..=4, nw in 0..=4.
pub const CQ: usize = 5 * 5 * 2;
/// Feature count.
pub const NF: usize = PAIRS + C2 + CT + CQ;

const OFF_C2: usize = PAIRS;
const OFF_CT: usize = OFF_C2 + C2;
const OFF_CQ: usize = OFF_CT + CT;

/// Feature names, in vector order (the offline tooling's column names).
#[allow(dead_code)]
#[must_use]
pub fn names() -> Vec<String> {
    let mut h: Vec<String> = ["p00", "p01", "p02", "p11", "p12", "p22"]
        .iter()
        .map(|s| (*s).to_string())
        .collect();
    for np in 1..=2 {
        for nw in 0..=2 {
            for k in ["n", "f"] {
                h.push(format!("c2_P{np}W{nw}S{}_{k}", 2 - nw));
            }
        }
    }
    for np in 0..=3 {
        for nw in 0..=3 {
            for k in ["n", "f"] {
                h.push(format!("ct_P{np}W{nw}S{}_{k}", 3 - nw));
            }
        }
    }
    for np in 0..=4 {
        for nw in 0..=4 {
            for k in ["n", "f"] {
                h.push(format!("cq_P{np}W{nw}S{}_{k}", 4 - nw));
            }
        }
    }
    h
}

/// The fixed topology: sorted CSR adjacency with edge ids.
pub struct Topo2 {
    edges: Arc<[(usize, usize)]>,
    n: usize,
    adj_start: Vec<usize>,
    adj: Vec<(u32, u32)>,
}

impl Topo2 {
    /// Build from an edge list over `n` nodes.
    #[allow(clippy::cast_possible_truncation)] // Device-sized graphs: ids fit u32.
    #[must_use]
    pub fn new(n: usize, edges: Arc<[(usize, usize)]>) -> Self {
        let mut deg = vec![0usize; n + 1];
        for &(u, v) in edges.iter() {
            deg[u + 1] += 1;
            deg[v + 1] += 1;
        }
        for i in 0..n {
            deg[i + 1] += deg[i];
        }
        let adj_start = deg.clone();
        let mut fill = deg;
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
        Self {
            edges,
            n,
            adj_start,
            adj,
        }
    }

    fn edge_id(&self, a: usize, b: usize) -> Option<usize> {
        let row = &self.adj[*self.adj_start.get(a)?..*self.adj_start.get(a + 1)?];
        u32::try_from(b)
            .ok()
            .and_then(|b| row.binary_search_by_key(&b, |x| x.0).ok())
            .map(|i| row[i].1 as usize)
    }

    /// The structure for `edges`, built once and shared while the edges stay
    /// the same.
    pub fn shared(n: usize, edges: &Arc<[(usize, usize)]>) -> Arc<Self> {
        static CACHE: Mutex<Option<Arc<Topo2>>> = Mutex::new(None);
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

/// One contracted half-edge: the far super-node, the near and far original
/// nodes, and `bits` = weak (bit 0) | J > 0 (bit 1).
#[derive(Clone, Copy)]
struct Half {
    y: u32,
    me: u32,
    ye: u32,
    bits: u8,
}

impl Half {
    fn weak(self) -> usize {
        usize::from(self.bits & 1)
    }
    fn pos(self) -> u8 {
        (self.bits >> 1) & 1
    }
}

/// The feature vector of one instance. `j` is aligned with the topology's
/// edges, `tiles` holds the four node positions of each selected tile.
/// `None` when a tile names a node or an edge the topology lacks.
#[allow(
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    clippy::many_single_char_names
)]
#[must_use]
pub fn features(topo: &Topo2, j: &[i32], tiles: &[[u32; 4]]) -> Option<[u32; NF]> {
    if j.len() != topo.edges.len() {
        return None;
    }
    let n = topo.n;
    let mut x = [0u32; NF];
    // 0 = W, 1 = P, 2 = S
    let mut ty = vec![0u8; j.len()];
    let mut membership = vec![0u8; n];
    for &[a, b, c, d] in tiles {
        for p in [a, b, c, d] {
            *membership.get_mut(p as usize)? += 1;
        }
        for (u, v) in [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)] {
            ty[topo.edge_id(u as usize, v as usize)?] = 2;
        }
    }
    let mut partner = vec![u32::MAX; n];
    for (k, &jj) in j.iter().enumerate() {
        if ty[k] != 2 && jj == -1000 {
            ty[k] = 1;
            let (u, v) = topo.edges[k];
            partner[u] = v as u32;
            partner[v] = u as u32;
            let (mu, mv) = (membership[u], membership[v]);
            let category = match (mu.min(mv), mu.max(mv)) {
                (0, 0) => 0,
                (0, 1) => 1,
                (0, 2) => 2,
                (1, 1) => 3,
                (1, 2) => 4,
                _ => 5,
            };
            x[category] += 1;
        }
    }
    // Super-node ids: a pair's two ends share one.
    let mut sid = vec![u32::MAX; n];
    let mut ns = 0usize;
    for v in 0..n {
        if sid[v] != u32::MAX {
            continue;
        }
        sid[v] = ns as u32;
        if partner[v] != u32::MAX {
            sid[partner[v] as usize] = ns as u32;
        }
        ns += 1;
    }
    let mut start = vec![0usize; ns + 1];
    for (k, &(u, v)) in topo.edges.iter().enumerate() {
        if ty[k] != 1 {
            start[sid[u] as usize + 1] += 1;
            start[sid[v] as usize + 1] += 1;
        }
    }
    for i in 0..ns {
        start[i + 1] += start[i];
    }
    let mut fill = start.clone();
    let mut half = vec![
        Half {
            y: 0,
            me: 0,
            ye: 0,
            bits: 0
        };
        start[ns]
    ];
    for (k, &(u, v)) in topo.edges.iter().enumerate() {
        if ty[k] == 1 {
            continue;
        }
        let bits = u8::from(ty[k] == 0) | (u8::from(j[k] > 0) << 1);
        let (su, sv) = (sid[u] as usize, sid[v] as usize);
        half[fill[su]] = Half {
            y: sv as u32,
            me: u as u32,
            ye: v as u32,
            bits,
        };
        fill[su] += 1;
        half[fill[sv]] = Half {
            y: su as u32,
            me: v as u32,
            ye: u as u32,
            bits,
        };
        fill[sv] += 1;
    }
    for s in 0..ns {
        half[start[s]..start[s + 1]].sort_unstable_by_key(|h| (h.y, h.me, h.ye));
    }
    let row = |s: usize| &half[start[s]..start[s + 1]];
    let mut add = |at: usize, frustrated: u8| {
        x[at] += 1;
        x[at + 1] += u32::from(frustrated);
    };
    // 2-cycles and triangles, X < Y (< Z).
    for s in 0..ns {
        let nx = row(s);
        let mut i = 0;
        while i < nx.len() {
            let y = nx[i].y as usize;
            let mut k = i;
            while k < nx.len() && nx[k].y as usize == y {
                k += 1;
            }
            if y > s {
                for a in i..k {
                    for b in a + 1..k {
                        let (h1, h2) = (nx[a], nx[b]);
                        let np = usize::from(h1.me != h2.me) + usize::from(h1.ye != h2.ye);
                        let nw = h1.weak() + h2.weak();
                        add(OFF_C2 + ((np - 1) * 3 + nw) * 2, h1.pos() ^ h2.pos());
                    }
                }
                for &h1 in &nx[i..k] {
                    for &h2 in row(y) {
                        let z = h2.y as usize;
                        if z <= y {
                            continue;
                        }
                        let lo = nx.partition_point(|h| (h.y as usize) < z);
                        for &h3 in nx[lo..].iter().take_while(|h| h.y as usize == z) {
                            // h3 runs s -> z: me on s, ye on z.
                            let np = usize::from(h1.me != h3.me)
                                + usize::from(h1.ye != h2.me)
                                + usize::from(h2.ye != h3.ye);
                            let nw = h1.weak() + h2.weak() + h3.weak();
                            add(OFF_CT + (np * 4 + nw) * 2, h1.pos() ^ h2.pos() ^ h3.pos());
                        }
                    }
                }
            }
            i = k;
        }
    }
    // 4-cycles X-Y-Z-W with X the smallest: pairs of wedges X-Y-Z, X-W-Z
    // with Y != W, each unordered pair once.
    // Wedges grouped by their far end Z with a counting pass (no sort).
    let mut wedges: Vec<(u32, Half, Half)> = Vec::new();
    let mut grouped: Vec<(u32, Half, Half)> = Vec::new();
    let mut slot = vec![0u32; ns];
    let mut zs: Vec<u32> = Vec::new();
    for s in 0..ns {
        wedges.clear();
        zs.clear();
        for &h1 in row(s) {
            let y = h1.y as usize;
            if y <= s {
                continue;
            }
            for &h2 in row(y) {
                let z = h2.y;
                if (z as usize) > s {
                    if slot[z as usize] == 0 {
                        zs.push(z);
                    }
                    slot[z as usize] += 1;
                    wedges.push((z, h1, h2));
                }
            }
        }
        // prefix offsets over the touched far ends
        let mut at = 0u32;
        for &z in &zs {
            let c = slot[z as usize];
            slot[z as usize] = at;
            at += c;
        }
        grouped.clear();
        grouped.resize(
            wedges.len(),
            wedges.first().copied().unwrap_or((
                0,
                Half {
                    y: 0,
                    me: 0,
                    ye: 0,
                    bits: 0,
                },
                Half {
                    y: 0,
                    me: 0,
                    ye: 0,
                    bits: 0,
                },
            )),
        );
        for &w in &wedges {
            let p = &mut slot[w.0 as usize];
            grouped[*p as usize] = w;
            *p += 1;
        }
        let mut lo = 0usize;
        for &z in &zs {
            let hi = slot[z as usize] as usize;
            slot[z as usize] = 0;
            let group = &grouped[lo..hi];
            for (a, &(_, h1, h2)) in group.iter().enumerate() {
                for &(_, h4, h3) in &group[a + 1..] {
                    if h4.y == h1.y {
                        continue;
                    }
                    // h1: s->y, h2: y->z, h4: s->w, h3: w->z
                    let np = usize::from(h1.me != h4.me)
                        + usize::from(h1.ye != h2.me)
                        + usize::from(h2.ye != h3.ye)
                        + usize::from(h4.ye != h3.me);
                    let nw = h1.weak() + h2.weak() + h3.weak() + h4.weak();
                    add(
                        OFF_CQ + (np * 5 + nw) * 2,
                        h1.pos() ^ h2.pos() ^ h3.pos() ^ h4.pos(),
                    );
                }
            }
            lo = hi;
        }
    }
    Some(x)
}

/// The model-2 score of one instance: `BIAS + sum WEIGHT[i] * x[i]`.
/// `None` when a tile names a node or an edge the topology lacks.
#[must_use]
pub fn score(topo: &Topo2, j: &[i32], tiles: &[[u32; 4]]) -> Option<f32> {
    let x = features(topo, j, tiles)?;
    let s = x
        .iter()
        .zip(WEIGHT.iter())
        .fold(BIAS, |acc, (&xi, &wi)| acc + wi * f64::from(xi));
    #[allow(clippy::cast_possible_truncation)]
    Some(s as f32)
}

// Ridge fit (lambda 3 on standardized features, folded back to raw counts)
// of z = (E - mean) / sd of the GPU Otsuka x512 best energy, on 12628
// labelled salts of 2026-11-13: 2533 from unscreened windows and 10095 from
// keep-0.1 windows. quip-build/agentH: export2.py, screen2_weights.json.
/// Raw weights per feature (ridge on standardized features, folded back).
#[rustfmt::skip]
#[allow(clippy::unreadable_literal, clippy::excessive_precision)]
const WEIGHT: [f64; NF] = [
    -0.01964289858533684, // p00
    -0.011297011632076756, // p01
    -0.006948688364839518, // p02
    0.0016463586263446666, // p11
    0.00352852142439075, // p12
    0.004837335188746461, // p22
    -0.014309567798431247, // c2_P1W0S2_n
    0.04649421765882299, // c2_P1W0S2_f
    -0.00545656079119212, // c2_P1W1S1_n
    0.01917944233944605, // c2_P1W1S1_f
    0.0027400043110819648, // c2_P1W2S0_n
    0.006981072939907464, // c2_P1W2S0_f
    -0.03508685888833003, // c2_P2W0S2_n
    0.06455459320124442, // c2_P2W0S2_f
    -0.010485513683504278, // c2_P2W1S1_n
    0.01868918100806894, // c2_P2W1S1_f
    -0.0022219373182698326, // c2_P2W2S0_n
    0.003773949984289615, // c2_P2W2S0_f
    0.0, // ct_P0W0S3_n
    0.0, // ct_P0W0S3_f
    -0.005284088828208799, // ct_P0W1S2_n
    0.006188593096191166, // ct_P0W1S2_f
    -0.0013195707306829889, // ct_P0W2S1_n
    0.002567834279773731, // ct_P0W2S1_f
    0.005776180285116035, // ct_P0W3S0_n
    0.0015806948901743868, // ct_P0W3S0_f
    -0.006515161831632682, // ct_P1W0S3_n
    0.009235178093763511, // ct_P1W0S3_f
    -0.004065193397682823, // ct_P1W1S2_n
    0.005911881808965104, // ct_P1W1S2_f
    -0.0008676218250281045, // ct_P1W2S1_n
    0.002213402329335339, // ct_P1W2S1_f
    -9.954507077482366e-05, // ct_P1W3S0_n
    0.000684650690801034, // ct_P1W3S0_f
    -0.006368027067479371, // ct_P2W0S3_n
    0.014785750315487688, // ct_P2W0S3_f
    -0.0031776351422874854, // ct_P2W1S2_n
    0.006683084000859165, // ct_P2W1S2_f
    -0.0012021534051971487, // ct_P2W2S1_n
    0.002187271358035017, // ct_P2W2S1_f
    -0.0002996145567922935, // ct_P2W3S0_n
    0.0008245034431115637, // ct_P2W3S0_f
    -0.012868214366300472, // ct_P3W0S3_n
    0.02195047827779446, // ct_P3W0S3_f
    -0.003302172807679744, // ct_P3W1S2_n
    0.006015664047409203, // ct_P3W1S2_f
    -0.0005903527976836876, // ct_P3W2S1_n
    0.0021167271841864136, // ct_P3W2S1_f
    -0.0002758446878909211, // ct_P3W3S0_n
    0.00045524529063740207, // ct_P3W3S0_f
    0.0038581526864954075, // cq_P0W0S4_n
    0.0026262981258407926, // cq_P0W0S4_f
    0.0013887518039659933, // cq_P0W1S3_n
    0.0007612575598146457, // cq_P0W1S3_f
    0.00045197735450008125, // cq_P0W2S2_n
    0.00046886388320287203, // cq_P0W2S2_f
    0.00043620407482827675, // cq_P0W3S1_n
    0.0002535839389026071, // cq_P0W3S1_f
    8.977444146498864e-05, // cq_P0W4S0_n
    0.00011801989128739058, // cq_P0W4S0_f
    0.00018435568676522133, // cq_P1W0S4_n
    0.0010736249216673492, // cq_P1W0S4_f
    -9.717863285331436e-05, // cq_P1W1S3_n
    0.0008143286179163062, // cq_P1W1S3_f
    -0.00015155996937645772, // cq_P1W2S2_n
    0.00042030219988739895, // cq_P1W2S2_f
    -6.508470126086385e-05, // cq_P1W3S1_n
    0.00017531463038117479, // cq_P1W3S1_f
    -4.288778040902094e-05, // cq_P1W4S0_n
    0.00011521225044431186, // cq_P1W4S0_f
    -0.0008064455273409343, // cq_P2W0S4_n
    0.001554340449698325, // cq_P2W0S4_f
    -0.0004889371322553636, // cq_P2W1S3_n
    0.0010230493480328491, // cq_P2W1S3_f
    -0.00023056383729857984, // cq_P2W2S2_n
    0.0004358469118477983, // cq_P2W2S2_f
    -6.076367220019183e-05, // cq_P2W3S1_n
    0.0001533994604370181, // cq_P2W3S1_f
    -3.9225649459704316e-05, // cq_P2W4S0_n
    4.983133489224475e-05, // cq_P2W4S0_f
    -0.000917622408787498, // cq_P3W0S4_n
    0.002761440302583575, // cq_P3W0S4_f
    -0.0003765426973688811, // cq_P3W1S3_n
    0.001003688183980515, // cq_P3W1S3_f
    -0.00021295691468679363, // cq_P3W2S2_n
    0.0004942165912340096, // cq_P3W2S2_f
    -0.00011301278300304656, // cq_P3W3S1_n
    0.00022954727138813636, // cq_P3W3S1_f
    -7.311227983120351e-05, // cq_P3W4S0_n
    0.00012666296410288372, // cq_P3W4S0_f
    -0.0020424928666418784, // cq_P4W0S4_n
    0.0035472692054541594, // cq_P4W0S4_f
    -0.0002522516763122665, // cq_P4W1S3_n
    0.00042744676815978887, // cq_P4W1S3_f
    -0.0002952894571520765, // cq_P4W2S2_n
    0.0004108925872801917, // cq_P4W2S2_f
    -0.0002928357965532823, // cq_P4W3S1_n
    0.0002685136714981431, // cq_P4W3S1_f
    4.3293937835935494e-06, // cq_P4W4S0_n
    0.00030491511233729827, // cq_P4W4S0_f
];
#[allow(clippy::unreadable_literal, clippy::excessive_precision)]
const BIAS: f64 = -475.96409412798357;
