//! Local build: `quip/twisted-ice-dimer@1` couplings with the nonce-independent
//! work cached.
//!
//! The canonical `FamilyLease::generate` recomputes the tile candidates and
//! their sides for every salt: about 9.5 of 13.5 ms per z(12,4) instance on
//! an i7-14700KF. This replays `ising_generators::ice::twisted_dimer::generate`
//! with those cached, drawing from the same named streams in the same order,
//! and returns only the couplings. Any failure returns `None`, and the caller
//! falls back to the canonical generator, which also classifies skips.
//! `tests::fast_dimer_matches_canonical` checks the two bit for bit.

use core::num::NonZeroU32;
use quip_solver_core::ising_generators::descriptor::{admit, params_digest, Value};
use quip_solver_core::ising_generators::family::{BaseTopology, Streams};
use quip_solver_core::ising_generators::ice::{
    candidates, select, sides, Candidates, Lattice, Sides, PAIR_MILLI, STRONG_MILLI, TWISTED_DIMER,
    TWISTED_DIMER_V1,
};
use quip_solver_core::ising_generators::rng::{shuffle, uniform};

pub(crate) struct FastDimer {
    descriptor_digest: [u8; 32],
    order_digest: [u8; 32],
    found: Candidates,
    split: Sides,
    /// The topology edges as `(min, max)`, ascending: the instance edge order.
    edges: Vec<(u32, u32)>,
    weak: i32,
    pairs: usize,
    nodes: usize,
}

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

impl FastDimer {
    /// `None` for any other family, or when the cached parts cannot be built.
    pub(crate) fn new(
        family_id: &str,
        revision: u32,
        params: &[u8],
        topology: &BaseTopology,
    ) -> Option<Self> {
        if family_id != TWISTED_DIMER_V1.family_id || revision != TWISTED_DIMER_V1.revision {
            return None;
        }
        let digest = params_digest(params);
        let admitted = admit(
            core::slice::from_ref(&TWISTED_DIMER_V1),
            family_id.as_bytes(),
            revision,
            params,
            &digest,
        )
        .ok()?;
        let p = admitted.params();
        let m = u16::try_from(unsigned(p.get(TAG_M))?).ok()?;
        let t = u8::try_from(unsigned(p.get(TAG_T))?).ok()?;
        let weak = i32::try_from(unsigned(p.get(TAG_WEAK))?).ok()?;
        let pairs = usize::try_from(unsigned(p.get(TAG_PAIRS))?).ok()?;
        let lattice = Lattice::from_shape(m, t).ok()?;
        let found = candidates(lattice, topology).ok()?;
        let split = sides(lattice, topology, &found).ok()?;
        let mut edges: Vec<(u32, u32)> = topology
            .edges()
            .iter()
            .map(|&(u, v)| (u.min(v), u.max(v)))
            .collect();
        edges.sort_unstable();
        Some(Self {
            descriptor_digest: admitted.descriptor().digest(),
            order_digest: *topology.order_digest(),
            found,
            split,
            edges,
            weak,
            pairs,
            nodes: topology.node_ids().len(),
        })
    }

    pub(crate) fn edges(&self) -> &[(u32, u32)] {
        &self.edges
    }

    pub(crate) fn nodes(&self) -> usize {
        self.nodes
    }

    fn shuffled(streams: &mut Streams<'_>, label: &str, len: usize) -> Option<Vec<u32>> {
        let mut order: Vec<u32> = (0..u32::try_from(len).ok()?).collect();
        let mut rng = streams.open(label)?;
        shuffle(&mut rng, &mut order).ok()?;
        Some(order)
    }

    /// The couplings of `nonce`, aligned with [`Self::edges`], and the node
    /// positions of each selected tile. Fields are zero and the offset is zero
    /// for this family.
    pub(crate) fn couplings(&self, nonce: [u8; 32]) -> Option<(Vec<i32>, Vec<[u32; 4]>)> {
        let labels = TWISTED_DIMER.labels;
        let mut streams = Streams::new(nonce, self.descriptor_digest, self.order_digest, labels);
        let left = Self::shuffled(&mut streams, labels[0], self.split.left.len())?;
        let right = Self::shuffled(&mut streams, labels[1], self.split.right.len())?;
        let tiles = select(&self.found, &self.split, &left, &right).ok()?;
        let two = NonZeroU32::MIN.saturating_add(1);
        let mut gauge_stream = streams.open(labels[2])?;
        let gauges: Vec<[i32; 4]> = tiles
            .iter()
            .map(|_| {
                [(); 4].map(|()| {
                    if uniform(&mut gauge_stream, two) == 0 {
                        1
                    } else {
                        -1
                    }
                })
            })
            .collect();
        let mut strong: Vec<Option<i32>> = vec![None; self.edges.len()];
        for (tile, g) in tiles.iter().zip(&gauges) {
            let [a, b, c, d] = tile.nodes;
            let [ga, gb, gc, gd] = *g;
            for ((u, v), coupling) in [
                ((a, b), ga * gb),
                ((a, c), ga * gc),
                ((a, d), ga * gd),
                ((b, c), gb * gc),
                ((b, d), gb * gd),
                ((c, d), gc * gd),
            ] {
                let at = self.edges.binary_search(&(u, v)).ok()?;
                *strong.get_mut(at)? = Some(STRONG_MILLI.checked_mul(coupling)?);
            }
        }
        let mut weak_stream = streams.open(labels[3])?;
        let mut j: Vec<i32> = strong
            .iter()
            .map(|fixed| match fixed {
                Some(coupling) => *coupling,
                None if uniform(&mut weak_stream, two) == 0 => self.weak,
                None => -self.weak,
            })
            .collect();
        let mut order: Vec<u32> = Vec::new();
        for (at, fixed) in strong.iter().enumerate() {
            if fixed.is_none() {
                order.push(u32::try_from(at).ok()?);
            }
        }
        let mut rng = streams.open(labels[4])?;
        shuffle(&mut rng, &mut order).ok()?;
        let mut used = vec![false; self.nodes];
        let mut taken = 0;
        for at in order {
            if taken >= self.pairs {
                break;
            }
            let at = at as usize;
            let (u, v) = *self.edges.get(at)?;
            let (u, v) = (u as usize, v as usize);
            if *used.get(u)? || *used.get(v)? {
                continue;
            }
            used[u] = true;
            used[v] = true;
            j[at] = PAIR_MILLI;
            taken += 1;
        }
        if taken < self.pairs {
            return None;
        }
        Some((j, tiles.iter().map(|tile| tile.nodes).collect()))
    }
}
