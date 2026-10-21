//! The station record: its sample grid, where a contribution lands on it, and the reports on
//! what did not fit.

use ndarray::{Array2, ArrayView2, s};

use crate::source::MomentWeight;

/// The record's time axis: `ndata` samples, `dt` apart, sample 1 at the origin time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SampleGrid {
    pub dt: f32,
    pub ndata: usize,
}

impl SampleGrid {
    /// The 1-based sample a contribution's first sample lands on.
    ///
    /// Can be negative: both terms truncate toward zero, so a window starting before the
    /// origin time gives a negative start. [`SampleGrid::place`] is what handles that, and
    /// relies on it.
    ///
    /// The one place this is formed. The pruning bound in `sim` and the real arrival must
    /// agree on it exactly for the bound to be exact.
    #[inline]
    pub fn start_sample(&self, rupture_time_s: f32, window_start_s: f32) -> i32 {
        (rupture_time_s / self.dt).trunc() as i32 + (window_start_s / self.dt).trunc() as i32
    }

    /// Whether a contribution starting at `start_sample` begins after the record ends.
    #[inline]
    pub fn starts_after_end(&self, start_sample: i32) -> bool {
        start_sample > self.ndata as i32
    }

    /// Where an `np2`-sample window starting at `start_sample` lands, after clipping to the
    /// record, or `None` if none of it lands.
    ///
    /// Sample 1 of the window lands on `start_sample`. Production placed every contribution
    /// one sample later (`start_sample + 1`); this does not reproduce that.
    ///
    /// # Both ends can miss, and they are different cases
    ///
    /// One contribution ends before the record begins (a large negative start); another begins
    /// after it ends (a long-path ray at a far station). The second case computes a negative
    /// length, which must be rejected before it is cast to `usize` or it wraps.
    pub fn place(&self, start_sample: i32, np2: usize) -> Option<Placement> {
        let last_sample = (start_sample + np2 as i32 - 1).min(self.ndata as i32);
        let first_sample = start_sample.max(1);
        if last_sample < first_sample {
            return None;
        }
        Some(Placement {
            skip: (first_sample - start_sample) as usize,
            offset: first_sample as usize - 1,
            count: (last_sample - first_sample + 1) as usize,
        })
    }
}

/// Where a contribution's window lands in the record, after clipping to it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Placement {
    /// How many of the contribution's own samples fall before the record starts.
    pub skip: usize,
    /// 0-based index in the record where the first surviving sample lands.
    pub offset: usize,
    /// How many samples land.
    pub count: usize,
}

/// One station's synthetic record.
pub struct Simulation {
    /// Samples per component.
    pub ndata: usize,
    pub dt: f32,
    /// Ground motion, shaped `(n_components, ndata)`, rows ordered 090, 000, vertical.
    pub acc: Array2<f32>,
    /// What did not fit in the record. See [`Clipping`].
    pub clipping: Clipping,
    /// How much of the work reached the record. See [`Census`].
    pub census: Census,
}

impl Simulation {
    /// A silent record of `components` rows on `grid`, with nothing reported yet.
    pub fn silent(grid: SampleGrid, components: usize) -> Self {
        Self {
            ndata: grid.ndata,
            dt: grid.dt,
            acc: Array2::zeros((components, grid.ndata)),
            clipping: Clipping::default(),
            census: Census::default(),
        }
    }

    /// Add one weighted contribution, all components at once.
    ///
    /// Every index here is already known to be inside both buffers: [`SampleGrid::place`] did
    /// that. What is left is the axpy, `y += alpha * x` over the clipped window.
    pub fn accumulate(
        &mut self,
        contribution: ArrayView2<'_, f32>,
        weight: MomentWeight,
        at: &Placement,
    ) {
        let &Placement {
            skip,
            offset,
            count,
        } = at;
        let mut destination = self.acc.slice_mut(s![.., offset..offset + count]);
        destination.scaled_add(weight.0, &contribution.slice(s![.., skip..skip + count]));
    }
}

/// How much of what a run computed actually reached the record.
///
/// [`Clipping`] answers "was the record long enough"; this answers the neighbouring question
/// "how much did that cost". The two are separate because a run can be entirely complete by
/// `Clipping`'s standard and still spend most of its time on samples that are discarded: the
/// shaping window at long path distance is set by the path-duration model and has no upper
/// cap, so it routinely runs longer than the record it is being placed into.
///
/// `samples_computed` and `samples_accumulated` are per component — all three share one
/// transform length and one placement, so the ratio between them is the same whether you count
/// one component or all three.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Census {
    /// `(subfault, ray)` pairs that passed the moment-weight threshold.
    pub pairs_attempted: usize,
    /// Of those, the ones whose window landed entirely outside the record — every sample
    /// computed for them was discarded.
    pub pairs_outside_record: usize,
    /// Transform samples computed, summed over pairs.
    pub samples_computed: usize,
    /// Transform samples that landed in the record.
    pub samples_accumulated: usize,
    /// Distinct transform lengths the station's [`crate::spectrum::PlanCache`] built.
    pub transform_lengths: usize,
}

impl Census {
    /// Fraction of computed samples that reached the record, in `0..=1`.
    ///
    /// Zero for a station that computed nothing, rather than a division by zero.
    pub fn useful_fraction(&self) -> f64 {
        if self.samples_computed == 0 {
            return 0.0;
        }
        self.samples_accumulated as f64 / self.samples_computed as f64
    }
}

/// Arrivals the record was too short to hold.
///
/// Some clipping is normal and this does not report it. Every subfault's envelope decays
/// to a fraction of a percent of its peak well before the end of its own buffer, and that tail
/// routinely falls past the end of the record; discarding it costs nothing.
///
/// What this counts is the case that is not normal: a subfault whose envelope peak lands
/// beyond the record, meaning the arrival itself was cut rather than its tail. The record
/// then understates the shaking, and looks like a station that stopped shaking early rather
/// than one whose record ran out.
///
/// It is reported rather than refused because the right length is the caller's to choose, and
/// a record deliberately cut short is a legitimate thing to ask for. The window has no upper
/// cap, so at long path distances (hundreds of km) it can easily exceed the requested
/// duration.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Clipping {
    /// Subfault-ray contributions whose envelope peak fell past the end of the record.
    pub peaks_lost: usize,
    /// How far past the end the latest of them fell, seconds. Zero when none were lost.
    ///
    /// A lower bound, not an exact figure. A contribution ruled out before it is traced is
    /// reported against the earliest start it could have had, and its real start is at or
    /// after that. So the record is short by at least this much. The count above is exact
    /// either way.
    pub worst_overrun_s: f32,
}

impl Clipping {
    /// True when every arrival landed inside the record.
    pub fn is_complete(&self) -> bool {
        self.peaks_lost == 0
    }

    /// Record one arrival: whether its envelope peak landed inside the record.
    ///
    /// The Saragoni–Hart envelope peaks `peak_delay_s` after the trace starts, so the peak
    /// sample is `start_sample + peak_delay_s/dt`. Everything past `ndata` is discarded by
    /// [`SampleGrid::place`]; this is the part of that discard worth telling the caller about.
    pub fn record_arrival(&mut self, grid: SampleGrid, start_sample: i32, peak_delay_s: f32) {
        let peak_sample = start_sample as f32 + peak_delay_s / grid.dt;
        let overrun_samples = peak_sample - grid.ndata as f32;
        if overrun_samples > 0.0 {
            self.peaks_lost += 1;
            self.worst_overrun_s = self.worst_overrun_s.max(overrun_samples * grid.dt);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The straightforward sample-at-a-time loop, as an independent check on the slice form.
    fn accumulate_reference(
        acc: &mut Array2<f32>,
        subfault_acc: &Array2<f32>,
        weight: f32,
        start_sample: i32,
        np2: usize,
        ndata: usize,
    ) {
        let kend = (start_sample + np2 as i32 - 1).min(ndata as i32);
        let mut li = start_sample;
        while li <= kend {
            if li >= 1 {
                let idx = (li - start_sample) as usize;
                let sample = li as usize - 1;
                for component in 0..acc.nrows() {
                    acc[[component, sample]] += weight * subfault_acc[[component, idx]];
                }
            }
            li += 1;
        }
    }

    /// The slice form must agree with the loop form at every alignment, including the
    /// two that put the window entirely outside the record.
    ///
    /// `start_sample > ndata` computes a negative length, and casting that to `usize` wraps.
    /// It is reachable: a Moho multiple can make the path long enough to start past the end
    /// of the record.
    #[test]
    fn accumulate_matches_the_reference_loop_at_every_alignment() {
        let np2 = 8usize;
        let grid = SampleGrid {
            dt: 0.01,
            ndata: 10,
        };
        let subfault_acc = Array2::from_shape_fn((3, np2), |(c, i)| (c * 100 + i + 1) as f32);

        // Well before the record, straddling both edges, and well past the end.
        for start in -12i32..=14 {
            let mut got = Simulation::silent(grid, 3);
            let mut want: Array2<f32> = Array2::zeros((3, grid.ndata));
            if let Some(at) = grid.place(start, np2) {
                got.accumulate(subfault_acc.view(), MomentWeight(2.0), &at);
            }
            accumulate_reference(&mut want, &subfault_acc, 2.0, start, np2, grid.ndata);
            assert_eq!(got.acc, want, "start_sample = {start}");
        }
    }
}
