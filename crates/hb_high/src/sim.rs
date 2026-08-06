//! Simulating one station: the finite-fault layer.
//!
//! This module is **Graves & Pitarka (2010)**, "Broadband ground-motion simulation using a
//! hybrid approach", *BSSA* 100(5A), 2095–2123 — specifically its high-frequency module,
//! equations 10 through 17. [`crate::stoc`] builds one subfault's spectrum (Boore 1983); this
//! walks the rupture and sums them.
//!
//! ```text
//! A_i(f) = Σ_j  C_ij · S_i(f) · G_ij(f) · P(f)          eq. 10
//! ```
//!
//! The loops here are that sum: over subfaults `i`, over ray paths `j`, and over the three
//! output components. See `PHYSICS.md` §7 for the assembly, and `papers/README.md` for the
//! equation-by-equation verification.
//!
//! # One station per call
//!
//! **A multi-station run is not a concatenation of single-station runs**, at least not in the
//! original: it shared one generator across its station loop, so each station's draws continued
//! from wherever the previous one stopped. Here each station gets an independent stream seeded
//! from its own seed, which makes a batch safe to reorder, subset or resume.
//!
//! # Cost note
//!
//! Each call re-does the slip-model normalisation and the air-layer insertion, both of which are
//! station-independent. Deliberate for now — it keeps the signature honest about what it needs.
//! A `Simulator` type holding the prepared model and the reusable buffers is the obvious next
//! step once a caller starts looping over many stations.

use crate::config::{
    HfConfig, PathDurationModel, RUPTURE_VELOCITY_FRACTION_MAX, RayKind, RayType,
    RuptureVelocityTaper,
};
use ndarray::{Array1, Array2, ArrayView2, ArrayViewMut2, s};
use std::f32::consts::PI;

use crate::fft::Complex32;
use crate::geom::{FaultPlane, GeoPoint, SubfaultGeometry, subfault_geometry};
use crate::input::{Segment, StochModel, insert_air_layer};
use crate::radiation::{
    RadiationAngles, horizontal_radiation_spectrum, vertical_radiation_spectrum,
};
use crate::ray::green_function;
use crate::rng::{DrawSource, Draws, fill_uniform_deviates, normal_deviate};
use crate::site::{apply_site_amplification, site_amplification_factors};
use crate::state::{Layer, RayState, VelocityModel, VelocityModelInput, WaveMode};
use crate::stoc::{RayPath, SourceModel, SpectrumPlan, radiate_and_invert, stochastic_spectrum};

/// The three output components, in the order they are computed — which is also the order a
/// caller receives them in.
///
/// **The order is load-bearing, and this enum does not make it safe to change.** The two
/// horizontals each draw 5,000 deviates from the shared stream inside
/// [`crate::radiation::horizontal_radiation_spectrum`]; the vertical draws none, reading a
/// pre-filled table instead. So the three are *not* interchangeable positions in a loop:
/// reordering them, or iterating them in anything that does not preserve declaration order,
/// moves every waveform. See `PHYSICS.md` §9.
///
/// An enum rather than a trait deliberately — a trait would have to smuggle the
/// horizontal/vertical difference through an associated type and would read worse. Match on it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Component {
    /// `090` — east, and the first computed.
    E090,
    /// `000` — north.
    N000,
    /// The vertical. Capped at 15 Hz where the horizontals are not.
    Vertical,
}

impl Component {
    /// Declaration order, which is stream order. Iterate this, never a collection built
    /// some other way.
    const ALL: [Component; 3] = [Self::E090, Self::N000, Self::Vertical];

    /// Index into the three-element per-component arrays.
    #[inline]
    fn index(self) -> usize {
        match self {
            Self::E090 => 0,
            Self::N000 => 1,
            Self::Vertical => 2,
        }
    }

    /// Which radiation routine this component uses, and the angle it needs.
    ///
    /// This was an `Option<f32>` whose `None` meant "vertical", so the doc comment had to
    /// explain that the absence of an angle was really a routine selector. Two named variants
    /// say it instead.
    #[inline]
    fn radiation_mode(self) -> RadiationMode {
        match self {
            Self::E090 => RadiationMode::Horizontal {
                azimuth_offset_deg: -90.0,
            },
            Self::N000 => RadiationMode::Horizontal {
                azimuth_offset_deg: 0.0,
            },
            Self::Vertical => RadiationMode::Vertical,
        }
    }

    /// `f_max`, capped for the vertical only.
    ///
    /// The cap is empirical: vertical-component spectra fall off from a lower corner than the
    /// horizontals do. It is the one place the component identity changes the *physics* rather
    /// than just which radiation routine runs.
    #[inline]
    fn capped_fmax(self, fmax_hz: f32) -> f32 {
        match self {
            Self::Vertical => fmax_hz.min(VERTICAL_FMAX_CEILING_HZ),
            _ => fmax_hz,
        }
    }
}

/// How a component's radiation pattern is obtained.
///
/// The two arms differ in more than an angle: the horizontal one draws 5,000 deviates from the
/// live stream, and the vertical one draws none, reading the pre-filled uniform tables instead.
/// That asymmetry is why [`Component::ALL`]'s order is load-bearing.
#[derive(Clone, Copy, Debug, PartialEq)]
enum RadiationMode {
    /// Project SH and SV onto a horizontal axis this far from the station azimuth.
    Horizontal { azimuth_offset_deg: f32 },
    /// No projection; the vertical takes `SV · sin(takeoff)`.
    Vertical,
}

/// Ceiling on `f_max` for the vertical component, Hz.
const VERTICAL_FMAX_CEILING_HZ: f32 = 15.0;

/// Boore (1983, p. 1869): the record is made about twice the duration of strong shaking, so
/// the windowed transient has room to decay inside it.
const WINDOW_DURATION_FACTOR: f32 = 2.12;

/// Bars·km³ to dyn·cm, the CGS moment unit Boore (1983) eq. 2 works in.
const BARS_KM3_TO_DYN_CM: f32 = 1.0e+21;

/// The subfault moments are relative, in units of `mu · area` with lengths in km; this brings
/// their sum to dyn·cm.
const RELATIVE_MOMENT_TO_DYN_CM: f32 = 1.0e+20;

/// Nominal `Q₀` for the straight-ray approximation, which does not trace the medium and so
/// cannot integrate the real per-layer attenuation.
const STRAIGHT_RAY_Q: f32 = 150.0;
/// Nominal shear velocity for the same, km/s.
const STRAIGHT_RAY_VELOCITY_KM_S: f32 = 3.7;
/// Where the window starts, as a fraction of the straight-ray travel time.
const STRAIGHT_RAY_WINDOW_START_FRACTION: f32 = 0.7;

/// Below this relative moment a subfault contributes nothing and is skipped outright. Applied
/// identically when counting subfaults for the normalisation and when walking them in the
/// subfault pass — the two must agree or `moment_scale`'s `N` counts subfaults that never
/// radiate.
const SUBFAULT_WEIGHT_THRESHOLD: MomentWeight = MomentWeight(0.001);

/// One station's synthetic record.
pub struct Simulation {
    /// Samples per component.
    pub ndata: usize,
    pub dt: f32,
    /// Ground motion, shaped `(n_components, ndata)`, rows ordered 090, 000, vertical.
    pub acc: Array2<f32>,
}

/// Why a simulation could not be produced.
///
/// The messages name the field a caller would set, since a caller is who has to fix it.
#[derive(Debug, thiserror::Error)]
pub enum SimError {
    /// Every segment must share the first one's subfault size, because `dl` is a single
    /// per-run quantity in the corner-frequency and duration models.
    #[error(
        "segment {segment} has {dimension} = {found} km, but segment 0 has {expected} km; \
         every segment must be diced the same way because the subfault size is one \
         quantity for the whole run"
    )]
    InconsistentSegments {
        /// 0-based index of the offending segment, matching `StochModel::segments`.
        segment: usize,
        /// Which dimension disagrees, as the field name a caller would set.
        dimension: &'static str,
        found: f32,
        expected: f32,
    },
}

/// A configured simulation, ready to run stations against.
///
/// # Why this is built once
///
/// Everything a run needs that does not depend on where the receiver is — the air layer, the
/// slip-model normalisation, the moment scaling, the rupture taper, the path-duration table,
/// the per-segment angles — is computed here. [`Simulator::run`] then does only the part that
/// genuinely varies: the source-to-station geometry and the passes over it.
///
/// # Stations are independent, and that is a contract
///
/// `run` takes `&self`, so a batch can be driven from several threads over one `Simulator`.
/// Each call builds its own generator from its own seed and its own accumulator, so no state
/// crosses between stations — which is what makes a batch safe to reorder, subset or resume.
pub struct Simulator {
    /// The slip model as given. **Not modified** — the derived moment weights live in
    /// [`Simulator::weights`], which is the whole point of them being a separate type.
    slip: StochModel,
    /// One entry per segment, each indexed by [`Segment::grid_index`].
    weights: Vec<SegmentWeights>,
    /// The working model, with the air layer inserted and the two `real*4` fields widened.
    vmod: VelocityModel,
    rupture: RuptureVelocityTaper,
    path_duration: PathDuration,
    /// One per segment, in `slip.segments` order. Station-independent: the angles are the
    /// fault's own orientation plus a corner-frequency coefficient.
    angles: Vec<SegmentAngles>,
    rayset: Vec<RayType>,
    run: RunScalars,
    /// `ln` of the site-amplification table's frequency axis.
    siteamp_log_freq: Array1<f32>,
}

impl Simulator {
    /// Prepare a simulation. Fails only if the slip model's segments disagree on subfault size.
    pub fn new(
        config: &HfConfig,
        slip: &StochModel,
        vmod_in: &VelocityModelInput,
    ) -> Result<Self, SimError> {
        // Boore (1983, p. 1869)'s own envelope-shape values, and the ones Graves & Pitarka (2010)
        // use: the peak sits at 0.2 of the duration, decayed to 0.05 of the peak by the end.
        let window_peak_fraction = 0.2f32;
        let window_end_fraction = 0.05f32;

        let conical_sample_count = 1000usize;

        let fn_hz: Array1<f32> = ndarray::array![
            0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00, 2.00, 3.00, 5.00,
            7.00, 10.00, 20.00, 30.00, 50.00, 70.00,
        ];
        let siteamp_log_freq: Array1<f32> = fn_hz.mapv(f32::ln);

        let czero = config.source.czero;
        let calpha = config.source.calpha;
        let (duration_s, dt, f_max_hz, kappa_s, q_exponent) = (
            config.record.duration_s,
            config.record.dt_s,
            config.site.f_max_hz,
            config.site.kappa_s,
            config.path.q_exponent,
        );

        // The slip model is normalised in place and the velocity model gains an air
        // layer, so both are worked on as copies.
        let slip_model = slip.clone();
        let vmod_in = vmod_in.clone();

        // Every segment must agree with the FIRST on subfault size, so the first is the
        // reference and the rest are the candidates -- `split_first` says that, where
        // re-indexing `segments[0]` inside a loop over `segments` left it to the reader to
        // notice the index was constant. The reported numbers stay 1-based, to match the input
        // file's own numbering.
        if let Some((reference, rest)) = slip_model.segments.split_first() {
            for (index, segment) in rest.iter().enumerate() {
                // 0-based, matching `segments`, where the original reported the deck's 1-based
                // numbering.
                let segment_index = index + 1;
                for (dimension, found, expected) in [
                    (
                        "subfault_length_km",
                        segment.subfault_length_km,
                        reference.subfault_length_km,
                    ),
                    (
                        "subfault_width_km",
                        segment.subfault_width_km,
                        reference.subfault_width_km,
                    ),
                ] {
                    if found != expected {
                        return Err(SimError::InconsistentSegments {
                            segment: segment_index,
                            dimension,
                            found,
                            expected,
                        });
                    }
                }
            }
        }

        // ------------------------------------------------- path duration model ---
        let path_duration = path_duration_table(config.path.path_duration);

        // ------------------------------------------------------- air layer -------
        let vmod_in = insert_air_layer(vmod_in);

        // Resolved here rather than at parse time: the deep transition depths depend on
        // the deepest hypocentre, which is only known once the slip model is read.
        let rupture = config
            .source
            .rupture_velocity
            .resolve(slip_model.max_hypocentre_depth_km);

        // ------------------------------------------------ source normalisation ---
        let (
            SourceScale {
                avg_subfault_km,
                total_moment_dyn_cm,
                subfault_count,
            },
            weights,
        ) = normalise_source(&slip_model, &vmod_in);

        // `sigma_p * dl^3` -- the subfault moment scale, the denominator of Graves & Pitarka (2010)
        // eq. 12's `F`. The 1e21 converts bars*km^3 to dyn*cm.
        let subevent_moment = config.source.stress_drop_bars
            * avg_subfault_km
            * avg_subfault_km
            * avg_subfault_km
            * BARS_KM3_TO_DYN_CM;

        // `F` in Graves & Pitarka (2010) eq. 12 -- Frankel's (1995) finite-fault factor. It scales
        // the subfault corner frequency towards the mainshock's while keeping the summed moment
        // right; `crate::stoc` is where it does its work, and the note on `frank` there shows the
        // algebra.
        //
        // DEVIATION FROM THE PUBLISHED METHOD, AND IT IS A PHYSICS CHOICE. G&P define
        // `F = M_o / (N * sigma_p * dl^3)` -- LINEAR in subfault count. This uses `sqrt(N)`.
        // Graves & Pitarka (2015) does not revise `F`, so neither paper licenses the square root;
        // it may come from the subfault-summation scheme instead. Three other candidate scalings
        // were tried and abandoned upstream, which is some evidence this was tuned rather than
        // derived:
        //
        //   M_o / (subevent_moment * subfault_count)          linear in N -- what G&P specify
        //   M_o / (subevent_moment * sqrt(subfault_count))    THE LIVE ONE
        //   (M_o / subevent_moment)^(2/3)
        //   (fce_avg / fcmain)^2
        //
        // Flagged in `papers/README.md` finding 4 and left alone: changing it would move every
        // waveform, and it wants a domain judgement rather than a tidy-up.
        //
        // The `1.0 *` forces the integer count through a real multiply before the sqrt, and is
        // load-bearing for the exact result.
        let moment_scale =
            total_moment_dyn_cm / (subevent_moment * (1.0 * subfault_count as f32).sqrt());

        // ------------------------------------------------------------ stations ---
        // No ceiling on the record length: buffers are sized from the requested duration. The
        // original clamped this to a compile-time maximum and SILENTLY TRUNCATED anything longer.
        let ndata = (duration_s / dt).trunc() as usize;

        // The working model widens the two `real*4` input fields to `real*8`.
        let vmod: VelocityModel = vmod_in.iter().copied().map(Layer::from).collect();

        // Everything the two passes need that is constant for the whole run, gathered once
        // so the extracted functions take one reference instead of eighteen scalars.
        let run = RunScalars {
            dt,
            fmax_hz: f_max_hz,
            kappa_s,
            q_exponent,
            window_peak_fraction,
            window_end_fraction,
            corner_const: czero,
            calpha,
            rupture_velocity_sigma: config.source.rupture_velocity.rv_sig1,
            avg_subfault_km,
            subevent_moment,
            moment_scale,
            conical_sample_count,
            site_table_len: fn_hz.len(),
            ndata,
        };

        let angles = slip_model
            .segments
            .iter()
            .map(|seg| SegmentAngles::for_segment(seg, run.calpha, run.corner_const))
            .collect();

        Ok(Self {
            slip: slip_model,
            weights,
            vmod,
            rupture,
            path_duration,
            angles,
            rayset: config.path.rayset.clone(),
            run,
            siteamp_log_freq,
        })
    }

    /// Samples per component in every record this produces.
    pub fn ndata(&self) -> usize {
        self.run.ndata
    }

    /// Simulate one station.
    pub fn run(&self, station: crate::input::Station, seed: u64) -> Simulation {
        let (mut rng, deviates) = seed_and_predraw(seed, self.run.conical_sample_count);
        // One row per component, which is the shape this is returned in.
        let mut acc: Array2<f32> = Array2::zeros((Component::ALL.len(), self.run.ndata));

        // ------------------------------------------------- the single station ---
        for ((seg, angles), weights) in self
            .slip
            .segments
            .iter()
            .zip(&self.angles)
            .zip(&self.weights)
        {
            let geom = subfault_geometry(
                &FaultPlane {
                    origin: GeoPoint {
                        lat_deg: seg.fault_lat_deg,
                        lon_deg: seg.fault_lon_deg,
                    },
                    strike_deg: seg.strike_deg,
                    dip_deg: seg.dip_deg,
                    top_depth_km: seg.top_depth_km,
                    along_strike_offset_km: seg.along_strike_offset_km,
                    subfault_length_km: seg.subfault_length_km,
                    subfault_width_km: seg.subfault_width_km,
                    along_strike_count: seg.along_strike_count,
                    down_dip_count: seg.down_dip_count,
                },
                GeoPoint {
                    lat_deg: station.latitude,
                    lon_deg: station.longitude,
                },
            );

            let windows = time_window_pass(
                seg,
                &geom,
                &self.vmod,
                &self.rupture,
                &self.path_duration,
                angles,
                &self.run,
            );
            // The minimum over ALL segments. The original re-initialised this inside the segment
            // loop, so on a multi-segment model the reported distance described only the last
            // segment -- invisible while every fixture was single-segment. Now the minimum,
            // which is what "closest subfault distance" means.

            let plan = SpectrumPlan::new(
                windows.tmax,
                self.run.dt,
                self.run.q_exponent,
                self.run.window_peak_fraction,
                self.run.window_end_fraction,
            );

            subfault_pass(
                &mut rng,
                acc.view_mut(),
                SegmentPass {
                    seg,
                    geom: &geom,
                    windows: &windows,
                    plan: &plan,
                    angles,
                    weights,
                },
                RunContext {
                    vmod: &self.vmod,
                    rupture: &self.rupture,
                    run: &self.run,
                    rayset: &self.rayset,
                    deviates: &deviates,
                    siteamp_log_freq: &self.siteamp_log_freq,
                },
            );
        }

        Simulation {
            ndata: self.run.ndata,
            dt: self.run.dt,
            acc,
        }
    }
}

// ---------------------------------------------------------------------------
// The pieces a run is made of
// ---------------------------------------------------------------------------

/// Everything derived from the deck and the slip model that is constant for the whole
/// run.
///
/// This exists because the alternative is an eighteen-argument function. Grouping does
/// not make the coupling smaller, but it does put every one of these under a name with a
/// unit, in one place, instead of spread across a 500-line body.
struct RunScalars {
    dt: f32,
    fmax_hz: f32,
    kappa_s: f32,
    q_exponent: f32,
    /// The Saragoni-Hart window shape: where the envelope peaks, as a fraction of the
    /// duration, and what fraction of the peak it has decayed to by the end.
    window_peak_fraction: f32,
    window_end_fraction: f32,
    /// `c₀` — the numerator of the corner-frequency coefficient.
    corner_const: f32,
    calpha: f32,
    /// Rupture-velocity randomisation sigma. Zero disables the perturbation *and* its
    /// deviate consumption.
    rupture_velocity_sigma: f32,
    /// Average subfault dimension, km.
    avg_subfault_km: f32,
    subevent_moment: f32,
    moment_scale: f32,
    /// Sample count for the conical radiation average, and a DRAW COUNT.
    conical_sample_count: usize,
    /// Length of the site-amplification frequency table.
    site_table_len: usize,
    ndata: usize,
}

/// Per-segment angles, plus the corner-frequency coefficient hoisted out of the subfault
/// loops.
struct SegmentAngles {
    strike_rad: f32,
    dip_rad: f32,
    rake_rad: f32,
    /// `c₀(1 + fcfac) / α_τ` — everything in Graves & Pitarka (2010) eq. 13's corner frequency
    /// that does not vary within a segment.
    ///
    /// **Hoisted**, and bit-identically so: `α_τ` is a pure function of three per-segment
    /// constants, so evaluating it once per segment rather than once per subfault per loop gives
    /// the identical `f32`.
    corner_coeff: f32,
}

impl SegmentAngles {
    fn for_segment(seg: &Segment, calpha: f32, corner_const: f32) -> Self {
        Self {
            strike_rad: seg.strike_deg.to_radians(),
            dip_rad: seg.dip_deg.to_radians(),
            rake_rad: seg.rake_deg.to_radians(),
            corner_coeff: corner_const / alpha_t(seg.dip_deg, seg.rake_deg, calpha),
        }
    }
}

/// The generator and the pre-drawn blocks. **The fill order is the draw order**, and the draw
/// order is part of the answer.
struct Deviates {
    /// `rna` / `rnb` — the vertical component's uniforms.
    radv_uniform_a: Vec<f32>,
    radv_uniform_b: Vec<f32>,
}

/// Seed, then make the two pre-draws.
///
/// **The order and the counts are the contract, not an implementation detail.** Seed,
/// then `nr` uniforms into `a`, then `nr` uniforms into `b`. Changing either moves every
/// sample downstream, which is why this is one function rather than two calls spread
/// through the setup.
///
/// The two uniform blocks must stay two sequential fills. Interleaving them into one pass
/// would put different deviates in different slots and change every vertical component.
fn seed_and_predraw(seed: u64, radv_sample_count: usize) -> (DrawSource, Deviates) {
    let mut rng = DrawSource::for_station(seed);

    // Exactly `radv_sample_count` values, which is what `vertical_radiation_spectrum` reads.
    // Filled by two SEPARATE sequential passes -- see the note there on why they must not be
    // interleaved.
    let mut radv_uniform_a = vec![0.0f32; radv_sample_count];
    let mut radv_uniform_b = vec![0.0f32; radv_sample_count];
    fill_uniform_deviates(&mut rng, radv_sample_count, &mut radv_uniform_a);
    fill_uniform_deviates(&mut rng, radv_sample_count, &mut radv_uniform_b);

    (
        rng,
        Deviates {
            radv_uniform_a,
            radv_uniform_b,
        },
    )
}

/// The velocity-model layer a source at `depth_km` sits in.
///
/// # The defect this replaces
///
/// The two subfault passes disagreed about what to do when a source is deeper than the
/// whole model, and **both answers were wrong**:
///
/// * the window pass had no fallback at all, so it silently reused the *previous
///   subfault's* velocity — and on the very first subfault of the first station the
///   original read uninitialised memory. Pinning that to 0.0 would be a choice rather than
///   a behaviour, and no more defensible.
/// * the subfault pass fell back to layer 0 for the velocity — which after
///   `insert_air_layer` is the **air layer**, `vsh = 0.0005 km/s` — and set `ksrc` to
///   `layer_count`, one PAST the model. That index then reached
///   `site_amplification_factors`, which read a zeroed `Layer`, computed
///   `ln(0 / (bz*pz)) = -inf`, and exponentiated it back to a gain of **zero**. A
///   subfault below the model contributed nothing at all.
///
/// Both now take the deepest real layer, which is the only physically sensible reading:
/// a source below the model is in the half-space, and the half-space is the bottom layer.
///
/// This is reachable — the `vs_moho=4.2` deck truncates the model at the Moho and lands
/// subfaults beneath it, and it is the one deck of 22 whose output moves.
fn source_layer_for(vmod: &VelocityModel, depth_km: f32) -> usize {
    vmod.iter()
        .position(|layer| layer.depth_km >= depth_km as f64)
        .unwrap_or(vmod.len() - 1)
}

/// What the time-window pass produces for one segment.
struct WindowPass {
    /// One entry per real subfault, indexed by [`Segment::grid_index`].
    window_s: Vec<f32>,
    /// Longest window over the segment; sizes the transform.
    tmax: f32,
}

/// Time-window pass — `hb_high_ref.f`'s first subfault loop.
///
/// **Depth-major: `j` outer, `i` inner.** That is the OPPOSITE order to
/// [`subfault_pass`], and the difference is load-bearing there, not here: this pass draws
/// nothing from the generator, so its order affects only the `tmax` max-reduction and the
/// `d10` min-reduction, both of which are order-independent in exact arithmetic and
/// preserved here anyway.
fn time_window_pass(
    seg: &Segment,
    geom: &SubfaultGeometry,
    vmod: &VelocityModel,
    rupture: &RuptureVelocityTaper,
    path_duration: &PathDuration,
    angles: &SegmentAngles,
    run: &RunScalars,
) -> WindowPass {
    let mut tmax = 0.0f32;
    let mut window_s = vec![0.0f32; seg.subfault_total()];

    for (i, j) in seg.depth_major() {
        let ray = geom.at(i, j);
        // A subfault below the whole model takes the DEEPEST layer -- see
        // [`source_layer_for`] for why neither of the original fallbacks was defensible.
        let shear_velocity_km_s = vmod[source_layer_for(vmod, ray.depth_km)].vsh_km_s as f32;

        let rupture_fraction = rupture.factor(ray.depth_km);

        // The last table segment this distance is past. `.last()`, NOT `.find()` -- the table
        // ascends, so the first match is the wrong end.
        //
        // Strict `>`, so a distance of exactly the first breakpoint (0.0) matches NOTHING and
        // the duration terms stay zero. That zero is a deliberate choice for an undefined
        // case, and `unwrap_or` is where it lives.
        let bin = path_duration
            .iter()
            .take_while(|s| ray.slant_km > s.start_km)
            .last()
            .copied()
            .unwrap_or(DurationSegment {
                start_km: 0.0,
                duration_s: 0.0,
                slope_s_per_km: 0.0,
            });

        // Graves & Pitarka (2010) eq. 13: `f_ci = c0 * V_Ri / (alpha_tau * pi * dl)`, with
        // `corner_coeff` carrying `c0 / alpha_tau` and `rupture_fraction * beta` being the local rupture
        // speed `V_Ri`.
        let corner_frequency_hz = angles.corner_coeff * rupture_fraction * shear_velocity_km_s
            / (run.avg_subfault_km * PI);
        // The window duration, eq. 17: `T_di = f_ci^-1 + c1*R_i`, a source term plus a path
        // term. Two details worth stating:
        //
        //   * the source term uses `sqrt(F)/f_ci`, which is `1/f_c_effective` for the RESCALED
        //     corner of eq. 12 -- see the `frank` note in `crate::stoc`. So the duration
        //     follows the mainshock-scaled corner, not the raw subfault corner.
        //   * the factor of 2.12 is Boore's -- see WINDOW_DURATION_FACTOR.
        //
        // No upper cap on the window length.
        let source_duration_s = run.moment_scale.sqrt() * (1.0 / corner_frequency_hz);
        let path_duration_s = bin.duration_s + bin.slope_s_per_km * (ray.slant_km - bin.start_km);
        let window = WINDOW_DURATION_FACTOR * (source_duration_s + path_duration_s);
        window_s[seg.grid_index(i, j)] = window;

        if window > tmax {
            tmax = window;
        }
    }

    WindowPass { window_s, tmax }
}

/// The five per-segment things the subfault pass reads.
///
/// Bundled with [`RunContext`] to get `subfault_pass` from thirteen positional arguments
/// to four. The two bundles are the natural cut: this one changes once per segment, that
/// one not at all.
#[derive(Clone, Copy)]
struct SegmentPass<'a> {
    seg: &'a Segment,
    geom: &'a SubfaultGeometry,
    windows: &'a WindowPass,
    plan: &'a SpectrumPlan,
    angles: &'a SegmentAngles,
    /// This segment's moment weights, indexed by [`Segment::grid_index`].
    weights: &'a [MomentWeight],
}

/// Everything the subfault pass reads that is fixed for the whole run.
#[derive(Clone, Copy)]
struct RunContext<'a> {
    vmod: &'a VelocityModel,
    rupture: &'a RuptureVelocityTaper,
    run: &'a RunScalars,
    /// Which ray paths to sum over — the one thing the pass reads from the caller's config.
    rayset: &'a [RayType],
    deviates: &'a Deviates,
    siteamp_log_freq: &'a Array1<f32>,
}

/// Subfault pass — `hb_high_ref.f`'s second subfault loop.
///
/// **Strike-major: `i` outer, `j` inner, the OPPOSITE of [`time_window_pass`], and here
/// the order IS the contract.** `irandcnt` advances once per surviving subfault and
/// indexes the pre-drawn normals, and every `stochastic_spectrum` and
/// `horizontal_radiation_spectrum` call draws from the live stream. Walking the grid the
/// other way pairs a different deviate with every subfault and changes every waveform.
/// See `PORTING_RULES.md` §5.
fn subfault_pass(
    rng: &mut impl Draws,
    mut acc: ArrayViewMut2<'_, f32>,
    segment: SegmentPass<'_>,
    ctx: RunContext<'_>,
) {
    let SegmentPass {
        seg,
        geom,
        windows,
        plan,
        angles,
        weights,
    } = segment;
    let RunContext {
        vmod,
        rupture,
        run,
        rayset,
        deviates,
        siteamp_log_freq,
    } = ctx;

    let np2 = plan.np2;
    // Fixed for the whole run, so it is built once here rather than per call. Six of
    // `stochastic_spectrum`'s nineteen former arguments were these, re-passed on every one
    // of the hundreds of thousands of calls in a run.
    let model = SourceModel {
        dt: run.dt,
        window_eps: run.window_peak_fraction,
        window_eta: run.window_end_fraction,
        subevent_moment: run.subevent_moment,
        kappa_s: run.kappa_s,
        moment_scale: run.moment_scale,
    };
    // One row per component, allocated once for the whole pass. Both are pure scratch: every
    // element is written before it is read, on every subfault and every ray.
    let components = Component::ALL.len();
    let mut spectrum: Array2<Complex32> = Array2::zeros((components, np2));
    let mut subfault_acc: Array2<f32> = Array2::zeros((components, np2));
    let mut radiation: Array1<f32> = Array1::zeros(plan.fold_count);
    let mut siteamp_factors: Array1<f32> = Array1::zeros(run.site_table_len);
    let mut ray = RayState::default();

    for (i, j) in seg.strike_major() {
        let subfault = seg.at(i, j);
        let weight = weights[seg.grid_index(i, j)];
        if weight < SUBFAULT_WEIGHT_THRESHOLD {
            continue; // goto 4 lands on the inner loop's terminator
        }
        let ray_geometry = geom.at(i, j);
        let subfault_window_s = windows.window_s[seg.grid_index(i, j)];

        let source_layer = source_layer_for(vmod, ray_geometry.depth_km);
        let shear_velocity_km_s = vmod[source_layer].vsh_km_s as f32;
        let density_g_cm3 = vmod[source_layer].density_g_cm3 as f32;

        let base_rvf = rupture.factor(ray_geometry.depth_km);
        let rupture_fraction = if run.rupture_velocity_sigma > 0.0 {
            (base_rvf * (normal_deviate(rng) * run.rupture_velocity_sigma).exp())
                .min(RUPTURE_VELOCITY_FRACTION_MAX)
        } else {
            base_rvf
        };

        let corner_frequency_hz =
            angles.corner_coeff * rupture_fraction * shear_velocity_km_s / run.avg_subfault_km / PI;

        for &ray_type in rayset {
            let kind = ray_type.kind();

            // The tracing runs even for a straight ray -- type 0 borrows type 1's tracing and
            // then the straight-line values below overwrite the results. Wasteful, but the
            // tracer also advances no random state, so removing it is safe only if you are
            // sure of that.
            let green = green_function(
                &mut ray,
                vmod,
                ray_geometry.depth_km,
                ray_geometry.horiz_km,
                ray_type.trace_type(),
                WaveMode::Sh,
            );
            // The straight-ray option discards the traced result entirely and substitutes a
            // geometric one, so the two are alternatives rather than a default and a patch.
            // The tracer still ran either way -- see the note at the loop head.
            // The travel time itself is not one of them: it only ever fed `window_start_s`,
            // which the two arms compute differently anyway.
            let (path_length_km, qbar, window_start_s) = if kind == RayKind::StraightRay {
                let path_length_km = ray_geometry.slant_km;
                let travel_time_s = path_length_km / STRAIGHT_RAY_VELOCITY_KM_S;
                (
                    path_length_km,
                    path_length_km / (shear_velocity_km_s * STRAIGHT_RAY_Q),
                    STRAIGHT_RAY_WINDOW_START_FRACTION * travel_time_s,
                )
            } else {
                (
                    green.rpath,
                    green.qbar,
                    green.stime - run.window_peak_fraction * subfault_window_s,
                )
            };

            // Three calls in component order: each draws `np2` normal deviates. The only
            // field that differs between them is `fmax_hz`, which the vertical caps. Each
            // writes a full row, so there is nothing to reset between rays.
            for component in Component::ALL {
                stochastic_spectrum(
                    rng,
                    plan,
                    &model,
                    &RayPath {
                        distance_km: path_length_km,
                        window_s: subfault_window_s,
                        shear_velocity_km_s,
                        density_g_cm3,
                        corner_frequency_hz,
                        fmax_hz: component.capped_fmax(run.fmax_hz),
                        qbar,
                    },
                    spectrum.row_mut(component.index()),
                );
            }

            // Unconditional. There is no run for which the quarter-wavelength site
            // amplification should be off, so it is not a choice a caller gets to make.
            site_amplification_factors(
                vmod,
                source_layer,
                siteamp_log_freq.view(),
                siteamp_factors.view_mut(),
            );
            for mut spec in spectrum.rows_mut() {
                apply_site_amplification(
                    spec.as_slice_mut().expect("a row of a C-order Array2"),
                    plan.log_frequency_hz.view(),
                    siteamp_log_freq.view(),
                    siteamp_factors.view(),
                );
            }

            // Incidence angle from the ray parameter: sin(i)/vs = ray_parameter.
            let ray_parameter = green.rp0;
            let incidence = if shear_velocity_km_s * ray_parameter > 1.0 {
                0.5 * PI
            } else {
                (shear_velocity_km_s * ray_parameter).asin()
            };
            let th = match kind {
                // The straight-ray approximation ignores the traced ray parameter and
                // uses the geometric take-off angle.
                RayKind::StraightRay => ray_geometry.takeoff_rad,
                RayKind::Upgoing => PI - incidence,
                RayKind::Downgoing => incidence,
            };
            let pa = ray_geometry.azimuth_rad;

            // The ONLY difference between the three components is which radiation
            // routine runs, and the `Option` carries it: `Some` is a horizontal, which
            // draws 5,000 deviates; `None` is the vertical, which draws none.
            let arrival = RadiationAngles {
                strike_rad: angles.strike_rad,
                dip_rad: angles.dip_rad,
                rake_rad: angles.rake_rad,
                azimuth_rad: pa,
                takeoff_rad: th,
            };
            for component in Component::ALL {
                match component.radiation_mode() {
                    RadiationMode::Horizontal { azimuth_offset_deg } => {
                        horizontal_radiation_spectrum(
                            rng,
                            &arrival,
                            plan.frequency_hz.view(),
                            azimuth_offset_deg.to_radians(),
                            run.conical_sample_count,
                            radiation.view_mut(),
                        )
                    }
                    RadiationMode::Vertical => vertical_radiation_spectrum(
                        &arrival,
                        plan.frequency_hz.view(),
                        &deviates.radv_uniform_a,
                        &deviates.radv_uniform_b,
                        run.conical_sample_count,
                        radiation.view_mut(),
                    ),
                };
                // In place on both rows: the inverse transform overwrites the spectrum,
                // and the real samples land straight in the accumulator's row. No
                // allocation and no copy anywhere on this path.
                radiate_and_invert(
                    spectrum.row_mut(component.index()),
                    radiation.view(),
                    subfault_acc.row_mut(component.index()),
                );
            }

            // Both terms truncate TOWARD ZERO, not toward negative infinity, so a negative
            // `window_start_s` makes `kst` smaller and possibly negative. `accumulate_subfault`
            // relies on that and clips; see `PORTING_RULES.md` §7.
            let kst = (subfault.rupture_time_s / run.dt).trunc() as i32
                + (window_start_s / run.dt).trunc() as i32;

            // ONE DRAW, AND IT MUST STAY. A sub-event loop here once drew a uniform and
            // turned it into a time offset, but the sub-event count was frozen at 1 and the
            // offset was then unconditionally zeroed -- computed and thrown away.
            //
            // The arithmetic is gone. THE DRAW IS NOT. It advances the shared generator once
            // per (subfault, ray), and every sample drawn after it depends on where the stream
            // ends up. Deleting this as obviously-dead code changes every waveform in the
            // program. See `PHYSICS.md` §9.
            let _stream_advance = rng.next_f32();

            if let Some(at) = Placement::clip(kst, np2, run.ndata) {
                accumulate_subfault(acc.view_mut(), subfault_acc.view(), weight, &at);
            }
        }
    }
}

/// Where a subfault's `np2`-sample window lands in the record, after clipping to it.
///
/// # A fixed off-by-one
///
/// Sample 1 of the subfault's trace lands on `start_sample`, not on `start_sample + 1`. The
/// original indexed one element before the column — which nothing wrote, and which aliasing
/// made read as zero rather than crash — so every subfault's contribution arrived one sample
/// late. Corrected here.
struct Placement {
    /// How many of the subfault's own samples fall before the record starts.
    skip: usize,
    /// 0-based index in the record where the first surviving sample lands.
    offset: usize,
    /// How many samples land.
    count: usize,
}

impl Placement {
    /// Clip a subfault's window to the record, or `None` if none of it lands.
    ///
    /// `start_sample` is a 1-based sample number and **can be negative**: truncation is toward
    /// zero and the window start can precede the origin time.
    ///
    /// # Both ends can miss, and they are different cases
    ///
    /// One contribution ends before the record begins (a large negative start); another begins
    /// after it ends (a long-path ray at a far station). AS AN EXPLICIT RANGE THE SECOND CASE
    /// COMPUTES A NEGATIVE LENGTH, which must be rejected before it is cast to `usize` or it
    /// wraps to something enormous. A gate caught exactly that once.
    fn clip(start_sample: i32, np2: usize, ndata: usize) -> Option<Self> {
        let last_sample = (start_sample + np2 as i32 - 1).min(ndata as i32);
        let first_sample = start_sample.max(1);
        if last_sample < first_sample {
            return None;
        }
        Some(Self {
            skip: (first_sample - start_sample) as usize,
            offset: first_sample as usize - 1,
            count: (last_sample - first_sample + 1) as usize,
        })
    }
}

/// Place one subfault's contribution into the station accumulator.
///
/// Every index here is already known to be inside both buffers: [`Placement::clip`] did that,
/// and it is the caller's job to have called it. What is left is the axpy.
fn accumulate_subfault(
    mut acc: ArrayViewMut2<'_, f32>,
    subfault_acc: ArrayView2<'_, f32>,
    weight: MomentWeight,
    at: &Placement,
) {
    let &Placement {
        skip,
        offset,
        count,
    } = at;
    // `scaled_add` IS this operation: `y += alpha * x`, the axpy every linear-algebra library
    // names. One call per component, over the clipped window of each.
    let mut destination = acc.slice_mut(s![.., offset..offset + count]);
    destination.scaled_add(weight.0, &subfault_acc.slice(s![.., skip..skip + count]));
}

/// One segment of the piecewise-linear duration-versus-distance table.
///
/// One value per segment rather than three parallel arrays: every read of one field is at the
/// same index as the other two, so they belong together.
#[derive(Clone, Copy, Debug, PartialEq)]
struct DurationSegment {
    /// `rdur` — distance at which this segment starts, km.
    start_km: f32,
    /// `dpth` — duration at that distance, s.
    duration_s: f32,
    /// `dpdr` — slope of this segment, s/km.
    slope_s_per_km: f32,
}

/// The path-duration model: a piecewise-linear duration-versus-distance table.
///
/// `len()` is the segment count. The largest model here uses eight segments.
type PathDuration = Vec<DurationSegment>;

/// Build the path-duration table.
fn path_duration_table(model: PathDurationModel) -> PathDuration {
    /// The single-segment models give their slope directly and have no breakpoints.
    fn constant_slope(slope_s_per_km: f32) -> PathDuration {
        vec![DurationSegment {
            start_km: 0.0,
            duration_s: 0.0,
            slope_s_per_km,
        }]
    }

    /// The multi-segment models give (distance, duration) breakpoints; the slopes are
    /// differenced from consecutive pairs.
    ///
    /// The last segment repeats the previous slope so distances past the table
    /// extrapolate rather than flatten — which is why the fold looks one short and then
    /// pushes a copy.
    fn from_breakpoints<const N: usize>(start_km: [f32; N], duration_s: [f32; N]) -> PathDuration {
        let mut table: PathDuration = start_km
            .windows(2)
            .zip(duration_s.windows(2))
            .map(|(r, d)| DurationSegment {
                start_km: r[0],
                duration_s: d[0],
                slope_s_per_km: (d[1] - d[0]) / (r[1] - r[0]),
            })
            .collect();
        let last_slope = table
            .last()
            .expect("a breakpoint table has at least two entries")
            .slope_s_per_km;
        table.push(DurationSegment {
            start_km: start_km[N - 1],
            duration_s: duration_s[N - 1],
            slope_s_per_km: last_slope,
        });
        table
    }

    match model {
        // Graves & Pitarka (2010) eq. 17: `T_di = f_ci^-1 + c1*R_i` with `c1 = 0.063`.
        PathDurationModel::Gp2010 => constant_slope(0.063),
        PathDurationModel::Wus => constant_slope(0.07),
        PathDurationModel::Ena => constant_slope(0.1),
        // Boore & Thompson (2014) Table 1, reproduced exactly: breakpoints at 0, 7, 45, 125,
        // 175, 270 km with durations 0, 2.4, 8.4, 10.9, 17.4, 34.2 s.
        //
        // KNOWN DEVIATION FROM THE PAPER, BEYOND 270 km. Table 1 specifies a tail slope of
        // 0.156 s/km for `R` past the last breakpoint. `from_breakpoints` instead copies the
        // slope of the final tabulated segment, `(34.2 - 17.4)/(270 - 175) = 0.177` -- about
        // 13% steeper. This reproduces the original faithfully (it did
        // `dpdr(ndur) = dpdr(ndur-1)`), so the deviation is in the model as implemented rather
        // than in this port, and it only bites for ray paths longer than 270 km. Recorded in
        // `papers/README.md`; not changed here, because it would move every long-path waveform.
        PathDurationModel::Bt2014Wus => from_breakpoints(
            [0.0, 7.0, 45.0, 125.0, 175.0, 270.0],
            [0.0, 2.4, 8.4, 10.9, 17.4, 34.2],
        ),
        PathDurationModel::Bt2015Ena => from_breakpoints(
            [0.0, 15.0, 35.0, 50.0, 125.0, 200.0, 392.0, 600.0],
            [0.0, 2.6, 17.5, 25.1, 25.1, 28.5, 46.0, 69.1],
        ),
    }
}

/// Scalars derived from the slip model before any station is simulated.
struct SourceScale {
    /// Average subfault dimension `sqrt(length * width)`, averaged over segments, km.
    avg_subfault_km: f32,
    /// `M_o` — total seismic moment, summed from the subfault moments, dyn·cm.
    total_moment_dyn_cm: f32,
    /// Count of subfaults whose relative moment exceeds 0.001 — the same
    /// threshold the subfault pass uses to skip a subfault entirely.
    subfault_count: usize,
}

/// A subfault's share of the total moment, normalised so the mean over contributing
/// subfaults is one.
///
/// This is what a subfault's trace is weighted by when it is summed into the record. It is
/// **derived from** [`crate::input::Slip`] and is not slip: the conversion multiplies by rigidity and area, then rescales the
/// whole model. Keeping the two as distinct types is what stops a caller weighting by
/// centimetres of slip, which is what a single reused `f32` field allowed.
#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct MomentWeight(pub f32);

/// One segment's moment weights, indexed by [`Segment::grid_index`].
type SegmentWeights = Vec<MomentWeight>;

/// Convert slip to relative moment, then normalise to unit average weight.
///
/// The slip model is **not modified**: the weights come back in their own storage, parallel to
/// the subfault grid.
///
/// Three passes over the subfault grid, in this order:
///
/// 1. average subfault size, and the maximum absolute slip (computed and discarded)
/// 2. slip → moment via the rigidity `xmu`, accumulating `xsum`, `fce_avg`,
///    `trise_avg` and a count of *all* subfaults
/// 3. re-count only the subfaults above 0.001 and rescale so their mean weight is 1
///
/// The two counts are different and both matter: pass 2's normalises the averages, pass 3's is
/// the one that reaches `moment_scale` — the `N` of Graves & Pitarka (2010) eq. 12. Only the
/// second is returned, because only it is read downstream.
fn normalise_source(
    stoch: &StochModel,
    vmod_in: &VelocityModelInput,
) -> (SourceScale, Vec<SegmentWeights>) {
    let segment_count = stoch.segments.len();

    // --- pass 1: average subfault size ----------------------------------------
    let avg_subfault_km = stoch
        .segments
        .iter()
        .map(|s| (s.subfault_length_km * s.subfault_width_km).sqrt())
        .sum::<f32>()
        / segment_count as f32;

    // --- pass 2: relative slip to relative moment -----------------------------
    // An average corner frequency and rise time were also accumulated here, at the cost of a
    // rupture-velocity taper, an `alpha_tau` evaluation and several more operations PER
    // SUBFAULT. All of it fed a quantity nothing reads:
    //
    //   fce_avg, trise_avg -> fcmain -> bigC3  (hb_high_ref.f:805)
    //
    // and `bigC` is assigned five times at :807-811 with `bigC = bigC1b` last, so
    // bigC3 never survives. Their only other consumers are the commented-out prints
    // at :684-685 and :817. Dropping them also removes `xnorm`, this pass's own
    // `subfault_count` count (which only normalised them), and `islip_weight_avg` -- a frozen
    // switch whose two branches only ever weighted these dead quantities.
    //
    // What remains live: the in-place slip-to-moment conversion, and the moment sum.
    let mut relative_moment_sum = 0.0f32;
    let mut weights: Vec<SegmentWeights> = Vec::with_capacity(stoch.segments.len());

    for segment in &stoch.segments {
        let mut segment_weights: SegmentWeights = Vec::with_capacity(segment.subfault_total());
        let row_depth_step_km = segment.subfault_width_km * segment.dip_deg.to_radians().sin();
        let top_depth_km = segment.top_depth_km;
        // THE WHOLE PRODUCT BELOW IS COMPUTED IN f64 and narrows only on assignment to `xmu`.
        // Narrowing earlier -- for instance by keeping the dimensions in `f32` -- shifts every
        // subfault moment by an ulp or two.
        let (length_km, width_km) = (
            segment.subfault_length_km as f64,
            segment.subfault_width_km as f64,
        );

        // Depth-major, which is storage order, so the sum accumulates without index
        // arithmetic. Rigidity is per depth row, not per subfault, so the row is the unit.
        for (row_index, row) in segment.depth_rows().enumerate() {
            let row_depth_km = top_depth_km + (row_index as f32 + 0.5) * row_depth_step_km;
            // A subfault below every layer is in the half-space, which is the bottom
            // layer -- the same reading as `source_layer_for` and `ray::source_layer`.
            let layer = vmod_in
                .iter()
                .position(|layer| row_depth_km <= layer.depth_km)
                .unwrap_or(vmod_in.len() - 1);
            // Rigidity times area: `mu = rho * beta^2`, so slip times this is moment.
            let rigidity_area = (vmod_in[layer].vsh_km_s
                * vmod_in[layer].vsh_km_s
                * vmod_in[layer].density_g_cm3
                * length_km
                * width_km) as f32;

            for subfault in row {
                let weight = MomentWeight(subfault.slip.0 * rigidity_area);
                relative_moment_sum += weight.0;
                segment_weights.push(weight);
            }
        }
        weights.push(segment_weights);
    }

    let total_moment_dyn_cm = RELATIVE_MOMENT_TO_DYN_CM * relative_moment_sum;

    // --- pass 3: normalise relative moments to average weight unity -----------
    // Depth-major again -- and `weights` was filled depth-major, so a flat walk of it is the
    // same order the original summed in.
    let mut weight_sum = 0.0f32;
    let mut subfault_count = 0usize;
    for weight in weights.iter().flatten() {
        if *weight > SUBFAULT_WEIGHT_THRESHOLD {
            weight_sum += weight.0;
            subfault_count += 1;
        }
    }
    let scale = subfault_count as f32 / weight_sum;
    for weight in weights.iter_mut().flatten() {
        weight.0 *= scale;
    }

    (
        SourceScale {
            avg_subfault_km,
            total_moment_dyn_cm,
            subfault_count,
        },
        weights,
    )
}

/// `α_τ`, the dip-and-rake corner-frequency and rise-time adjustment.
///
/// Graves & Pitarka (2015) eq. 3, `α_T = 1 + F_D·F_R·c_α`. **Returns the RECIPROCAL of that**,
/// because the caller divides `c₀` by it and eq. 13 has `α_τ` in the denominator — so the value
/// returned here is `α_τ` itself, ≤ 1, and smaller for a shallow-dipping thrust. Physically that
/// means such a fault gets a *higher* corner frequency and a shorter rise time, which is the
/// observed trend (Graves & Pitarka 2010, p. 2099, citing Somerville 1998: shorter rise times for
/// thrust events imply relatively high dynamic stress drops).
///
/// Graves & Pitarka (2010) eq. 9 parameterised this on **dip alone**, piecewise — 1 above 60°,
/// 0.82 below 45°. The continuous dip-and-rake form below is the 2015 revision, and is another
/// marker that this code tracks the later method — as does the default `c₀` of 2.0 rather than
/// the 2010 paper's 2.1. See `papers/README.md` finding 5.
///
/// `fD` tapers with dip above 45 degrees; `fR` peaks at a rake of 90 degrees. The rake is first
/// wrapped into
/// `[-180, 180]` by repeated addition or subtraction of 360.
fn alpha_t(dip_deg: f32, rake_deg: f32, calpha: f32) -> f32 {
    let dip_factor = if (45.0..=90.0).contains(&dip_deg) {
        1.0 - (dip_deg - 45.0) / 45.0
    } else if (0.0..=45.0).contains(&dip_deg) {
        1.0
    } else {
        0.0
    };

    // Wrapped into [-180, 180]. `rem_euclid` folds into [0, 360) in one step where the
    // original stepped by 360 in a loop, and the shift back is the second line.
    let wrapped_rake_deg = {
        let folded = rake_deg.rem_euclid(360.0);
        if folded > 180.0 {
            folded - 360.0
        } else {
            folded
        }
    };

    let rake_factor = if (0.0..=180.0).contains(&wrapped_rake_deg) {
        1.0 - (wrapped_rake_deg - 90.0).abs() / 90.0
    } else {
        0.0
    };

    1.0 / (1.0 + dip_factor * rake_factor * calpha)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `alpha_t` wraps the rake into `[-180, 180]`. That was a pair of `while` loops stepping
    /// by 360 and is now `rem_euclid`, which is NOT the same function at the boundary:
    /// `-180` wraps to `+180` here where the loop left it at `-180`.
    ///
    /// It does not matter, and this is why. The wrapped rake is only ever used as
    /// `1 - |rake - 90|/90` inside `0..=180`, and the two boundary values agree there:
    /// `-180` falls outside the range and contributes 0, while `+180` falls inside and
    /// computes `1 - 1 = 0`. Both are exactly zero, so `alpha_t` is unchanged.
    #[test]
    fn rake_wrapping_agrees_with_the_stepping_loop_it_replaced() {
        fn stepped(mut rake_deg: f32) -> f32 {
            while rake_deg < -180.0 {
                rake_deg += 360.0;
            }
            while rake_deg > 180.0 {
                rake_deg -= 360.0;
            }
            if (0.0..=180.0).contains(&rake_deg) {
                1.0 - (rake_deg - 90.0).abs() / 90.0
            } else {
                0.0
            }
        }
        fn wrapped(rake_deg: f32) -> f32 {
            let folded = rake_deg.rem_euclid(360.0);
            let rake_deg = if folded > 180.0 {
                folded - 360.0
            } else {
                folded
            };
            if (0.0..=180.0).contains(&rake_deg) {
                1.0 - (rake_deg - 90.0).abs() / 90.0
            } else {
                0.0
            }
        }
        // Every boundary and every multiple of 90 across four turns.
        for step in -720i32..=720 {
            let rake_deg = step as f32;
            assert_eq!(
                stepped(rake_deg).to_bits(),
                wrapped(rake_deg).to_bits(),
                "rake {rake_deg} deg"
            );
        }
    }

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
    /// `start_sample > ndata` is the case that was once wrong: it computes a negative length,
    /// and casting that to `usize` wraps. It is reachable -- a Moho multiple can make the path
    /// long enough to start past the end of the record -- and this pins it directly rather
    /// than relying on a deck that happens to trigger it.
    #[test]
    fn accumulate_matches_the_reference_loop_at_every_alignment() {
        let np2 = 8usize;
        let ndata = 10usize;
        let subfault_acc = Array2::from_shape_fn((3, np2), |(c, i)| (c * 100 + i + 1) as f32);

        // Well before the record, straddling both edges, and well past the end.
        for start in -12i32..=14 {
            let mut got: Array2<f32> = Array2::zeros((3, ndata));
            let mut want: Array2<f32> = Array2::zeros((3, ndata));
            if let Some(at) = Placement::clip(start, np2, ndata) {
                accumulate_subfault(got.view_mut(), subfault_acc.view(), MomentWeight(2.0), &at);
            }
            accumulate_reference(&mut want, &subfault_acc, 2.0, start, np2, ndata);
            assert_eq!(got, want, "start_sample = {start}");
        }
    }
}
