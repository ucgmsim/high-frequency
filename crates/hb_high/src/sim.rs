//! Simulating one station: the finite-fault layer.
//!
//! This module is **Graves & Pitarka (2010)**, "Broadband ground-motion simulation using a
//! hybrid approach", *BSSA* 100(5A), 2095–2123 — specifically its high-frequency module,
//! equations 10 through 17. [`crate::spectrum`] builds one subfault's spectrum (Boore 1983);
//! this walks the rupture and sums them.
//!
//! ```text
//! A_i(f) = Σ_j  C_ij · S_i(f) · G_ij(f) · P(f)          eq. 10
//! ```
//!
//! The loops here are that sum: over segments, over subfaults `i`, over ray paths `j`, and over
//! the three output components. See `PHYSICS.md` §7 for the assembly.
//!
//! # How a station is simulated
//!
//! [`Simulator::new`] does everything that does not depend on the station — see
//! [`crate::source`] — and [`Simulator::run`] does the rest, one segment at a time:
//!
//! 1. **geometry** — every subfault's distance, azimuth and depth as seen from the station;
//! 2. **window lengths** — each subfault's shaping-window duration, eq. 17, whose maximum
//!    sizes the segment's scratch;
//! 3. **the subfault walk** — for each subfault and each ray path: trace the ray, place it in
//!    time, and if it can reach the record, synthesise three components and add them in.
//!
//! # One station per call
//!
//! Each station gets an independent stream seeded from its own seed, and every
//! `(subfault, ray)` a sub-stream seeded from its own identity, which makes a batch safe to
//! reorder, subset or resume.

use ndarray::{Array1, Array2, s};
use std::f32::consts::PI;

use crate::config::HfConfig;
use crate::fft::Complex32;
use crate::geom::{GeoPoint, SubfaultGeometry, SubfaultRay, subfault_geometry};
use crate::path_duration::PathDuration;
use crate::radiation::{
    CONICAL_SAMPLE_COUNT, RadiationAngles, horizontal_radiation_spectrum,
    vertical_radiation_spectrum,
};
use crate::ray::{Onset, RayState, RayType, TracedPath, trace};
use crate::record::SampleGrid;
pub use crate::record::Simulation;
use crate::rng::{Draws, Pcg, ray_stream_seed, substream_seed};
use crate::site::{SiteResponse, apply_site_amplification};
use crate::slip_model::{Segment, SlipModel};
use crate::source::{
    MomentWeight, RuptureVelocityTaper, SUBFAULT_WEIGHT_THRESHOLD, SegmentAngles, SegmentWeights,
    SourceScale, scale_source,
};
use crate::spectrum::{
    PlanCache, SourceModel, SpectrumInputs, SpectrumPlan, SpectrumShape, WindowShape,
    radiate_and_invert, stochastic_spectrum,
};
use crate::velocity::{VelocityModel, VelocityModelInput, layer_containing, working_model};

/// The three output components, in the order they are computed — which is also the order a
/// caller receives them in.
///
/// The order is load-bearing. The two horizontals each draw 5,000 deviates from the shared
/// stream inside [`crate::radiation::horizontal_radiation_spectrum`]; the vertical draws none,
/// reading a pre-filled table instead. Reordering them, or iterating them in anything that
/// does not preserve declaration order, moves every waveform. See `PHYSICS.md` §9.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Component {
    /// `090` — east, and the first computed.
    E090,
    /// `000` — north.
    N000,
    /// The vertical. Capped at 15 Hz where the horizontals are not.
    Vertical,
}

/// How many components a record has.
pub const COMPONENT_COUNT: usize = Component::ALL.len();

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
/// That asymmetry is why [`Component::ALL`]'s order matters.
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
        /// 0-based index of the offending segment, matching `SlipModel::segments`.
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
/// Everything a run needs that does not depend on where the receiver is — the working velocity
/// model, the moment weights and scaling, the rupture taper, the path-duration table, the
/// per-segment angles — is computed here. [`Simulator::run`] then does only the part that
/// genuinely varies: the source-to-station geometry and the passes over it.
///
/// # Stations are independent
///
/// `run` takes `&self`, so a batch can be driven from several threads over one `Simulator`.
/// Each call builds its own generator from its own seed and its own accumulator, so no state
/// crosses between stations — which is what makes a batch safe to reorder, subset or resume.
pub struct Simulator {
    /// One entry per segment of the slip model, in order.
    segments: Vec<SegmentSource>,
    /// The working model, with the air layer inserted.
    vmod: VelocityModel,
    site: SiteResponse,
    rupture: RuptureVelocityTaper,
    path_duration: PathDuration,
    /// Which ray paths to sum over.
    rayset: Vec<RayType>,
    scale: SourceScale,
    /// The spectral-model constants every subfault's spectrum shares.
    spectrum: SourceModel,
    q_frequency_exponent: f32,
    fmax_hz: f32,
    grid: SampleGrid,
    /// The fastest shear-wave velocity in the model, km/s.
    ///
    /// Not physics but a bound: no traced ray can travel faster than the quickest medium it
    /// could possibly cross, so straight-line distance over this is a lower bound on any
    /// subfault's travel time, and that is what lets an arrival be ruled out before it is
    /// traced. See [`SubfaultSource::earliest_start_sample`].
    max_shear_velocity_km_s: f32,
}

/// One segment of the slip model, with what the simulation derives from it.
struct SegmentSource {
    segment: Segment,
    /// This segment's moment weights, indexed by [`Segment::grid_index`].
    weights: SegmentWeights,
    /// Station-independent: the fault's own orientation plus a corner-frequency coefficient.
    angles: SegmentAngles,
}

impl Simulator {
    /// Prepare a simulation. Fails only if the slip model's segments disagree on subfault size.
    pub fn new(
        config: &HfConfig,
        slip: &SlipModel,
        vmod_in: &VelocityModelInput,
    ) -> Result<Self, SimError> {
        check_subfault_sizes(slip)?;

        let vmod = working_model(vmod_in);
        let (scale, weights) = scale_source(slip, &vmod, config.source.stress_drop_bars);

        let segments = slip
            .segments
            .iter()
            .zip(weights)
            .map(|(segment, weights)| SegmentSource {
                angles: SegmentAngles::for_segment(
                    segment,
                    config.source.corner_frequency_alpha,
                    config.source.corner_frequency_constant,
                ),
                weights,
                segment: segment.clone(),
            })
            .collect();

        let dt = config.record.dt_s;
        // No ceiling on the record length: buffers are sized from the requested duration.
        let grid = SampleGrid {
            dt,
            ndata: (config.record.duration_s / dt).trunc() as usize,
        };

        Ok(Self {
            segments,
            site: SiteResponse::new(&vmod),
            // Resolved here rather than at parse time: the deep transition depths depend on
            // the deepest hypocentre, which is only known once the slip model is read.
            rupture: RuptureVelocityTaper::new(
                config.source.rupture_velocity,
                slip.max_hypocentre_depth_km,
            ),
            path_duration: PathDuration::new(config.path.path_duration),
            rayset: config.path.rayset.clone(),
            spectrum: SourceModel {
                dt,
                window: WindowShape::BOORE_1983,
                subevent_moment: scale.subevent_moment,
                kappa_s: config.site.kappa_s,
                moment_scale: scale.moment_scale,
            },
            scale,
            q_frequency_exponent: config.path.q_frequency_exponent,
            fmax_hz: config.site.fmax_hz,
            grid,
            // The air layer is the slowest thing in the model, so the maximum is unaffected by
            // its presence.
            max_shear_velocity_km_s: vmod
                .iter()
                .map(|layer| layer.vsh_km_s as f32)
                .fold(f32::MIN, f32::max),
            vmod,
        })
    }

    /// Samples per component in every record this produces.
    pub fn ndata(&self) -> usize {
        self.grid.ndata
    }

    /// Simulate one station, drawing from the production generator, [`Pcg`].
    pub fn run(&self, station: GeoPoint, seed: u64) -> Simulation {
        self.run_with::<Pcg>(station, seed)
    }

    /// Simulate one station, drawing every stream from `D`.
    ///
    /// The choice of generator is part of the answer — see [`crate::rng`] — so it is a type
    /// parameter rather than a setting: the snapshot test runs [`crate::rng::FixtureDraws`].
    pub fn run_with<D: Draws>(&self, station: GeoPoint, seed: u64) -> Simulation {
        let mut run = StationRun {
            seed,
            vertical: VerticalUniforms::draw::<D>(seed),
            // Shared across every segment of this station, because the lengths repeat across
            // segments as well as within one. See `PlanCache` for why it is per station.
            plans: PlanCache::new(
                self.grid.dt,
                self.q_frequency_exponent,
                self.spectrum.window,
            ),
            out: Simulation::silent(self.grid, COMPONENT_COUNT),
        };

        for (index, segment) in self.segments.iter().enumerate() {
            let geometry = subfault_geometry(&segment.segment.fault_plane(), station);
            self.sum_segment::<D>(&mut run, index, segment, &geometry);
        }

        // A property of the station's one cache, so it is read here rather than accumulated
        // out of the per-segment passes.
        run.out.census.transform_lengths = run.plans.len();
        run.out
    }

    /// Window-length pass: the shaping-window length of every subfault in a segment.
    ///
    /// Depth-major. The order does not matter: this pass draws nothing from the generator and
    /// its only reduction is the maximum.
    fn window_lengths(
        &self,
        segment: &SegmentSource,
        geometry: &SubfaultGeometry,
    ) -> WindowLengths {
        let mut longest_s = 0.0f32;
        let mut window_s = vec![0.0f32; segment.segment.subfault_total()];

        for (i, j) in segment.segment.depth_major() {
            let ray = geometry.at(i, j);
            let shear_velocity_km_s =
                self.vmod[layer_containing(&self.vmod, ray.depth_km)].vsh_km_s as f32;
            let rupture_fraction = self.rupture.factor(ray.depth_km);

            // Graves & Pitarka (2010) eq. 13: `f_ci = c0 * V_Ri / (alpha_tau * pi * dl)`, with
            // `corner_coeff` carrying `c0 / alpha_tau` and `rupture_fraction * beta` being the
            // local rupture speed `V_Ri`. Unperturbed, unlike `Simulator::subfault_source`'s — and
            // associated differently, `/ (dl * pi)` against `/ dl / pi`, which rounds
            // differently; both are pinned.
            let corner_frequency_hz =
                segment.angles.corner_coeff * rupture_fraction * shear_velocity_km_s
                    / (self.scale.avg_subfault_km * PI);
            // The window duration, eq. 17: `T_di = f_ci^-1 + c1*R_i`, a source term plus a
            // path term. Two details worth stating:
            //
            //   * the source term uses `sqrt(F)/f_ci`, which is `1/f_c_effective` for the
            //     rescaled corner of eq. 12 -- see the `frank` note in `crate::spectrum`. So the
            //     duration follows the mainshock-scaled corner, not the raw subfault corner.
            //   * the factor of 2.12 is Boore's -- see WINDOW_DURATION_FACTOR.
            //
            // No upper cap on the window length.
            let source_duration_s = self.scale.moment_scale.sqrt() * (1.0 / corner_frequency_hz);
            let path_duration_s = self.path_duration.at(ray.slant_km);
            let window = WINDOW_DURATION_FACTOR * (source_duration_s + path_duration_s);
            window_s[segment.segment.grid_index(i, j)] = window;

            if window > longest_s {
                longest_s = window;
            }
        }

        WindowLengths {
            window_s,
            longest_s,
        }
    }

    /// Sum every subfault of one segment into the station's record.
    ///
    /// # The walk order is free
    ///
    /// Each `(subfault, ray)` draws from a stream seeded by its own identity (see
    /// [`substream_seed`]), so a contribution that cannot reach the record can be skipped
    /// without changing the ones that can. Strike-major is used because it is the grid's
    /// layout order.
    fn sum_segment<D: Draws>(
        &self,
        run: &mut StationRun,
        index: usize,
        segment: &SegmentSource,
        geometry: &SubfaultGeometry,
    ) {
        let windows = self.window_lengths(segment, geometry);
        let grid = self.grid;
        let peak_fraction = self.spectrum.window.peak_fraction;
        let mut scratch = SegmentScratch::for_plan(
            run.plans
                .for_length(SpectrumPlan::length_for(windows.longest_s, grid.dt)),
        );

        for (i, j) in segment.segment.strike_major() {
            let subfault_seed = substream_seed(run.seed, index, segment.segment.grid_index(i, j));
            let mut subfault_rng = D::from_seed(subfault_seed);

            let Some(source) =
                self.subfault_source(&mut subfault_rng, segment, geometry, &windows, i, j)
            else {
                continue;
            };

            // Ruled out before a single ray is traced; the bound is exact, see
            // `SubfaultSource::earliest_start_sample`. Every ray of this subfault shares the
            // bound, because it is built from the straight-line distance rather than from any
            // particular path.
            let earliest = source.earliest_start_sample(self);
            if grid.starts_after_end(earliest) {
                run.out.census.pairs_attempted += self.rayset.len();
                run.out.census.pairs_outside_record += self.rayset.len();
                // The peak is past the end for all of them, and by at least this much: the
                // real start is at or after `earliest`, so the real overrun is at or above the
                // one reported here. See `Clipping::worst_overrun_s`.
                for _ in &self.rayset {
                    run.out.clipping.record_arrival(
                        grid,
                        earliest,
                        peak_fraction * source.window_s,
                    );
                }
                continue;
            }

            for (ray_index, &ray_type) in self.rayset.iter().enumerate() {
                let mut rng = D::from_seed(ray_stream_seed(subfault_seed, ray_index));

                let ray = trace(
                    &mut scratch.ray_state,
                    &self.vmod,
                    ray_type,
                    &source.geometry,
                    source.shear_velocity_km_s,
                );
                let window_start_s = match ray.onset {
                    // The envelope leads its own peak by `ε` of the window, so the window
                    // starts that far ahead of the arrival and can start before the origin.
                    Onset::Arrival { travel_time_s } => {
                        travel_time_s - peak_fraction * source.window_s
                    }
                    Onset::WindowStart { window_start_s } => window_start_s,
                };
                let start_sample = grid.start_sample(source.rupture_time_s, window_start_s);

                run.out.clipping.record_arrival(
                    grid,
                    start_sample,
                    peak_fraction * source.window_s,
                );
                run.out.census.pairs_attempted += 1;

                // Decided before a transform length is chosen, because no length makes a
                // contribution that begins past the end of the record reach it.
                if grid.starts_after_end(start_sample) {
                    run.out.census.pairs_outside_record += 1;
                    continue;
                }

                // This subfault's own transform length, clipped to what can land. See
                // `SpectrumPlan::length_for_arrival` for why that is Boore's factor of two
                // rather than a truncation of it. Built at most once per distinct length per
                // station.
                let plan = run.plans.for_length(SpectrumPlan::length_for_arrival(
                    source.window_s,
                    grid.dt,
                    start_sample,
                    grid.ndata,
                ));

                // The other way a contribution can miss: it ends before the record begins.
                let Some(placement) = grid.place(start_sample, plan.np2) else {
                    run.out.census.pairs_outside_record += 1;
                    continue;
                };
                run.out.census.samples_computed += plan.np2;
                run.out.census.samples_accumulated += placement.count;

                let angles = RadiationAngles {
                    strike_rad: segment.angles.strike_rad,
                    dip_rad: segment.angles.dip_rad,
                    rake_rad: segment.angles.rake_rad,
                    azimuth_rad: source.geometry.azimuth_rad,
                    takeoff_rad: ray.takeoff_rad,
                };
                self.synthesise(
                    &mut rng,
                    &mut scratch,
                    plan,
                    &source,
                    &ray,
                    &angles,
                    &run.vertical,
                );
                run.out.accumulate(
                    scratch.contribution.slice(s![.., ..plan.np2]),
                    source.weight,
                    &placement,
                );
            }
        }
    }

    /// Gather one subfault as a source, or `None` when its moment is too small to radiate.
    ///
    /// Draws exactly one normal deviate, and only when the rupture-velocity sigma is
    /// non-zero; see [`RuptureVelocityTaper::perturbed_factor`]. It comes from the subfault's
    /// own stream, so it is a function of which subfault this is rather than of how many ran
    /// before it.
    fn subfault_source(
        &self,
        rng: &mut impl Draws,
        segment: &SegmentSource,
        geometry: &SubfaultGeometry,
        windows: &WindowLengths,
        along_strike: usize,
        down_dip: usize,
    ) -> Option<SubfaultSource> {
        let grid = segment.segment.grid_index(along_strike, down_dip);
        let weight = segment.weights[grid];
        if weight < SUBFAULT_WEIGHT_THRESHOLD {
            return None;
        }

        let geometry = geometry.at(along_strike, down_dip);
        let layer = layer_containing(&self.vmod, geometry.depth_km);
        let shear_velocity_km_s = self.vmod[layer].vsh_km_s as f32;
        let rupture_fraction = self.rupture.perturbed_factor(rng, geometry.depth_km);

        Some(SubfaultSource {
            geometry,
            weight,
            rupture_time_s: segment.segment.at(along_strike, down_dip).rupture_time_s,
            window_s: windows.window_s[grid],
            layer,
            shear_velocity_km_s,
            density_g_cm3: self.vmod[layer].density_g_cm3 as f32,
            // Graves & Pitarka (2010) eq. 13, with this subfault's perturbed rupture speed.
            // See `window_lengths` for why the association differs from the one there.
            corner_frequency_hz: segment.angles.corner_coeff
                * rupture_fraction
                * shear_velocity_km_s
                / self.scale.avg_subfault_km
                / PI,
        })
    }

    /// Synthesise one `(subfault, ray)`'s three components into `scratch.contribution`.
    ///
    /// The draw order is the answer: three phase realisations in component order, each `np2`
    /// normals, then the two horizontal radiation averages, 5,000 uniforms each.
    #[allow(clippy::too_many_arguments)]
    fn synthesise(
        &self,
        rng: &mut impl Draws,
        scratch: &mut SegmentScratch,
        plan: &SpectrumPlan,
        source: &SubfaultSource,
        ray: &TracedPath,
        angles: &RadiationAngles,
        vertical: &VerticalUniforms,
    ) {
        let (np2, fold_count) = (plan.np2, plan.fold_count);

        // Three phase realisations in component order: each draws `np2` normal deviates, and
        // each writes a full row, so there is nothing to reset between rays.
        //
        // The shape behind them is rebuilt only when `f_max` changes it, which means twice at
        // most and — whenever `f_max` is already at or below the vertical's ceiling — once.
        // `capped_fmax` is the only per-component input to it.
        let mut built_for = f32::NAN;
        for component in Component::ALL {
            let fmax_hz = component.capped_fmax(self.fmax_hz);
            // NaN compares unequal to everything, so the first component always builds.
            if fmax_hz != built_for {
                scratch.shape.refresh(
                    plan,
                    &self.spectrum,
                    &SpectrumInputs {
                        distance_km: ray.path_length_km,
                        window_s: source.window_s,
                        shear_velocity_km_s: source.shear_velocity_km_s,
                        density_g_cm3: source.density_g_cm3,
                        corner_frequency_hz: source.corner_frequency_hz,
                        fmax_hz,
                        qbar: ray.qbar,
                    },
                );
                built_for = fmax_hz;
            }
            stochastic_spectrum(
                rng,
                plan,
                &mut scratch.shape,
                scratch
                    .spectrum
                    .row_mut(component.index())
                    .slice_mut(s![..np2]),
            );
        }

        // Quarter-wavelength site amplification is always applied. The curve depends only on
        // the source layer and the transform length, so it is built once for the three
        // components.
        self.site.gain_curve(
            &self.vmod,
            source.layer,
            plan.log_frequency_hz.view(),
            scratch.site_gain.slice_mut(s![..fold_count]),
        );
        for mut row in scratch.spectrum.rows_mut() {
            apply_site_amplification(
                row.slice_mut(s![..np2])
                    .into_slice()
                    .expect("a prefix of a row of a C-order Array2 is contiguous"),
                scratch.site_gain.slice(s![..fold_count]),
            );
        }

        // The only difference between the three components is which radiation routine runs: a
        // horizontal draws 5,000 deviates, the vertical draws none and reads the pre-filled
        // tables instead.
        for component in Component::ALL {
            let radiation = scratch.radiation.slice_mut(s![..fold_count]);
            match component.radiation_mode() {
                RadiationMode::Horizontal { azimuth_offset_deg } => horizontal_radiation_spectrum(
                    rng,
                    angles,
                    plan.frequency_hz.view(),
                    azimuth_offset_deg.to_radians(),
                    CONICAL_SAMPLE_COUNT,
                    radiation,
                ),
                RadiationMode::Vertical => vertical_radiation_spectrum(
                    angles,
                    plan.frequency_hz.view(),
                    &vertical.uniform_a,
                    &vertical.uniform_b,
                    CONICAL_SAMPLE_COUNT,
                    radiation,
                ),
            };
            // In place on both rows: the inverse transform overwrites the spectrum, and the
            // real samples land straight in the contribution's row.
            radiate_and_invert(
                scratch
                    .spectrum
                    .row_mut(component.index())
                    .slice_mut(s![..np2]),
                scratch.radiation.slice(s![..fold_count]),
                plan.taper.view(),
                scratch
                    .contribution
                    .row_mut(component.index())
                    .slice_mut(s![..np2]),
            );
        }
    }
}

/// Every segment must agree with the first on subfault size.
fn check_subfault_sizes(slip: &SlipModel) -> Result<(), SimError> {
    let Some((reference, rest)) = slip.segments.split_first() else {
        return Ok(());
    };
    for (index, segment) in rest.iter().enumerate() {
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
                    // 0-based, matching `segments`.
                    segment: index + 1,
                    dimension,
                    found,
                    expected,
                });
            }
        }
    }
    Ok(())
}

/// What one station's run carries from segment to segment.
struct StationRun {
    /// The station's seed, from which every sub-stream is derived.
    seed: u64,
    vertical: VerticalUniforms,
    plans: PlanCache,
    /// The record being accumulated, and its reports.
    out: Simulation,
}

/// The vertical component's pre-drawn uniform tables: the only draws taken from the station's
/// own stream.
///
/// The order and the counts are part of the output: seed, then [`CONICAL_SAMPLE_COUNT`]
/// uniforms into `a`, then as many into `b`. They must stay two sequential fills; interleaving
/// them into one pass would put different deviates in different slots and change every vertical
/// component.
struct VerticalUniforms {
    uniform_a: Vec<f32>,
    uniform_b: Vec<f32>,
}

impl VerticalUniforms {
    fn draw<D: Draws>(seed: u64) -> Self {
        let mut rng = D::from_seed(seed);
        let mut uniform_a = vec![0.0f32; CONICAL_SAMPLE_COUNT];
        let mut uniform_b = vec![0.0f32; CONICAL_SAMPLE_COUNT];
        rng.fill_uniform(&mut uniform_a);
        rng.fill_uniform(&mut uniform_b);
        Self {
            uniform_a,
            uniform_b,
        }
    }
}

/// What the window-length pass produces for one segment.
struct WindowLengths {
    /// One entry per subfault, indexed by [`Segment::grid_index`].
    window_s: Vec<f32>,
    /// Longest window over the segment; sizes the transform.
    longest_s: f32,
}

/// The buffers one segment's subfaults are synthesised in.
///
/// Sized for the longest window in the segment, then used a prefix at a time: each subfault
/// works in `spectrum[.., ..np2]` for its own `np2`, so there is one allocation per segment.
/// The tail beyond a subfault's own prefix holds an earlier subfault's numbers. Every read goes
/// through a view sliced to the current `np2`, so the stale tail is unreachable.
struct SegmentScratch {
    spectrum: Array2<Complex32>,
    /// One subfault's three components, before weighting.
    contribution: Array2<f32>,
    radiation: Array1<f32>,
    /// The site gain resampled onto this subfault's frequency axis — see
    /// [`crate::site::site_gain_curve`] for why it is built once per (subfault, ray) rather
    /// than once per component.
    site_gain: Array1<f32>,
    shape: SpectrumShape,
    ray_state: RayState,
}

impl SegmentScratch {
    fn for_plan(longest: &SpectrumPlan) -> Self {
        let (np2_max, fold_max) = (longest.np2, longest.fold_count);
        Self {
            spectrum: Array2::zeros((COMPONENT_COUNT, np2_max)),
            contribution: Array2::zeros((COMPONENT_COUNT, np2_max)),
            radiation: Array1::zeros(fold_max),
            site_gain: Array1::zeros(fold_max),
            shape: SpectrumShape::with_capacity(np2_max),
            ray_state: RayState::default(),
        }
    }
}

/// One subfault as a source: where it is, how strong it is, and the medium it sits in.
///
/// Everything here is fixed before a ray path is chosen, so the ray loop reads as a pipeline:
/// trace, place, synthesise, accumulate.
struct SubfaultSource {
    /// Source-to-station geometry.
    geometry: SubfaultRay,
    /// This subfault's share of the total moment.
    weight: MomentWeight,
    /// Rupture arrival at the subfault, s after origin.
    rupture_time_s: f32,
    /// Shaping-window length, Boore (1983) `T_w`.
    window_s: f32,
    /// Velocity-model layer the subfault sits in.
    layer: usize,
    /// `β` and `ρ` at the subfault, not at the station.
    shear_velocity_km_s: f32,
    density_g_cm3: f32,
    /// `f_ci`, Graves & Pitarka (2010) eq. 13, carrying this subfault's perturbed rupture
    /// speed.
    corner_frequency_hz: f32,
}

impl SubfaultSource {
    /// The earliest sample any ray from this subfault could start on.
    ///
    /// # Why this is exact rather than heuristic
    ///
    /// The real start is formed from the traced travel time by the same
    /// [`SampleGrid::start_sample`]. No traced path is shorter than the straight line between
    /// its endpoints, and no medium on it is faster than the fastest layer in the model, so
    /// `slant_km / max_shear_velocity` is a lower bound on that travel time. Truncation toward
    /// zero is monotonic, so the bound survives it, and the envelope lead-in `ε·window` is
    /// subtracted identically in both. A contribution ruled out here is ruled out for real:
    /// this prunes, it does not approximate.
    ///
    /// # What it saves
    ///
    /// It gates the ray tracing, not just the synthesis. Tracing runs an iterative root-find
    /// for the stationary ray parameter and a per-layer spreading integral, and a contribution
    /// that cannot reach the record needs neither. Two divisions decide it.
    fn earliest_start_sample(&self, simulator: &Simulator) -> i32 {
        let travel_time_s = self.geometry.slant_km / simulator.max_shear_velocity_km_s;
        let window_start_s =
            travel_time_s - simulator.spectrum.window.peak_fraction * self.window_s;
        simulator
            .grid
            .start_sample(self.rupture_time_s, window_start_s)
    }
}
