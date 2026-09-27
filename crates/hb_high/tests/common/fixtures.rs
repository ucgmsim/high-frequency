//! Fixed simulation inputs shared by the snapshot test, the property tests and the
//! whole-program bench, so the bench times the same workload the snapshot pins.
//!
//! Included by path rather than through `common`, so the bench does not link the golden
//! readers.

// Each including binary uses a different subset.
#![allow(dead_code)]

use hb_high::config::{
    HfConfig, PathParameters, RecordParameters, RuptureVelocity, SiteParameters, SourceParameters,
};
use hb_high::path_duration::PathDurationModel;
use hb_high::ray::{RayShape, RayType};
use hb_high::slip_model::{Segment, Slip, SlipModel, Subfault};
use hb_high::velocity::{InputLayer, VelocityModelInput, build_velocity_model};

/// The direct upgoing ray — what production runs.
pub const DIRECT_RAY: RayType = RayType::Traced(RayShape::Upgoing { multiples: 0 });

/// A uniform-slip single-segment fault, `along` by `down` subfaults.
pub fn uniform_fault(along: usize, down: usize) -> SlipModel {
    let segment = Segment::builder()
        .fault_lon_deg(173.0)
        .fault_lat_deg(-43.0)
        .along_strike_count(along)
        .down_dip_count(down)
        .subfault_length_km(1.5)
        .subfault_width_km(1.5)
        .strike_deg(220.0)
        .dip_deg(60.0)
        .rake_deg(160.0)
        .top_depth_km(1.0)
        .hypocentre_along_strike_km(0.0)
        .hypocentre_down_dip_km(1.5)
        .subfaults(vec![
            Subfault {
                slip: Slip(50.0),
                rise_time_s: 0.5,
                rupture_time_s: 0.0
            };
            along * down
        ])
        .build();
    SlipModel::new(vec![segment])
}

/// A smoothly graded crustal model: thin slow layers near the surface, thickening and speeding
/// up with depth, zero-thickness base as the reader expects. The thin first layer makes
/// `insert_air_layer` fire as it does on every production model.
pub fn crustal_model(layers: usize) -> VelocityModelInput {
    let built: Vec<InputLayer> = (0..layers)
        .map(|k| {
            let frac = k as f64 / (layers - 1) as f64;
            let vsh_km_s = 0.5 + 4.1 * frac;
            let qs = 50.0 + 150.0 * frac;
            InputLayer {
                // Derived by build_velocity_model, which accumulates it down the column.
                depth_km: 0.0,
                thickness_km: if k == layers - 1 {
                    0.0
                } else {
                    (0.05 + 3.0 * frac) as f32
                },
                vp_km_s: vsh_km_s * 1.75,
                vsh_km_s,
                density_g_cm3: 1.81 + 1.5 * frac,
                attenuation_p: (2.0 * qs) as f32,
                attenuation_s: qs as f32,
            }
        })
        .collect();
    build_velocity_model(&built, 999.9).expect("valid velocity model")
}

/// The production configuration, at a chosen record length.
pub fn production_config(duration_s: f32) -> HfConfig {
    HfConfig {
        source: SourceParameters {
            stress_drop_bars: 50.0,
            corner_frequency_constant: 2.0,
            corner_frequency_alpha: 0.1,
            rupture_velocity: RuptureVelocity {
                fraction: 0.8,
                shallow_factor: 0.6,
                deep_factor: 0.6,
                sigma: 0.1,
            },
        },
        path: PathParameters {
            rayset: vec![DIRECT_RAY],
            q_frequency_exponent: 0.6,
            path_duration: PathDurationModel::Gp2010,
        },
        site: SiteParameters {
            kappa_s: 0.045,
            fmax_hz: 10.0,
        },
        record: RecordParameters {
            duration_s,
            dt_s: 0.005,
        },
    }
}
