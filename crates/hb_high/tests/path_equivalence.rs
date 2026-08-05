//! The deck path and the array path must produce identical waveforms.
//!
//! # Why this replaces a statistical certification
//!
//! §4.2 was planned as one more LONG campaign against production Fortran, to certify the
//! per-station seeding change. That turned out to be the wrong instrument for two reasons.
//!
//! The seeding widening from `i32` to `u64` is **bit-exact for deck-sourced seeds** — the
//! `run_cheap.sh` replay gate confirms 22/22 decks within 1e-6 — so a LONG run would spend
//! an hour and a half re-certifying a stream Stage 3 already certified, byte for byte. And
//! the thing that genuinely is new, per-station independence, **cannot be exercised through
//! a deck at all**: `nsite != 1` was refused precisely because the Fortran's station loop
//! shared one generator.
//!
//! So the real question is not "does the port still match the Fortran" — Stage 3 answered
//! that at n=5100 with 373/375 certified and 0 refuted, beside a null control. It is "does
//! the array path compute the same thing the certified deck path computes". With a fixed
//! draw source that is a **bit-exact** question, which is a far stronger claim than any
//! distributional test could make, and it costs milliseconds instead of an hour.
//!
//! Certification then transfers: deck path ≡ Fortran (Stage 3, statistically), array path ≡
//! deck path (here, exactly), therefore array path ≡ Fortran.
//!
//! # What this licenses
//!
//! §4.3 deletes `read_stoch` and `read_velocity_model`. This is the test that says the
//! constructors replacing them lose nothing. It should be deleted along with them, having
//! done its job — there will be nothing left to compare against.

use hb_high::config::{
    HfConfig, PathDurationModel, RayType, RuptureVelocity, StressParamAdjust, DEG_TO_RAD,
};
use hb_high::input::{
    build_velocity_model, read_stoch, read_velocity_model, Segment, Station, StochModel,
    Subfault,
};
use hb_high::sim::simulate;
use hb_high::state::{InputLayer, VelocityModelInput};

/// Rebuild a parsed slip model through the array constructors.
///
/// Reads only the public surface a Python caller has — the geometry fields and `at()` —
/// so if that surface cannot express something the reader captured, this fails to
/// reproduce it and the waveform comparison catches it.
fn rebuild_through_constructors(parsed: &StochModel) -> StochModel {
    let segments = parsed
        .segments
        .iter()
        .map(|seg| {
            // 1-based, as `at` documents. Strike index fastest, one row per down-dip
            // index, which is the order the constructor expects and every accumulation
            // over the grid runs in.
            let subfaults: Vec<Subfault> = (1..=seg.down_dip_count)
                .flat_map(|j| (1..=seg.along_strike_count).map(move |i| (i, j)))
                .map(|(i, j)| seg.at(i, j))
                .collect();
            Segment::builder()
                .fault_lon_deg(seg.fault_lon_deg)
                .fault_lat_deg(seg.fault_lat_deg)
                .along_strike_count(seg.along_strike_count)
                .down_dip_count(seg.down_dip_count)
                .subfault_length_km(seg.subfault_length_km)
                .subfault_width_km(seg.subfault_width_km)
                .strike_deg(seg.strike_deg)
                .dip_deg(seg.dip_deg)
                .rake_deg(seg.rake_deg)
                .top_depth_km(seg.top_depth_km)
                .hypocentre_along_strike_km(seg.hypocentre_along_strike_km)
                .hypocentre_down_dip_km(seg.hypocentre_down_dip_km)
                .subfaults(subfaults)
                .build()
        })
        .collect();
    StochModel::new(segments, DEG_TO_RAD)
}

/// Rebuild a parsed velocity model through `build_velocity_model`.
///
/// Note the reader is given `vs_moho` and applies truncation itself, so what comes back is
/// already truncated. Feeding those layers to `build_velocity_model` with a `vs_moho` that
/// cannot truncate further reproduces it without truncating twice.
fn rebuild_velocity_model(parsed: &VelocityModelInput, layer_count: usize) -> (VelocityModelInput, usize) {
    let layers: Vec<InputLayer> = (0..layer_count)
        .map(|i| InputLayer {
            // Derived by the constructor. Left at zero so a constructor that failed to
            // accumulate it cannot pass by copying the reader's answer.
            depth_km: 0.0,
            ..parsed[i]
        })
        .collect();
    let mut rebuilt = VelocityModelInput::new();
    // The reader already truncated, and it zeroed the base layer's thickness while doing
    // so. `build_velocity_model` re-derives depths from thicknesses, so a zero-thickness
    // base is reproduced identically -- but a `vs_moho` low enough to truncate again would
    // shorten the model, hence f64::INFINITY.
    let count = build_velocity_model(&mut rebuilt, &layers, f64::INFINITY)
        .expect("the reader already accepted this model");
    (rebuilt, count)
}

fn production_config(seed: u64, duration: f32) -> HfConfig {
    HfConfig {
        stress_drop: 50.0,
        rayset: vec![RayType(1)],
        site_amp: true,
        seed,
        duration,
        dt: 0.005,
        fmax: 10.0,
        kappa: 0.045,
        qfexp: 0.6,
        rupture_velocity: RuptureVelocity { frac: None, shallow: None, deep: None },
        czero: None,
        calpha: None,
        moment: None,
        rupture_velocity_override: None,
        vs_moho: None,
        // -99, not -1: `insert_air_layer` increments this, and it is a flag not a count.
        nl_skip: -99,
        fa_sig1: 0.0,
        fa_sig2: 0.0,
        rv_sig1: 0.1,
        path_duration: PathDurationModel::Gp2010,
        stress_param_adjust: StressParamAdjust::None,
        target_magnitude: None,
        fault_area: None,
    }
}

/// Every fixture fault, every seed, both paths, bit for bit.
///
/// The faults span the shapes that matter: 4 subfaults, 112, and a 2827-subfault
/// multi-kilometre rupture. `alpine_base_r1` is the one that would catch an error in the
/// aggregate derivations (`fault_area_km2`, `max_hypocentre_depth_km`), because they scale
/// with the segment count and grid size.
#[test]
fn the_array_path_reproduces_the_deck_path_exactly() {
    let velocity_text = std::fs::read_to_string("../../harness/fixtures/velocity_model")
        .expect("fixture velocity model");

    // alpine_base_r1 is 2827 subfaults, np2 = 16384 each: 392 s in release and tens of
    // minutes in debug. Behind SLOW=1, the same convention run_parity.sh uses for the same
    // fixture. Worth having available -- the aggregate derivations (fault_area_km2,
    // max_hypocentre_depth_km) scale with grid size, so a summation-order error there shows
    // on the big fault first, and it HAS passed there. The two small faults are 4 and 112
    // subfaults and run in about a second, which is what belongs in a per-commit gate.
    let mut faults = vec!["2012p578973", "2013p543824"];
    if std::env::var_os("SLOW").is_some() {
        faults.push("alpine_base_r1");
    } else {
        eprintln!("  skip  alpine_base_r1 (set SLOW=1 to include; 2827 subfaults, ~7 min)");
    }

    for fault in faults {
        let stoch_text =
            std::fs::read_to_string(format!("../../harness/fixtures/stoch/{fault}.stoch"))
                .expect("fixture stoch model");

        let from_deck = read_stoch(&stoch_text, DEG_TO_RAD).expect("valid stoch");
        let from_arrays = rebuild_through_constructors(&from_deck);

        // The aggregates first, because a mismatch here explains any waveform difference
        // and is far easier to read than a sample index.
        assert_eq!(
            from_deck.subfault_count, from_arrays.subfault_count,
            "{fault}: subfault_count"
        );
        assert_eq!(
            from_deck.fault_area_km2, from_arrays.fault_area_km2,
            "{fault}: fault_area_km2"
        );
        assert_eq!(
            from_deck.max_hypocentre_depth_km, from_arrays.max_hypocentre_depth_km,
            "{fault}: max_hypocentre_depth_km"
        );

        let mut vmod_deck = VelocityModelInput::new();
        let layer_count = read_velocity_model(&velocity_text, &mut vmod_deck, 999.9)
            .expect("valid velocity model");
        let (vmod_arrays, layer_count_arrays) = rebuild_velocity_model(&vmod_deck, layer_count);
        assert_eq!(layer_count, layer_count_arrays, "{fault}: layer count");

        // The station is placed relative to THIS fault, not at a fixed point. The three
        // fixtures span the country -- 2012p578973 is off East Cape, alpine_base_r1 is in
        // Westland -- so one hardcoded station sits ~1000 km from some of them and the
        // record contains nothing but zeros. The silence guard below caught exactly that.
        let origin = &from_deck.segments[0];
        let (stlon, stlat) = (origin.fault_lon_deg + 0.3, origin.fault_lat_deg + 0.3);

        // Two seeds and two record lengths. One seed could agree by accident on a path
        // that ignores the seed; the longer record reaches arrivals the shorter truncates.
        for (seed, duration) in [(12345u64, 40.0f32), (987654321, 60.0)] {
            let station = Station { stlon, stlat, cap: "TEST".to_string() };
            let deck = simulate(
                &production_config(seed, duration),
                &from_deck,
                &vmod_deck,
                layer_count,
                station.clone(),
            )
            .expect("deck path simulates");
            let arrays = simulate(
                &production_config(seed, duration),
                &from_arrays,
                &vmod_arrays,
                layer_count_arrays,
                station,
            )
            .expect("array path simulates");

            assert_eq!(deck.ndata, arrays.ndata, "{fault} seed {seed}: sample count");
            // A record too short for the arrival is all zeros, and two all-zero records
            // compare equal -- so this assertion would pass vacuously. Rule it out before
            // trusting it. (The 10 s version of this test did exactly that.)
            let peak = deck.acc.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            assert!(
                peak > 0.0,
                "{fault} seed {seed}: the deck path produced silence at {duration} s, so \
                 comparing the two paths proves nothing -- lengthen the record"
            );
            assert_eq!(
                deck.acc, arrays.acc,
                "{fault} seed {seed}: the two paths disagree (peak {peak} cm/s^2)"
            );
        }
    }
}
