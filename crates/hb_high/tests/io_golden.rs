//! Bit-identity gate for the input readers.
//!
//! `harness/kernels/io_driver.f` contains the read blocks copied **verbatim**
//! from the vendored original, so this checks the port against what the Fortran
//! actually parses. The unit tests in `input.rs` only check it against my
//! reading of the format, which is a weaker claim.
//!
//! Regenerate with `harness/kernels/gen_io_golden.sh`.

use hb_high::input::{insert_air_layer, read_stations, read_stoch, read_velocity_model, Subfault};
use hb_high::state::VelocityModelInput;
use std::path::PathBuf;

mod common;
use common::*;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn check(golden: &str, stoch_name: &str) {
    let mut r = Golden::open("io", golden);
    let pu = hb_high::config::DEG_TO_RAD;

    let stoch_text = std::fs::read_to_string(
        root().join("harness/fixtures/stoch").join(stoch_name),
    )
    .unwrap();
    let m = read_stoch(&stoch_text, pu).unwrap();

    let want_nevnt = r.i32() as usize;
    let want_nstot = r.i32() as usize;
    assert_eq!(m.segments.len(), want_nevnt, "nevnt");
    assert_eq!(m.subfault_count, want_nstot, "nstot");
    eq32("farea_in", m.fault_area_km2, r.f32());
    eq32("zhyp_max", m.max_hypocentre_depth_km, r.f32());

    for (k, s) in m.segments.iter().enumerate() {
        assert_eq!(s.along_strike_count, r.i32() as usize, "seg {k} nx");
        assert_eq!(s.down_dip_count, r.i32() as usize, "seg {k} nw");
        eq32(&format!("seg {k} elonq"), s.fault_lon_deg, r.f32());
        eq32(&format!("seg {k} elatq"), s.fault_lat_deg, r.f32());
        eq32(&format!("seg {k} dx"), s.subfault_length_km, r.f32());
        eq32(&format!("seg {k} dw"), s.subfault_width_km, r.f32());
        eq32(&format!("seg {k} strq"), s.strike_deg, r.f32());
        eq32(&format!("seg {k} dipq"), s.dip_deg, r.f32());
        eq32(&format!("seg {k} rakeq"), s.rake_deg, r.f32());
        eq32(&format!("seg {k} dtop"), s.top_depth_km, r.f32());
        eq32(&format!("seg {k} shyp"), s.hypocentre_along_strike_km, r.f32());
        eq32(&format!("seg {k} dhyp"), s.hypocentre_down_dip_km, r.f32());
        eq32(&format!("seg {k} astop"), s.along_strike_offset_km, r.f32());
        // Driver dump order: ((arr(iv,i,j), i=1,nx), j=1,nw)
        type Field = (&'static str, fn(&Subfault) -> f32);
        let fields: [Field; 3] = [
            ("sddp", |sub| sub.slip),
            ("rist", |sub| sub.rise_time_s),
            ("rupt", |sub| sub.rupture_time_s),
        ];
        for (name, field) in fields {
            for j in 1..=s.down_dip_count {
                for i in 1..=s.along_strike_count {
                    eq32(&format!("seg {k} {name}({i},{j})"), field(&s.at(i, j)), r.f32());
                }
            }
        }
    }

    // Velocity model, including the air-layer insertion.
    let vel_text =
        std::fs::read_to_string(root().join("harness/fixtures/velocity_model")).unwrap();
    let mut v = VelocityModelInput::new();
    let j0 = read_velocity_model(&vel_text, &mut v, 999.9).unwrap();
    let (j0, nlskip) = insert_air_layer(&mut v, j0, -99);

    let want_j0 = r.i32() as usize;
    let want_nlskip = r.i32();
    assert_eq!(j0, want_j0, "j0 after Moho truncation and air layer");
    assert_eq!(nlskip, want_nlskip, "nlskip");
    // Layers are 0-based since §2.3; the golden's dump order is unchanged, so `i` here is
    // the storage index and `i + 1` is the Fortran layer number the label reports.
    for i in 0..j0 { eq32(&format!("depth_km[{}]", i + 1), v[i].depth_km, r.f32()); }
    for i in 0..j0 { eq32(&format!("thickness_km[{}]", i + 1), v[i].thickness_km, r.f32()); }
    for i in 0..j0 { eq64(&format!("vp_km_s[{}]", i + 1), v[i].vp_km_s, r.f64()); }
    for i in 0..j0 { eq64(&format!("vsh_km_s[{}]", i + 1), v[i].vsh_km_s, r.f64()); }
    for i in 0..j0 { eq64(&format!("density_g_cm3[{}]", i + 1), v[i].density_g_cm3, r.f64()); }
    for i in 0..j0 { eq32(&format!("attenuation_p[{}]", i + 1), v[i].attenuation_p, r.f32()); }
    for i in 0..j0 { eq32(&format!("attenuation_s[{}]", i + 1), v[i].attenuation_s, r.f32()); }

    // Station list.
    let st_text =
        std::fs::read_to_string(root().join("harness/golden/io/stations.ll")).unwrap();
    let stations = read_stations(&st_text, 3).unwrap();
    let want_head = r.i32();
    assert_eq!(want_head, 2, "the fixture has two comment header lines");
    assert_eq!(stations.len(), 3);
    for (k, s) in stations.iter().enumerate() {
        eq32(&format!("station {k} lon"), s.stlon, r.f32());
        eq32(&format!("station {k} lat"), s.stlat, r.f32());
        // character*12, blank-padded by the Fortran.
        let want = r.chars(12);
        assert_eq!(s.cap, want.trim_end(), "station {k} name");
    }

    r.assert_exhausted();
}

#[test]
fn readers_match_fortran_on_the_minimal_fault() {
    check("mini.bin", "2012p578973.stoch");
}

#[test]
fn readers_match_fortran_on_the_alpine_fault() {
    // 257 along strike by 11 down dip -- 2827 subfaults, and the case where a compact
    // layout matters: the Fortran's sddp/rist/rupt are 720 MB of mostly-unused array.
    check("alpine.bin", "alpine_base_r1.stoch");
}
