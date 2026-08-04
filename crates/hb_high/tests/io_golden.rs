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

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

struct Reader {
    buf: Vec<u8>,
    pos: usize,
}

impl Reader {
    fn open(name: &str) -> Self {
        let path = root().join("harness/golden/io").join(name);
        let buf = std::fs::read(&path).unwrap_or_else(|e| {
            panic!("reading {}: {e}. Run harness/kernels/gen_io_golden.sh", path.display())
        });
        Self { buf, pos: 0 }
    }
    fn f32(&mut self) -> f32 {
        let v = f32::from_le_bytes(self.buf[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        v
    }
    fn f64(&mut self) -> f64 {
        let v = f64::from_le_bytes(self.buf[self.pos..self.pos + 8].try_into().unwrap());
        self.pos += 8;
        v
    }
    fn i32(&mut self) -> i32 {
        let v = i32::from_le_bytes(self.buf[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        v
    }
    fn chars(&mut self, n: usize) -> String {
        let s = String::from_utf8_lossy(&self.buf[self.pos..self.pos + n]).to_string();
        self.pos += n;
        s
    }
    fn assert_exhausted(&self) {
        assert_eq!(self.pos, self.buf.len(),
                   "consumed {} of {} bytes; record layout disagrees with the driver",
                   self.pos, self.buf.len());
    }
}

#[track_caller]
fn eq32(what: &str, got: f32, want: f32) {
    assert_eq!(got.to_bits(), want.to_bits(),
               "{what}: rust {got:?} (0x{:08x}) vs fortran {want:?} (0x{:08x})",
               got.to_bits(), want.to_bits());
}

#[track_caller]
fn eq64(what: &str, got: f64, want: f64) {
    assert_eq!(got.to_bits(), want.to_bits(),
               "{what}: rust {got:?} vs fortran {want:?}");
}

fn check(golden: &str, stoch_name: &str) {
    let mut r = Reader::open(golden);
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
    for i in 0..j0 { eq32(&format!("depth_km[{}]", i + 1), v.depth_km[i], r.f32()); }
    for i in 0..j0 { eq32(&format!("thickness_km[{}]", i + 1), v.thickness_km[i], r.f32()); }
    for i in 0..j0 { eq64(&format!("vp_km_s[{}]", i + 1), v.vp_km_s[i], r.f64()); }
    for i in 0..j0 { eq64(&format!("vsh_km_s[{}]", i + 1), v.vsh_km_s[i], r.f64()); }
    for i in 0..j0 { eq64(&format!("density_g_cm3[{}]", i + 1), v.density_g_cm3[i], r.f64()); }
    for i in 0..j0 { eq32(&format!("attenuation_p[{}]", i + 1), v.attenuation_p[i], r.f32()); }
    for i in 0..j0 { eq32(&format!("attenuation_s[{}]", i + 1), v.attenuation_s[i], r.f32()); }

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
