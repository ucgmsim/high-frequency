//! `hb_high` command-line driver.
//!
//! Everything here is about the *interface* the Fortran happens to have: a 22-line
//! list-directed deck on stdin, three input files named inside it, `ndata * 3`
//! interleaved `f32` written to a file at a byte offset, and one `(1x,f10.4)`
//! distance per station on stderr. The simulation itself is `hb_high::sim::simulate`.
//!
//! Keeping this split matters for two reasons. It is what makes the crate usable as
//! a library (and so wrappable from Python without going through a temp file), and
//! `read_deck` is the only remaining tie to the Fortran oracle — `harness/run_parity.sh`
//! drives this binary with generated decks, which is the gate every Stage 1 change
//! is verified against.

use std::io::Write;

use hb_high::config::{
    HfConfig, PathDurationModel, RayType, RuptureVelocity, StressParamAdjust, DEG_TO_RAD,
};
use hb_high::deck::ListReader;
use hb_high::input::{read_stations, read_stoch, read_velocity_model};
use hb_high::sim::simulate;
use hb_high::state::VelocityModelInput;



/// The parts of the deck that say where data comes from and where it goes, as
/// opposed to what to simulate. Everything physical lives in [`HfConfig`].
///
/// This split is the point of the exercise: `HfConfig` is what a library caller or
/// the future Python wrapper supplies, while `DeckIo` is an artifact of being driven
/// by a text deck that writes to a file.
struct DeckIo {
    /// `asite` — station list path.
    station_file: String,
    /// Opened **without** truncation and seeked into, so several invocations can
    /// fill disjoint ranges of one shared file. See `PORTING_RULES.md` §9.
    output: String,
    slip_model: String,
    velocity_model: String,
    /// Always 1 in production — `hf_sim.py` runs one process per station. Anything
    /// else is refused; see `run`.
    nsite: usize,
    seek_bytes: i64,
    /// `ift`. Non-zero would call `filter3d`, which is dead here.
    iftt: i32,
    /// `fhi` — reaches only the `fhigh must be <` warning.
    fhil: f32,
}



/// Read the deck, in the Fortran's order.
///
/// Loading the files it names (`slip_model`, `velfile`, `asite`) is deliberately
/// *not* done here. The Fortran interleaves those reads with the deck reads, but
/// nothing in deck parsing depends on their contents, so hoisting the whole deck
/// ahead of them is behaviour-preserving and makes both halves legible.
fn read_deck(
    deck: &mut ListReader,
) -> Result<(HfConfig, DeckIo), Box<dyn std::error::Error>> {
    let stress_average = deck.f32()?;
    let asite = deck.read_filename()?;
    let outname = deck.read_filename()?;

    // `read(5,*) nrtyp,(irtype(i),i=1,nrtyp)` is ONE read whose length depends on its own
    // first item.
    let (nrtyp, irtype_items) = deck.read_count_and_list()?;
    let rayset = irtype_items
        .iter()
        .take(nrtyp.max(0) as usize)
        .map(|item| Ok(RayType(hb_high::deck::parse_i32(item.as_deref().unwrap_or(""))?)))
        .collect::<Result<Vec<_>, hb_high::deck::DeckError>>()?;

    let isite_amp = deck.i32()?;

    let (iftt, fhil) = {
        let mut r = deck.read_items(4)?;
        r.skip(1); // nbu
        let iftt = r.i32()?;
        r.skip(1); // flol -- clobbered by the radiation routines and only read by filter3d
        (iftt, r.f32()?)
    };

    let irand = deck.i32()?;
    let nsite = deck.i32()? as usize;

    let (duration, dt, fmx, akapp, qfexp) = {
        let mut r = deck.read_items(5)?;
        (r.f32()?, r.f32()?, r.f32()?, r.f32()?, r.f32()?)
    };

    let (rupture_velocity, czero, calpha) = {
        let mut r = deck.read_items(5)?;
        (
            RuptureVelocity {
                frac: r.defaulted_f32()?,
                shallow: r.defaulted_f32()?,
                deep: r.defaulted_f32()?,
            },
            r.defaulted_f32()?,
            r.defaulted_f32()?,
        )
    };

    let (moment, rupture_velocity_override) = {
        let mut r = deck.read_items(2)?;
        // Note the moment's convention differs from `derived`: the Fortran tests
        // `sm < 0.0`, so an explicit zero is honoured rather than derived.
        (r.derived_allowing_zero_f32()?, r.derived_f32()?)
    };

    let slip_model = deck.read_filename()?;
    let velocity_model = deck.read_filename()?;
    let vsmoho = deck.f64()?; // pre-set to -1.0, then read
    let vs_moho = if vsmoho <= 0.0 { None } else { Some(vsmoho) };

    let nl_skip = {
        let mut r = deck.read_items(6)?;
        let nlskip = r.i32()?;
        // Velocity-model perturbation sigmas and icflag: all dead under the production
        // deck (every sigma is 0.0), so read past them. The original also normalises
        // icflag to 1 when it is neither 0 nor 1, which cannot matter once nothing reads
        // it.
        r.skip(5);
        nlskip
    };
    let _velname = deck.read_filename()?;

    let (fa_sig1, fa_sig2, rv_sig1) = {
        let mut r = deck.read_items(3)?;
        (r.f32()?, r.f32()?, r.f32()?)
    };

    let ipdur_model = deck.i32()?;
    let path_duration = PathDurationModel::from_deck(ipdur_model).ok_or_else(|| {
        format!(
            "ipdur_model = {ipdur_model} leaves ndur undefined in the Fortran \
             (accepted values are <=0, 1, 2, 11, 12)"
        )
    })?;

    // NOTE: this read wants THREE items and the deck supplies a bare `0` line followed by
    // three more, so it takes the `0` as ispar_adjust and two from the next record --
    // leaving the third to be discarded when the next read starts a fresh record. That is
    // why production's tect_type never arrives and why targ_mag and fault_area are
    // swapped relative to what hf_sim.py intends. Reproduced deliberately; see
    // REFACTOR.md §1.1.
    let (stress_param_adjust, target_magnitude, fault_area) = {
        let mut r = deck.read_items(3)?;
        (
            StressParamAdjust::from_deck(r.i32()?),
            r.derived_f32()?,
            r.derived_f32()?,
        )
    };

    let seek_bytes = {
        // Pre-set to 0, then read.
        let mut r = deck.read_items(1)?;
        r.f64().map(|x| x as i64).unwrap_or(0)
    };

    Ok((
        HfConfig {
            stress_drop: stress_average,
            rayset,
            site_amp: isite_amp != 0,
            seed: irand,
            duration,
            dt,
            fmax: fmx,
            kappa: akapp,
            qfexp,
            rupture_velocity,
            czero,
            calpha,
            moment,
            rupture_velocity_override,
            vs_moho,
            nl_skip,
            fa_sig1,
            fa_sig2,
            rv_sig1,
            path_duration,
            stress_param_adjust,
            target_magnitude,
            fault_area,
        },
        DeckIo {
            station_file: asite,
            output: outname,
            slip_model,
            velocity_model,
            nsite,
            seek_bytes,
            iftt,
            fhil,
        },
    ))
}

fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("hb_high: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut stdin = String::new();
    std::io::Read::read_to_string(&mut std::io::stdin(), &mut stdin)?;
    let (config, io) = read_deck(&mut ListReader::new(&stdin))?;

    if io.nsite != 1 {
        // The Fortran's station loop shares one generator: `uniform_deviates` fills the
        // `vertical_radiation_spectrum` uniforms once before it, and each station's
        // `normal_deviates` draw continues from wherever the previous station
        // left the stream. A multi-station run is therefore NOT a concatenation of
        // single-station runs, and `simulate` -- one station, seeded from
        // `config.seed` -- cannot express it.
        //
        // Production always writes nsite = 1 (`hf_sim.py` runs one process per
        // station) and no parity deck uses anything else, so this is refused rather
        // than silently computed differently.
        return Err(format!(
            "nsite = {} is not supported: the station loop shares one RNG stream, so \
             stations are not independent. Production runs one station per process. \
             See REFACTOR.md §1.1.",
            io.nsite
        )
        .into());
    }

    let slip = {
        let text = std::fs::read_to_string(&io.slip_model)
            .map_err(|e| format!("opening slip model {}: {e}", io.slip_model))?;
        read_stoch(&text, DEG_TO_RAD)?
    };

    let mut vmod_in = VelocityModelInput::new();
    let j0 = {
        let text = std::fs::read_to_string(&io.velocity_model)
            .map_err(|e| format!("opening velocity model {}: {e}", io.velocity_model))?;
        read_velocity_model(&text, &mut vmod_in, config.vs_moho())?
    };

    let stations = {
        let text = std::fs::read_to_string(&io.station_file)
            .map_err(|e| format!("opening station file {}: {e}", io.station_file))?;
        read_stations(&text, io.nsite)?
    };

    if io.iftt > 0 {
        return Err(format!(
            "ift = {} would call filter3d, which is not ported: it is dead under the \
             production BINMOD/VERSION1 configuration.",
            io.iftt
        )
        .into());
    }
    if io.fhil > 1.0 / 2.0 / config.dt {
        eprintln!(
            "fhigh must be < {} {} {}",
            1.0 / 2.0 / config.dt,
            io.fhil,
            config.dt
        );
    }

    let mut out = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&io.output)
        .map_err(|e| format!("opening output {}: {e}", io.output))?;
    // No status= in the Fortran open, so an existing file is NOT truncated;
    // combined with the seek this lets several invocations fill disjoint ranges of
    // one shared file. See PORTING_RULES.md §9.
    std::io::Seek::seek(&mut out, std::io::SeekFrom::Start(io.seek_bytes as u64))?;

    let mut stderr = std::io::stderr();
    for station in stations {
        // The Fortran's "np2 > mm, need to recompile with larger array size" exit is
        // gone: §2.6b sizes the buffers from the deck, so there is no compiled ceiling
        // left to exceed.
        let sim = simulate(&config, &slip, &vmod_in, j0, station)?;

        // Streamed through a BufWriter rather than materialised: the previous form built
        // a second Vec the size of the whole record -- 12000 samples x 3 x 4 bytes for a
        // 20 s record at dt=0.005 -- allocated fresh inside the station loop, purely to
        // hand it to one write_all. `to_le_bytes` per sample keeps the byte order
        // explicit rather than depending on the host's.
        let mut writer = std::io::BufWriter::new(&mut out);
        for v in &sim.acc {
            writer.write_all(&v.to_le_bytes())?;
        }
        writer.flush()?;
        drop(writer);

        // `(1x,f10.4)`: a leading blank from the 1x, THEN a width-10 field.
        // Dropping the 1x would give 10 characters instead of 11 -- harmless to
        // hf_sim.py, which does float(stderr.strip()), but still an interface
        // difference, and the parity gate flags it.
        writeln!(stderr, " {:>10.4}", sim.d10_km)?;
    }

    Ok(())
}
