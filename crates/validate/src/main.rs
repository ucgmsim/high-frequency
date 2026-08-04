//! Scientific equivalence campaign runner.
//!
//! ```text
//! validate --tier b            paired IMs vs the oracle, matched seeds
//! validate --tier c --cell a   distributional IMs vs production Fortran
//! validate --tier d --cell a   inter-frequency correlation
//! ```
//!
//! Tiers B, C and D must all pass on the current build, because it is bit-identical
//! to the oracle. A failure now means the *measurement* is wrong, not the port —
//! which is the point of running them before anything is optimised.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use im::COMPONENTS;
use validate::campaign::{
    self, DeckTemplate, Fault, Realisation, Stratum, StratumResult, Workspace, ALPINE,
    MEDIUM, MINI,
};
use validate::stats::{self, Equivalence, Verdict};

const DT: f64 = 0.005;
/// Record length comes from the stratum; see `campaign::LADDER` for why it is
/// 100 s rather than the 20 s the parity gate uses.
const DURATION_NOTE: &str = "dt=0.005 s, duration per stratum";

#[derive(Clone, Copy, PartialEq, Debug)]
enum Tier {
    /// Paired, matched seeds, against the oracle. Same RNG, so the same
    /// realisation — near-zero variance and correspondingly sensitive.
    B,
    /// Distributional, against production Fortran. Different RNG, so unpaired.
    C,
    /// Inter-frequency correlation. Needs the same sample as C.
    D,
}

/// One inter-frequency-correlation test, held until the whole family is known.
///
/// Tier D cannot decide pass/fail one test at a time: the decision depends on how
/// many tests are in the family, so every test is collected first and adjudicated
/// once at the end. See `stats::holm_adjusted`.
struct IfcTest {
    label: String,
    component: &'static str,
    bands: usize,
    observed: f64,
    p: f64,
}

struct Cell {
    name: &'static str,
    fault: Fault,
    distances: usize,
    seeds: usize,
    /// Resolution this cell can certify, for honest reporting.
    note: &'static str,
}

const CELL_A: Cell =
    Cell { name: "a", fault: MINI, distances: 5, seeds: 600, note: "+/-2% resolution" };
const CELL_B: Cell =
    Cell { name: "b", fault: MEDIUM, distances: 3, seeds: 200, note: "+/-3.5% resolution" };
const CELL_C: Cell = Cell {
    name: "c",
    fault: ALPINE,
    distances: 1,
    seeds: 20,
    note: "+/-11% -- SMOKE TEST ONLY, not a certification",
};

fn main() -> std::process::ExitCode {
    match run() {
        Ok(true) => std::process::ExitCode::SUCCESS,
        Ok(false) => {
            eprintln!("\nFAIL: at least one endpoint is not equivalent");
            std::process::ExitCode::FAILURE
        }
        Err(e) => {
            eprintln!("validate: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

fn arg(name: &str) -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    a.iter().position(|x| x == name).and_then(|i| a.get(i + 1).cloned())
}

fn run() -> Result<bool, Box<dyn std::error::Error>> {
    let tier = match arg("--tier").as_deref() {
        Some("b") => Tier::B,
        Some("c") => Tier::C,
        Some("d") => Tier::D,
        _ => {
            return Err(
                "usage: validate --tier b|c|d [--cell a|b|c] [--band 0.02] [--seeds N]".into(),
            )
        }
    };
    let cell = match arg("--cell").as_deref().unwrap_or("a") {
        "a" => CELL_A,
        "b" => CELL_B,
        "c" => CELL_C,
        other => return Err(format!("unknown cell {other}").into()),
    };
    let band: f64 =
        arg("--band").and_then(|s| s.parse().ok()).unwrap_or(stats::DEFAULT_BAND);
    let n_seeds: usize = arg("--seeds").and_then(|s| s.parse().ok()).unwrap_or(cell.seeds);

    // A/A calibration: run ONE binary and split its realisations in half. Every
    // flag is then a known false alarm, so the flag rate measures whether the
    // permutation test is calibrated for a max-over-435-pairs statistic. Without
    // this, a Tier D failure cannot be attributed between "the codes differ" and
    // "the test over-rejects".
    let aa = std::env::args().any(|a| a == "--aa");
    if aa && tier != Tier::D {
        return Err("--aa is only meaningful for tier d".into());
    }

    let root = repo_root();
    let rust = root.join("target/release/hb_high");
    // Tier B compares against the ORACLE, which shares the port's PCG32 stream, so
    // matched seeds give matched realisations. Tier C/D compare against PRODUCTION,
    // which uses gfortran's generator -- necessarily unpaired.
    let reference = match tier {
        Tier::B => root.join("reference/build/hb_ref"),
        Tier::C | Tier::D => root.join("reference/build/hb_prod"),
    };
    for p in [&rust, &reference] {
        if !p.exists() {
            return Err(format!(
                "{} is missing -- run harness/build_ref.sh, `cargo build --release`, \
                 and harness/bench_vs_fortran.sh (which builds hb_prod)",
                p.display()
            )
            .into());
        }
    }

    let strata = campaign::distance_ladder(cell.fault, cell.distances);
    let seed_list = campaign::seeds(n_seeds);
    let fas_edges = im::fas::default_bin_edges();

    println!(
        "tier {tier:?}  cell {}  fault {} ({} subfaults)",
        cell.name, cell.fault.name, cell.fault.subfaults
    );
    println!("reference: {}", reference.display());
    println!(
        "{} strata x {} seeds x 2 binaries = {} runs   [{}]",
        strata.len(),
        seed_list.len(),
        2 * strata.len() * seed_list.len(),
        DURATION_NOTE
    );
    println!("band +/-{:.1}%   cell note: {}\n", 100.0 * band, cell.note);

    let mut verdicts: Vec<(String, Equivalence)> = Vec::new();
    let mut all_pass = true;
    let mut rows: Vec<String> = vec![
        "stratum,reported_km,component,endpoint,n_a,n_b,gm_ratio,ci_lo,ci_hi,\
         half_width_ln,sd_ratio,ks,verdict"
            .to_string(),
    ];
    let mut ifc: Vec<IfcTest> = Vec::new();

    for st in &strata {
        let a = collect(&rust, st, &seed_list, &fas_edges, &root, "rust")?;
        // A/A splits `a`, so the reference binary is not run at all -- which also
        // halves the wall clock, and the reference is the slow one (it pays FFTW
        // planning per process).
        let b = if aa {
            None
        } else {
            Some(collect(&reference, st, &seed_list, &fas_edges, &root, "ref")?)
        };
        // Fail loudly on a stratum with no signal rather than reporting NaN
        // endpoints as non-equivalent.
        a.check_has_signal().map_err(|e| format!("rust: {e}"))?;
        if let Some(b) = &b {
            b.check_has_signal().map_err(|e| format!("reference: {e}"))?;
        }
        let km = a.realisations.first().map_or(f64::NAN, |r| r.reported_km);
        let label = format!("{}@{:.0}km", st.fault.name, km);

        if tier == Tier::D {
            for (ci, cname) in COMPONENTS.iter().enumerate() {
                let (bands_a, m_all_a) = a.fas_matrix(ci);

                // In A/A mode both halves come from the SAME binary, so any flag is
                // by construction a false alarm and the flag rate measures the
                // test's calibration rather than the port.
                let (ma, mb) = if aa {
                    let half = m_all_a.len() / 2;
                    (m_all_a[..half].to_vec(), m_all_a[half..].to_vec())
                } else {
                    let (bands_b, m_all_b) =
                        b.as_ref().expect("non-aa mode collects the reference").fas_matrix(ci);
                    // Compare the same band set or not at all. Band occupancy is set
                    // by np2, which is deterministic given the deck and therefore
                    // identical between binaries -- so this should never fire, and if
                    // it does the comparison would be silently misaligned rather than
                    // wrong in an obvious way.
                    if bands_a != bands_b {
                        return Err(format!(
                            "{label} {cname}: the two binaries kept different FAS \
                             bands ({} vs {}), so the correlation matrices are not \
                             comparable column-for-column",
                            bands_a.len(),
                            bands_b.len()
                        )
                        .into());
                    }
                    (m_all_a, m_all_b)
                };

                // Deterministic permutation seed derived from the stratum: the
                // result is reproducible, but not identical across strata.
                let seed = 0x00C0FFEE_u64 ^ ((ci as u64) << 32) ^ (st.nominal_km as u64);
                let (p, observed) = stats::permutation_p_max_delta(&ma, &mb, 200, seed);
                // A LOW p means the two correlation structures differ by more than
                // relabelling explains, so equivalence wants a HIGH p. This is the
                // one place the campaign uses a p-value, and it is calibrating a
                // max statistic over hundreds of correlated pairs, which no closed
                // form covers.
                //
                // The pass/fail decision is deferred: gating on raw p here would give
                // the family a 53.7% false-alarm rate. See stats::holm_adjusted.
                println!(
                    "  {label:16} {cname:4} IFC  bands={:2}  n={:3}v{:3}  \
                     max|dRho|={observed:.4}  p={p:.3}",
                    bands_a.len(),
                    ma.len(),
                    mb.len(),
                );
                ifc.push(IfcTest {
                    label: label.clone(),
                    component: cname,
                    bands: bands_a.len(),
                    observed,
                    p,
                });
            }
            continue;
        }

        let mut stratum_verdicts: Vec<(String, Equivalence)> = Vec::new();
        for (ci, cname) in COMPONENTS.iter().enumerate() {
            let names: Vec<String> = a.realisations[0].measures[ci]
                .endpoints()
                .into_iter()
                .map(|(n, _)| n)
                .collect();
            for name in &names {
                let xa = a.endpoint(ci, name);
                let xb = b
                    .as_ref()
                    .expect("tiers B and C always collect the reference")
                    .endpoint(ci, name);
                let e = match tier {
                    Tier::B => stats::equivalence_paired(&xa, &xb, band),
                    _ => stats::equivalence_unpaired(&xa, &xb, band),
                };
                // The gate is "nothing REFUTED". An undetermined endpoint means the
                // sample could not decide, which is a statement about the sample, not
                // about the port -- see stats::Verdict.
                all_pass &= e.verdict != Verdict::Refuted;
                rows.push(format!(
                    "{label},{km:.4},{cname},{name},{},{},{:.6},{:.6},{:.6},{:.6},\
                     {:.4},{:.4},{}",
                    e.n_a,
                    e.n_b,
                    e.gm_ratio,
                    e.ci_lo,
                    e.ci_hi,
                    e.achieved_half_width,
                    e.sd_ratio,
                    e.ks,
                    e.verdict.as_str()
                ));
                stratum_verdicts.push((format!("{label} {cname} {name}"), e));
            }
        }
        // One line per stratum, not per endpoint: 3 components x 25 endpoints is 75
        // lines per stratum, which would bury the failures.
        if let Some((n, e)) =
            stratum_verdicts.iter().max_by(|x, y| dev(&x.1).partial_cmp(&dev(&y.1)).unwrap())
        {
            println!(
                "  {label:16} worst of {}: {n:38} gm={:.5} CI[{:.5},{:.5}] {}",
                stratum_verdicts.len(),
                e.gm_ratio,
                e.ci_lo,
                e.ci_hi,
                e.verdict.as_str()
            );
        }
        verdicts.extend(stratum_verdicts);
    }

    // Report the ACHIEVED resolution, not the design one. The sample size was sized
    // from sigma ~ 0.2 inferred from 24 samples in the Phase 0c work; if the real
    // scatter is wider, the band we can actually certify is wider too, and saying so
    // is the difference between a measurement and a claim.
    if tier != Tier::D && !verdicts.is_empty() {
        let mut hw: Vec<f64> = verdicts
            .iter()
            .map(|(_, e)| e.achieved_half_width)
            .filter(|v| v.is_finite())
            .collect();
        hw.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let max_hw = hw.last().copied().unwrap_or(f64::NAN);
        let med = hw.get(hw.len() / 2).copied().unwrap_or(f64::NAN);
        println!(
            "\nachieved CI half-width (ln): median {med:.6}, worst {max_hw:.6}  \
             -> resolves +/-{:.3}% typical, +/-{:.3}% worst",
            100.0 * med.exp_m1(),
            100.0 * max_hw.exp_m1()
        );
        let count = |v: Verdict| verdicts.iter().filter(|(_, e)| e.verdict == v).count();
        let (cert, refu, undet) = (
            count(Verdict::Certified),
            count(Verdict::Refuted),
            count(Verdict::Undetermined),
        );
        println!(
            "\nverdicts at +/-{:.1}%:  {cert} certified   {refu} REFUTED   \
             {undet} undetermined   (of {})",
            100.0 * band,
            verdicts.len()
        );

        // Pooled bias across endpoints. Individual endpoints are noisy and highly
        // correlated (neighbouring pSA periods especially), but a systematic
        // difference between the codes would show here even when no single endpoint
        // reaches significance.
        let lg: Vec<f64> = verdicts
            .iter()
            .filter(|(_, e)| e.gm_ratio.is_finite())
            .map(|(_, e)| e.gm_ratio.ln())
            .collect();
        if !lg.is_empty() {
            let m = lg.iter().sum::<f64>() / lg.len() as f64;
            let sd = (lg.iter().map(|x| (x - m) * (x - m)).sum::<f64>()
                / (lg.len() - 1) as f64)
                .sqrt();
            println!(
                "pooled bias over endpoints: {:+.3}%  (endpoint-to-endpoint sd {:.3}%)",
                100.0 * m.exp_m1(),
                100.0 * sd
            );
        }

        for (n, e) in verdicts.iter().filter(|(_, e)| e.verdict == Verdict::Refuted) {
            println!(
                "  REFUTED  {n:40} gm={:.5} CI[{:.5},{:.5}]",
                e.gm_ratio, e.ci_lo, e.ci_hi
            );
        }
        if undet > 0 {
            // Show which endpoints could not be decided and what it would take,
            // rather than leaving the reader to infer it.
            let worst = verdicts
                .iter()
                .filter(|(_, e)| e.verdict == Verdict::Undetermined)
                .map(|(_, e)| e.achieved_half_width)
                .fold(0.0f64, f64::max);
            let need = (worst / ((1.0 + band).ln() / 2.0)).powi(2) * n_seeds as f64;
            println!(
                "  {undet} undetermined: the sample cannot decide these. Worst achieved \
                 resolution +/-{:.2}%;\n  certifying them all at this band would need \
                 roughly n = {:.0} per stratum.",
                100.0 * worst.exp_m1(),
                need
            );
        }
    }

    // ------------------------------------------- tier D: adjudicate the family ---
    let mut corr_rows: Vec<String> = vec![
        "stratum,component,bands,max_abs_delta_rho,permutation_p,holm_adjusted_p,verdict"
            .to_string(),
    ];
    if tier == Tier::D && !ifc.is_empty() {
        let raw: Vec<f64> = ifc.iter().map(|t| t.p).collect();
        let adj = stats::holm_adjusted(&raw);
        let flagged: Vec<usize> = (0..ifc.len()).filter(|&i| adj[i] <= 0.05).collect();
        let raw_flagged = raw.iter().filter(|p| **p < 0.05).count();

        println!();
        for (i, t) in ifc.iter().enumerate() {
            let differs = adj[i] <= 0.05;
            if differs {
                all_pass = false;
            }
            println!(
                "  {:16} {:4} max|dRho|={:.4}  p={:.3}  holm={:.3}  {}",
                t.label,
                t.component,
                t.observed,
                t.p,
                adj[i],
                if differs { "DIFFER" } else { "ok" }
            );
            corr_rows.push(format!(
                "{},{},{},{:.6},{:.4},{:.4},{}",
                t.label,
                t.component,
                t.bands,
                t.observed,
                t.p,
                adj[i],
                if differs { "DIFFER" } else { "ok" }
            ));
        }

        println!(
            "\nfamily of {} tests: {raw_flagged} flagged at raw p<0.05, \
             {} after Holm (FWER 0.05)",
            ifc.len(),
            flagged.len()
        );
        // The naive gate's false-alarm rate, stated so the corrected number has
        // something to be compared against.
        println!(
            "  gating on raw p would carry a family-wise false-alarm rate of {:.1}%",
            100.0 * (1.0 - 0.95f64.powi(ifc.len() as i32))
        );

        // Enrichment. Holm answers "is any single test significant"; this answers
        // the different question "does the family as a whole look uniform", which is
        // what would reveal a weak effect spread across many endpoints.
        let k = raw_flagged;
        let m = ifc.len();
        let tail: f64 = (0..k)
            .map(|j| {
                let c = (0..j).fold(1.0f64, |acc, t| acc * (m - t) as f64 / (t + 1) as f64);
                c * 0.05f64.powi(j as i32) * 0.95f64.powi((m - j) as i32)
            })
            .sum();
        println!(
            "  enrichment: P(>= {k} of {m} below 0.05 | calibrated null) = {:.4}",
            1.0 - tail
        );

        if aa {
            println!(
                "\nA/A CALIBRATION -- both halves came from the same binary, so every\n\
                 flag above is a known false alarm. A well-calibrated test should show\n\
                 about {:.1} of {m} at raw p<0.05 and 0 after Holm; it showed \
                 {raw_flagged} and {}.",
                0.05 * m as f64,
                flagged.len()
            );
        }
    }

    let out = root.join(format!(
        "harness/science_tier{}_cell{}{}.csv",
        format!("{tier:?}").to_lowercase(),
        cell.name,
        if aa { "_aa" } else { "" }
    ));
    std::fs::write(
        &out,
        if tier == Tier::D { corr_rows.join("\n") } else { rows.join("\n") },
    )?;
    println!("\nwrote {}", out.display());
    Ok(all_pass)
}

/// How far an endpoint's CI reaches from unity, for ranking the worst case.
fn dev(e: &Equivalence) -> f64 {
    if !e.gm_ratio.is_finite() {
        return f64::INFINITY;
    }
    (e.ci_lo - 1.0).abs().max((e.ci_hi - 1.0).abs())
}

/// Run one binary over every seed in a stratum, in parallel across threads.
fn collect(
    exe: &Path,
    st: &Stratum,
    seed_list: &[i32],
    fas_edges: &[f64],
    root: &Path,
    tag: &str,
) -> Result<StratumResult, Box<dyn std::error::Error>> {
    let nthreads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(16);
    // Each worker gets its own directory: the station file and the output file are
    // per-run state, and sharing them across threads would interleave writes.
    let spaces: Vec<Workspace> = (0..nthreads)
        .map(|i| Workspace::new(root, &format!("{tag}{i}")))
        .collect::<std::io::Result<_>>()?;
    // Building the template also writes that worker's station file.
    let templates: Vec<DeckTemplate> = spaces
        .iter()
        .map(|w| DeckTemplate::build(root, st, &w.station()))
        .collect::<std::io::Result<_>>()?;

    let next = AtomicUsize::new(0);
    let results: Mutex<Vec<Realisation>> = Mutex::new(Vec::with_capacity(seed_list.len()));
    let errors: Mutex<Vec<String>> = Mutex::new(Vec::new());

    std::thread::scope(|s| {
        for w in 0..nthreads {
            let (next, results, errors) = (&next, &results, &errors);
            let (space, tmpl) = (&spaces[w], &templates[w]);
            s.spawn(move || {
                let out = space.output("o");
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= seed_list.len() {
                        break;
                    }
                    let seed = seed_list[i];
                    let deck = tmpl.with(seed, &out);
                    match campaign::run_one(exe, &deck, &out, DT, fas_edges) {
                        Ok((km, measures, log_fas)) => results.lock().unwrap().push(
                            Realisation { seed, reported_km: km, measures, log_fas },
                        ),
                        Err(e) => errors.lock().unwrap().push(format!("seed {seed}: {e}")),
                    }
                }
            });
        }
    });

    let errs = errors.into_inner().unwrap();
    if !errs.is_empty() {
        return Err(format!(
            "{} of {} runs failed, first: {}",
            errs.len(),
            seed_list.len(),
            errs.first().map_or("", |s| s.as_str())
        )
        .into());
    }
    let mut realisations = results.into_inner().unwrap();
    // Threads finish out of order; sort so the paired comparison lines up seed for
    // seed rather than by completion order.
    realisations.sort_by_key(|r| r.seed);
    Ok(StratumResult { stratum: st.clone(), realisations })
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}
