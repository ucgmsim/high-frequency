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
    /// Distributional, against production Fortran. Unpaired.
    C,
    /// Inter-frequency correlation. Needs the same sample as C.
    D,
}

/// Band on the ratio of log-space standard deviations.
///
/// Deliberately wider than the mean band, because a standard deviation is estimated far
/// less precisely than a mean: its relative standard error is about `1/sqrt(2n)`, which
/// at n = 2500 is 1.4%. A ±2% gate on scatter would be testing the estimator, not the
/// port. ±10% still catches the thing this exists for — a change in *sampling* that moves
/// the spread while leaving the centre alone.
const SD_BAND: f64 = 0.10;

/// Quantiles gated alongside the mean. The median catches a shift the mean can absorb
/// through outliers; the two extremes catch a change in spread or skew that neither the
/// mean nor the sd ratio need show.
const GATED_QUANTILES: [f64; 3] = [0.05, 0.50, 0.95];

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
        Some("c") => Tier::C,
        Some("d") => Tier::D,
        _ => {
            return Err(
                "usage: validate --tier c|d [--cell a|b|c] [--band 0.02] \
                 [--shape-band 0.02] [--seeds N] [--ref-bin PATH] [--aa] [--baseline]"
                    .into(),
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
    // The quantile gate's band, separate from the mean's.
    //
    // The +/-2% band is a deliberate physics choice about IM MEANS -- roughly 0.04 of a
    // typical GMM aleatory sigma. Reusing it on q05/q95 was a category error on my part: a
    // tail quantile is a fundamentally noisier statistic than a mean at the same n, so the
    // same numeric band is a materially stricter test. Defaults to `band` so nothing
    // changes until an A/A run says what the null actually supports.
    let shape_band: f64 = arg("--shape-band").and_then(|s| s.parse().ok()).unwrap_or(band);
    let n_seeds: usize = arg("--seeds").and_then(|s| s.parse().ok()).unwrap_or(cell.seeds);

    // A/A calibration: run ONE binary and split its realisations in half. Every
    // flag is then a known false alarm, so the flag rate measures whether the test is
    // calibrated rather than whether the port is correct. Without it, a failure cannot be
    // attributed between "the codes differ" and "the test over-rejects".
    //
    // This used to be refused for anything but tier D, and that restriction cost a whole
    // campaign. Stage 3's LONG returned 373/375 certified on the mean and 0 refuted, but
    // its NEW quantile gate flagged 14 of 375 -- with no way to ask how many it flags when
    // both sides are the same program. Tier D was never in that position: it prints its own
    // family-wise false-alarm rate (53.7%) beside every verdict, which is the only reason
    // its lone p=0.005 could be read as expected rather than alarming.
    //
    // Note the sample-size arithmetic. Splitting one run in half gives each side n/2, so an
    // A/A that matches the A/B's RESOLUTION needs 2x the seeds -- and false-alarm rates
    // depend on the estimator's spread, so matching resolution is the whole point. It costs
    // the same total work: 2n seeds against one binary, versus n seeds against two.
    let aa = std::env::args().any(|a| a == "--aa");

    // Opt in to overwriting the recorded baseline CSV; see the write site.
    let baseline = std::env::args().any(|a| a == "--baseline");

    let root = repo_root();
    let rust = root.join("target/release/hb_high");
    // Everything compares against PRODUCTION Fortran, which uses gfortran's generator and
    // is therefore necessarily unpaired.
    //
    // Tier B used to compare against the ORACLE, which shared the port's PCG32 stream so
    // that matched seeds gave matched realisations. Stage 3 retired it: replacing the
    // generator destroys the pairing, and a desynced paired test does not fail loudly --
    // it goes Undetermined everywhere and PASSES. See `Tier`.
    //
    // `--ref-bin` overrides both. Stage 3 needs it because the interesting comparison
    // stops being Rust-vs-Fortran and becomes Rust-vs-Rust: an earlier commit's binary,
    // built by `run_selfparity.sh` into `target/selfparity-target/release/hb_high`. Both
    // halves of that have existed since §2.8 and had never been connected -- the path was
    // hardcoded here, so there was no way to point the campaign at anything else.
    let reference = match arg("--ref-bin") {
        Some(p) => std::path::PathBuf::from(p),
        None => root.join("reference/build/hb_prod"),
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
    // Endpoints whose *distribution shape* differs, even where the mean agrees.
    let mut shape_failures: Vec<String> = Vec::new();
    // Whether the sample was ever large enough for a shape gate to have an opinion.
    let mut shape_gated = false;

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
                // A/A splits one binary's realisations down the middle, exactly as tier D
                // does with its correlation matrices. Both halves are then the same
                // program, so every flag below is a known false alarm.
                let (xa, xb) = match &b {
                    Some(b) => (a.endpoint(ci, name), b.endpoint(ci, name)),
                    None => {
                        let all = a.endpoint(ci, name);
                        let half = all.len() / 2;
                        (all[..half].to_vec(), all[half..].to_vec())
                    }
                };
                let e = stats::equivalence_unpaired(&xa, &xb, band);
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
                // THE SHAPE GATES. The mean is the first moment, and a change in
                // sampling is exactly what moves a distribution's shape while leaving its
                // centre alone. Two distributions can agree on the mean to 0.1% and
                // disagree materially in the upper tail -- which, for ground motion, is
                // the part anyone cares about.
                let where_ = format!("{label} {cname} {name}");
                // A ratio of standard deviations has relative standard error about
                // 1/sqrt(2n) in log space -- 7.1% at n=100 against a 10% band, which
                // would flag one endpoint in seven by chance. Require the estimator to be
                // at least three times tighter than the band before letting it refuse.
                let sd_se = 1.0 / (2.0 * e.n_a.min(e.n_b) as f64).sqrt();
                let sd_can_decide = sd_se * 3.0 < SD_BAND;
                shape_gated |= sd_can_decide;
                if sd_can_decide && e.sd_ratio.is_finite() && (e.sd_ratio - 1.0).abs() > SD_BAND {
                    all_pass = false;
                    shape_failures.push(format!(
                        "{where_:44} scatter ratio {:.4} outside +/-{:.0}%",
                        e.sd_ratio,
                        100.0 * SD_BAND
                    ));
                }
                for q in GATED_QUANTILES {
                    let qe = stats::quantile_equivalence(&xa, &xb, q, shape_band);
                    // The same rule the mean gate follows: a test that cannot resolve the
                    // band does not get to refute on it.
                    //
                    // `verdict_of` calls an endpoint Refuted when the POINT ESTIMATE is
                    // outside the band, whatever the interval. That is right for a mean,
                    // which converges quickly. It is badly wrong for an extreme quantile
                    // at small n: the 5th percentile of 100 draws is the 5th smallest
                    // value, and its scatter alone puts it well outside a 10% band. At
                    // n=100 that produced 220 "failures" of 375, none of them real.
                    //
                    // So a quantile only refutes when it had the resolution to say so.
                    let can_decide = qe.achieved_half_width < (1.0 + band).ln();
                    if qe.verdict == Verdict::Refuted && can_decide {
                        all_pass = false;
                        shape_failures.push(format!(
                            "{where_:44} q{:.0} ratio {:.4} CI[{:.4},{:.4}]",
                            100.0 * q,
                            qe.gm_ratio,
                            qe.ci_lo,
                            qe.ci_hi
                        ));
                    }
                }
                stratum_verdicts.push((where_, e));
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

        // THE RESOLUTION GATE. A tier that cannot resolve the band it claims to test has
        // not passed -- it has ABSTAINED, and abstention must not be spelled the same way
        // as success.
        //
        // This closes a hole that would otherwise have swallowed the whole of Stage 3.
        // The verdict gate below accepts `Undetermined` on purpose, because an
        // undetermined endpoint is a statement about the sample rather than the port. But
        // nothing checked that the sample could decide ANYTHING. A paired tier whose
        // stream had desynchronised -- exactly what replacing the RNG does -- produces
        // differences of unrelated realisations, a half-width inflated from ~1e-6 to
        // ~0.066 (+-6.8% against a +-2% band), `Undetermined` on every endpoint, and a
        // PASS. The CSV looks entirely plausible.
        //
        // The threshold is the TYPICAL half-width, not the worst: a handful of genuinely
        // noisy endpoints (PGA and short-period pSA are extreme-value statistics) should
        // not condemn a run that resolved the other 370. Failing here means the sample is
        // too small, the reference is wrong, or the comparison has come unpaired -- and
        // all three want a human, not a green tick.
        if med.exp_m1() > band {
            all_pass = false;
            println!(
                "  NOT EQUIVALENT: this run resolves only +/-{:.3}% typical, which cannot \
                 decide a +/-{:.1}% band.\n  \
                 It has abstained, not passed. Suspect sample size, the reference binary, \
                 or a desynchronised paired comparison.",
                100.0 * med.exp_m1(),
                100.0 * band
            );
        }
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

        if !shape_gated {
            println!(
                "shape gates: SKIPPED -- n={} cannot resolve them. They need roughly \
                 n >= {} for scatter; run the full tier.",
                n_seeds,
                (4.5 / (SD_BAND * SD_BAND)).ceil() as usize
            );
        } else if shape_failures.is_empty() {
            println!(
                "shape gates: all endpoints within +/-{:.0}% on scatter and \
                 +/-{:.1}% at q5/q50/q95",
                100.0 * SD_BAND,
                100.0 * shape_band
            );
        } else {
            println!(
                "\nSHAPE NOT EQUIVALENT -- {} endpoint(s). The mean can agree while the \
                 distribution does not:",
                shape_failures.len()
            );
            for f in shape_failures.iter().take(20) {
                println!("  {f}");
            }
            if shape_failures.len() > 20 {
                println!("  ... and {} more", shape_failures.len() - 20);
            }
        }

        // The number that makes the count above interpretable. 14 of 375 means nothing
        // until you know what the same gate returns when both sides are the same program.
        let total_endpoints = verdicts.len();
        if aa {
            println!(
                "\nA/A CALIBRATION -- both halves came from the SAME binary, so all {} \
                 shape flag(s) of {} endpoint(s) are known FALSE ALARMS.\n  \
                 false-alarm rate {:.2}% at a +/-{:.1}% quantile band.\n  \
                 If this is near the A/B count, the band is too tight for a tail \
                 statistic and does not measure the port. If it is near zero, the A/B \
                 flags are real.",
                shape_failures.len(),
                total_endpoints,
                100.0 * shape_failures.len() as f64 / total_endpoints.max(1) as f64,
                100.0 * shape_band
            );
        } else if !shape_failures.is_empty() {
            println!(
                "  {} of {} endpoints ({:.2}%). This count is UNINTERPRETABLE alone -- \
                 rerun with --aa to measure how often the same gate fires on identical \
                 programs.",
                shape_failures.len(),
                total_endpoints,
                100.0 * shape_failures.len() as f64 / total_endpoints.max(1) as f64
            );
        }

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

    // Writing the canonical filename is OPT-IN, via `--baseline`. Everything else writes
    // to `..._n<seeds>.csv`.
    //
    // Bisection runs at reduced `n` constantly, and an underpowered run silently
    // overwriting the certified CSV is how a baseline gets lost without anyone noticing.
    // (Learned by doing exactly that, one command before writing this.) Inferring it from
    // the seed count does not work -- cell A's default is 600 but the campaign runs 2500 --
    // so it has to be said out loud.
    let suffix = if baseline { String::new() } else { format!("_n{n_seeds}") };
    let out = root.join(format!(
        "harness/science_tier{}_cell{}{}{}.csv",
        format!("{tier:?}").to_lowercase(),
        cell.name,
        if aa { "_aa" } else { "" },
        suffix
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
