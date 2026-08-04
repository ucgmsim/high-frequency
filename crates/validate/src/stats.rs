//! Statistics for equivalence testing, hand-rolled.
//!
//! `scipy` is not installed in the system Python (numpy is), and depending on
//! another repository's virtualenv would be fragile. Everything here is
//! elementary enough that implementing it is cheaper than acquiring a dependency,
//! and it keeps the whole campaign inside one `cargo run`.
//!
//! # Why equivalence testing rather than a null-hypothesis test
//!
//! The natural instinct is to test H0: "the two distributions are the same" and
//! pass when it is not rejected. That is the wrong test, in both directions:
//!
//! * Failing to reject is not evidence of equivalence. With a small sample almost
//!   nothing is rejected, so the gate passes for lack of data rather than for
//!   agreement — the test gets *easier* the less evidence you gather.
//! * With a large sample, any difference becomes significant. A 0.1% bias no
//!   seismologist could detect fails at n = 10⁴.
//!
//! Equivalence testing inverts the logic: state a band that is scientifically
//! negligible, then require the confidence interval for the difference to lie
//! **inside** it. Now more data makes the test harder to pass in the right way —
//! it narrows the interval — and the verdict is tied to a number someone can
//! argue about, rather than to a sample size.
//!
//! It also composes across endpoints. Requiring *every* IM × period × component
//! endpoint to be equivalent is an intersection-union test, whose type-I error is
//! bounded by that of any single component test. So no multiplicity correction is
//! needed, unlike the difference-testing framing where hundreds of endpoints would
//! demand Bonferroni or similar.

/// The two-sided equivalence band, as a fractional deviation of the geometric
/// mean ratio. ±2% ≈ 0.04 of a typical ground-motion-model aleatory sigma.
pub const DEFAULT_BAND: f64 = 0.02;

/// z for a one-sided 95% bound, i.e. the 90% two-sided interval used by TOST.
const Z_90: f64 = 1.6448536269514722;

/// Summary of one endpoint's comparison.
#[derive(Clone, Debug)]
pub struct Equivalence {
    pub n_a: usize,
    pub n_b: usize,
    /// Geometric-mean ratio A/B.
    pub gm_ratio: f64,
    /// 90% CI on the geometric-mean ratio.
    pub ci_lo: f64,
    pub ci_hi: f64,
    /// Band tested against, as a fraction.
    pub band: f64,
    /// True when the whole CI lies inside `[1-band, 1+band]`.
    pub equivalent: bool,
    /// Ratio of standard deviations of ln IM (A/B). Reported, not gated.
    pub sd_ratio: f64,
    /// Two-sample Kolmogorov-Smirnov statistic. Diagnostic, not gated.
    pub ks: f64,
    /// Half-width of the CI in ln units — the resolution actually achieved.
    pub achieved_half_width: f64,
}

fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

fn var(x: &[f64]) -> f64 {
    if x.len() < 2 {
        return 0.0;
    }
    let m = mean(x);
    x.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / (x.len() - 1) as f64
}

/// Unpaired equivalence test on the geometric-mean ratio.
///
/// Works in log space, where the ratio becomes a difference of means and the
/// distribution of IM values is far closer to normal than in linear space.
/// Welch's standard error is used because the two simulators need not have equal
/// scatter — and if they do not, that is itself worth seeing, so `sd_ratio` is
/// reported alongside.
pub fn equivalence_unpaired(a: &[f64], b: &[f64], band: f64) -> Equivalence {
    let la: Vec<f64> = a.iter().filter(|v| **v > 0.0).map(|v| v.ln()).collect();
    let lb: Vec<f64> = b.iter().filter(|v| **v > 0.0).map(|v| v.ln()).collect();

    let (na, nb) = (la.len(), lb.len());
    if na < 2 || nb < 2 {
        return Equivalence {
            n_a: na, n_b: nb, gm_ratio: f64::NAN, ci_lo: f64::NAN, ci_hi: f64::NAN,
            band, equivalent: false, sd_ratio: f64::NAN, ks: f64::NAN,
            achieved_half_width: f64::NAN,
        };
    }

    let (ma, mb) = (mean(&la), mean(&lb));
    let (va, vb) = (var(&la), var(&lb));
    let d = ma - mb;
    let se = (va / na as f64 + vb / nb as f64).sqrt();
    let hw = Z_90 * se;

    let (lo, hi) = (d - hw, d + hw);
    // The band is fractional; compare in ln space against ln(1 +/- band).
    let (band_lo, band_hi) = ((1.0 - band).ln(), (1.0 + band).ln());

    Equivalence {
        n_a: na,
        n_b: nb,
        gm_ratio: d.exp(),
        ci_lo: lo.exp(),
        ci_hi: hi.exp(),
        band,
        equivalent: lo > band_lo && hi < band_hi,
        sd_ratio: if vb > 0.0 { (va / vb).sqrt() } else { f64::NAN },
        ks: ks_2samp(&la, &lb),
        achieved_half_width: hw,
    }
}

/// Paired equivalence test, for when both codes run the same RNG stream and
/// matched seeds therefore give matched realisations.
///
/// Vastly more sensitive than the unpaired form: the between-realisation variance
/// cancels, leaving only the difference the change actually made. Detects a 0.01%
/// bias with n on the order of ten, where the unpaired test needs hundreds for 2%.
pub fn equivalence_paired(a: &[f64], b: &[f64], band: f64) -> Equivalence {
    assert_eq!(a.len(), b.len(), "paired test needs equal-length inputs");
    let d: Vec<f64> = a
        .iter()
        .zip(b)
        .filter(|(x, y)| **x > 0.0 && **y > 0.0)
        .map(|(x, y)| x.ln() - y.ln())
        .collect();
    let n = d.len();
    if n < 2 {
        return Equivalence {
            n_a: n, n_b: n, gm_ratio: f64::NAN, ci_lo: f64::NAN, ci_hi: f64::NAN,
            band, equivalent: false, sd_ratio: f64::NAN, ks: f64::NAN,
            achieved_half_width: f64::NAN,
        };
    }
    let m = mean(&d);
    let se = (var(&d) / n as f64).sqrt();
    let hw = Z_90 * se;
    let (lo, hi) = (m - hw, m + hw);
    let (band_lo, band_hi) = ((1.0 - band).ln(), (1.0 + band).ln());

    Equivalence {
        n_a: n,
        n_b: n,
        gm_ratio: m.exp(),
        ci_lo: lo.exp(),
        ci_hi: hi.exp(),
        band,
        equivalent: lo > band_lo && hi < band_hi,
        sd_ratio: f64::NAN, // meaningless for a paired difference
        ks: f64::NAN,
        achieved_half_width: hw,
    }
}

/// Two-sample Kolmogorov-Smirnov statistic: the largest gap between the two
/// empirical CDFs.
///
/// Reported as a diagnostic only. It answers "are these distributions
/// distinguishable", which is the null-hypothesis question we are deliberately
/// not gating on — but a large D alongside a passing equivalence test is a useful
/// signal that the shapes differ even though the means agree.
pub fn ks_2samp(a: &[f64], b: &[f64]) -> f64 {
    let mut sa: Vec<f64> = a.to_vec();
    let mut sb: Vec<f64> = b.to_vec();
    sa.sort_by(|x, y| x.partial_cmp(y).unwrap());
    sb.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let (na, nb) = (sa.len() as f64, sb.len() as f64);
    let (mut i, mut j) = (0usize, 0usize);
    let mut d = 0.0f64;
    while i < sa.len() && j < sb.len() {
        let x = sa[i].min(sb[j]);
        while i < sa.len() && sa[i] <= x {
            i += 1;
        }
        while j < sb.len() && sb[j] <= x {
            j += 1;
        }
        d = d.max((i as f64 / na - j as f64 / nb).abs());
    }
    d
}

/// Pearson correlation.
pub fn pearson(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let n = x.len();
    if n < 3 {
        return f64::NAN;
    }
    let (mx, my) = (mean(x), mean(y));
    let mut sxy = 0.0;
    let mut sxx = 0.0;
    let mut syy = 0.0;
    for k in 0..n {
        let dx = x[k] - mx;
        let dy = y[k] - my;
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    if sxx <= 0.0 || syy <= 0.0 {
        return f64::NAN;
    }
    sxy / (sxx * syy).sqrt()
}

/// Inter-frequency correlation matrix of log-amplitude residuals.
///
/// `spectra[r][f]` is realisation `r`'s log amplitude in band `f`. The residual is
/// taken about the mean **over realisations at that frequency**, so what remains is
/// the realisation-to-realisation fluctuation whose frequency structure we want —
/// not the average spectral shape, which is common to both codes and would swamp
/// the correlation.
pub fn interfrequency_correlation(spectra: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let nf = spectra.first().map_or(0, |s| s.len());
    let nr = spectra.len();
    if nr < 3 || nf == 0 {
        return vec![vec![f64::NAN; nf]; nf];
    }
    // Residuals about the per-frequency mean.
    let mut resid = vec![vec![0.0f64; nr]; nf];
    for f in 0..nf {
        let col: Vec<f64> = (0..nr).map(|r| spectra[r][f]).collect();
        let m = mean(&col);
        for r in 0..nr {
            resid[f][r] = col[r] - m;
        }
    }
    let mut rho = vec![vec![0.0f64; nf]; nf];
    for i in 0..nf {
        for j in 0..nf {
            rho[i][j] = if i == j { 1.0 } else { pearson(&resid[i], &resid[j]) };
        }
    }
    rho
}

/// Largest absolute difference between two correlation matrices.
pub fn max_abs_delta(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    let mut m = 0.0f64;
    for i in 0..a.len() {
        for j in 0..a[i].len() {
            let d = (a[i][j] - b[i][j]).abs();
            if d.is_finite() && d > m {
                m = d;
            }
        }
    }
    m
}

/// A deterministic 64-bit generator for permutation and bootstrap resampling.
///
/// Seeded explicitly so the entire campaign — including its resampling — is
/// reproducible. A "statistical" gate that gives a different answer each run
/// cannot be used as a gate; fixing this seed is what makes tight thresholds
/// safe in CI.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        // splitmix64
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// Permutation p-value for `max |rho_A - rho_B|`.
///
/// A fixed threshold on a max statistic would be arbitrary: with 30 frequency
/// bands there are 435 distinct pairs, so the maximum of 435 correlated sampling
/// fluctuations sits several standard errors above zero even when nothing differs.
/// Pooling the realisations and relabelling them at random builds the null
/// distribution of that maximum directly, which handles both the multiplicity and
/// the correlation between pairs without needing a formula for either.
pub fn permutation_p_max_delta(
    a: &[Vec<f64>],
    b: &[Vec<f64>],
    iterations: usize,
    seed: u64,
) -> (f64, f64) {
    let observed = max_abs_delta(
        &interfrequency_correlation(a),
        &interfrequency_correlation(b),
    );
    let mut pool: Vec<&Vec<f64>> = a.iter().chain(b.iter()).collect();
    let na = a.len();
    let mut rng = Rng::new(seed);
    let mut ge = 0usize;

    for _ in 0..iterations {
        // Fisher-Yates on the pooled realisations.
        for i in (1..pool.len()).rev() {
            let j = rng.below(i + 1);
            pool.swap(i, j);
        }
        let sa: Vec<Vec<f64>> = pool[..na].iter().map(|v| (*v).clone()).collect();
        let sb: Vec<Vec<f64>> = pool[na..].iter().map(|v| (*v).clone()).collect();
        let d = max_abs_delta(
            &interfrequency_correlation(&sa),
            &interfrequency_correlation(&sb),
        );
        if d >= observed {
            ge += 1;
        }
    }
    // Add-one smoothing: a p-value of exactly 0 overstates the evidence available
    // from a finite number of permutations.
    ((ge + 1) as f64 / (iterations + 1) as f64, observed)
}

/// Percentile bootstrap CI for the geometric mean of a sample.
pub fn bootstrap_gm_ci(x: &[f64], iterations: usize, seed: u64) -> (f64, f64) {
    let l: Vec<f64> = x.iter().filter(|v| **v > 0.0).map(|v| v.ln()).collect();
    if l.len() < 2 {
        return (f64::NAN, f64::NAN);
    }
    let mut rng = Rng::new(seed);
    let mut means = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let mut s = 0.0;
        for _ in 0..l.len() {
            s += l[rng.below(l.len())];
        }
        means.push(s / l.len() as f64);
    }
    means.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let lo = means[(0.05 * iterations as f64) as usize];
    let hi = means[((0.95 * iterations as f64) as usize).min(iterations - 1)];
    (lo.exp(), hi.exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic lognormal sample: ln x ~ N(mu, sigma).
    fn lognormal(n: usize, mu: f64, sigma: f64, seed: u64) -> Vec<f64> {
        let mut r = Rng::new(seed);
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            // Box-Muller from two uniforms.
            let u1 = (r.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
            let u2 = (r.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
            if u1 <= 0.0 {
                continue;
            }
            let m = (-2.0 * u1.ln()).sqrt();
            out.push((mu + sigma * m * (2.0 * std::f64::consts::PI * u2).cos()).exp());
            if out.len() < n {
                out.push((mu + sigma * m * (2.0 * std::f64::consts::PI * u2).sin()).exp());
            }
        }
        out.truncate(n);
        out
    }

    #[test]
    fn identical_samples_are_equivalent() {
        let a = lognormal(600, 0.0, 0.2, 1);
        let e = equivalence_unpaired(&a, &a, DEFAULT_BAND);
        assert!(e.equivalent, "a sample must be equivalent to itself: {e:?}");
        assert!((e.gm_ratio - 1.0).abs() < 1e-12);
    }

    #[test]
    fn a_five_percent_bias_is_not_equivalent_at_two_percent() {
        let a = lognormal(600, 0.0, 0.2, 1);
        let b = lognormal(600, (1.05f64).ln(), 0.2, 2);
        let e = equivalence_unpaired(&a, &b, DEFAULT_BAND);
        assert!(!e.equivalent, "5% bias must fail a 2% band: {e:?}");
    }

    #[test]
    fn achieved_half_width_matches_the_design_calculation() {
        // The plan sized n = 600 from sigma ~ 0.2 and 1.645*sigma*sqrt(2/n) < 0.0198.
        // Check the estimator agrees with that arithmetic, so the design is not
        // resting on a formula nobody ever evaluated.
        let a = lognormal(600, 0.0, 0.2, 3);
        let b = lognormal(600, 0.0, 0.2, 4);
        let e = equivalence_unpaired(&a, &b, DEFAULT_BAND);
        let expected = 1.645 * 0.2 * (2.0f64 / 600.0).sqrt();
        assert!(
            (e.achieved_half_width / expected - 1.0).abs() < 0.15,
            "half width {} vs design {expected}",
            e.achieved_half_width
        );
    }

    #[test]
    fn underpowered_sample_reports_a_wide_interval_rather_than_passing() {
        // The failure mode of null-hypothesis testing: n = 5 would fail to reject
        // almost anything. Equivalence testing must NOT pass here.
        let a = lognormal(5, 0.0, 0.2, 5);
        let b = lognormal(5, 0.0, 0.2, 6);
        let e = equivalence_unpaired(&a, &b, DEFAULT_BAND);
        assert!(
            !e.equivalent,
            "5 samples cannot certify a 2% band; got CI [{}, {}]",
            e.ci_lo, e.ci_hi
        );
    }

    #[test]
    fn paired_is_far_more_sensitive_than_unpaired() {
        // Same realisations with a 0.5% multiplicative offset. Unpaired at n=20
        // cannot resolve it; paired at n=20 detects it easily.
        let a = lognormal(20, 0.0, 0.2, 7);
        let b: Vec<f64> = a.iter().map(|v| v * 1.005).collect();
        let unpaired = equivalence_unpaired(&a, &b, 0.002);
        let paired = equivalence_paired(&a, &b, 0.002);
        assert!(
            paired.achieved_half_width < unpaired.achieved_half_width / 10.0,
            "paired hw {} should be far below unpaired hw {}",
            paired.achieved_half_width,
            unpaired.achieved_half_width
        );
        assert!(!paired.equivalent, "paired must catch a 0.5% offset at a 0.2% band");
    }

    #[test]
    fn ks_is_zero_for_identical_and_one_for_disjoint() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(ks_2samp(&a, &a), 0.0);
        let b = vec![10.0, 11.0, 12.0, 13.0];
        assert!((ks_2samp(&a, &b) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn correlation_matrix_is_unit_diagonal_and_symmetric() {
        let mut r = Rng::new(9);
        let spectra: Vec<Vec<f64>> = (0..200)
            .map(|_| (0..10).map(|_| (r.below(1000) as f64) / 1000.0).collect())
            .collect();
        let rho = interfrequency_correlation(&spectra);
        for i in 0..10 {
            assert!((rho[i][i] - 1.0).abs() < 1e-12);
            for j in 0..10 {
                assert!((rho[i][j] - rho[j][i]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn permutation_p_is_large_when_both_samples_come_from_one_process() {
        let mut r = Rng::new(11);
        let make = |r: &mut Rng| -> Vec<Vec<f64>> {
            (0..120)
                .map(|_| {
                    // A shared factor plus per-band noise gives realistic
                    // positive inter-frequency correlation.
                    let common = (r.below(2000) as f64) / 1000.0 - 1.0;
                    (0..8)
                        .map(|_| common + ((r.below(2000) as f64) / 1000.0 - 1.0) * 0.5)
                        .collect()
                })
                .collect()
        };
        let a = make(&mut r);
        let b = make(&mut r);
        let (p, observed) = permutation_p_max_delta(&a, &b, 200, 4242);
        assert!(p > 0.05, "same process should not be flagged (p={p}, max|d|={observed})");
    }
}
