//! Fourier amplitude spectra, for the inter-frequency correlation analysis.
//!
//! Reuses `hb_high::fft::fast` rather than adding a second transform: that routine
//! is already gated bit-identical against the Fortran across 13 lengths and both
//! directions, and it already requires the power-of-two lengths we need here.
//!
//! Two consequences of that reuse, both benign for this purpose:
//!
//! * It is `f32`. A 4096-point `f32` transform carries relative error of order
//!   1e-6, which is four orders of magnitude below the ±2% equivalence band, so
//!   it cannot influence a verdict.
//! * Records are zero-padded up to the next power of two. Padding interpolates in
//!   frequency rather than adding information, and it is applied identically to
//!   both codes, so it cannot bias a comparison.

use hb_high::fft::fast;
use hb_high::fort::Complex32;

/// Default frequency bins for the correlation analysis: 30 log-spaced bands from
/// 0.1 to 20 Hz.
///
/// The band is chosen for where the stochastic high-frequency method actually
/// carries the solution. Below ~0.1 Hz the hybrid hands over to the low-frequency
/// model and the amplitudes here are not meant to be used; above 20 Hz the
/// deck's `fmax = 10 Hz` high-cut has removed most of the energy, so ratios there
/// measure the taper rather than the physics.
pub fn default_bin_edges() -> Vec<f64> {
    let (lo, hi, n) = (0.1f64, 20.0f64, 30usize);
    (0..=n)
        .map(|i| lo * (hi / lo).powf(i as f64 / n as f64))
        .collect()
}

/// Single-sided Fourier amplitude spectrum.
///
/// Returns `(frequencies_hz, amplitudes)` for bins `0..=n/2`, where `n` is the
/// zero-padded length. Amplitudes are scaled by `dt`, giving the usual
/// cm/s² · s = cm/s units for acceleration input.
pub fn fas(acc: &[f32], dt: f64) -> (Vec<f64>, Vec<f64>) {
    let n = acc.len().next_power_of_two().max(2);
    let mut z = vec![Complex32::ZERO; n];
    for (slot, &a) in z.iter_mut().zip(acc) {
        *slot = Complex32::new(a, 0.0);
    }
    // ind = -1 is the analysis direction; see hb_high::fft::fast.
    fast(z.as_mut_slice(), -1);

    let nf = n / 2 + 1;
    let df = 1.0 / (n as f64 * dt);
    let mut freqs = Vec::with_capacity(nf);
    let mut amps = Vec::with_capacity(nf);
    for (bin, z) in z.iter().enumerate().take(nf) {
        freqs.push(bin as f64 * df);
        amps.push(z.norm() as f64 * dt);
    }
    (freqs, amps)
}

/// Geometric-mean FAS within each `edges[i]..edges[i+1]` band.
///
/// The geometric mean is used because the analysis works in log amplitude — the
/// residuals whose correlation we want are differences of logs, so binning must
/// happen in the same space or the two operations do not commute.
///
/// Bands containing no usable bin yield `None`, which the caller must handle
/// rather than silently treating as zero.
pub fn fas_binned(acc: &[f32], dt: f64, edges: &[f64]) -> Vec<Option<f64>> {
    let (freqs, amps) = fas(acc, dt);
    let mut out = Vec::with_capacity(edges.len().saturating_sub(1));
    for w in edges.windows(2) {
        let (lo, hi) = (w[0], w[1]);
        let mut sum_ln = 0.0f64;
        let mut count = 0usize;
        for (&f, &a) in freqs.iter().zip(&amps) {
            if f >= lo && f < hi && a > 0.0 {
                sum_ln += a.ln();
                count += 1;
            }
        }
        out.push(if count == 0 {
            None
        } else {
            Some((sum_ln / count as f64).exp())
        });
    }
    out
}

/// Band centre frequencies (geometric), for labelling and for the
/// frequency-ratio binning of the correlation matrix.
pub fn bin_centres(edges: &[f64]) -> Vec<f64> {
    edges.windows(2).map(|w| (w[0] * w[1]).sqrt()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sine_peaks_at_its_own_frequency() {
        let dt = 0.005;
        let f0 = 5.0;
        let n = 4096;
        let a: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * f0 * i as f64 * dt).sin() as f32)
            .collect();
        let (freqs, amps) = fas(&a, dt);
        let peak = amps
            .iter()
            .enumerate()
            .skip(1)
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert!(
            (freqs[peak] - f0).abs() < 1.0 / (n as f64 * dt) * 1.5,
            "peak at {} Hz, expected {f0} Hz",
            freqs[peak]
        );
    }

    #[test]
    fn zero_padding_does_not_change_the_peak_frequency() {
        // 3000 samples pads to 4096. The peak must stay put.
        let dt = 0.005;
        let f0 = 4.0;
        let a: Vec<f32> = (0..3000)
            .map(|i| (2.0 * std::f64::consts::PI * f0 * i as f64 * dt).sin() as f32)
            .collect();
        let (freqs, amps) = fas(&a, dt);
        let peak = amps
            .iter()
            .enumerate()
            .skip(1)
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert!((freqs[peak] - f0).abs() < 0.2, "peak at {} Hz", freqs[peak]);
    }

    #[test]
    fn bins_are_log_spaced_and_cover_the_band() {
        let e = default_bin_edges();
        assert_eq!(e.len(), 31);
        assert!((e[0] - 0.1).abs() < 1e-12);
        assert!((e[30] - 20.0).abs() < 1e-9);
        // Constant ratio between successive edges.
        let r0 = e[1] / e[0];
        for w in e.windows(2) {
            assert!((w[1] / w[0] / r0 - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn empty_bands_report_none_rather_than_zero() {
        // dt = 0.005 gives df = 1/(4096*0.005) ~ 0.049 Hz, so a band far below
        // that resolution contains no bin.
        let a: Vec<f32> = (0..4096).map(|i| (i as f32 * 0.1).sin()).collect();
        let out = fas_binned(&a, 0.005, &[1.0e-4, 2.0e-4, 1.0, 2.0]);
        assert_eq!(out[0], None, "a band narrower than df must be None");
        assert!(out[2].is_some());
    }
}
