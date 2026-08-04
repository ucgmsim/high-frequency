//! The gamma function.

/// `Gamma(x)`.
///
/// Replaces `FUNCTION DGAMM(X)` (`hb_high_ref.f:2745`), which was a 20-term
/// Chebyshev-like series with hand-rolled argument reduction into `[-0.5, 0.5]`,
/// plus two error paths that wrote a diagnostic to unit 6 and returned `1.0e75`.
///
/// # Why this was safe to swap
///
/// The one live call is `stochastic_spectrum`'s `gamma(2b+1)`, where `b` comes from
/// the time-window shape `(eps, eta)`. On the production path those are hardcoded, so
/// the argument is always `3.5062997341156006`, and there:
///
/// ```text
/// DGAMM             3.346549271566832    (0x400ac5bb9fdea847)
/// libm::tgamma      3.346549271566831    (0x400ac5bb9fdea844)
/// ```
///
/// Three ulps of `f64`. The result is consumed as
/// `aa = sqrt((2c)^(2b+1) / gm) as f32`, and `f32` keeps 24 mantissa bits against
/// `f64`'s 53, so the difference is annihilated by the narrowing. It is *not*
/// bit-identical by construction, only in effect — which is why the parity ladder is
/// evidence here rather than a guarantee.
///
/// # What changed in behaviour
///
/// The `1.0e75` sentinel is gone. `stochastic_spectrum` never checked for it, so a
/// pole or an overflow used to propagate a plausible-looking finite number straight
/// into the spectrum. `libm::tgamma` returns infinity or NaN instead, which is louder
/// and cannot be mistaken for a value. Neither is reachable from a real deck: the
/// argument is a constant of the window shape.
///
/// `DGAMM` also refused any `x > 57`. `tgamma` is happy to about 171 before
/// overflowing, so that artificial ceiling is gone too.
#[inline]
pub fn gamma(x: f64) -> f64 {
    libm::tgamma(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The value the port actually needs, pinned so that a future change of gamma
    /// implementation has to look at the one argument that matters rather than at the
    /// thousand it does not.
    #[test]
    fn the_production_argument_survives_narrowing_to_f32() {
        let gsa = 3.5062997341156006f64;
        let fortran = 3.346549271566832f64;
        let got = gamma(gsa);
        assert!(
            (got - fortran).abs() / fortran < 1e-15,
            "gamma({gsa}) = {got}, Fortran gave {fortran}"
        );
        // The consumer narrows to f32; show the difference does not survive that.
        assert_eq!((got as f32).to_bits(), (fortran as f32).to_bits());
    }

    /// Poles are now loud rather than a plausible finite number.
    #[test]
    fn poles_are_not_finite() {
        for pole in [0.0, -1.0, -2.0, -3.0] {
            assert!(
                !gamma(pole).is_finite(),
                "gamma({pole}) = {} should not be a usable value",
                gamma(pole)
            );
        }
    }
}
