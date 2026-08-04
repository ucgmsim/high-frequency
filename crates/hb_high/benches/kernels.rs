//! Per-subprogram microbenchmarks.
//!
//! Baseline for Phase 3 optimisation work: nothing gets optimised until its cost
//! is on record here. Conventions follow `IM_calculation`'s Rust benches —
//! `harness = false`, `criterion_group!`/`criterion_main!`, `BenchmarkId::new`,
//! `Throughput` where a natural unit exists, `std::hint::black_box`, and
//! `sample_size(10)` for groups where a single iteration is already milliseconds.
//!
//! Two notes on method:
//!
//! * Kernels that draw from the RNG (`stochastic_spectrum`, `horizontal_radiation_spectrum`,
//!   `normal_deviates`) advance the generator across iterations, so
//!   successive iterations see different values. The *work* is identical
//!   regardless — the only value-dependent branch anywhere is
//!   `normal_deviates`'s zero-rejection retry, which fires with probability
//!   2^-24 per draw. Reseeding per iteration would only add setup cost.
//! * Inputs are built to production shape (35-layer velocity model, SH mode, the
//!   ray topology `green_function` actually produces) rather than to whatever is
//!   convenient, so the numbers here transfer to the real workload.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;

use hb_high::fft::{fast, remove_quadratic_trend};
use hb_high::fort::{Array1, Complex32, Complex64};
use hb_high::geom::subfault_geometry;
use hb_high::highcor::apply_radiation_and_invert;
use hb_high::radiation::{horizontal_radiation_spectrum, vertical_radiation_spectrum, radiation_pattern};
use hb_high::ray::{cagniard_time, vertical_slowness, cagniard_time_derivative, geometric_spreading, green_function, stationary_ray_parameter, build_ray_path, travel_time};
use hb_high::rng::{fill_normal_deviates, fill_uniform_deviates, Pcg32};
use hb_high::site::{site_amplification_factors, apply_site_amplification};
use hb_high::state::{params, RayState, VelocityModel};
use hb_high::stoc::stochastic_spectrum;

/// Transform lengths the program actually produces. `np2` is built by doubling
/// from 2 until it exceeds `2*tmax/dt`, so it is always a power of two; 65536 is
/// what the 2827-subfault alpine fault reaches.
const NP2S: &[usize] = &[1024, 4096, 16384, 65536];

/// `nr` in the main program: the sample count for the conical radiation average.
const NR: usize = 1000;

const DT: f32 = 0.005;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// A 35-layer model shaped like the production `velocity_model` fixture after the
/// air layer is inserted: thin slow layers near the surface, thickening and
/// speeding up with depth, zero-thickness base.
fn vmod(j0: usize) -> VelocityModel {
    let mut v = VelocityModel::new();
    let mut dep = 0.0f64;
    for k in 1..=j0 {
        let frac = (k - 1) as f64 / (j0 - 1) as f64;
        v.thickness_km[k] = 0.05 + 3.0 * frac;
        v.vsh_km_s[k] = 0.5 + 4.1 * frac;
        v.vp_km_s[k] = v.vsh_km_s[k] * 1.75;
        v.density_g_cm3[k] = 1.81 + 1.5 * frac;
        v.attenuation_s[k] = (50.0 + 150.0 * frac) as f32;
        v.attenuation_p[k] = 2.0 * v.attenuation_s[k];
        dep += v.thickness_km[k];
        v.depth_km[k] = dep;
    }
    v.thickness_km[j0] = 0.0;
    v
}

/// A ray in the shape `green_function` builds for `itype=1`: segments running from the
/// source layer up to `krec = 2`, all SH (`nm = 4`).
fn ray_state(ksrc: usize) -> RayState {
    let mut st = RayState::default();
    let mut l = 0usize;
    let mut j = ksrc as i64;
    while j >= 2 {
        l += 1;
        st.rays.nh[l] = j as i32;
        st.rays.nm[l] = 4;
        j -= 1;
    }
    st.rays.nd[1] = l as i32;
    st.rays.ndeg[1] = 1;
    st
}

/// `/travel/` and `/coff/` filled by the real `build_ray_path`, so the downstream ray
/// kernels see self-consistent state.
fn ray_state_after_trav(ksrc: usize, v: &VelocityModel) -> RayState {
    let mut st = ray_state(ksrc);
    let mut depsum = 0.0f64;
    for k in 1..=ksrc {
        depsum += v.thickness_km[k];
    }
    let hs = depsum - 0.5 * v.thickness_km[ksrc];
    build_ray_path(&mut st, v, 1, hs, v.thickness_km[1]);
    st
}

/// The frequency axis the main program builds: `dfr(i) = df*(i-1)`.
fn dfr_axis(np2: usize) -> Array1<f32> {
    let mut dfr = Array1::<f32>::new(np2);
    let df = 1.0 / (np2 as f32 * DT);
    for i in 1..=np2 / 2 + 1 {
        dfr[i] = df * (i - 1) as f32;
    }
    dfr
}

/// Complex spectrum of plausible magnitude, deterministic so runs are comparable.
fn spectrum(np2: usize) -> Array1<Complex32> {
    let mut cw = Array1::filled(np2, Complex32::ZERO);
    let (mut g, _) = Pcg32::seed(20260804);
    for i in 1..=np2 {
        cw[i] = Complex32::new(g.next_f32() - 0.5, g.next_f32() - 0.5);
    }
    cw
}

/// The 20-entry log-frequency site-amplification table from `:196-218`.
fn site_table() -> (Array1<f32>, Array1<f32>) {
    const HZ: [f32; 20] = [
        0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00, 2.00,
        3.00, 5.00, 7.00, 10.00, 20.00, 30.00, 50.00, 70.00,
    ];
    let mut fn_ = Array1::<f32>::new(params::NLAYMAX);
    let mut an = Array1::<f32>::new(params::NLAYMAX);
    let (mut g, _) = Pcg32::seed(11);
    for i in 1..=20 {
        fn_[i] = HZ[i - 1].ln();
        an[i] = 0.5 * g.next_f32();
    }
    (fn_, an)
}

// ---------------------------------------------------------------------------
// FFT
// ---------------------------------------------------------------------------

/// The dominant cost. Prior EMOD3D profiling attributed ~19% of total runtime to
/// `cexpf`/`sincosf` computing twiddles inside this routine alone, and measured a
/// twiddle table as worth 1.35–1.43x while staying bit-identical. This group is
/// the baseline that claim will be re-checked against.
fn bench_fft(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft");
    for &np2 in NP2S {
        group.throughput(Throughput::Elements(np2 as u64));
        let src = spectrum(np2);
        for (name, ind) in [("analysis", -1i32), ("synthesis", 1i32)] {
            group.bench_with_input(
                BenchmarkId::new(name, np2),
                &(np2, ind),
                |b, &(_n, i)| {
                    b.iter_batched_ref(
                        || src.clone(),
                        |ace| fast(ace.as_mut_slice(), black_box(i)),
                        criterion::BatchSize::SmallInput,
                    )
                },
            );
        }
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// RNG
// ---------------------------------------------------------------------------

fn bench_rng(c: &mut Criterion) {
    let mut group = c.benchmark_group("rng");

    group.throughput(Throughput::Elements(1));
    group.bench_function("next_u32", |b| {
        let (mut g, _) = Pcg32::seed(1);
        b.iter(|| black_box(g.next_u32()))
    });
    group.bench_function("next_f32", |b| {
        let (mut g, _) = Pcg32::seed(1);
        b.iter(|| black_box(g.next_f32()))
    });

    // normal_deviates is called once per stochastic_spectrum, i.e. three times per
    // subfault per ray, with n = np2. Box-Muller plus a full renormalisation
    // pass, so it is not a trivial wrapper.
    for &n in &[1024usize, 4096, 65536] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("normal", n), &n, |b, &n| {
            let (mut g, _) = Pcg32::seed(1);
            let mut acc = Array1::<f32>::new(n);
            b.iter(|| fill_normal_deviates(&mut g, black_box(n), &mut acc))
        });
        group.bench_with_input(BenchmarkId::new("uniform_deviates", n), &n, |b, &n| {
            let (mut g, _) = Pcg32::seed(1);
            let mut rn = Array1::<f32>::new(n);
            b.iter(|| fill_uniform_deviates(&mut g, black_box(n), &mut rn))
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Radiation
// ---------------------------------------------------------------------------

/// `horizontal_radiation_spectrum` is the second-largest cost after the FFT and the largest RNG
/// consumer: `nr = 1000` iterations, each drawing five deviates and calling
/// `radiation_pattern`, so 5,000 draws and 5,000 `radiation_pattern` calls per invocation — twice per
/// subfault per ray. `radiation_pattern` is benched separately so the split is visible.
fn bench_radiation(c: &mut Criterion) {
    let mut group = c.benchmark_group("radiation");

    group.throughput(Throughput::Elements(1));
    group.bench_function("radiation_pattern", |b| {
        b.iter(|| {
            black_box(radiation_pattern(
                black_box(1.2),
                black_box(0.9),
                black_box(-0.4),
                black_box(2.1),
                black_box(2.6),
            ))
        })
    });

    let np2 = 4096usize;
    let nfold = np2 / 2 + 1;
    let dfr = dfr_axis(np2);
    let mut rdna = Array1::<f32>::new(np2);

    // Per call, not per draw: the useful comparison is against one FFT of the
    // same np2, since both happen the same number of times per subfault.
    group.throughput(Throughput::Elements(1));
    group.bench_function(BenchmarkId::new("horizontal_radiation_spectrum", format!("nr{NR}")), |b| {
        let (mut g, _) = Pcg32::seed(7);
        b.iter(|| {
            horizontal_radiation_spectrum(
                &mut g, 1.2, 0.9, -0.4, 2.1, 2.6, &dfr, nfold,
                black_box(-90.0 * (3.1415926 / 180.0)), NR, &mut rdna,
            )
        })
    });

    let (mut g, _) = Pcg32::seed(3);
    let mut rna = Array1::<f32>::new(NR);
    let mut rnb = Array1::<f32>::new(NR);
    fill_uniform_deviates(&mut g, NR, &mut rna);
    fill_uniform_deviates(&mut g, NR, &mut rnb);
    group.bench_function(BenchmarkId::new("vertical_radiation_spectrum", format!("nr{NR}")), |b| {
        b.iter(|| {
            vertical_radiation_spectrum(1.2, 0.9, -0.4, 2.1, 2.6, &dfr, nfold, &rna, &rnb, NR, &mut rdna)
        })
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Ray theory
// ---------------------------------------------------------------------------

/// `stationary_ray_parameter` is interesting because it always bisects and always exits on the
/// `|dtau/dp| <= 0.01` tolerance rather than the 40-iteration cap, and every
/// iteration calls `cagniard_time_derivative`, which calls `vertical_slowness` once per layer. So its cost is
/// `iterations x layers x vertical_slowness`.
fn bench_ray(c: &mut Criterion) {
    let mut group = c.benchmark_group("ray");
    let v = vmod(35);
    let st = ray_state_after_trav(18, &v);
    let p = Complex64::new(0.15, 0.0);

    group.throughput(Throughput::Elements(1));
    group.bench_function("vertical_slowness", |b| {
        b.iter(|| black_box(vertical_slowness(black_box(p), black_box(3.2))))
    });
    group.bench_function("cagniard_time", |b| {
        b.iter(|| black_box(cagniard_time(&st, &v, black_box(p), 1, black_box(60.0))))
    });
    group.bench_function("cagniard_time_derivative", |b| {
        b.iter(|| black_box(cagniard_time_derivative(&st, &v, black_box(p), 1, black_box(60.0))))
    });
    group.bench_function("stationary_ray_parameter", |b| {
        b.iter(|| black_box(stationary_ray_parameter(&st, &v, 1, black_box(60.0))))
    });
    group.bench_function("travel_time", |b| {
        b.iter(|| black_box(travel_time(&st, &v, 1, black_box(0.15), 0.0, black_box(60.0))))
    });
    group.bench_function("geometric_spreading", |b| {
        b.iter(|| black_box(geometric_spreading(&st, &v, black_box(30.0), black_box(0.15), 1)))
    });
    group.bench_function("build_ray_path", |b| {
        b.iter_batched_ref(
            || ray_state(18),
            |st| build_ray_path(st, &v, 1, black_box(30.0), black_box(v.thickness_km[1])),
            criterion::BatchSize::SmallInput,
        )
    });
    // The whole cluster, as the main program calls it once per subfault per ray.
    group.bench_function("green_function", |b| {
        b.iter_batched_ref(
            RayState::default,
            |st| {
                black_box(green_function(
                    st, &v, 35, black_box(28.0), black_box(60.0), 1, 4,
                ))
            },
            criterion::BatchSize::SmallInput,
        )
    });
    group.finish();
}

// ---------------------------------------------------------------------------
// Spectrum shaping
// ---------------------------------------------------------------------------

fn bench_spectrum(c: &mut Criterion) {
    let mut group = c.benchmark_group("spectrum");
    group.sample_size(20);

    for &np2 in NP2S {
        let nf = np2 / 2 + 1;
        let mf = np2 / 2 - 1;
        let dfr = dfr_axis(np2);
        let src = spectrum(np2);
        let (fn_, an) = site_table();
        let mut rdna = Array1::<f32>::new(np2);
        for i in 1..=nf {
            rdna[i] = 0.7;
        }

        group.throughput(Throughput::Elements(np2 as u64));

        // stochastic_spectrum: one FFT plus np2 normal draws plus the per-bin spectral shape.
        group.bench_with_input(BenchmarkId::new("stochastic_spectrum", np2), &np2, |b, &np2| {
            let (mut g, _) = Pcg32::seed(5);
            let mut cw = Array1::filled(np2, Complex32::ZERO);
            b.iter(|| {
                stochastic_spectrum(
                    &mut g, np2, 60.0, 2.0, 0.2, 0.05, 3.2, 2.7, DT, 3.0e22, 0.0,
                    1.5, 10.0, 0.045, &mut cw, &dfr, 0.02, 0.6, 2.1,
                )
            })
        });

        // apply_radiation_and_invert: the radiation multiply, one inverse FFT, the scale and the
        // raised-cosine taper.
        group.bench_with_input(BenchmarkId::new("apply_radiation_and_invert", np2), &np2, |b, &np2| {
            let mut stdd = Array1::<f32>::new(np2);
            b.iter_batched_ref(
                || src.clone(),
                |cw| apply_radiation_and_invert(nf, mf, np2, cw, &mut stdd, &rdna),
                criterion::BatchSize::SmallInput,
            )
        });

        group.bench_with_input(BenchmarkId::new("apply_site_amplification", np2), &np2, |b, &np2| {
            b.iter_batched_ref(
                || src.clone(),
                |cw| apply_site_amplification(np2, cw, &dfr, 20, &fn_, &an),
                criterion::BatchSize::SmallInput,
            )
        });

        group.bench_with_input(BenchmarkId::new("remove_quadratic_trend", np2), &np2, |b, &np2| {
            let mut a = Array1::<f32>::new(np2);
            for i in 1..=np2 {
                a[i] = (i as f32 * 0.01).sin();
            }
            b.iter_batched_ref(
                || a.clone(),
                |a| remove_quadratic_trend(DT, a.as_mut_slice()),
                criterion::BatchSize::SmallInput,
            )
        });
    }

    // Cheap and called once per subfault per ray; included so it can be ruled out
    // rather than assumed negligible.
    let v = vmod(35);
    let (fn_, mut an) = site_table();
    group.throughput(Throughput::Elements(1));
    group.bench_function("site_amplification_factors", |b| {
        b.iter(|| site_amplification_factors(&v, black_box(20), black_box(20), &fn_, &mut an))
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

/// `subfault_geometry` runs once per segment per station and calls `DELAZ5` once per
/// subfault, so it scales with the subfault count rather than with `np2`.
fn bench_geom(c: &mut Criterion) {
    let mut group = c.benchmark_group("geom");
    for &(nx, nw) in &[(2usize, 2usize), (16, 7), (257, 11)] {
        group.throughput(Throughput::Elements((nx * nw) as u64));
        group.bench_with_input(
            BenchmarkId::new("subfault_geometry", format!("{nx}x{nw}")),
            &(nx, nw),
            // `subfault_geometry` now allocates its five (nq, np) arrays itself and returns
            // them, so this timing INCLUDES ~1.2 MB of allocation per call where it
            // previously hoisted them out of the loop. That is the honest number:
            // the driver allocates per segment too, so the old form was
            // under-measuring real use. Expect a step change against baselines
            // recorded before this.
            |b, &(nx, nw)| {
                b.iter(|| {
                    subfault_geometry(
                        173.0, -43.0, 173.1, -43.0, 220.0, 70.0, 5.0,
                        0.5 * nx as f32 * 1.5, 1.5, 1.5, nx, nw,
                    )
                })
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// The 1-based array wrapper
// ---------------------------------------------------------------------------

/// What the transliteration's indexing discipline costs.
///
/// `Array1`'s `Index` impl asserts `i >= 1` and then bounds-checks the `Vec`, so
/// every element access in every kernel carries two branches. This quantifies
/// that against a plain slice sum, which tells us whether removing the wrapper in
/// hot loops during Phase 3 is worth the risk of touching index arithmetic.
fn bench_array(c: &mut Criterion) {
    let mut group = c.benchmark_group("array");
    let n = 65536usize;
    let mut a = Array1::<f32>::new(n);
    for i in 1..=n {
        a[i] = i as f32;
    }
    group.throughput(Throughput::Elements(n as u64));
    group.bench_function("indexed_sum", |b| {
        b.iter(|| {
            let mut s = 0.0f32;
            for i in 1..=n {
                s += a[i];
            }
            black_box(s)
        })
    });
    group.bench_function("slice_sum", |b| {
        let s = a.as_slice();
        b.iter(|| black_box(s.iter().sum::<f32>()))
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_fft,
    bench_rng,
    bench_radiation,
    bench_ray,
    bench_spectrum,
    bench_geom,
    bench_array,
);
criterion_main!(benches);
