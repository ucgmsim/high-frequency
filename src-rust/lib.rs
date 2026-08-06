//! Python bindings for the `hb_high` stochastic high-frequency generator.

use hb_high::config::{
    HfConfig, PathDurationModel, PathParameters, RayType, RecordParameters, RuptureVelocity,
    SiteParameters, SourceParameters,
};
use hb_high::input::{build_velocity_model, Segment, Station, StochModel, Subfault};
use hb_high::sim::simulate;
use hb_high::state::{InputLayer, VelocityModelInput};
use numpy::ndarray::Array3;
use numpy::{IntoPyArray, PyArray3, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

/// Components per station: 090, 000, vertical.
const COMPONENT_COUNT: usize = 3;

/// One fault segment: geometry plus the three subfault grids.
///
/// Grids are `(down_dip, along_strike)` for `slip`/`rise`/`trup` and the row
/// order the `.stoch` format itself uses.
#[pyclass(frozen, name = "FaultSegment")]
pub struct PyFaultSegment {
    inner: Segment,
}

#[pymethods]
impl PyFaultSegment {
    #[new]
    #[pyo3(signature = (*, longitude_deg, latitude_deg, strike_deg, dip_deg, rake_deg,
                        top_depth_km, subfault_length_km, subfault_width_km,
                        hypocentre_along_strike_km, hypocentre_down_dip_km,
                        slip, rise_time_s, rupture_time_s))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        longitude_deg: f32,
        latitude_deg: f32,
        strike_deg: f32,
        dip_deg: f32,
        rake_deg: f32,
        top_depth_km: f32,
        subfault_length_km: f32,
        subfault_width_km: f32,
        hypocentre_along_strike_km: f32,
        hypocentre_down_dip_km: f32,
        slip: PyReadonlyArray2<f32>,
        rise_time_s: PyReadonlyArray2<f32>,
        rupture_time_s: PyReadonlyArray2<f32>,
    ) -> PyResult<Self> {
        let (slip, rise, rupture) = (
            slip.as_array(),
            rise_time_s.as_array(),
            rupture_time_s.as_array(),
        );
        // Keyword-only above, so the twelve scalars cannot be transposed positionally. The
        // grids still can be, which is what this checks.
        if slip.dim() != rise.dim() || slip.dim() != rupture.dim() {
            return Err(PyValueError::new_err(format!(
                "slip {:?}, rise_time_s {:?} and rupture_time_s {:?} must have the same \
                 shape -- they describe the same subfaults",
                slip.dim(),
                rise.dim(),
                rupture.dim()
            )));
        }
        let (down_dip_count, along_strike_count) = slip.dim();
        if down_dip_count == 0 || along_strike_count == 0 {
            return Err(PyValueError::new_err(
                "a segment needs at least one subfault; got an empty grid",
            ));
        }

        // Strike index fastest, one row per down-dip index -- the order every accumulation
        // over the grid runs in, so it is built that way rather than transposed later.
        let subfaults = (0..down_dip_count)
            .flat_map(|j| {
                (0..along_strike_count).map(move |i| Subfault {
                    slip: slip[[j, i]],
                    rise_time_s: rise[[j, i]],
                    rupture_time_s: rupture[[j, i]],
                })
            })
            .collect();

        Ok(Self {
            inner: Segment::builder()
                .fault_lon_deg(longitude_deg)
                .fault_lat_deg(latitude_deg)
                .along_strike_count(along_strike_count)
                .down_dip_count(down_dip_count)
                .subfault_length_km(subfault_length_km)
                .subfault_width_km(subfault_width_km)
                .strike_deg(strike_deg)
                .dip_deg(dip_deg)
                .rake_deg(rake_deg)
                .top_depth_km(top_depth_km)
                .hypocentre_along_strike_km(hypocentre_along_strike_km)
                .hypocentre_down_dip_km(hypocentre_down_dip_km)
                .subfaults(subfaults)
                .build(),
        })
    }
}

/// A whole slip model: one or more [`PyFaultSegment`].
#[pyclass(frozen, name = "SlipModel")]
pub struct PySlipModel {
    inner: StochModel,
}

#[pymethods]
impl PySlipModel {
    #[new]
    fn new(segments: Vec<PyRef<'_, PyFaultSegment>>) -> PyResult<Self> {
        if segments.is_empty() {
            return Err(PyValueError::new_err(
                "a slip model needs at least one segment",
            ));
        }
        let segments = segments.iter().map(|s| s.inner.clone()).collect();
        Ok(Self {
            inner: StochModel::new(segments),
        })
    }

    /// Total subfaults across all segments
    #[getter]
    fn subfault_count(&self) -> usize {
        self.inner.subfault_count
    }
}

/// The 1-D velocity model, already truncated at the Moho.
#[pyclass(frozen, name = "VelocityModel1D")]
pub struct PyVelocityModel {
    input: VelocityModelInput,
}

#[pymethods]
impl PyVelocityModel {
    #[new]
    #[pyo3(signature = (*, thickness_km, vp_km_s, vsh_km_s, density_g_cm3,
                        quality_factor_p, quality_factor_s, vs_moho_km_s))]
    fn new(
        thickness_km: PyReadonlyArray1<f32>,
        vp_km_s: PyReadonlyArray1<f64>,
        vsh_km_s: PyReadonlyArray1<f64>,
        density_g_cm3: PyReadonlyArray1<f64>,
        quality_factor_p: PyReadonlyArray1<f32>,
        quality_factor_s: PyReadonlyArray1<f32>,
        vs_moho_km_s: f64,
    ) -> PyResult<Self> {
        let (thickness, vp, vsh) = (
            thickness_km.as_slice()?,
            vp_km_s.as_slice()?,
            vsh_km_s.as_slice()?,
        );
        let (density, qp, qs) = (
            density_g_cm3.as_slice()?,
            quality_factor_p.as_slice()?,
            quality_factor_s.as_slice()?,
        );
        let lengths = [
            thickness.len(),
            vp.len(),
            vsh.len(),
            density.len(),
            qp.len(),
            qs.len(),
        ];
        if lengths.iter().any(|&n| n != lengths[0]) {
            return Err(PyValueError::new_err(format!(
                "every layer property must have one entry per layer. Got lengths {lengths:?} \
                 for (thickness_km, vp_km_s, vsh_km_s, density_g_cm3, quality_factor_p, \
                 quality_factor_s)"
            )));
        }

        let layers: Vec<InputLayer> = (0..lengths[0])
            .map(|i| InputLayer {
                // Derived by build_velocity_model, which accumulates it down the column.
                depth_km: 0.0,
                thickness_km: thickness[i],
                vp_km_s: vp[i],
                vsh_km_s: vsh[i],
                density_g_cm3: density[i],
                attenuation_p: qp[i],
                attenuation_s: qs[i],
            })
            .collect();

        let input = build_velocity_model(&layers, vs_moho_km_s)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { input })
    }

    /// Layers remaining after Moho truncation.
    #[getter]
    fn layer_count(&self) -> usize {
        self.input.len()
    }
}

/// Simulate a batch of stations against one source and one velocity model.
///
/// Returns acceleration in cm/s^2, shaped `(3, n_station, n_time)` with components ordered
/// 090/000/vertical.
///
#[pyfunction]
#[pyo3(signature = (
    slip_model, velocity_model, *,
    latitude_deg, longitude_deg, station_seed,
    duration_s, dt, stress_drop_bars, fmax_hz, kappa_s, q_frequency_exponent,
    rayset,
    rupture_velocity_fraction, rupture_velocity_shallow, rupture_velocity_deep,
    rupture_velocity_sigma, corner_frequency_constant, corner_frequency_alpha,
    path_duration_model,
))]
#[allow(clippy::too_many_arguments)]
fn _simulate_stations<'py>(
    py: Python<'py>,
    slip_model: &PySlipModel,
    velocity_model: &PyVelocityModel,
    latitude_deg: PyReadonlyArray1<f32>,
    longitude_deg: PyReadonlyArray1<f32>,
    station_seed: PyReadonlyArray1<u64>,
    duration_s: f32,
    dt: f32,
    stress_drop_bars: f32,
    fmax_hz: f32,
    kappa_s: f32,
    q_frequency_exponent: f32,
    rayset: Vec<i32>,
    rupture_velocity_fraction: f32,
    rupture_velocity_shallow: f32,
    rupture_velocity_deep: f32,
    rupture_velocity_sigma: f32,
    corner_frequency_constant: f32,
    corner_frequency_alpha: f32,
    path_duration_model: i32,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let (latitude, longitude, seeds) = (
        latitude_deg.as_slice()?,
        longitude_deg.as_slice()?,
        station_seed.as_slice()?,
    );
    if latitude.len() != longitude.len() || latitude.len() != seeds.len() {
        return Err(PyValueError::new_err(format!(
            "latitude_deg ({}), longitude_deg ({}) and station_seed ({}) must have one \
             entry per station",
            latitude.len(),
            longitude.len(),
            seeds.len()
        )));
    }
    if rayset.is_empty() {
        return Err(PyValueError::new_err(
            "rayset must name at least one ray type",
        ));
    }
    let path_duration = PathDurationModel::from_deck(path_duration_model).ok_or_else(|| {
        PyValueError::new_err(format!(
            "path_duration_model {path_duration_model} is not one of 0, 1, 2, 11, 12 -- \
             the Fortran left `ndur` undefined for every other value"
        ))
    })?;

    let config = HfConfig {
        source: SourceParameters {
            stress_drop_bars,
            czero: corner_frequency_constant,
            calpha: corner_frequency_alpha,
            rupture_velocity: RuptureVelocity {
                frac: rupture_velocity_fraction,
                shallow: rupture_velocity_shallow,
                deep: rupture_velocity_deep,
                rv_sig1: rupture_velocity_sigma,
            },
        },
        path: PathParameters {
            rayset: rayset.into_iter().map(RayType).collect(),
            q_exponent: q_frequency_exponent,
            path_duration,
        },
        site: SiteParameters {
            kappa_s,
            f_max_hz: fmax_hz,
        },
        record: RecordParameters {
            duration_s,
            dt_s: dt,
        },
    };

    let station_count = latitude.len();
    let waveform: Array3<f32> = py.detach(|| {
        // One station first, to learn n_time before allocating the batch. Every station
        // shares the deck's duration and dt, so ndata is the same for all of them.
        let mut waveform: Option<Array3<f32>> = None;

        for (index, ((&stlat, &stlon), &seed)) in
            latitude.iter().zip(longitude).zip(seeds).enumerate()
        {
            let station = Station {
                latitude: stlat,
                longitude: stlon,
                name: format!("station-{index}"),
            };
            let sim = simulate(
                &config,
                &slip_model.inner,
                &velocity_model.input,
                station,
                seed,
            )
            .map_err(|e| PyRuntimeError::new_err(format!("station {index}: {e}")))?;

            let out = waveform
                .get_or_insert_with(|| Array3::zeros((COMPONENT_COUNT, station_count, sim.ndata)));
            if sim.ndata * COMPONENT_COUNT != sim.acc.len() {
                return Err(PyRuntimeError::new_err(format!(
                    "station {index} returned {} samples for {} x {} expected",
                    sim.acc.len(),
                    COMPONENT_COUNT,
                    sim.ndata
                )));
            }
            // `acc` is interleaved component-fastest; this is the de-interleave.
            for (sample, chunk) in sim.acc.chunks_exact(COMPONENT_COUNT).enumerate() {
                for (component, &value) in chunk.iter().enumerate() {
                    out[[component, index, sample]] = value;
                }
            }
        }

        waveform.ok_or_else(|| -> PyErr {
            PyValueError::new_err("no stations given, so there is nothing to simulate")
        })
    })?;
    Ok(waveform.into_pyarray(py))
}

#[pymodule]
#[pyo3(name = "_hf_simulation")]
fn hf_simulation(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFaultSegment>()?;
    m.add_class::<PySlipModel>()?;
    m.add_class::<PyVelocityModel>()?;
    m.add_function(wrap_pyfunction!(_simulate_stations, m)?)?;
    Ok(())
}
