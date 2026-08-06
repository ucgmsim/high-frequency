//! Python bindings for the `hb_high` stochastic high-frequency generator.

use hb_high::config::{
    HfConfig, PathDurationModel, PathParameters, RayType, RecordParameters, RuptureVelocity,
    SiteParameters, SourceParameters,
};
use hb_high::input::{Segment, Slip, Station, StochModel, Subfault, build_velocity_model};
use hb_high::sim::Simulator;
use hb_high::state::{InputLayer, VelocityModelInput};
use numpy::ndarray::{Array3, s};
use numpy::{IntoPyArray, PyArray3, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
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
                    slip: Slip(slip[[j, i]]),
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

/// The earthquake source: how strong the high-frequency radiation is, and how fast the
/// rupture travels.
#[pyclass(frozen, name = "SourceParameters")]
pub struct PySourceParameters {
    inner: SourceParameters,
}

#[pymethods]
impl PySourceParameters {
    #[new]
    #[pyo3(signature = (*, stress_drop_bars, corner_frequency_constant, corner_frequency_alpha,
                        rupture_velocity_fraction, rupture_velocity_shallow,
                        rupture_velocity_deep, rupture_velocity_sigma))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        stress_drop_bars: f32,
        corner_frequency_constant: f32,
        corner_frequency_alpha: f32,
        rupture_velocity_fraction: f32,
        rupture_velocity_shallow: f32,
        rupture_velocity_deep: f32,
        rupture_velocity_sigma: f32,
    ) -> Self {
        Self {
            inner: SourceParameters {
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
        }
    }
}

/// The path from source to site: which rays, and how the medium attenuates along them.
#[pyclass(frozen, name = "PathParameters")]
pub struct PyPathParameters {
    inner: PathParameters,
}

#[pymethods]
impl PyPathParameters {
    #[new]
    #[pyo3(signature = (*, rayset, q_frequency_exponent, path_duration_model))]
    fn new(
        rayset: Vec<i32>,
        q_frequency_exponent: f32,
        path_duration_model: i32,
    ) -> PyResult<Self> {
        // Only the checks Python cannot make for itself: this one decodes a non-contiguous
        // integer set that the Rust enum owns.
        let path_duration = PathDurationModel::from_deck(path_duration_model).ok_or_else(|| {
            PyValueError::new_err(format!(
                "path_duration_model {path_duration_model} is not one of 0, 1, 2, 11, 12"
            ))
        })?;
        Ok(Self {
            inner: PathParameters {
                rayset: rayset.into_iter().map(RayType).collect(),
                q_exponent: q_frequency_exponent,
                path_duration,
            },
        })
    }
}

/// The near-surface: what happens in the last few hundred metres.
///
/// Quarter-wavelength site amplification is always applied and is not a field here.
#[pyclass(frozen, name = "SiteParameters")]
pub struct PySiteParameters {
    inner: SiteParameters,
}

#[pymethods]
impl PySiteParameters {
    #[new]
    #[pyo3(signature = (*, kappa_s, fmax_hz))]
    fn new(kappa_s: f32, fmax_hz: f32) -> Self {
        Self {
            inner: SiteParameters {
                kappa_s,
                f_max_hz: fmax_hz,
            },
        }
    }
}

/// The shape of the record to produce.
#[pyclass(frozen, name = "RecordParameters")]
pub struct PyRecordParameters {
    inner: RecordParameters,
}

#[pymethods]
impl PyRecordParameters {
    #[new]
    #[pyo3(signature = (*, duration_s, dt))]
    fn new(duration_s: f32, dt: f32) -> Self {
        Self {
            inner: RecordParameters {
                duration_s,
                dt_s: dt,
            },
        }
    }
}

/// Everything needed to simulate, with nothing about where the inputs came from.
#[pyclass(frozen, name = "HfConfig")]
pub struct PyHfConfig {
    inner: HfConfig,
}

#[pymethods]
impl PyHfConfig {
    #[new]
    #[pyo3(signature = (*, source, path, site, record))]
    fn new(
        source: &PySourceParameters,
        path: &PyPathParameters,
        site: &PySiteParameters,
        record: &PyRecordParameters,
    ) -> Self {
        Self {
            inner: HfConfig {
                source: source.inner.clone(),
                path: path.inner.clone(),
                site: site.inner.clone(),
                record: record.inner.clone(),
            },
        }
    }
}

/// A configured simulation, ready to run stations against.
///
/// Built once per source; the station-independent work — the air layer, the slip-model
/// normalisation, the moment scaling — happens here rather than per station.
///
/// # Threads, not processes
///
/// `run_stations` releases the GIL and takes `&self`, so one of these can be shared across a
/// **dask thread pool**. It is deliberately not picklable: `dask.distributed` would need to
/// send it between processes, and a silent re-normalisation on the far side is a worse
/// failure than a loud one here.
#[pyclass(frozen, name = "Simulator")]
pub struct PySimulator {
    inner: Simulator,
}

#[pymethods]
impl PySimulator {
    #[new]
    #[pyo3(signature = (config, slip_model, velocity_model))]
    fn new(
        config: &PyHfConfig,
        slip_model: &PySlipModel,
        velocity_model: &PyVelocityModel,
    ) -> PyResult<Self> {
        Simulator::new(&config.inner, &slip_model.inner, &velocity_model.input)
            .map(|inner| Self { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Simulate a batch of stations.
    ///
    /// Returns acceleration in cm/s^2, shaped `(3, n_station, n_time)` with components
    /// ordered 090/000/vertical.
    #[pyo3(signature = (*, latitude_deg, longitude_deg, station_seed))]
    fn run_stations<'py>(
        &self,
        py: Python<'py>,
        latitude_deg: PyReadonlyArray1<f32>,
        longitude_deg: PyReadonlyArray1<f32>,
        station_seed: PyReadonlyArray1<u64>,
    ) -> PyResult<Bound<'py, PyArray3<f32>>> {
        // `as_slice` is the one check Rust must own: it fails for a non-contiguous array,
        // which Python cannot see from the outside.
        let (latitude, longitude, seeds) = (
            latitude_deg.as_slice()?,
            longitude_deg.as_slice()?,
            station_seed.as_slice()?,
        );

        let station_count = latitude.len();
        let waveform: Array3<f32> = py.detach(|| {
            let ndata = self.inner.ndata();
            let mut waveform = Array3::zeros((COMPONENT_COUNT, station_count, ndata));

            for (index, ((&stlat, &stlon), &seed)) in
                latitude.iter().zip(longitude).zip(seeds).enumerate()
            {
                let sim = self.inner.run(
                    Station {
                        latitude: stlat,
                        longitude: stlon,
                        name: format!("station-{index}"),
                    },
                    seed,
                );
                // `sim.acc` is already (n_components, n_time), so this is a row copy per
                // component into the batch's station slot.
                for (component, trace) in sim.acc.rows().into_iter().enumerate() {
                    waveform.slice_mut(s![component, index, ..]).assign(&trace);
                }
            }

            waveform
        });
        Ok(waveform.into_pyarray(py))
    }
}

#[pymodule]
#[pyo3(name = "_hf_simulation")]
fn hf_simulation(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFaultSegment>()?;
    m.add_class::<PySlipModel>()?;
    m.add_class::<PyVelocityModel>()?;
    m.add_class::<PySourceParameters>()?;
    m.add_class::<PyPathParameters>()?;
    m.add_class::<PySiteParameters>()?;
    m.add_class::<PyRecordParameters>()?;
    m.add_class::<PyHfConfig>()?;
    m.add_class::<PySimulator>()?;
    Ok(())
}
