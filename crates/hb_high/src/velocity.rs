//! The 1-D velocity model: the layers as read, Moho truncation, the air layer, and the
//! widened working model the physics reads.
//!
//! File parsing lives in Python (`workflow.realisations.HFVelocityModel1D`).

/// One layer of the working velocity model.
///
/// The mixed precision is deliberate: widening `attenuation_s` to `f64` would change
/// `geometric_spreading`'s single-precision accumulation, which matches the original code.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Layer {
    /// Cumulative depth to the base of this layer.
    pub depth_km: f64,
    /// Layer thickness.
    pub thickness_km: f64,
    /// P velocity.
    pub vp_km_s: f64,
    /// S velocity.
    pub vsh_km_s: f64,
    /// Density.
    pub density_g_cm3: f64,
    pub attenuation_p: f32,
    pub attenuation_s: f32,
}

/// The working velocity model: layers from the surface down, 0-based. `len()` is the layer
/// count.
pub type VelocityModel = Vec<Layer>;

/// One layer as read from file.
///
/// `depth_km` and `thickness_km` are `f32` here and `f64` in [`Layer`] deliberately: the
/// original code held them in single precision as read, and rounding through `f32` preserves
/// its output.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct InputLayer {
    pub depth_km: f32,
    pub thickness_km: f32,
    pub vp_km_s: f64,
    pub vsh_km_s: f64,
    pub density_g_cm3: f64,
    pub attenuation_p: f32,
    pub attenuation_s: f32,
}

/// The velocity model as read, before the air layer and the working-model widening.
pub type VelocityModelInput = Vec<InputLayer>;

impl From<InputLayer> for Layer {
    /// The unperturbed path: the input layer verbatim, widening the two `f32` fields the
    /// working model holds in `f64`.
    fn from(l: InputLayer) -> Self {
        Self {
            depth_km: l.depth_km as f64,
            thickness_km: l.thickness_km as f64,
            vp_km_s: l.vp_km_s,
            vsh_km_s: l.vsh_km_s,
            density_g_cm3: l.density_g_cm3,
            attenuation_p: l.attenuation_p,
            attenuation_s: l.attenuation_s,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("velocity model has no layers")]
    NoLayers,
    #[error(
        "the first layer already reaches vs_moho = {vs_moho_km_s} km/s, so there is no \
         model above the Moho to simulate"
    )]
    MohoAtFirstLayer { vs_moho_km_s: f64 },
}

/// Build the velocity model from layer records, truncated at the Moho.
///
/// Depth accumulation, truncation at the first layer reaching `vs_moho_km_s`, and the
/// zero-thickness bottom layer that makes reflected rays come out right.
///
/// The returned model is exactly as long as the truncation left it, so its `len()` is the
/// layer count. There is no ceiling on how many layers a caller may supply.
pub fn build_velocity_model(
    layers: &[InputLayer],
    vs_moho_km_s: f64,
) -> Result<VelocityModelInput, ModelError> {
    if layers.is_empty() {
        return Err(ModelError::NoLayers);
    }

    let mut model = VelocityModelInput::with_capacity(layers.len());
    for (i, layer) in layers.iter().enumerate() {
        let mut built = *layer;
        built.depth_km = layer.thickness_km;
        if i > 0 {
            built.depth_km += model[i - 1].depth_km;
        }

        if layer.vsh_km_s >= vs_moho_km_s {
            if i == 0 {
                return Err(ModelError::MohoAtFirstLayer { vs_moho_km_s });
            }
            // Truncate here: this layer is the half-space, and it is the last.
            built.thickness_km = 0.0;
            built.depth_km = model[i - 1].depth_km;
            model.push(built);
            return Ok(model);
        }
        model.push(built);
    }

    // Untruncated: the deepest layer is the half-space.
    let last = model.len() - 1;
    model[last].thickness_km = 0.0;
    Ok(model)
}

/// Insert the thin "air" layer at the top of the model.
///
/// Needed to get the correct free-surface reflection coefficient for
/// surface-reflected rays. The model grows by one layer.
///
/// # The air layer's Q is never set, and it does not matter
///
/// The air layer keeps the original first layer's `attenuation_p` and `attenuation_s` rather
/// than getting air-like ones. Neither field is ever read at index 0: `attenuation_p` has no
/// reader, and `attenuation_s` is read only in `geometric_spreading`, at ray-segment layer
/// indices that are never below the receiver layer, 1.
pub fn insert_air_layer(mut vmod_in: VelocityModelInput) -> VelocityModelInput {
    if !(vmod_in[0].depth_km > 0.001 && vmod_in[0].vp_km_s > 0.01) {
        return vmod_in;
    }

    // Starting from a copy of the old first layer: the five fields set below are exactly the
    // five that differ (see above for why the attenuation values are left).
    let mut air = vmod_in[0];
    air.depth_km = 0.0001;
    air.thickness_km = 0.0001;
    // Rounded through `f32` to reproduce the single-precision constants of the original code:
    // it stores 0.0010000000474974513, not 0.001.
    air.vp_km_s = 0.001f32 as f64;
    air.vsh_km_s = 0.0005f32 as f64;
    air.density_g_cm3 = 0.001f32 as f64;

    vmod_in.insert(0, air);
    vmod_in
}

/// The working model a simulation reads: the air layer inserted, then every layer widened.
///
/// See [`insert_air_layer`] for the first step and [`Layer`]'s `From` impl for the second.
pub fn working_model(input: &VelocityModelInput) -> VelocityModel {
    insert_air_layer(input.clone())
        .into_iter()
        .map(Layer::from)
        .collect()
}

/// The layer a source at `depth_km` sits in: the first whose base is at or below it.
///
/// A source deeper than the whole model takes the deepest layer: it is in the half-space, and
/// the half-space is the bottom layer. Reachable when a model is truncated above the fault's
/// base (e.g. at the Moho).
///
/// This is the lookup for the *medium* at a source. The ray tracer places its source
/// differently, nudging it clear of interfaces and out of the zero-thickness half-space; see
/// `ray::source_layer`. The two can disagree for a source on or near an interface.
pub fn layer_containing(vmod: &VelocityModel, depth_km: f32) -> usize {
    vmod.iter()
        .position(|layer| layer.depth_km >= depth_km as f64)
        .unwrap_or(vmod.len() - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Layer records built the way a Python caller builds them:
    /// `(thickness_km, vp, vsh, density, qp, qs)` per layer.
    fn layers(rows: &[(f32, f64, f64, f64, f32, f32)]) -> Vec<InputLayer> {
        rows.iter()
            .map(
                |&(thickness_km, vp_km_s, vsh_km_s, density_g_cm3, qp, qs)| {
                    InputLayer {
                        // Derived by build_velocity_model, so deliberately not supplied.
                        depth_km: 0.0,
                        thickness_km,
                        vp_km_s,
                        vsh_km_s,
                        density_g_cm3,
                        attenuation_p: qp,
                        attenuation_s: qs,
                    }
                },
            )
            .collect()
    }

    #[test]
    fn a_moho_in_the_first_layer_is_an_error_not_a_panic() {
        // Returns an error rather than panicking: this faces untrusted input from Python.
        let layers = [InputLayer {
            depth_km: 0.0,
            thickness_km: 1.0,
            vp_km_s: 8.0,
            vsh_km_s: 4.6,
            density_g_cm3: 3.3,
            attenuation_p: 400.0,
            attenuation_s: 200.0,
        }];
        assert!(matches!(
            build_velocity_model(&layers, 4.0),
            Err(ModelError::MohoAtFirstLayer { .. })
        ));
        assert!(matches!(
            build_velocity_model(&[], 999.9),
            Err(ModelError::NoLayers)
        ));
    }

    #[test]
    fn velocity_model_truncates_at_the_moho_and_zeroes_the_base() {
        let model = layers(&[
            (1.0, 2.0, 1.0, 2.0, 100.0, 50.0),
            (2.0, 4.0, 2.5, 2.5, 200.0, 100.0),
            (3.0, 8.0, 4.6, 3.3, 400.0, 200.0),
        ]);
        // vsmoho below the third layer's 4.6 truncates there.
        let v = build_velocity_model(&model, 4.0).unwrap();
        assert_eq!(v.len(), 3);
        let base = v.len() - 1;
        assert_eq!(v[base].thickness_km, 0.0, "the Moho layer is zeroed");
        assert_eq!(v[base].depth_km, v[base - 1].depth_km);
    }

    #[test]
    fn velocity_model_without_moho_still_zeroes_the_base() {
        let model = layers(&[
            (1.0, 2.0, 1.0, 2.0, 100.0, 50.0),
            (2.0, 4.0, 2.5, 2.5, 200.0, 100.0),
        ]);
        let v = build_velocity_model(&model, 999.9).unwrap();
        assert_eq!(v.len(), 2);
        // The base of a 2-layer model is index 1, and the first layer's cumulative depth is
        // index 0.
        assert_eq!(v[1].thickness_km, 0.0);
        assert_eq!(v[0].depth_km, 1.0);
    }

    #[test]
    fn air_layer_is_inserted_for_a_realistic_model() {
        let model = layers(&[
            (0.05, 1.8, 0.5, 1.81, 116.0, 58.0),
            (2.0, 4.0, 2.5, 2.5, 200.0, 100.0),
        ]);
        let built = build_velocity_model(&model, 999.9).unwrap();
        let layers_before = built.len();
        let qp1_before = built[0].attenuation_p;
        let v = insert_air_layer(built);
        assert_eq!(
            v.len(),
            layers_before + 1,
            "production models do get the air layer"
        );
        assert_eq!(v[0].thickness_km, 0.0001);
        // Not 0.001f64: the original constant was single precision.
        assert_eq!(v[0].vp_km_s, 0.001f32 as f64);
        assert_eq!(v[0].vsh_km_s, 0.0005f32 as f64);
        assert_eq!(v[0].density_g_cm3, 0.001f32 as f64);
        assert_eq!(
            v[1].thickness_km, 0.05,
            "the original first layer shifted down"
        );
        // The air layer starts as a copy of the old first layer and only five of its seven
        // fields are set, so Q stays put.
        assert_eq!(
            v[0].attenuation_p, qp1_before,
            "the air layer's Q is left as-is; nothing reads it, so this pins the shift \
             rather than a physical choice"
        );
    }

    #[test]
    fn air_layer_is_skipped_when_the_model_starts_at_the_surface() {
        let model = layers(&[
            (0.0, 1.8, 0.5, 1.81, 116.0, 58.0),
            (2.0, 4.0, 2.5, 2.5, 200.0, 100.0),
        ]);
        let built = build_velocity_model(&model, 999.9).unwrap();
        let layers_before = built.len();
        let v = insert_air_layer(built);
        assert_eq!(v.len(), layers_before);
    }
}
