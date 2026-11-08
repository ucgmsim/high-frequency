"""Batch invariants: the properties only a batched API can have.

None of these asserts a computed sample value; they assert the *relationships* a caller
depends on, which must hold for any correct implementation. Per-station seeding is what
makes the three waveform invariants possible.
"""

from collections.abc import Iterable

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hf_simulation import (
    COMPONENTS,
    FaultSegment,
    HfConfig,
    PathParameters,
    RecordParameters,
    Simulator,
    SlipModel,
    VelocityModel1D,
    station_seeds,
)

STATION_NAMES = ["CHCH", "LINC", "REHS", "SHFC"]
STATION_LATITUDE = np.array([-43.4, -43.6, -43.5, -43.3], np.float32)
STATION_LONGITUDE = np.array([172.6, 172.4, 172.7, 172.5], np.float32)
# Deliberately different per station, so a parameter applied to the wrong station shows.
STATION_KAPPA_S = np.array([0.03, 0.045, 0.06, 0.02], np.float32)
STATION_Q_EXPONENT = np.array([0.5, 0.6, 0.7, 0.55], np.float32)

# 40 s, not 20. The stations sit ~55 km out and shear velocity is ~3.5 km/s, so the S
# arrival lands near 16 s; a 20 s record technically contains it and a 10 s record is
# silence. Getting this wrong produces an all-zero array rather than an error, which is
# exactly why `assert_not_silent` exists below.
RECORD_DURATION_S = 40.0


def fault_segment(
    stress_drop_bars: np.ndarray | None = None,
    dip_deg: float = 70.0,
    top_depth_km: float = 0.0,
) -> FaultSegment:
    """A 4-by-3 subfault segment with uniform slip.

    Parameters
    ----------
    stress_drop_bars : np.ndarray, optional
        Per-subfault stress parameter, shaped ``(3, 4)``. Defaults to 50 bars everywhere.
    dip_deg : float, optional
        Dip of the segment. Defaults to 70 degrees.
    top_depth_km : float, optional
        Depth of the segment's top edge. Defaults to 0 km.

    Returns
    -------
    FaultSegment
        The segment.
    """
    return FaultSegment(
        longitude_deg=173.0,
        latitude_deg=-43.0,
        strike_deg=220.0,
        dip_deg=dip_deg,
        rake_deg=160.0,
        top_depth_km=top_depth_km,
        subfault_length_km=1.5,
        subfault_width_km=1.5,
        hypocentre_along_strike_km=0.0,
        hypocentre_down_dip_km=1.5,
        slip=np.full((3, 4), 50.0, np.float32),
        rise_time_s=np.full((3, 4), 0.5, np.float32),
        rupture_time_s=np.zeros((3, 4), np.float32),
        stress_drop_bars=(
            np.full((3, 4), 50.0, np.float32)
            if stress_drop_bars is None
            else stress_drop_bars
        ),
    )


@pytest.fixture(scope="module")
def slip_model() -> SlipModel:
    """A single-segment fault with uniform slip and stress parameter.

    Returns
    -------
    SlipModel
        A 4-by-3 subfault grid.
    """
    return SlipModel([fault_segment()])


@pytest.fixture(scope="module")
def velocity_model() -> VelocityModel1D:
    """A four-layer crustal model with no Moho truncation.

    Returns
    -------
    VelocityModel1D
        The model.
    """
    return VelocityModel1D(
        thickness_km=np.array([1.0, 2.0, 5.0, 0.0], np.float32),
        vp_km_s=np.array([3.0, 5.0, 6.0, 7.5]),
        vsh_km_s=np.array([1.5, 2.8, 3.5, 4.2]),
        density_g_cm3=np.array([2.1, 2.5, 2.7, 3.0]),
        quality_factor_p=np.array([100.0, 200.0, 400.0, 500.0], np.float32),
        quality_factor_s=np.array([50.0, 100.0, 200.0, 250.0], np.float32),
        vs_moho_km_s=999.9,
    )


def simulate(
    slip_model: SlipModel,
    velocity_model: VelocityModel1D,
    indices: Iterable[int],
) -> np.ndarray:
    """Simulate the stations named by ``indices``, in that order.

    Parameters
    ----------
    slip_model : SlipModel
        The fault.
    velocity_model : VelocityModel1D
        The velocity structure.
    indices : Sequence of int
        Which of :data:`STATION_NAMES` to run, in the order to run them.

    Returns
    -------
    np.ndarray
        Waveforms, shaped ``(3, len(indices), n_time)``.
    """
    simulator = Simulator(
        slip_model,
        velocity_model,
        HfConfig(record=RecordParameters(duration_s=RECORD_DURATION_S)),
    )
    return simulator.run_stations(
        latitude_deg=STATION_LATITUDE[list(indices)],
        longitude_deg=STATION_LONGITUDE[list(indices)],
        station_seed=station_seeds(1234, [STATION_NAMES[i] for i in indices]),
        kappa_s=STATION_KAPPA_S[list(indices)],
        q_frequency_exponent=STATION_Q_EXPONENT[list(indices)],
    )


def assert_not_silent(waveform: np.ndarray) -> None:
    """Fail if any station is entirely zero, or any sample is not finite.

    A record too short to contain the S arrival comes back as exact zeros rather than as
    an error, so every test that compares waveforms has to rule that out first —
    otherwise ``all zeros == all zeros`` passes every equality assertion below.

    Parameters
    ----------
    waveform : np.ndarray
        Shaped ``(3, n_station, n_time)``.
    """
    assert np.isfinite(waveform).all(), "waveform contains non-finite samples"
    peak = np.abs(waveform).max(axis=(0, 2))
    assert (peak > 0).all(), f"station(s) produced silence: peak amplitudes {peak}"


def test_shape_and_components(
    slip_model: SlipModel, velocity_model: VelocityModel1D
) -> None:
    """The result is (component, station, time) with three components."""
    waveform = simulate(slip_model, velocity_model, range(4))
    assert waveform.shape == (
        len(COMPONENTS),
        4,
        int(RECORD_DURATION_S / RecordParameters(duration_s=RECORD_DURATION_S).dt),
    )
    assert waveform.dtype == np.float32
    assert_not_silent(waveform)


def test_station_order_does_not_matter(
    slip_model: SlipModel, velocity_model: VelocityModel1D
) -> None:
    """Permuting the stations permutes the rows and changes nothing else.

    This is the invariant that makes a dask chunking strategy free: if order mattered,
    every rechunk would silently change the science.
    """
    order = [2, 0, 3, 1]
    reference = simulate(slip_model, velocity_model, range(4))
    assert_not_silent(reference)
    permuted = simulate(slip_model, velocity_model, order)
    np.testing.assert_array_equal(permuted, reference[:, order, :])


def test_subsetting_equals_slicing(
    slip_model: SlipModel, velocity_model: VelocityModel1D
) -> None:
    """Simulating a subset gives exactly the corresponding slice of the whole batch.

    This is what makes a failed run resumable: re-running the stations that are missing
    cannot disagree with the ones already written.
    """
    reference = simulate(slip_model, velocity_model, range(4))
    assert_not_silent(reference)
    subset = simulate(slip_model, velocity_model, [1, 2])
    np.testing.assert_array_equal(subset, reference[:, 1:3, :])


def test_a_batch_of_one_matches_the_batch(
    slip_model: SlipModel, velocity_model: VelocityModel1D
) -> None:
    """One station alone equals that station inside a batch.

    Fails if stations share one generator, so that a station's waveform depends on those
    simulated before it.
    """
    reference = simulate(slip_model, velocity_model, range(4))
    assert_not_silent(reference)
    alone = simulate(slip_model, velocity_model, [3])
    np.testing.assert_array_equal(alone, reference[:, 3:4, :])


def test_different_stations_get_different_waveforms(
    slip_model: SlipModel, velocity_model: VelocityModel1D
) -> None:
    """Independent seeds must actually produce independent realisations.

    Guards the failure mode where every station accidentally shares one stream — which
    would satisfy all three invariants above and still be wrong.
    """
    waveform = simulate(slip_model, velocity_model, range(4))
    assert_not_silent(waveform)
    for i in range(4):
        for j in range(i + 1, 4):
            assert not np.array_equal(waveform[:, i, :], waveform[:, j, :]), (
                f"{STATION_NAMES[i]} and {STATION_NAMES[j]} produced identical "
                "waveforms, so they are sharing an RNG stream"
            )


def run_batch(
    slip_model: SlipModel,
    velocity_model: VelocityModel1D,
    kappa_s: np.ndarray,
    q_frequency_exponent: np.ndarray,
) -> np.ndarray:
    """Simulate all four stations with the given per-station parameters.

    Parameters
    ----------
    slip_model : SlipModel
        The fault.
    velocity_model : VelocityModel1D
        The velocity structure.
    kappa_s : np.ndarray
        Kappa, one per station.
    q_frequency_exponent : np.ndarray
        Q exponent, one per station.

    Returns
    -------
    np.ndarray
        Waveforms, shaped ``(3, 4, n_time)``.
    """
    simulator = Simulator(
        slip_model,
        velocity_model,
        HfConfig(record=RecordParameters(duration_s=RECORD_DURATION_S)),
    )
    return simulator.run_stations(
        latitude_deg=STATION_LATITUDE,
        longitude_deg=STATION_LONGITUDE,
        station_seed=station_seeds(1234, STATION_NAMES),
        kappa_s=kappa_s,
        q_frequency_exponent=q_frequency_exponent,
    )


def test_per_station_parameters_change_only_their_station(
    slip_model: SlipModel, velocity_model: VelocityModel1D
) -> None:
    """Changing one station's kappa or Q exponent changes that station and no other."""
    reference = run_batch(
        slip_model, velocity_model, STATION_KAPPA_S, STATION_Q_EXPONENT
    )
    assert_not_silent(reference)

    kappa, q_exponent = STATION_KAPPA_S.copy(), STATION_Q_EXPONENT.copy()
    kappa[2] *= 1.5
    q_exponent[2] *= 1.5
    others = [0, 1, 3]
    for name, waveform in [
        ("kappa_s", run_batch(slip_model, velocity_model, kappa, STATION_Q_EXPONENT)),
        (
            "q_frequency_exponent",
            run_batch(slip_model, velocity_model, STATION_KAPPA_S, q_exponent),
        ),
    ]:
        assert not np.array_equal(waveform[:, 2, :], reference[:, 2, :]), (
            f"{name} had no effect on its station"
        )
        np.testing.assert_array_equal(waveform[:, others, :], reference[:, others, :])


def test_stress_drop_is_per_subfault(velocity_model: VelocityModel1D) -> None:
    """One subfault's stress parameter moves the record; raising it everywhere adds energy.

    The stress parameter sets each subfault's high-frequency level linearly, so doubling it
    everywhere should roughly quadruple the record's energy. The bound is loose because the
    window durations shorten too, which changes the draws.
    """
    uniform = np.full((3, 4), 50.0, np.float32)
    reference = simulate(SlipModel([fault_segment(uniform)]), velocity_model, range(4))
    assert_not_silent(reference)

    one_hot = uniform.copy()
    one_hot[1, 2] = 100.0
    changed = simulate(SlipModel([fault_segment(one_hot)]), velocity_model, range(4))
    assert not np.array_equal(changed, reference), (
        "one subfault's stress parameter had no effect"
    )

    doubled = simulate(
        SlipModel([fault_segment(2 * uniform)]), velocity_model, range(4)
    )
    energy_ratio = np.sum(doubled.astype(np.float64) ** 2) / np.sum(
        reference.astype(np.float64) ** 2
    )
    assert energy_ratio > 2.0, (
        f"doubling the stress parameter scaled energy by {energy_ratio}"
    )


@pytest.mark.parametrize("dip_deg", [0.0, 0.005])
def test_a_subfault_in_the_air_layer_is_placed_in_the_first_real_layer(
    velocity_model: VelocityModel1D, dip_deg: float
) -> None:
    """A subfault within 0.1 m of the surface behaves as one just below the air layer.

    The inserted air layer is 0.1 m thick and is not ground. A near-horizontal fault at
    the surface puts subfault centres inside it, where they used to take the air's shear
    velocity and density, or panic in the ray tracer. Moving the fault 0.2 m deeper, below
    the air layer, must make no material difference.

    LINC is left out: it is ~83 km away, and from a source in the 1.5 km/s surface layer
    its S arrival falls after the end of the record.
    """
    stations = [0, 2, 3]
    surface = simulate(
        SlipModel([fault_segment(dip_deg=dip_deg, top_depth_km=0.0)]),
        velocity_model,
        stations,
    )
    assert_not_silent(surface)
    deeper = simulate(
        SlipModel([fault_segment(dip_deg=dip_deg, top_depth_km=0.0002)]),
        velocity_model,
        stations,
    )
    assert_not_silent(deeper)
    assert surface.shape == deeper.shape
    np.testing.assert_allclose(
        np.abs(surface).max(axis=(0, 2)), np.abs(deeper).max(axis=(0, 2)), rtol=1e-3
    )


@pytest.mark.parametrize(
    ("stress_drop_bars", "message"),
    [
        (np.full((2, 4), 50.0, np.float32), "same shape"),
        (np.zeros((3, 4), np.float32), "positive and finite"),
        (np.full((3, 4), np.nan, np.float32), "positive and finite"),
    ],
)
def test_a_bad_stress_drop_grid_is_rejected(
    stress_drop_bars: np.ndarray, message: str
) -> None:
    """The grid must match the slip grid, and be positive and finite throughout."""
    with pytest.raises(ValueError, match=message):
        fault_segment(stress_drop_bars)


@pytest.mark.parametrize(
    ("grid", "value", "message"),
    [
        ("slip", np.nan, "slip must be non-negative and finite"),
        ("slip", np.inf, "slip must be non-negative and finite"),
        ("slip", -1.0, "slip must be non-negative and finite"),
        ("rupture_time_s", np.nan, "rupture_time_s must be finite"),
        ("rupture_time_s", np.inf, "rupture_time_s must be finite"),
    ],
)
def test_a_bad_slip_or_rupture_time_cell_is_rejected(
    grid: str, value: float, message: str
) -> None:
    """One non-finite (or, for slip, negative) cell is an error, not a silent record.

    A NaN slip poisons the total moment and so every subfault; a NaN rupture time
    truncates to zero and silently moves its subfault to origin time.
    """
    grids = {
        "slip": np.full((3, 4), 50.0, np.float32),
        "rise_time_s": np.full((3, 4), 0.5, np.float32),
        "rupture_time_s": np.zeros((3, 4), np.float32),
        "stress_drop_bars": np.full((3, 4), 50.0, np.float32),
    }
    grids[grid][1, 2] = value
    with pytest.raises(ValueError, match=message):
        FaultSegment(
            longitude_deg=173.0,
            latitude_deg=-43.0,
            strike_deg=220.0,
            dip_deg=70.0,
            rake_deg=160.0,
            top_depth_km=0.0,
            subfault_length_km=1.5,
            subfault_width_km=1.5,
            hypocentre_along_strike_km=0.0,
            hypocentre_down_dip_km=1.5,
            **grids,
        )


@pytest.mark.parametrize(
    ("duration_s", "dt"),
    [
        (RECORD_DURATION_S, 0.0),
        (RECORD_DURATION_S, -0.005),
        (RECORD_DURATION_S, float("nan")),
        (RECORD_DURATION_S, float("inf")),
        (0.0, 0.005),
        (-1.0, 0.005),
        (float("nan"), 0.005),
        (float("inf"), 0.005),
        (0.001, 0.005),
    ],
)
def test_a_bad_record_shape_is_rejected(
    slip_model: SlipModel,
    velocity_model: VelocityModel1D,
    duration_s: float,
    dt: float,
) -> None:
    """Duration and `dt` must be positive and finite, with room for at least one sample."""
    with pytest.raises(ValueError, match="duration_s and dt"):
        Simulator(
            slip_model,
            velocity_model,
            HfConfig(record=RecordParameters(duration_s=duration_s, dt=dt)),
        )


@pytest.mark.parametrize(
    ("rayset", "message"),
    [((), "rayset is empty"), ((-1,), "ray type -1 is negative")],
)
def test_a_bad_rayset_is_rejected(
    slip_model: SlipModel,
    velocity_model: VelocityModel1D,
    rayset: tuple[int, ...],
    message: str,
) -> None:
    """An empty rayset or a negative ray type is a ValueError when the simulator is built.

    Otherwise an empty rayset silently gives an all-zero record, and a negative ray type
    panics per station inside `run_stations`.
    """
    config = HfConfig(
        record=RecordParameters(duration_s=RECORD_DURATION_S),
        # -1 is not a Ray; it is passed on purpose to reach the Rust-side check.
        path=PathParameters(rayset=rayset),  # ty: ignore[invalid-argument-type]
    )
    with pytest.raises(ValueError, match=message):
        Simulator(slip_model, velocity_model, config)


@pytest.mark.parametrize("thickness_km", [0.0, 10.0])
def test_a_lone_half_space_is_rejected(thickness_km: float) -> None:
    """A single layer below the Moho is all half-space, with no model above it to trace."""
    with pytest.raises(ValueError, match="no layer above the half-space"):
        VelocityModel1D(
            thickness_km=np.array([thickness_km], np.float32),
            vp_km_s=np.array([6.0]),
            vsh_km_s=np.array([3.5]),
            density_g_cm3=np.array([2.7]),
            quality_factor_p=np.array([400.0], np.float32),
            quality_factor_s=np.array([200.0], np.float32),
            vs_moho_km_s=999.9,
        )


def test_an_empty_batch_is_an_error(
    slip_model: SlipModel, velocity_model: VelocityModel1D
) -> None:
    """Zero stations cannot yield an array of unknown time length."""
    empty32 = np.array([], np.float32)
    simulator = Simulator(
        slip_model,
        velocity_model,
        HfConfig(record=RecordParameters(duration_s=RECORD_DURATION_S)),
    )
    with pytest.raises(ValueError, match="nothing to simulate"):
        simulator.run_stations(
            latitude_deg=empty32,
            longitude_deg=empty32,
            station_seed=np.array([], np.uint64),
            kappa_s=empty32,
            q_frequency_exponent=empty32,
        )


def test_mismatched_station_arrays_are_rejected(
    slip_model: SlipModel, velocity_model: VelocityModel1D
) -> None:
    """One entry per station, in every array, or an error naming every length."""
    simulator = Simulator(
        slip_model,
        velocity_model,
        HfConfig(record=RecordParameters(duration_s=RECORD_DURATION_S)),
    )
    with pytest.raises(ValueError, match="one entry per station"):
        simulator.run_stations(
            latitude_deg=STATION_LATITUDE,
            longitude_deg=STATION_LONGITUDE[:2],
            station_seed=station_seeds(1234, STATION_NAMES),
            kappa_s=STATION_KAPPA_S,
            q_frequency_exponent=STATION_Q_EXPONENT,
        )
    with pytest.raises(ValueError, match="one entry per station"):
        simulator.run_stations(
            latitude_deg=STATION_LATITUDE,
            longitude_deg=STATION_LONGITUDE,
            station_seed=station_seeds(1234, STATION_NAMES),
            kappa_s=STATION_KAPPA_S[:2],
            q_frequency_exponent=STATION_Q_EXPONENT,
        )


# ---------------------------------------------------------------------------
# station_seeds -- the same three invariants, one level down
# ---------------------------------------------------------------------------

names_strategy = st.lists(
    st.text(min_size=1, max_size=12), min_size=1, max_size=30, unique=True
)


@given(names=names_strategy, root=st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=50, deadline=None)
def test_station_seeds_do_not_depend_on_order(names: list[str], root: int) -> None:
    """A name's seed is a function of the name, not of where it sits in the list."""
    reference = dict(zip(names, station_seeds(root, names)))
    shuffled = list(reversed(names))
    for name, seed in zip(shuffled, station_seeds(root, shuffled)):
        assert seed == reference[name]


@given(names=names_strategy, root=st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=50, deadline=None)
def test_station_seeds_subset_equals_slice(names: list[str], root: int) -> None:
    """``station_seeds(root, names[a:b]) == station_seeds(root, names)[a:b]``."""
    whole = station_seeds(root, names)
    half = len(names) // 2
    np.testing.assert_array_equal(station_seeds(root, names[half:]), whole[half:])


@given(names=names_strategy, root=st.integers(min_value=0, max_value=2**31 - 2))
@settings(max_examples=50, deadline=None)
def test_adjacent_roots_share_no_seeds(names: list[str], root: int) -> None:
    """Adjacent root seeds give unrelated station seeds.

    A bare XOR of root and name hash would fail this: incrementing the root would flip one
    bit of every station seed.
    """
    assert not (
        set(station_seeds(root, names).tolist())
        & set(station_seeds(root + 1, names).tolist())
    )


@given(names=names_strategy, root=st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=50, deadline=None)
def test_station_seeds_are_distinct_and_unsigned(names: list[str], root: int) -> None:
    """Distinct names give distinct seeds, and no seed is negative."""
    seeds = station_seeds(root, names)
    assert seeds.dtype == np.uint64
    assert len(set(seeds.tolist())) == len(names)


def test_a_negative_root_seed_is_rejected() -> None:
    """``SeedSequence`` cannot take negative entropy, so say so rather than crash."""
    with pytest.raises(ValueError, match="must be non-negative"):
        station_seeds(-1, ["CHCH"])


def test_station_arrays_of_any_numeric_dtype_are_converted(
    slip_model: SlipModel, velocity_model: VelocityModel1D
) -> None:
    """float64 coordinates and parameters, and int64 seeds, are cast rather than refused.

    The extension itself only takes float32 and uint64 (it rejects anything else with a
    TypeError), so it is the wrapper's conversion that makes the natural NumPy defaults
    usable -- and the cast must not change the result.
    """
    simulator = Simulator(
        slip_model,
        velocity_model,
        HfConfig(record=RecordParameters(duration_s=RECORD_DURATION_S)),
    )
    seeds = np.array([11, 22, 33, 44])
    reference = simulator.run_stations(
        latitude_deg=STATION_LATITUDE,
        longitude_deg=STATION_LONGITUDE,
        station_seed=seeds.astype(np.uint64),
        kappa_s=STATION_KAPPA_S,
        q_frequency_exponent=STATION_Q_EXPONENT,
    )
    assert_not_silent(reference)
    # Deliberately not float32 / uint64: these are the dtypes NumPy picks by default.
    converted = simulator.run_stations(
        latitude_deg=STATION_LATITUDE.astype(np.float64),  # ty: ignore[invalid-argument-type]
        longitude_deg=STATION_LONGITUDE.astype(np.float64),  # ty: ignore[invalid-argument-type]
        station_seed=seeds.astype(np.int64),  # ty: ignore[invalid-argument-type]
        kappa_s=STATION_KAPPA_S.astype(np.float64),  # ty: ignore[invalid-argument-type]
        q_frequency_exponent=STATION_Q_EXPONENT.astype(np.float64),  # ty: ignore[invalid-argument-type]
    )
    assert converted.dtype == np.float32
    np.testing.assert_array_equal(converted, reference)


@pytest.mark.parametrize(
    ("short", "expected"),
    [
        ("latitude_deg", "'latitude_deg': (2,), 'longitude_deg': (4,)"),
        ("longitude_deg", "'latitude_deg': (4,), 'longitude_deg': (2,)"),
    ],
)
def test_the_mismatch_error_names_each_array_and_its_shape(
    slip_model: SlipModel, velocity_model: VelocityModel1D, short: str, expected: str
) -> None:
    """The error reports every array by its argument name, so the bad one can be found."""
    simulator = Simulator(
        slip_model,
        velocity_model,
        HfConfig(record=RecordParameters(duration_s=RECORD_DURATION_S)),
    )
    arrays = {
        "latitude_deg": STATION_LATITUDE,
        "longitude_deg": STATION_LONGITUDE,
        "station_seed": station_seeds(1234, STATION_NAMES),
        "kappa_s": STATION_KAPPA_S,
        "q_frequency_exponent": STATION_Q_EXPONENT,
    }
    arrays[short] = arrays[short][:2]
    with pytest.raises(ValueError) as raised:
        simulator.run_stations(**arrays)
    message = str(raised.value)
    assert expected in message
    assert "'station_seed': (4,), 'kappa_s': (4,), 'q_frequency_exponent': (4,)" in (
        message
    )


def test_station_seeds_are_pinned() -> None:
    """Seeds are stable across releases, so a past run's waveforms can be reproduced.

    Any change to the name hash (byte order, digest size, encoding) or to how the seed is
    drawn from the ``SeedSequence`` (word size) would silently re-seed every station of
    every run. These values were produced by the current implementation.
    """
    seeds = station_seeds(1234, ["CHCH", "LINC", "Ōtautahi"])
    assert seeds.dtype == np.uint64
    assert seeds.tolist() == [
        1905741496502012377,
        13603888227495392929,
        10769600984917523404,
    ]
    # A root seed wider than 32 bits is accepted and mixed in.
    assert station_seeds(2**40, ["CHCH"]).tolist() == [9577342823655924923]


def test_no_station_names_give_an_empty_uint64_array() -> None:
    """An empty name list still yields seeds of the documented dtype, not float64."""
    seeds = station_seeds(1234, [])
    assert seeds.shape == (0,)
    assert seeds.dtype == np.uint64


def test_the_negative_root_seed_error_says_why() -> None:
    """The message gives the offending value and the reason, verbatim."""
    with pytest.raises(
        ValueError,
        match=r"^root_seed must be non-negative, got -5; SeedSequence rejects "
        r"negative entropy$",
    ):
        station_seeds(-5, ["CHCH"])


def test_the_station_array_errors_are_verbatim(
    slip_model: SlipModel, velocity_model: VelocityModel1D
) -> None:
    """Both batch-shape errors read exactly as written, from start to end.

    The mismatch message lists the arguments it checks, then every shape. The empty-batch
    message says why nothing can be returned.
    """
    simulator = Simulator(
        slip_model,
        velocity_model,
        HfConfig(record=RecordParameters(duration_s=RECORD_DURATION_S)),
    )
    with pytest.raises(ValueError) as raised:
        simulator.run_stations(
            latitude_deg=STATION_LATITUDE,
            longitude_deg=STATION_LONGITUDE,
            station_seed=station_seeds(1234, STATION_NAMES),
            kappa_s=STATION_KAPPA_S,
            q_frequency_exponent=STATION_Q_EXPONENT[:1],
        )
    assert str(raised.value) == (
        "latitude_deg, longitude_deg, station_seed, kappa_s and q_frequency_exponent "
        "must have one entry per station; got shapes {'latitude_deg': (4,), "
        "'longitude_deg': (4,), 'station_seed': (4,), 'kappa_s': (4,), "
        "'q_frequency_exponent': (1,)}"
    )

    empty32 = np.array([], np.float32)
    with pytest.raises(ValueError) as raised:
        simulator.run_stations(
            latitude_deg=empty32,
            longitude_deg=empty32,
            station_seed=np.array([], np.uint64),
            kappa_s=empty32,
            q_frequency_exponent=empty32,
        )
    assert str(raised.value) == "no stations given, so there is nothing to simulate"
