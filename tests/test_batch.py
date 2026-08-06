"""Batch invariants: the properties only a batched API can have.

These are the tests that make a refactor confident rather than nervous. None of them
asserts a computed sample value, so none of them obstructs changing how the numbers are
produced — they assert the *relationships* a caller depends on, which must hold for any
correct implementation.

The three waveform invariants exist because the Fortran could not have them. It shared one
RNG stream across its station loop, which is why ``nsite != 1`` had to be refused outright:
a multi-station run was not the concatenation of single-station runs. Per-station seeding is
what buys them, and these tests are what keep them.
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
    Simulator,
    SlipModel,
    VelocityModel1D,
    station_seeds,
)

STATION_NAMES = ["CHCH", "LINC", "REHS", "SHFC"]
STATION_LATITUDE = np.array([-43.4, -43.6, -43.5, -43.3], np.float32)
STATION_LONGITUDE = np.array([172.6, 172.4, 172.7, 172.5], np.float32)

# 40 s, not 20. The stations sit ~55 km out and shear velocity is ~3.5 km/s, so the S
# arrival lands near 16 s; a 20 s record technically contains it and a 10 s record is
# silence. Getting this wrong produces an all-zero array rather than an error, which is
# exactly why `assert_not_silent` exists below.
RECORD_DURATION_S = 40.0


@pytest.fixture(scope="module")
def slip_model() -> SlipModel:
    """A single-segment fault with uniform slip.

    Returns
    -------
    SlipModel
        A 4-by-3 subfault grid.
    """
    return SlipModel(
        [
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
                slip=np.full((3, 4), 50.0, np.float32),
                rise_time_s=np.full((3, 4), 0.5, np.float32),
                rupture_time_s=np.zeros((3, 4), np.float32),
            )
        ]
    )


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
        slip_model, velocity_model, HfConfig(duration_s=RECORD_DURATION_S)
    )
    return simulator.run_stations(
        latitude_deg=STATION_LATITUDE[list(indices)],
        longitude_deg=STATION_LONGITUDE[list(indices)],
        station_seed=station_seeds(1234, [STATION_NAMES[i] for i in indices]),
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
        int(RECORD_DURATION_S / HfConfig(duration_s=RECORD_DURATION_S).dt),
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

    The Fortran could not satisfy this, which is the whole reason ``nsite != 1`` was
    refused: its station loop shared one generator, so the second station's waveform
    depended on the first.
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


def test_an_empty_batch_is_an_error(
    slip_model: SlipModel, velocity_model: VelocityModel1D
) -> None:
    """Zero stations cannot yield an array of unknown time length."""
    empty32 = np.array([], np.float32)
    simulator = Simulator(
        slip_model, velocity_model, HfConfig(duration_s=RECORD_DURATION_S)
    )
    with pytest.raises(ValueError, match="nothing to simulate"):
        simulator.run_stations(
            latitude_deg=empty32,
            longitude_deg=empty32,
            station_seed=np.array([], np.uint64),
        )


def test_mismatched_station_arrays_are_rejected(
    slip_model: SlipModel, velocity_model: VelocityModel1D
) -> None:
    """One entry per station, in every array, or an error naming all three lengths."""
    simulator = Simulator(
        slip_model, velocity_model, HfConfig(duration_s=RECORD_DURATION_S)
    )
    with pytest.raises(ValueError, match="one entry per station"):
        simulator.run_stations(
            latitude_deg=STATION_LATITUDE,
            longitude_deg=STATION_LONGITUDE[:2],
            station_seed=station_seeds(1234, STATION_NAMES),
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

    The property ``int32(root) ^ stable_hash(name)`` lacked: a bare XOR is not mixing, so
    incrementing the root flipped one bit of every station seed.
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
