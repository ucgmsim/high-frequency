"""The stub, the dataclass and the extension must all describe the same call.

``simulate_stations`` passes ``HfConfig`` through as ``**{field: value}``, so the dataclass's
fields *are* the extension's keyword arguments. That coupling is invisible at both ends: add
a field to ``HfConfig`` and the extension raises ``TypeError`` deep inside an unrelated test;
add an argument to the Rust signature and nothing complains at all until a caller wants it.

These tests make the coupling explicit, which is what let ``pyo3-stub-gen`` be declined. A
generated stub would type the surface; this pins it, in three places at once, and needs no
dependency.
"""

import ast
import dataclasses
import inspect
import pathlib

import pytest

from hf_simulation import HfConfig, simulate_stations

# Arguments `_simulate_stations` takes that are NOT HfConfig fields: the two positional
# models and the three per-station arrays, which are data rather than configuration.
NON_CONFIG_ARGUMENTS = {
    "slip_model",
    "velocity_model",
    "latitude_deg",
    "longitude_deg",
    "station_seed",
}


def stub_signature_arguments() -> set[str]:
    """Parameter names of ``_simulate_stations`` as declared in the ``.pyi``.

    Parsed rather than imported, because a stub file is never executed — nothing else
    would ever notice if it drifted.

    Returns
    -------
    set of str
        Every parameter name, positional and keyword-only.
    """
    stub = pathlib.Path(__file__).parent.parent / "hf_simulation" / "_hf_simulation.pyi"
    tree = ast.parse(stub.read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_simulate_stations":
            return {
                argument.arg for argument in (*node.args.args, *node.args.kwonlyargs)
            }
    pytest.fail("_simulate_stations is not declared in _hf_simulation.pyi")


def test_the_stub_matches_the_extension() -> None:
    """Every argument the stub declares is one the extension actually accepts.

    A stub is never executed, so without this it can say anything at all.
    """
    config_fields = {field.name for field in dataclasses.fields(HfConfig)}
    expected = config_fields | NON_CONFIG_ARGUMENTS
    assert stub_signature_arguments() == expected, (
        "the .pyi and HfConfig disagree; symmetric difference "
        f"{stub_signature_arguments() ^ expected}"
    )


def test_every_config_field_reaches_the_extension() -> None:
    """A field added to ``HfConfig`` must be accepted by the Rust signature.

    The splat in ``simulate_stations`` means an unmatched field is a ``TypeError`` from
    pyo3 — correct, but reported as a confusing failure in whichever test ran first. This
    names the problem instead.
    """
    # `simulate_stations` forwards `rayset` explicitly (list, not tuple) and everything
    # else by name, so the wrapper's own signature is not the thing to check -- the
    # dataclass is.
    config_fields = {field.name for field in dataclasses.fields(HfConfig)}
    missing = config_fields - stub_signature_arguments()
    assert not missing, (
        f"HfConfig fields not accepted by _simulate_stations: {sorted(missing)}. "
        "Add them to the #[pyo3(signature = ...)] in src-rust/lib.rs."
    )


def test_the_public_wrapper_keeps_the_arrays_keyword_only() -> None:
    """``latitude_deg``/``longitude_deg``/``station_seed`` must stay keyword-only.

    Three same-length 1-D arrays in a row is exactly the shape where a positional call
    silently transposes two of them. Latitude and longitude are both plausible floats in
    the same range, so swapping them moves a station rather than raising.
    """
    parameters = inspect.signature(simulate_stations).parameters
    for name in ("latitude_deg", "longitude_deg", "station_seed"):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY, (
            f"{name} must be keyword-only: a positional call could transpose it with "
            "its neighbour and merely relocate the station"
        )
