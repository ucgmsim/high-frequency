"""The stub, the dataclass and the extension must all describe the same call.

``HfConfig`` is a Python dataclass holding the defaults; ``HfConfig._to_rust`` spreads its
fields across the four Rust parameter classes. That coupling is invisible at both ends: add
a field to the dataclass and nothing consumes it, or rename a Rust argument and nothing
complains until a caller wants it.

These tests make the coupling explicit, which is what let ``pyo3-stub-gen`` be declined. A
generated stub would type the surface; this pins it, in three places at once, and needs no
dependency.
"""

import ast
import dataclasses
import inspect
import pathlib

import pytest

from hf_simulation import HfConfig
from hf_simulation import _hf_simulation as extension

STUB = pathlib.Path(__file__).parent.parent / "hf_simulation" / "_hf_simulation.pyi"

# The Rust classes `HfConfig._to_rust` builds, and the placeholder each argument gets when
# this test constructs one. Every value is a float because every argument of these four is.
PARAMETER_CLASSES = (
    "SourceParameters",
    "PathParameters",
    "SiteParameters",
    "RecordParameters",
)


def stub_init_arguments(class_name: str) -> set[str]:
    """Keyword-argument names of a class's ``__init__`` as declared in the ``.pyi``.

    Parsed rather than imported, because a stub file is never executed — nothing else
    would ever notice if it drifted.

    Parameters
    ----------
    class_name : str
        The class to look up.

    Returns
    -------
    set of str
        Every parameter name except ``self``.
    """
    tree = ast.parse(STUB.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    named = (*item.args.args, *item.args.kwonlyargs)
                    return {
                        argument.arg for argument in named if argument.arg != "self"
                    }
    pytest.fail(f"{class_name}.__init__ is not declared in {STUB.name}")


@pytest.mark.parametrize("class_name", PARAMETER_CLASSES)
def test_the_stub_matches_the_extension(class_name: str) -> None:
    """Every argument the stub declares is one the extension actually accepts.

    A stub is never executed, so without this it can say anything at all. pyo3 raises
    ``TypeError`` for an unknown *or* a missing keyword argument, so constructing with
    exactly the stub's argument set checks both directions at once.
    """
    arguments = stub_init_arguments(class_name)
    cls = getattr(extension, class_name)
    # `rayset` is the one non-float; everything else these four take is a number, and the
    # values do not matter because nothing is simulated here.
    values = {name: [1] if name == "rayset" else 1.0 for name in arguments}
    if "path_duration_model" in values:
        values["path_duration_model"] = 0
    cls(**values)


def test_every_config_field_reaches_rust() -> None:
    """No dataclass field is silently dropped on the way across the boundary.

    ``_to_rust`` names each field explicitly, so a field added to ``HfConfig`` and forgotten
    there would default away silently — the Rust side has no defaults to fall back on, so
    the value would simply be whatever the forgotten call passed instead.
    """
    source = inspect.getsource(HfConfig._to_rust)
    missing = [
        field.name
        for field in dataclasses.fields(HfConfig)
        if f"self.{field.name}" not in source
    ]
    assert not missing, f"HfConfig fields never read by _to_rust: {missing}"


def test_the_config_builds_its_rust_counterpart() -> None:
    """The default configuration crosses the boundary without a TypeError.

    This is the test that fails first if a Rust argument is renamed: `_to_rust` passes
    keywords, and pyo3 rejects an unknown one.
    """
    assert HfConfig(duration_s=10.0)._to_rust() is not None


def test_simulator_run_stations_is_keyword_only() -> None:
    """The three per-station arrays stay keyword-only.

    They are three same-typed arrays in a row, so positional order would be silent to swap.
    """
    arguments = stub_init_arguments("Simulator")
    assert arguments == {"config", "slip_model", "velocity_model"}

    tree = ast.parse(STUB.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Simulator":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "run_stations":
                    assert not [a.arg for a in item.args.args if a.arg != "self"], (
                        "run_stations must take no positional arguments"
                    )
                    assert {a.arg for a in item.args.kwonlyargs} == {
                        "latitude_deg",
                        "longitude_deg",
                        "station_seed",
                    }
                    return
    pytest.fail("Simulator.run_stations is not declared in the stub")
