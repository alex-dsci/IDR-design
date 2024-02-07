"""Common items for the test suite."""
import json
import os
from typing import Any, Optional

from pathlib import Path
import pytest

from idrdesigner.core.consts import AMINOACIDS
from idrdesigner.core.exceptions import IDRDesignerException

tests_path = Path(os.path.dirname(os.path.abspath(__file__)))
data_path = tests_path.joinpath("data")
small_config_file_path = data_path.joinpath("small_config_file.json")
plain_text_path = data_path.joinpath("plain_text.txt")

all_feats_cache: dict[str, Optional[float]]
with open(
    data_path.joinpath("default_expected.json"), "rt", encoding="utf-8"
) as file:
    all_feats_cache = json.load(file)
small_feats_cache: dict[str, Optional[float]]
with open(small_config_file_path, "rt", encoding="utf-8") as file:
    loaded: Any = json.load(file)
    small_feats_cache = dict(
        sum(
            (
                [(feat, None) for feat in feat_type]
                for feat_type in loaded.values()
            ),
            start=[],  # type: ignore
        )
        + [(f"_{aa}", 0) for aa in AMINOACIDS]
    )


def assert_fails(f: Any, errors: list[type]):
    """
    A DRY-compliant abstraction for asserting that
    a function raises the provided chain of exception types.

    Does not catch non-IDRDesignerException types.

    Parameters
    ----------
    f : Callable[[], None]
        A callable taking no arguments (most commonly a `functools.partial`).
        Expected to raise `IDRDesignerException` at the top.

    errors : list[type]
        A list of exception types, where the first exception types to be raised
        are the last exception types in the list.

    Raises
    ------
    AssertionErrors

    Note
    ----
    Ignores exception class hierarchy; i.e. the list of errors cannot be
        `[Exception, Exception, Exception, ...]` even though
        each error encountered on the way is a subclass of `Exception`.
    """
    try:
        f()
        assert False, f"Call to {f} did not raise any of the expected errors!"
    except IDRDesignerException as e:
        for n, exception_type in enumerate(errors):
            assert (
                type(e) is exception_type  # noqa: E501, pylint: disable=unidiomatic-typecheck,line-too-long
            ), (
                f"Expected error type {exception_type.__name__} at index {n}, "
                + f"got {type(e).__name__} instead."
            )
            assert e is not None, f"Expected error at index {n}, got None"
            e = e.__cause__  # type: ignore
        assert e is None, (
            "Expected no more errors at end of cause chain, "
            + f"got {type(e).__name__}"
        )


def test_assert_fails():
    """Tests `assert_fails`, above. Run with `> pytest tests/__init__.py`."""

    def f():
        try:
            try:
                raise ArithmeticError()
            except ArithmeticError as e:
                raise FloatingPointError from e
        except ArithmeticError as e:
            raise IDRDesignerException from e

    # Correct usage.
    assert_fails(f, [IDRDesignerException, FloatingPointError, ArithmeticError])

    # Extra error type at the end.
    with pytest.raises(AssertionError):
        assert_fails(
            f,
            [
                IDRDesignerException,
                FloatingPointError,
                ArithmeticError,
                ArithmeticError,
            ],
        )

    # Wrong error type at index 2: ArithmeticError != FloatingPointError.
    with pytest.raises(AssertionError):
        assert_fails(
            f, [IDRDesignerException, FloatingPointError, FloatingPointError]
        )

    # Wrong error type at index 1: FloatingPointError != ArithmeticError.
    # Also shows that subclass hierarchy is ignored.
    assert isinstance(FloatingPointError(), ArithmeticError)
    with pytest.raises(AssertionError):
        assert_fails(
            f, [IDRDesignerException, ArithmeticError, ArithmeticError]
        )

    # Did not fully unwrap error.
    with pytest.raises(AssertionError):
        assert_fails(f, [IDRDesignerException, ArithmeticError])

    def g():
        try:
            try:
                raise IDRDesignerException()
            except IDRDesignerException as e:
                raise FloatingPointError from e
        except FloatingPointError as e:
            raise ArithmeticError from e

    # Native (non IDRDesignerException) errors at the top not handled.
    with pytest.raises(ArithmeticError):
        assert_fails(
            g, [ArithmeticError, FloatingPointError, IDRDesignerException]
        )

    def h():
        pass

    # Fails if no errors raised.
    with pytest.raises(AssertionError):
        assert_fails(h, [IDRDesignerException])
