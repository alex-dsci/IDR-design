"""Tests for `libs.libbasesfc.calculator`."""

from typing import Any, Optional
from functools import partial
from pathlib import Path
from copy import copy
from math import sqrt, log1p
import json
from numpy import float64
from numpy.typing import NDArray
import pytest

from idrdesigner.core.exceptions import IDRDesignerException
from idrdesigner.libs.libbasesfc.sequences import (
    BaseSeqRepr,
    BaseExtendedSequence,
)
from idrdesigner.libs.libbasesfc.calculator import BaseSeqFeatCalc
from . import small_config_file_path, data_path, plain_text_path, assert_fails


@pytest.mark.parametrize(
    "args,kwargs,num_feats",
    [
        ([], {}, 94),
        ([small_config_file_path], {}, 7),
        ([], {"feat_config_path": small_config_file_path}, 7),
        ([data_path.joinpath("empty.json")], {}, 0),
        ([data_path.joinpath("empty_non_modular.json")], {}, 0),
    ],
)
def test_BFC_init(
    args: list[Any],
    kwargs: dict[str, Any],
    num_feats: int,
):  # pylint: disable=invalid-name
    """Test that `BaseSeqFeatCalc.__init__` runs
    and produces a reasonable output."""
    assert len(BaseSeqFeatCalc(*args, **kwargs).features) == num_feats


@pytest.mark.parametrize(
    "bad_json,errors",
    [
        (
            data_path.joinpath("bad_config_count_bad_key.json"),
            [IDRDesignerException, IDRDesignerException],
        ),
        (
            data_path.joinpath("bad_config_count_too_many.json"),
            [IDRDesignerException, IDRDesignerException],
        ),
        (
            data_path.joinpath("bad_config_length_missing_key.json"),
            [IDRDesignerException, IDRDesignerException],
        ),
        (
            data_path.joinpath("bad_config_length_bad_key.json"),
            [IDRDesignerException, IDRDesignerException],
        ),
        (
            data_path.joinpath("bad_config_log_ratio_bad_key.json"),
            [IDRDesignerException, IDRDesignerException],
        ),
        (
            data_path.joinpath("bad_config_log_ratio_missing_key.json"),
            [IDRDesignerException, IDRDesignerException],
        ),
        (
            data_path.joinpath("bad_config_non_modular.json"),
            [IDRDesignerException, IDRDesignerException],
        ),
        (
            Path("non_existent_file.json"),
            [IDRDesignerException, FileNotFoundError],
        ),
        (plain_text_path, [IDRDesignerException, json.JSONDecodeError]),
    ],
)
def test_BFC_init_fails(
    bad_json: Path,
    errors: list[type],
):  # pylint: disable=invalid-name
    """Test that `BaseSeqFeatCalc.__init__` fails when
    config JSON is incorrectly specified."""
    assert_fails(partial(BaseSeqFeatCalc, bad_json), errors)


def test_BFC_call_with_no_prev():  # pylint: disable=invalid-name
    """Test that `BaseSeqFeatCalc.__call__` behaves as expected using
    a BaseSeqRepr with no mutation, meaning features should be cached there."""

    target: BaseSeqRepr = BaseSeqRepr(
        BaseExtendedSequence("METER", feat_config_path=small_config_file_path)
    )
    empty_feat_cache: dict[str, Optional[float]] = copy(target.inner.feat_cache)
    expected_feats: dict[str, Optional[float]] = copy(empty_feat_cache)

    expected_feats.update(
        {
            "scd": (sqrt(2) - sqrt(3) - sqrt(1)) / 5,
            "fcr": 3 / 5,
            "cold_regex_pattern": 0,
            "A_minus_G": 0,
            "percent_P_or_T": 1 / 5,
            "cold_regex_length": 0,
            "ED_ratio": log1p(2),
        }
    )

    calculator: BaseSeqFeatCalc = BaseSeqFeatCalc(small_config_file_path)
    target_feats: NDArray[float64] = calculator(target)
    with pytest.raises(AttributeError):
        _: BaseExtendedSequence = calculator.next_seq_cache
    assert (
        target.inner.feat_cache  # noqa: E501
        == pytest.approx(expected_feats)  # type: ignore
    )

    assert all(
        target.inner.feat_cache[feat_name] == val
        for val, feat_name in zip(target_feats, calculator.features)
    )


def test_BFC_call_with_prev():  # pylint: disable=invalid-name
    """Test that `BaseSeqFeatCalc.__call__` behaves as expected using
    a BaseSeqRepr with mutation, meaning features should be cached in the
    `next_seq_cache`."""
    prev_inner: BaseExtendedSequence = BaseExtendedSequence(
        "PETER",
        feat_config_path=small_config_file_path,
    )
    expected_feats: dict[str, Optional[float]] = copy(prev_inner.feat_cache)

    expected_feats.update(
        {
            "scd": (sqrt(2) - sqrt(3) - sqrt(1)) / 5,
            "fcr": 3 / 5,
            "cold_regex_pattern": 0,
            "A_minus_G": 0,
            "percent_P_or_T": 1 / 5,
            "cold_regex_length": 0,
            "ED_ratio": log1p(2),
            "_M": 1,
            "_P": 0,
        }
    )
    prev_inner.feat_cache.update(
        {
            "scd": (sqrt(2) - sqrt(3) - sqrt(1)) / 5,
            "fcr": 3 / 5,
            "cold_regex_pattern": 0,
            "A_minus_G": 0,
            "percent_P_or_T": 2 / 5,
            "cold_regex_length": 0,
            "ED_ratio": log1p(2),
        }
    )
    prev: BaseSeqRepr = BaseSeqRepr(prev_inner, (0, "M"))
    calculator: BaseSeqFeatCalc = BaseSeqFeatCalc(small_config_file_path)
    target_feats = calculator(prev)
    assert (
        calculator.next_seq_cache.feat_cache  # noqa: E501
        == pytest.approx(expected_feats)  # type: ignore
    )
    assert all(
        calculator.next_seq_cache.feat_cache[feat_name] == val
        for val, feat_name in zip(target_feats, calculator.features)
    )


def test_BFC_call_w_subset():  # pylint: disable=invalid-name
    """Test that `BaseSeqFeatCalc.__call__` works with a subset of features."""
    target: BaseSeqRepr = BaseSeqRepr(
        BaseExtendedSequence("METER", feat_config_path=small_config_file_path)
    )
    empty_feat_cache: dict[str, Optional[float]] = copy(target.inner.feat_cache)
    expected_feats: dict[str, Optional[float]] = copy(empty_feat_cache)

    expected_feats.update(
        {
            "scd": (sqrt(2) - sqrt(3) - sqrt(1)) / 5,
            "fcr": 3 / 5,
            "percent_P_or_T": 1 / 5,
            "ED_ratio": log1p(2),
        }
    )

    calculator: BaseSeqFeatCalc = BaseSeqFeatCalc(small_config_file_path)
    target_feats: NDArray[float64] = calculator(
        target, ["scd", "fcr", "percent_P_or_T", "ED_ratio"]
    )
    with pytest.raises(AttributeError):
        _: BaseExtendedSequence = calculator.next_seq_cache
    assert (
        target.inner.feat_cache  # noqa: E501
        == pytest.approx(expected_feats)  # type: ignore
    )

    assert all(
        target.inner.feat_cache[feat_name] == val
        for val, feat_name in zip(
            target_feats, ["scd", "fcr", "percent_P_or_T", "ED_ratio"]
        )
    )


def test_BFC_call_fails_w_bad_subset():  # pylint: disable=invalid-name
    """Test that `BaseSeqFeatCalc.__call__` fails with a subset of features
    that contains a key which is not supported."""
    target: BaseSeqRepr = BaseSeqRepr(
        BaseExtendedSequence("METER", feat_config_path=small_config_file_path)
    )

    calculator: BaseSeqFeatCalc = BaseSeqFeatCalc(small_config_file_path)
    assert_fails(
        partial(
            calculator,
            target,
            ["scd", "fcr", "percent_P_or_T", "not_a_feature"],
        ),
        [IDRDesignerException],
    )
