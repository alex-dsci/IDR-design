# pyright: reportPrivateUsage=false
"""Tests for libs.libbasesfc.features"""
from typing import Any, Callable, Generator, Optional
from unittest.mock import MagicMock, patch
from functools import partial
from math import sqrt, log1p, factorial

import pytest

from idrdesigner.core.consts import PKAS_ALL
from idrdesigner.core.exceptions import IDRDesignerException
from idrdesigner.libs.libbasesfc.features import (
    SFCValueException,
    _wrap_cache_logic,
    score_pat,
    count_pat,
    length_pat,
    log_ratio,
    fcr,
    scd,
    _custom_neighbors,
    custom_kappa,
    custom_omega,
    complexity,
    _continuous_charge,
    isoelectric_point,
)
from idrdesigner.libs.libbasesfc.sequences import (
    BaseExtendedSequence,
    BaseSeqRepr,
)

from . import assert_fails


def generate_tests_wrap_cache_logic() -> (
    Generator[
        tuple[
            Optional[list[Any]],
            Optional[dict[str, Any]],
            BaseExtendedSequence,
            float,
            Callable[..., None],
        ],
        None,
        None,
    ]
):
    """
    Generates test cases for `_wrap_cache_logic`.
    """
    # Test when the result is already cached.
    target1 = BaseExtendedSequence("METER", feat_cache={"test_feature": 2024.1})
    yield None, None, target1, 2024.1, MagicMock.assert_not_called

    def curry_called_with(
        other_args: list[Any],
        other_kwargs: dict[str, Any],
        mock: Any,
    ):
        MagicMock.assert_called_once_with(mock, *other_args, **other_kwargs)

    # Test when the result needs to be calculated.
    target2 = BaseExtendedSequence("METER", feat_cache={"test_feature": None})
    yield (
        None,
        None,
        target2,
        2024.2,
        partial(curry_called_with, ["METER"], {}),
    )
    # Test when the function requires additional arguments
    # and keyword arguments.
    target3 = BaseExtendedSequence("METER", feat_cache={"test_feature": None})
    fargs = [1, 2]
    fkwargs = {"keyword": "value"}
    yield (
        fargs,
        fkwargs,
        target3,
        2024.3,
        partial(
            curry_called_with,
            ["METER"] + fargs,
            fkwargs,
        ),
    )


@pytest.mark.parametrize(
    "fargs, fkwargs, target, expected_result, check_calls",
    generate_tests_wrap_cache_logic(),
)
def test_wrap_cache_logic(
    fargs: Optional[list[Any]],
    fkwargs: Optional[dict[str, Any]],
    target: BaseExtendedSequence,
    expected_result: float,
    check_calls: Callable[..., None],
):  # pylint: disable=too-many-arguments
    """Test `_wrap_cache_logic` accesses the cache when expected."""

    mock_f: MagicMock = MagicMock(return_value=expected_result)
    assert (
        _wrap_cache_logic(
            mock_f, target, "test_feature", fargs=fargs, fkwargs=fkwargs
        )
        == target.feat_cache["test_feature"]
        == expected_result
    )
    check_calls(mock_f)


@pytest.mark.parametrize(
    "args, expected_result",
    [
        (
            [
                BaseExtendedSequence(
                    "METER", feat_cache={"test_feature": None}
                ),
                "test_feature",
                {"E.": 1, "ER": 3},
            ],
            5,
        ),
        (
            [
                BaseExtendedSequence(
                    "METER", feat_cache={"test_feature": None}
                ),
                "test_feature",
                {"E.": 1, "ER": 3},
                True,
            ],
            1,
        ),
        (
            [
                BaseExtendedSequence("", feat_cache={"test_feature": None}),
                "test_feature",
                {"E.": 1, "ER": 3},
            ],
            0,
        ),
    ],
)
def test_score_pat(args: list[Any], expected_result: float):
    """Test that `score_pat` output is reasonable."""
    target: BaseExtendedSequence = args[0]
    assert (
        score_pat(*args) == target.feat_cache["test_feature"] == expected_result
    )


def test_score_pat_fails():
    """Test that `score_pat` fails on bad input."""
    target = BaseExtendedSequence("", feat_cache={"test_feature": None})

    assert_fails(
        partial(score_pat, target, "test_feature", {"E.": 1, "ER": 3}, True),
        [SFCValueException],
    )


@pytest.mark.parametrize(
    "args, expected_result",
    [
        (
            [
                BaseExtendedSequence(
                    "METER", feat_cache={"test_feature": None}
                ),
                "test_feature",
                "E",
            ],
            2,
        ),
        (
            [
                BaseExtendedSequence(
                    "METER", feat_cache={"test_feature": None}
                ),
                "test_feature",
                "E",
                True,
            ],
            2 / 5,
        ),
    ],
)
def test_count_pat(args: list[Any], expected_result: float):
    """Test that `count_pat` output is reasonable."""
    target: BaseExtendedSequence = args[0]
    assert (
        count_pat(*args) == target.feat_cache["test_feature"] == expected_result
    )


@pytest.mark.parametrize(
    "args, expected_result",
    [
        (
            # M ET ER
            #   -- --
            [
                BaseExtendedSequence(
                    "METER", feat_cache={"test_feature": None}
                ),
                "test_feature",
                "E.",
            ],
            4,
        ),
        (
            # M ETE R
            #   ---
            # Could've had ET ER both match but
            # the nature of the algorithm is greedy.
            [
                BaseExtendedSequence(
                    "METER", feat_cache={"test_feature": None}
                ),
                "test_feature",
                "E..?",
            ],
            3,
        ),
        (
            # M E T E R
            # -   -   -
            [
                BaseExtendedSequence(
                    "METER", feat_cache={"test_feature": None}
                ),
                "test_feature",
                "[MTR]",
            ],
            3,
        ),
        (
            # ME TE R
            # -- --
            [
                BaseExtendedSequence(
                    "METER", feat_cache={"test_feature": None}
                ),
                "test_feature",
                "[MTR]E",
            ],
            4,
        ),
        (
            # ME TE R
            # -- -- -
            [
                BaseExtendedSequence(
                    "METER", feat_cache={"test_feature": None}
                ),
                "test_feature",
                "[MTR]E?",
            ],
            5,
        ),
    ],
)
def test_length_pat(args: list[Any], expected_result: float):
    """Test that `length_pat` output is reasonable."""
    target: BaseExtendedSequence = args[0]
    assert (
        length_pat(*args)
        == target.feat_cache["test_feature"]
        == expected_result
    )


@pytest.mark.parametrize(
    "args, expected_result",
    [
        (
            [
                BaseExtendedSequence(
                    "METER", feat_cache={"test_feature": None}
                ),
                "_E",
                "This argument should not matter!",
                "_D",
                "This argument should not matter!",
                "test_feature",
            ],
            log1p(2),
        ),
        (
            [
                BaseExtendedSequence(
                    "METER",
                    feat_cache={
                        key: None
                        for key in ["test_feature", "test_num", "test_denom"]
                    },
                ),
                "test_num",
                "[MTR]",
                "test_denom",
                "E",
                "test_feature",
            ],
            log1p(3) - log1p(2),
        ),
    ],
)
def test_log_ratio(args: list[Any], expected_result: float):
    """Test for `log_ratio`."""
    target: BaseExtendedSequence = args[0]
    assert (
        log_ratio(*args) == target.feat_cache["test_feature"] == expected_result
    )


def test_fcr():
    """Test for `fcr`."""
    target: BaseExtendedSequence = BaseExtendedSequence("METER")
    assert target.feat_cache["fcr"] is None
    assert fcr(target) == target.feat_cache["fcr"] == 3 / 5


def test_fcr_fails():
    """Test that `fcr` fails with empty input."""
    assert_fails(partial(fcr, BaseExtendedSequence("")), [SFCValueException])


def generate_tests_scd() -> (
    Generator[
        tuple[BaseExtendedSequence, Optional[BaseSeqRepr], float], None, None
    ]
):
    """Generate tests for the `scd` function."""
    # To calculate SCD,
    # Square root the following numbers, then subtract the bracketed
    # (diff-charge) entries from the unbracketed (same-charge) ones.
    #   M  E- T  E- R+
    # M    0  0  0  0
    # E-      0  2 (3)
    # T          0  0
    # E-           (1)
    # R+
    # In this case, that's:
    scd_meter: float = sqrt(2) - sqrt(3) - sqrt(1)
    # And then divide by the length of the sequence
    scd_meter /= 5
    yield BaseExtendedSequence("METER"), None, scd_meter

    # Testing a mutation which does not affect charge distribution.
    prev_repr1: BaseSeqRepr = BaseSeqRepr("METER", (0, "P"))
    prev_repr1.inner.feat_cache["scd"] = scd_meter
    yield BaseExtendedSequence(prev_repr1), prev_repr1, scd_meter

    # Testing a mutation which adds a charged residue.
    prev_repr2: BaseSeqRepr = BaseSeqRepr("METER", (2, "E"))
    prev_repr2.inner.feat_cache["scd"] = scd_meter
    #   M  E- E- E- R+
    # M    0  0  0  0
    # E-      1  2 (3)
    # E-         1 (2)
    # E-           (1)
    # R+
    scd_meeer: float = sqrt(1) + sqrt(2) - sqrt(3) + sqrt(1) - sqrt(2) - sqrt(1)
    scd_meeer /= 5
    yield BaseExtendedSequence(prev_repr2), prev_repr2, scd_meeer

    # Testing a mutation which takes away a charged residue.
    prev_repr3: BaseSeqRepr = BaseSeqRepr("MEEER", (2, "T"))
    prev_repr3.inner.feat_cache["scd"] = scd_meeer
    yield BaseExtendedSequence(prev_repr3), prev_repr3, scd_meter


@pytest.mark.parametrize(
    "target, prev, expected_result",
    generate_tests_scd(),
)
def test_scd(
    target: BaseExtendedSequence,
    prev: Optional[BaseSeqRepr],
    expected_result: float,
):
    """Test for `scd`."""
    prev_feat: Optional[float] = (
        None if prev is None else prev.inner.feat_cache["scd"]
    )
    assert (
        scd(target, prev)
        == target.feat_cache["scd"]
        == pytest.approx(expected_result)  # type: ignore
    )

    new_prev_feat: Optional[float] = (
        None if prev is None else prev.inner.feat_cache["scd"]
    )
    assert prev_feat == new_prev_feat


@pytest.mark.parametrize(
    "target, prev, errors",
    [
        (
            # Empty sequence yields automatic failure.
            BaseExtendedSequence(""),
            None,
            [SFCValueException],
        ),
        (
            # prev BaseSeqRepr needs mutation information
            BaseExtendedSequence("METER"),
            BaseSeqRepr("METER", None),
            [IDRDesignerException],
        ),
        (
            # prev BaseSeqRepr needs cached "scd"
            BaseExtendedSequence("PETER"),
            BaseSeqRepr("METER", (0, "P")),
            [IDRDesignerException],
        ),
    ],
)
def test_scd_fails(
    target: BaseExtendedSequence,
    prev: Optional[BaseSeqRepr],
    errors: list[type],
):
    """Tests that `scd` fails on bad inputs."""
    assert_fails(partial(scd, target, prev), errors)


def generate_tests_custom_neighbors() -> (
    Generator[
        tuple[str, list[int], Callable[[str, int, int], bool], int], None, None
    ]
):
    """
    Generates tests for `test_custom_neighbors`.
    Used to avoid lambdas but still have local fns
    in the @pytest.mark.parameterize fixture.
    """

    def same_as_neighbor(s: str, i: int, j: int) -> bool:
        return s[i] == s[j]

    yield "PETER", [0, 3, 4], same_as_neighbor, 0
    yield "METER", [1, 3, 4], same_as_neighbor, 1
    yield "METER", [1, 2, 3, 4], same_as_neighbor, 0
    yield "MEEER", [1, 2, 3, 4], same_as_neighbor, 2

    def different_than_neighbor(s: str, i: int, j: int) -> bool:
        return s[i] != s[j]

    yield "PETER", [0, 3, 4], different_than_neighbor, 2
    yield "METER", [1, 3, 4], different_than_neighbor, 1
    yield "METER", [1, 2, 3, 4], different_than_neighbor, 3
    yield "MEEER", [1, 2, 3, 4], different_than_neighbor, 1


@pytest.mark.parametrize(
    "seq, candidates, neighbor_criteria_met, expected_result",
    generate_tests_custom_neighbors(),
)
def test_custom_neighbors(
    seq: str,
    candidates: list[int],
    neighbor_criteria_met: Callable[[str, int, int], bool],
    expected_result: int,
):
    """Test `custom_neighbors` is reasonably well behaved."""
    assert (
        _custom_neighbors(seq, candidates, neighbor_criteria_met)
        == expected_result
    )


@pytest.mark.parametrize(
    "seq, candidates, errors",
    # Zero or one arguments doesn't give enough pairs to iterate over.
    [("METER", [], [SFCValueException]), ("METER", [0], [SFCValueException])],
)
def test_custom_neighbors_fails(
    seq: str,
    candidates: list[int],
    errors: list[type],
):
    """Tests that `_custom_neighbors` fails on bad inputs."""

    # Really just here to avoid using a lambda for my pylint.
    def irrelevant_local_function(_s: str, _i: int, _j: int):
        return True

    assert_fails(
        partial(_custom_neighbors, seq, candidates, irrelevant_local_function),
        errors,
    )


def generate_tests_custom_kappa():
    """
    Generator function for test cases in the `test_custom_kappa` function.
    Used primarily for walking readers through how custom kappa is calculated.
    """
    # Let's use the example PRETTYMYSTERIESANDMAGIC, length 23.
    target = BaseExtendedSequence("PRETTYMYSTERIESANDMAGIC")
    # The charged residues of PRETTYMYSTERI(E)SAN(D)MAGIC are here:
    #                          ^^       ^^  ^     ^
    # There are 2 Rs, and 4 negative ones, for a total of 6 charged residues.

    # The only same charged pair of neighbors within 5 residues of each other
    # is that E and D in brackets.
    # So the count of neighbors is 1.
    counted_neighbors = 1

    # The probability of running into another charged residue in 5 or less
    # is about:
    p_ch = (6 / 23) * sum((1 - 6 / 23) ** i for i in range(5))

    # The probability of a charged residue being neighbors with a differently
    # charged residue is about the same as picking two differently charged
    # residues out of all the charged residues, i.e. 2 * (2 / 6) * (4 / 6)
    p_diff = 2 * (2 / 6) * (4 / 6)

    # This is roughly the probability that a charged residue is a
    # similarly-charged neighbor:
    p = p_ch * (1 - p_diff)

    # The rest of the calculations are pretty standard, modelling the event
    # as a Bernoulli variable. The custom_kappa is the z_score of that variable.
    exp_neighbors = p * 6
    sd_neighbors = sqrt(p * (1 - p) * 6)
    z_score = (counted_neighbors - exp_neighbors) / sd_neighbors

    # Here is its literal value.
    assert z_score == pytest.approx(-1.3166359432452748)  # type: ignore

    # Test that the above discussion is not nonsense.
    yield (target, None, -1.3166359432452748)

    # Test that a mutation which does not affect charge distribution uses
    # the same kappa value.
    prev_repr1 = BaseSeqRepr(target, (0, "M"))

    yield (BaseExtendedSequence(prev_repr1), prev_repr1, -1.3166359432452748)

    # For code coverage,
    # make a mutation that turns a charged into a proline.
    prev_repr2 = BaseSeqRepr("DRETTYMYSTERIESANDMAGIC", (0, "P"))
    yield (BaseExtendedSequence(prev_repr2), prev_repr2, -1.3166359432452748)


@pytest.mark.parametrize(
    "target, prev, expected_result",
    generate_tests_custom_kappa(),
)
def test_custom_kappa(
    target: BaseExtendedSequence,
    prev: Optional[BaseSeqRepr],
    expected_result: float,
):
    """
    Test for the `custom_kappa` function.
    """
    prev_feat: Optional[float] = (
        None if prev is None else prev.inner.feat_cache["custom_kappa"]
    )
    assert (
        custom_kappa(target, prev)
        == target.feat_cache["custom_kappa"]
        == pytest.approx(expected_result)  # type: ignore
    )

    new_prev_feat: Optional[float] = (
        None if prev is None else prev.inner.feat_cache["custom_kappa"]
    )
    assert prev_feat == new_prev_feat


@pytest.mark.parametrize(
    "target, prev, errors",
    [
        (
            # Empty sequence yields automatic failure.
            BaseExtendedSequence(""),
            None,
            [SFCValueException, SFCValueException],
        ),
        (
            # Valid target sequence, provided previous sequence,
            # but no mutation information.
            BaseExtendedSequence("METER"),
            BaseSeqRepr("METER", None),
            [IDRDesignerException],
        ),
        (
            # Valid target sequence, provided previous sequence,
            # but no cached custom_kappa.
            BaseExtendedSequence("METER"),
            BaseSeqRepr("PETER", (0, "M")),
            [IDRDesignerException],
        ),
        (
            # Valid target sequence with only charged residues.
            BaseExtendedSequence("KKKK"),
            None,
            [SFCValueException],
        ),
    ],
)
def test_custom_kappa_fails(
    target: BaseExtendedSequence,
    prev: Optional[BaseSeqRepr],
    errors: list[type],
):
    """Tests that `custom_kappa` fails on bad inputs."""
    assert_fails(partial(custom_kappa, target, prev), errors)


def generate_tests_custom_omega():
    """
    Generator function for test cases in the `test_custom_omega` function.
    Used primarily for walking readers through how custom omega is calculated.
    """
    # Let's use the example PRETTYMYSTERIESANDMAGIC, length 23.
    target = BaseExtendedSequence("PRETTYMYSTERIESANDMAGIC")
    # The charged residues of PRETTYMYSTERIESANDMAGIC are here:
    #                         ^^^       ^^ ^   ^
    # There is 1 proline and 6 charged residues.
    # The PRE at the beginning yields two neighbor pairs,
    # and the (ER)I(E)SAN(D) yields three neighbor pairs.
    # There are 5 neighbors.
    counted_neighbors = 5

    # Doing the same analysis as in `generate_tests_custom_kappa`,
    # we get:
    p = 7 / 23 * sum((1 - 7 / 23) ** i for i in range(5))
    exp_neighbors = p * 7
    sd_neighbors = sqrt(p * (1 - p) * 7)
    z_score = (counted_neighbors - exp_neighbors) / sd_neighbors

    # Here's the value:
    assert z_score == pytest.approx(-0.8797922492709175)  # type: ignore

    # Test that the above discussion is not nonsense:
    yield (target, None, -0.8797922492709175)

    # And test that a non pro/charge distribution affecting mutation
    # yields the same value.
    prev_repr1 = BaseSeqRepr(target, (3, "C"))
    yield (BaseExtendedSequence(prev_repr1), prev_repr1, -0.8797922492709175)

    # For code coverage,
    # make a mutation that turns a non pro/charged into a proline.
    prev_repr2 = BaseSeqRepr("MRETTYMYSTERIESANDMAGIC", (0, "P"))
    yield (BaseExtendedSequence(prev_repr2), prev_repr2, -0.8797922492709175)


@pytest.mark.parametrize(
    "target, prev, expected_result",
    generate_tests_custom_omega(),
)
def test_custom_omega(
    target: BaseExtendedSequence,
    prev: Optional[BaseSeqRepr],
    expected_result: float,
):
    """
    Test for the `custom_omega` function.
    """
    prev_feat: Optional[float] = (
        None if prev is None else prev.inner.feat_cache["custom_omega"]
    )
    assert (
        custom_omega(target, prev)
        == target.feat_cache["custom_omega"]
        == pytest.approx(expected_result)  # type: ignore
    )

    new_prev_feat: Optional[float] = (
        None if prev is None else prev.inner.feat_cache["custom_omega"]
    )
    assert prev_feat == new_prev_feat


@pytest.mark.parametrize(
    "target, prev, errors",
    [
        (
            # Empty sequence yields automatic failure.
            BaseExtendedSequence(""),
            None,
            [SFCValueException, SFCValueException],
        ),
        (
            # Valid target sequence, provided previous sequence,
            # but no mutation information.
            BaseExtendedSequence("METER"),
            BaseSeqRepr("METER", None),
            [IDRDesignerException],
        ),
        (
            # Valid target sequence, provided previous sequence,
            # but no cached custom_omega.
            BaseExtendedSequence("METER"),
            BaseSeqRepr("SETER", (0, "M")),
            [IDRDesignerException],
        ),
        (
            # Valid target sequence with only prolines and charged residues.
            BaseExtendedSequence("PKPK"),
            None,
            [SFCValueException],
        ),
    ],
)
def test_custom_omega_fails(
    target: BaseExtendedSequence,
    prev: Optional[BaseSeqRepr],
    errors: list[type],
):
    """Tests that `custom_omega` fails on bad inputs."""
    assert_fails(partial(custom_omega, target, prev), errors)


def test_complexity():
    """Tests that `complexity` runs as expected."""
    target: BaseExtendedSequence = BaseExtendedSequence("METER")
    assert target.feat_cache["complexity"] is None

    def logfactorial(x: int) -> float:
        return log1p(factorial(x) - 1)

    _complexity: float = logfactorial(5) - 3 * logfactorial(1) - logfactorial(2)
    _complexity /= 5
    assert complexity(target) == target.feat_cache["complexity"] == _complexity


def test_complexity_fails():
    """Tests that `complexity` fails as expected."""
    assert_fails(
        partial(complexity, BaseExtendedSequence("")), [SFCValueException]
    )


@pytest.mark.parametrize(
    "ph, num_basic_res, counts_and_pkas, expected_result",
    [
        (7, 1, [], -0.23989838581069733),
        (7, 3, [(2, 9), (3, 10)], 1.73730263099428),
    ],
)
def test_continuous_charge(
    ph: float,
    num_basic_res: int,
    counts_and_pkas: list[tuple[int, float]],
    expected_result: float,
):
    """
    Tests that `_continuous_charge` runs.
    The validity of the calculation is obviously not tested.
    """
    assert (
        _continuous_charge(ph, num_basic_res, counts_and_pkas)  # type: ignore
        == pytest.approx(expected_result)  # type: ignore
    )


def test_continuous_charge_fails():
    """For code coverage. `_continuous_charge` will fail if provided with
    and invalid number of basic residues."""
    assert_fails(partial(_continuous_charge, 7, 0, []), [IDRDesignerException])


@pytest.mark.parametrize(
    "target,prev,charge_f_args",
    [
        (
            BaseExtendedSequence("METER"),
            None,
            [2, [(2, PKAS_ALL["E"]), (1, PKAS_ALL["R"])]],
        ),
        (
            BaseExtendedSequence("METER"),
            BaseSeqRepr(
                BaseExtendedSequence(
                    "PETER", feat_cache={"isoelectric_point": 7}
                ),
                (0, "M"),
            ),
            [2, [(2, PKAS_ALL["E"]), (1, PKAS_ALL["R"])]],
        ),
    ],
)
def test_isoelectric_point(
    target: BaseExtendedSequence,
    prev: Optional[BaseSeqRepr],
    charge_f_args: list[Any],
):
    """Tests that `isoelectric_point` output is reasonable."""
    assert target.feat_cache["isoelectric_point"] is None

    assert (
        isoelectric_point(target, prev)
        == target.feat_cache["isoelectric_point"]
    )
    pi: Optional[float] = target.feat_cache["isoelectric_point"]
    assert pi is not None
    assert (
        _continuous_charge(pi, *charge_f_args)  # noqa: E501
        == pytest.approx(0)  # type: ignore
    )


@pytest.mark.parametrize(
    "target, prev, errors",
    [
        (
            # Valid target sequence, provided previous sequence,
            # but no cached isoelectric point.
            BaseExtendedSequence("METER"),
            BaseSeqRepr("SETER", (0, "M")),
            [IDRDesignerException],
        ),
        (
            # Valid target sequence, provided previous sequence,
            # but no cached isoelectric point.
            BaseExtendedSequence("METER"),
            BaseSeqRepr("METER", None),
            [IDRDesignerException],
        ),
        (
            # At pH = 14, the residual positive charge should make
            # the protein net positive, yielding a pH curve without solutions
            # below 14.
            BaseExtendedSequence("R" * 200),
            None,
            [SFCValueException],
        ),
    ],
)
def test_isoelectric_point_fails(
    target: BaseExtendedSequence,
    prev: Optional[BaseSeqRepr],
    errors: list[type],
):
    """Tests that `isoelectric_point` fails on bad inputs."""

    assert_fails(partial(isoelectric_point, target, prev), errors)


class _MockRootResults:  # pylint: disable=too-few-public-methods
    """
    Helper mock class mimicking scipy's `RootResults`.
    Meant to access fail case of `isoelectric_point`
    ."""

    converged: bool = False
    flag: str = "Fake error message from fake scipy."


@patch("idrdesigner.libs.libbasesfc.features.root_scalar")
def test_isoelectric_point_with_third_party_fail(mock_root_finder: MagicMock):
    """
    Tests that `isoelectric_point` wraps scipy failures
    in an `IDRDesignerException`.
    """

    mock_root_finder.return_value = _MockRootResults()
    assert_fails(
        partial(isoelectric_point, BaseExtendedSequence("METER")),
        [IDRDesignerException],
    )
