"""
A library of functions to calculate the base bulk molecular features.

Author: Tian Hao Huang (@alex-dsci)
Date: Feb 3rd, 2024
"""

import sys
import re
from typing import Any, Callable, Optional, cast, Iterable, TypeVar

import pprint
from enum import Enum, auto

from functools import partial
from math import sqrt, log1p
from scipy.optimize import root_scalar  # type: ignore
from scipy.special import loggamma  # type: ignore

from idrdesigner.core.exceptions import IDRDesignerException
from idrdesigner.core.consts import (
    BINARY_CHARGE,
    CHARGED_RES,
    AMINOACIDS,
    ACID_BASE_RES,
    PKA_C_TERM,
    PKA_N_TERM,
    BASIC_RES,
    PKAS_ALL,
)
from idrdesigner.libs.libbasesfc.sequences import (
    BaseExtendedSequence,
    BaseSeqRepr,
)

if sys.version_info >= (3, 10):
    from itertools import pairwise
else:
    from itertools import tee

    _T = TypeVar("_T")

    def pairwise(iterable: Iterable[_T]) -> Iterable[tuple[_T, _T]]:
        "s -> (s0,s1), (s1,s2), (s2, s3), ..."
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)


class SFCValueException(IDRDesignerException):
    """
    Sequence Feature Calculator's custom Value Exception.
    Raised when an inputted sequence to a calculator represents an invalid
    calculation.

    Uses in this library include:
      - Charge spacing calculation on something with less than two charged
        residues.
      - Most calculations on an empty input sequence.
    """

    class HandleBy(Enum):
        """
        The internal (to `SFCValueException`)
        `HandleBy` enum gives potential modes of ignoring these
        exceptions:
        1.  `RAISING_IMMEDIATELY` for fatal/unignorable errors, such as empty
            input sequences.
        2.  `DEFAULTING` for returning a default float, such as returning
            average kappa.
        """

        RAISING_IMMEDIATELY = auto()
        DEFAULTING = auto()

    handle_by: HandleBy

    def __init__(self, h: HandleBy, /, *args: object) -> None:
        self.handle_by = h
        super().__init__(*args)


def _wrap_cache_logic(
    f: Callable[..., float],
    target: BaseExtendedSequence,
    feat_name: str,
    fargs: Optional[list[Any]] = None,
    fkwargs: Optional[dict[str, Any]] = None,
):
    """
    A DRY-compliant abstraction of lazily evaluating a function if it is not
    already cached and then caching it there.

    Parameters
    ----------
    f : Callable[[str, ...], float]
        A function which returns the result of the calculation for this feature.

        Must take a string as first argument, and then other arguments can
        be customized based on compatibility with fargs and fkwargs.

        If function does not need sequence (e.g. see `isoelectric_point` in the
        same file), drop the first argument by using a function whose signature
        starts as `f(_: Any, ...)`

    target : BaseExtendedSequence
        Target sequence from which either the cached value will be retrieved or
        a new calculation will be performed.

    feat_name : str
        The name of this feature, also the key where the value will be cached.

    fargs: Optional[list[Any]]
        Optional list of positional arguments the function is going to need.
        For similar use cases as partial.

    fkwargs: Optional[dict[str, Any]]
        Optional dict of keyword arguments the function is going to need.
        For similar use cases as partial.
    """
    cached_result: Optional[float] = target.feat_cache[feat_name]
    if cached_result is not None:
        return cached_result
    if fargs is None:
        fargs = []
    if fkwargs is None:
        fkwargs = {}
    result: float = f(target.seq, *fargs, **fkwargs)
    target.feat_cache[feat_name] = result
    return result


def score_pat(
    target: BaseExtendedSequence,
    feat_name: str,
    scores: dict[str, float],
    average: bool = False,
) -> float:
    """
    Calculates a weighted count or average (weights specified by `score`)
    pattern occurrences in a target sequence.

    Parameters
    ----------
    target : BaseExtendedSequence
        Target sequence from which either the cached value will be retrieved or
        a new calculation will be performed.

    feat_name : str
        The name of this feature, also the key where the value will be cached.

    scores : dict[str, float]
        A dictionary containing the patterns to look for and the weights they
        contribute to the count.

    average : bool
        Whether to divide by sequence length at the end.
        Defaults to `False`.

    Raises
    ------
    SFCValueException
        If `average` is True and the provided sequence is empty.
    """
    if average:

        def f(seq: str) -> float:
            if not seq:
                raise SFCValueException(
                    SFCValueException.HandleBy.RAISING_IMMEDIATELY,
                    "Tried calling avg_pat with empty sequence!\n",
                )
            return sum(
                scores[pat] * len(re.findall(pat, seq)) for pat in scores.keys()
            ) / len(seq)
    else:

        def f(seq: str) -> float:
            return sum(
                scores[pat] * len(re.findall(pat, seq)) for pat in scores.keys()
            )

    return _wrap_cache_logic(f, target, feat_name)


def count_pat(
    target: BaseExtendedSequence,
    feat_name: str,
    pattern: str,
    average: bool = False,
) -> float:
    """
    Calculates a count or average (average used for percent composition) of
    pattern occurrences in a target sequence.

    Parameters
    ----------
    target : BaseExtendedSequence
        Target sequence from which either the cached value will be retrieved or
        a new calculation will be performed.

    feat_name : str
        The name of this feature, also the key where the value will be cached.

    pattern : str
        The pattern to be searched for.

    average : bool
        Whether to divide by sequence length at the end.
        Defaults to `False`

    Raises
    ------
    SFCValueException
        If `average` is True and the provided sequence is empty.
        (from score_pat)
    """
    return score_pat(target, feat_name, {pattern: 1}, average)


def length_pat(
    target: BaseExtendedSequence,
    feat_name: str,
    pattern: str,
) -> float:
    """
    Calculates the total length spanned by patterns in a target sequence.

    Parameters
    ----------
    target : BaseExtendedSequence
        Target sequence from which either the cached value will be retrieved or
        a new calculation will be performed.

    feat_name : str
        The name of this feature, also the key where the value will be cached.

    pattern : str
        The pattern to be searched for.
    """

    def f(seq: str) -> float:
        return sum(
            match.span()[1] - match.span()[0]
            for match in re.finditer(pattern, seq)
        )

    return _wrap_cache_logic(f, target, feat_name)


def log_ratio(  # pylint: disable=too-many-arguments
    target: BaseExtendedSequence,
    num_name: str,
    num_pat: str,
    denom_name: str,
    denom_pat: str,
    feat_name: str,
) -> float:
    """
    Calculate `log(1 + num) - log(1 - denom)` for two patterns in a target
    sequence. Uses natural log.

    Parameters
    ----------
    target : BaseExtendedSequence
        Target sequence from which either the cached value will be retrieved or
        a new calculation will be performed.

    num_name : str
        The name/cache key of numerator pattern.

    num_pat : str
        The pattern to be counted in the numerator.

    denom_name : str
        The name/cache key of denominator pattern.

    denom_pat : str
        The pattern to be counted in the denominator.

    feat_name : str
        This (log ratio) feature's name / key for cache retrieval.
    """

    def f(_: Any) -> float:
        num_count: float = count_pat(target, num_name, num_pat)
        denom_count: float = count_pat(target, denom_name, denom_pat)
        return log1p(num_count) - log1p(denom_count)

    return _wrap_cache_logic(f, target, feat_name)


def fcr(target: BaseExtendedSequence) -> float:
    """
    Calculate the fraction of charged residues in a target sequence. Defined
    as a separate function because charged residues are cached by default,
    likely over-optimizing.

    Parameters
    ----------
    target : BaseExtendedSequence
        Target sequence from which either the cached value will be retrieved or
        a new calculation will be performed.

    Raises
    ------
    SFCValueException
        If the provided sequence is empty.
    """

    def f(seq: str) -> float:
        if not seq:
            raise SFCValueException(
                SFCValueException.HandleBy.RAISING_IMMEDIATELY,
                "Tried calling fcr with empty sequence!\n",
            )
        return len(target.charged_res) / len(seq)

    return _wrap_cache_logic(f, target, "fcr")


def scd(
    target: BaseExtendedSequence,
    prev: Optional[BaseSeqRepr] = None,
) -> float:
    """
    Calculate the SCD (Sequence Charge Decoration) on a target sequence.

    Also able to speed up calculations using the previous SCD value
    of a sequence which is one point mutation away, using the `prev` parameter.


    Parameters
    ----------
    target : BaseExtendedSequence
        Target sequence from which either the cached value will be retrieved or
        a new calculation will be performed.

    prev : Optional[BaseSeqRepr]
        Optional previous sequence for faster SCD calculation.
        By default, not provided.

        The previous sequence should be one point mutation away from the target,
        and must have cached SCD information already calculated.

        By default, the SCD algorithm takes O(N^2) time. The `prev` parameter
        provides the previous SCD value and the point mutation to
        calculate SCD in O(N) time.

    Raises
    ------
    SFCValueException
        If the target sequence is empty.

    IDRDesignerException (not derived subclass)
        If previous sequence information is passed in but does not contain
        either of:
        * A point mutation that converts the previous sequence to the target
        * A cached SCD value
    """

    def f(seq: str) -> float:
        if not seq:
            raise SFCValueException(
                SFCValueException.HandleBy.RAISING_IMMEDIATELY,
                "Tried calling scd with empty sequence!\n",
            )
        if prev is not None:
            mut = prev.mutation
            if mut is None:
                raise IDRDesignerException(
                    f"Passed a {type(prev).__name__} with no mutation to scd:\n"
                    + f"{pprint.pformat(prev)}\n",
                )
            cached_result: Optional[float] = prev.inner.feat_cache["scd"]
            if cached_result is None:
                raise IDRDesignerException(
                    f"Passed a prev {type(prev).__name__} with uncached scd:\n"
                    + f"{pprint.pformat(prev)}\n",
                )
            result: float = cached_result
            sum_delta: float = 0
            if BINARY_CHARGE.get(mut.start_aa, 0) == BINARY_CHARGE.get(
                mut.end_aa, 0
            ):
                return result
            # O(N) calls to sqrt
            for i in prev.inner.charged_res:
                delta: float = (
                    BINARY_CHARGE.get(seq[i], 0)
                    * (
                        BINARY_CHARGE.get(mut.end_aa, 0)
                        - BINARY_CHARGE.get(mut.start_aa, 0)
                    )
                    * sqrt(abs(mut.loc - i))
                )
                sum_delta += delta
            result += sum_delta / len(seq)
            return result
        result: float = 0
        # O(N^2) calls to sqrt
        for i, loc_i in enumerate(target.charged_res):
            for loc_j in target.charged_res[:i]:
                result += (
                    BINARY_CHARGE[seq[loc_i]]
                    * BINARY_CHARGE[seq[loc_j]]
                    * sqrt(loc_i - loc_j)
                )
        return result / len(seq)

    return _wrap_cache_logic(f, target, "scd")


def _custom_neighbors(
    seq: str,
    candidates: list[int],
    neighbor_criteria_met: Callable[[str, int, int], bool],
    handle_empty: SFCValueException.HandleBy = SFCValueException.HandleBy.RAISING_IMMEDIATELY,  # noqa: E501, pylint: disable=line-too-long
) -> int:
    """
    A DRY-compliant abstraction for counting the number of neighbouring
    residue pairs satisfying some criteria.

    Parameters
    ----------
    seq : str
        The sequence to calculate over.

    candidates : list[int]
        List of indices representing residues considered in the calculation.

    neighbor_criteria_met : Callable[[AminoAcidSequence, int, int], bool]
        Function for determining whether a neighbor contributes to the count or
        not.

    handle_empty : SFCValueException.HandleBy
        Corrective behaviour to take if the sequence or candidate list is empty.
        Default is to raise immediately to the user.

    Raises
    ------
    SFCValueException
        If the sequence is empty or does not contain enough candidates.
    """
    candidate_pairs = list(pairwise(candidates))
    if not candidate_pairs:
        raise SFCValueException(
            handle_empty,
            "_custom_neighbors failed because seq does not contain enough"
            + f"candidates!\nCandidates:\n{pprint.pformat(candidates)}\n"
            + f"Sequence: {pprint.pformat(seq)}\n",
        )
    return sum(
        1 for i, j in candidate_pairs if neighbor_criteria_met(seq, i, j)
    )


def custom_kappa(
    target: BaseExtendedSequence,
    prev: Optional[BaseSeqRepr] = None,
    blob: int = 5,
    handle_extreme: SFCValueException.HandleBy = SFCValueException.HandleBy.RAISING_IMMEDIATELY,  # noqa: E501, pylint: disable=line-too-long
) -> float:
    """
    Calculate a parameter similar to kappa, relating to charge blockiness.
    The value of this feature goes up the more like-charged residues are
    clustered together.

    If given the context of a previously cached kappa and a mutation that
    doesn't affect charge patterning, can skip this calculation altogether.
    May be over-optimizing.

    Parameters
    ----------
    target : BaseExtendedSequence
        Target sequence from which either the cached value will be retrieved or
        a new calculation will be performed.

    prev : Optional[BaseSeqRepr]
        Optional previous sequence for faster custom_kappa score calculation.
        By default, not provided.

        The previous sequence should be one point mutation away from the target,
        and must have cached custom_kappa already calculated.

    blob : int
        Size of the charge pattern blob for considering neighboring charges.
        Defaults to `5`.

    handle_extreme : SFCValueException.HandleBy
        Corrective behaviour to take if no/only charged residues are present.
        Default is to raise immediately to the user.

    Raises
    ------
    SFCValueException
        If the sequence is empty or does not contain enough candidates
        (from _custom_neighbors); OR

        If the sequence contains only charged residues.

    IDRDesignerException (not derived subclass)
        If previous sequence information is passed in but does not contain
        either of:
        * A point mutation that converts the previous sequence to the target
        * A cached custom_kappa value
    """

    def f(seq: str) -> float:
        if prev is not None:
            mut = prev.mutation
            if mut is None:
                raise IDRDesignerException(
                    f"Passed a {type(prev).__name__} "
                    + "with no mutation into custom_kappa:\n"
                    + f"{pprint.pformat(prev)}\n",
                )
            if BINARY_CHARGE.get(mut.start_aa, 0) == BINARY_CHARGE.get(
                mut.end_aa, 0
            ):
                if prev.inner.feat_cache["custom_kappa"] is None:
                    raise IDRDesignerException(
                        f"Passed a prev {type(prev).__name__} "
                        + "with uncached kappa:\n"
                        + f"{pprint.pformat(prev)}\n",
                    )
                return prev.inner.feat_cache["custom_kappa"]
        candidates: list[int] = target.charged_res

        def kappa_neighbors_criteria(seq: str, i: int, j: int) -> bool:
            return (
                abs(i - j) <= blob
                and BINARY_CHARGE[seq[i]] == BINARY_CHARGE[seq[j]]
            )

        try:
            count_neighbors: int = _custom_neighbors(
                seq, candidates, kappa_neighbors_criteria, handle_extreme
            )
        except SFCValueException as e:
            raise SFCValueException(
                handle_extreme,
                "A kappa calculation cannot be done on a sequence "
                + "containing no charged residues!\n"
                + f"seq:\n{pprint.pformat(seq)}\n",
            ) from e

        def prob_neighbor_given_candidate() -> float:
            proportion_charged: float = len(candidates) / len(seq)
            # It shouldn't be zero either, but that should be caught and raise
            # an error in _custom_neighbors
            if proportion_charged == 1:
                raise SFCValueException(
                    handle_extreme,
                    "A kappa calculation cannot be done on a sequence "
                    + "containing only charged residues!\n"
                    + f"seq:\n{pprint.pformat(seq)}\n",
                )
            count_pos: int = sum(
                1 for i in candidates if BINARY_CHARGE[seq[i]] == 1
            )
            count_neg: int = len(candidates) - count_pos
            prob_next_charge_in_blob: float = proportion_charged * sum(
                (1 - proportion_charged) ** i for i in range(blob)
            )
            prob_charges_are_diff: float = (
                2 * count_neg * count_pos / (len(candidates) ** 2)
            )
            return prob_next_charge_in_blob * (1 - prob_charges_are_diff)

        p: float = prob_neighbor_given_candidate()
        mean_neighbors: float = p * len(candidates)
        sd_neighbors: float = sqrt(p * (1 - p) * len(candidates))
        return (count_neighbors - mean_neighbors) / sd_neighbors

    return _wrap_cache_logic(f, target, "custom_kappa")


def custom_omega(
    target: BaseExtendedSequence,
    prev: Optional[BaseSeqRepr] = None,
    blob: int = 5,
    handle_extreme: SFCValueException.HandleBy = SFCValueException.HandleBy.RAISING_IMMEDIATELY,  # noqa: E501, pylint: disable=line-too-long
) -> float:
    """
    Calculate a parameter similar to omega, relating to proline and charge
    separation.
    The value of this feature goes up the more charged residues and prolines are
    clustered together.

    If given the context of a previously cached omega and a mutation that
    doesn't affect proline/charge patterning, can skip this calculation
    altogether. May be over-optimizing.

    Parameters
    ----------
    target : BaseExtendedSequence
        Target sequence from which either the cached value will be retrieved or
        a new calculation will be performed.

    prev : Optional[BaseSeqRepr]
        Optional previous sequence for faster custom_omega score calculation.
        By default, not provided.

        The previous sequence should be one point mutation away from the target,
        and must have cached custom_omega already calculated.

    blob : int
        Size of the charge pattern blob for considering neighboring pro/charge
        residues.
        Defaults to `5`.

    handle_extreme : SFCValueException.HandleBy
        Corrective behaviour to take if no/only pro/charge residues are present.
        Default is to raise immediately to the user.

    Raises
    ------
    SFCValueException
        If the sequence is empty or does not contain enough candidates
        (from _custom_neighbors); OR

        If the sequence contains only charged residues.

    IDRDesignerException (not derived subclass)
        If previous sequence information is passed in but does not contain
        either of:
        * A point mutation that converts the previous sequence to the target
        * A cached custom_kappa value
    """

    def f(seq: str) -> float:
        if prev is not None:
            mut = prev.mutation
            if mut is None:
                raise IDRDesignerException(
                    f"Passed a {type(prev).__name__} "
                    + "with no mutation into custom_omega:\n"
                    + f"{pprint.pformat(prev)}\n",
                )
            if (mut.start_aa in CHARGED_RES + ["P"]) == (
                mut.end_aa in CHARGED_RES + ["P"]
            ):
                if prev.inner.feat_cache["custom_omega"] is None:
                    raise IDRDesignerException(
                        f"Passed a prev {type(prev).__name__} "
                        + "with uncached kappa:\n"
                        + f"{pprint.pformat(prev)}\n",
                    )
                return prev.inner.feat_cache["custom_omega"]
        candidates: list[int] = target.procharged_res

        def omega_neighbors_criteria(_: Any, i: int, j: int) -> bool:
            return abs(i - j) <= blob

        try:
            count_neighbors: int = _custom_neighbors(
                seq, candidates, omega_neighbors_criteria, handle_extreme
            )
        except SFCValueException as e:
            raise SFCValueException(
                handle_extreme,
                "An omega calculation cannot be done on a sequence "
                + "containing no prolines or charged residues!\n"
                + f"seq:\n{pprint.pformat(seq)}\n",
            ) from e

        def prob_neighbor_given_candidate() -> float:
            proportion_procharge: float = len(candidates) / len(seq)
            # It shouldn't be zero either, but that should be caught and raise
            # an error in _custom_neighbors
            if proportion_procharge == 1:
                raise SFCValueException(
                    handle_extreme,
                    "An omega calculation cannot be done on a sequence "
                    + "containing only prolines and charged residues!\n"
                    + f"seq:\n{pprint.pformat(seq)}\n",
                )
            return proportion_procharge * sum(
                (1 - proportion_procharge) ** i for i in range(blob)
            )

        p: float = prob_neighbor_given_candidate()
        mean_neighbors: float = p * len(candidates)
        sd_neighbors: float = sqrt(p * (1 - p) * len(candidates))
        return (count_neighbors - mean_neighbors) / sd_neighbors

    return _wrap_cache_logic(f, target, "custom_omega")


def complexity(target: BaseExtendedSequence) -> float:
    """
    Calculate the complexity (entropy-like feature) of a target sequence.
    Uses natural log.

    Parameters
    ----------
    target : BaseExtendedSequence
        Target sequence from which either the cached value will be retrieved or
        a new calculation will be performed.

    Raises
    ------
    SFCValueException
        If the sequence is empty.
    """

    def f(seq: str) -> float:
        if not seq:
            raise SFCValueException(
                SFCValueException.HandleBy.RAISING_IMMEDIATELY,
                "Tried calling complexity with empty sequence!\n",
            )
        log_gamma_sum: float = 0
        # Use the convention of _X as the feat_name of counting amino acid X.
        counts: list[float] = [
            count_pat(target, f"_{aa}", aa) for aa in AMINOACIDS
        ]
        for count in counts:
            log_gamma_sum += loggamma(1 + count)
        return (loggamma(1 + len(seq)) - log_gamma_sum) / len(seq)

    return _wrap_cache_logic(f, target, "complexity")


def _continuous_charge(
    ph: float, num_basic_res: int, counts_and_pkas: Iterable[tuple[int, float]]
) -> float:
    """
    Calculates a very accurate net charge based on pKa formulas.

    The logic behind this function is that one counts the basic sites
    (positively charged in their protonated state) as the default charge.

    As you increase the pH, a proportion of those sites become deprotonated,
    decreasing the charge. The proportion of deprotonated sites of species
    with a fixed pka is:
            1 / (1 + 10 ** (pka - ph))

    So you subtract the expected number of free protons (corresponding to
    deprotonated sites) from the number of basic sites.

    Parameters
    ----------
    ph : float
        The pH to be calculated at.

    num_basic_res: int
        The number of sites which are positively charged when protonated,
        including the basic N terminus site. i.e.
        0 is not a valid value because there is always a basic N terminus site.

    counts_and_pkas : list[tuple[int, float]]
        The number of sites of a specific pKa, for each pKa.

    Raises
    ------
    IDRDesignerException
        If an invalid (non-positive) number of residues is passed in.
    """
    if num_basic_res < 1:
        raise IDRDesignerException(
            " _continuous_charge was called with an invalid parameter:\n"
            + f"num_basic_res={num_basic_res}\n"
        )
    free_protons: float = 0
    for count, pka in counts_and_pkas:
        proportion_protonated = 1 / (1 + 10 ** (ph - pka))
        free_protons += count * (1 - proportion_protonated)
    proportion_protonated = 1 / (1 + (10 ** (ph - PKA_N_TERM)))
    free_protons += 1 - proportion_protonated
    proportion_protonated = 1 / (1 + (10 ** (ph - PKA_C_TERM)))
    free_protons += 1 - proportion_protonated
    return num_basic_res - free_protons


def isoelectric_point(
    target: BaseExtendedSequence,
    prev: Optional[BaseSeqRepr] = None,
    handle_bad_curve: SFCValueException.HandleBy = SFCValueException.HandleBy.RAISING_IMMEDIATELY,  # noqa: E501, pylint: disable=line-too-long
) -> float:
    """
    Calculate the isoelectric point of a target sequence, searching in the
    interval 0 to 14. Uses previously cached isoelectric point,
    if provided, as a guess to start the root finding.

    Parameters
    ----------
    target : BaseExtendedSequence
        Target sequence from which either the cached value will be retrieved or
        a new calculation will be performed.

    prev : Optional[BaseSeqRepr], default=None
        Optional previous sequence for faster isoelectric point calculation.
        By default, not provided.

        The previous sequence should be one point mutation away from the target,
        and must have cached isoelectric point.

    Raises
    ------
    SFCValueException
        If the charge curve is negative at the acidic end or positive at the
        basic end, it is guaranteed not to have a root on the interval.

    IDRDesignerException
        If previous sequence information is passed in but does not contain
        a cached isoelectric point;

        OR if the scipy library fails to converge on a root after it is called.
    """

    def f(_: Any) -> float:
        # Use the convention of _X as the feat_name of counting amino acid X.
        counts: list[float] = [
            count_pat(
                target,
                f"_{aa}",
                aa,
            )
            for aa in ACID_BASE_RES
        ]
        counts_and_pkas: list[tuple[int, float]] = []
        num_basic_res: int = 1
        for aa, count in zip(ACID_BASE_RES, counts):
            count = cast(int, count)
            counts_and_pkas.append((count, PKAS_ALL[aa]))
            if aa in BASIC_RES:
                num_basic_res += count

        continuous_charge: Callable[[float], float] = partial(
            _continuous_charge,
            num_basic_res=num_basic_res,
            counts_and_pkas=counts_and_pkas,
        )
        if (continuous_charge(0) <= 0) or (continuous_charge(14) >= 0):
            raise SFCValueException(
                handle_bad_curve,
                "Charge curve is guaranteed not to have a root on the pH range "
                + "0-14\ncounts_and_pKAs:\n"
                + f"{pprint.pformat(counts_and_pkas)}\n",
            )

        guess: Optional[float] = None
        if prev is not None:
            if prev.inner.feat_cache["isoelectric_point"] is None:
                raise IDRDesignerException(
                    f"Passed a prev {type(prev).__name__} "
                    + "with uncached isoelectric_point:\n"
                    + f"{pprint.pformat(prev)}\n"
                )
            guess = prev.inner.feat_cache["isoelectric_point"]
        scipy_result = root_scalar(  # type: ignore
            continuous_charge, method="brenth", bracket=(0, 14), x0=guess
        )
        if not scipy_result.converged:  # type: ignore
            flag: str = scipy_result.flag  # type: ignore
            raise IDRDesignerException(
                "Isoelectric point calculation unexpectedly failed to "
                + f"converge, raising the following flag:\n{flag}\n"
                + "counts_and_pKAs:\n"
                + f"{pprint.pformat(counts_and_pkas)}\n"
                + f"Initial guess = {guess}\n"
            )
        return scipy_result.root  # type: ignore

    return _wrap_cache_logic(f, target, "isoelectric_point")
