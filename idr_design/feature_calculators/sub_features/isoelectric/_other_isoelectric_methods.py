from idr_design.timeout_decorator import timeout
from typing import Callable as func, Iterator
from math import log
from scipy.optimize import root_scalar
from idr_design.feature_calculators.sub_features.isoelectric.main import ACID_BASE_RES, BASIC_RES, PKAS_ALL, PKA_N_TERM, PKA_C_TERM, _charge_at_pH
import os

path_to_this_file = os.path.dirname(os.path.realpath(__file__))
TURNING_PTS: list[float] = [4.25, 6, 9.5, 12]
FLAT_PTS: list[float] = list(map(lambda x: (x[0] + x[1]) / 2, zip(TURNING_PTS[:-1],TURNING_PTS[1:])))
PKAS_VALUES = PKAS_ALL.values()

def _quick_guess(charge: func[[float], float]) -> float:
    guess: float = TURNING_PTS[0]
    if (charge(guess) > 0):
        for i in range(len(FLAT_PTS)):
            guess = FLAT_PTS[i]
            if (charge(guess) < 0):
                return TURNING_PTS[i]
        else:
            return TURNING_PTS[-1]
    return guess


@timeout(3, "Halley's algorithm timed out!")
def _halley_pI(_counts: dict[str, int], converge_thres: float) -> float:

    # Setup calculation
    counts: list[int] = list(map(lambda res: _counts[res], ACID_BASE_RES)) 
    N_basic: int = sum([_counts[res] for res in BASIC_RES])
    charge: float
    halley_delta: float
    guess = _quick_guess(lambda pH: _charge_at_pH(pH, N_basic, zip(counts, PKAS_VALUES)))

    with open(f"{path_to_this_file}/pI_output.txt", "w") as output_file:
        # Halley's method
        charge, halley_delta = _charge_and_halleys_delta(guess, N_basic, zip(counts, PKAS_VALUES))
        while abs(charge) > converge_thres:
            print(f"guess pH, charge: {guess}, {charge}", file = output_file)
            guess += halley_delta
            charge, halley_delta = _charge_and_halleys_delta(guess, N_basic, zip(counts, PKAS_VALUES))
        print(f"guess pH, charge: {guess}, {charge}", file = output_file) 

    return guess

@timeout(3, "scipy timed out!")
def _scipy_suite_pI(_counts: dict[str, int], guess: bool = False, **kwargs) -> float:
    counts: list[int] = list(map(lambda res: _counts[res], ACID_BASE_RES)) 
    N_basic: int = sum([_counts[res] for res in BASIC_RES])
    to_optimize: func[[float], float] = lambda pH: _charge_at_pH(pH, N_basic, zip(counts, PKAS_VALUES))
    scipy_root: float
    if guess:
        x0: float = _quick_guess(to_optimize)
        if kwargs["method"] == "halley":
            scipy_root = root_scalar(lambda pH: _calc_charge_w_halleys(pH, N_basic, zip(counts, PKAS_VALUES)), x0=x0, fprime=True).root
        else:
            scipy_root = root_scalar(to_optimize, x0=x0, **kwargs).root 
    else:
        scipy_root = root_scalar(to_optimize, **kwargs).root
    return scipy_root

def _calc_charge_w_halleys(pH: float, num_basic_res: int, counts_and_pKAs: Iterator[tuple[int, float]]) -> tuple[float, float, float]:
    deriv_0: float
    deriv_1: float
    deriv_2: float
    deriv_0 = deriv_1 = deriv_2 = 0
    for count, pKA in counts_and_pKAs:
        proportion_protonated = 1 / (1 + 10 ** (pH - pKA))
        free_protons = count * (1 - proportion_protonated)
        deriv_0 += free_protons
        deriv_1 += free_protons * proportion_protonated
        deriv_2 += free_protons * proportion_protonated * (proportion_protonated - 0.5)
    proportion_protonated = 1 / (1 + 10 ** (pH - PKA_N_TERM))
    free_protons = (1 - proportion_protonated)
    deriv_0 += free_protons
    deriv_1 += free_protons * proportion_protonated
    deriv_2 += free_protons * proportion_protonated * (proportion_protonated - 0.5)
    proportion_protonated = 1 / (1 + 10 ** (pH - PKA_C_TERM))
    free_protons = (1 - proportion_protonated)
    deriv_0 += free_protons
    deriv_1 += free_protons * proportion_protonated
    deriv_2 += free_protons * proportion_protonated * (proportion_protonated - 0.5)
    charge: float = (1 + num_basic_res) - deriv_0
    return charge, - deriv_1 * log(10), deriv_2 * 2 * (log(10) ** 2)

def _charge_and_halleys_delta(pH: float, num_basic_res: int, counts_and_pKAs: Iterator[tuple[int, float]]) -> tuple[float, float]:
    charge: float
    deriv_1: float
    deriv_2: float
    charge, deriv_1, deriv_2 = _calc_charge_w_halleys(pH, num_basic_res, counts_and_pKAs)
    return charge, - charge * deriv_1 / (deriv_1 * deriv_1 + charge * deriv_2 * 0.5)

