from idr_design.timeout_decorator import timeout
from typing import Callable as func, Iterator
from scipy.optimize import root_scalar

PKAS_BASIC: dict[str, float] = { 
    "K": 10.0, 
    "R": 12.0, 
    "H": 5.98 
}
BASIC_RES: list[str] = list(PKAS_BASIC.keys())
PKA_N_TERM: float = 7.5
PKAS_ACIDIC: dict[str, float] = { 
    "D": 4.05, 
    "E": 4.45, 
    "C": 9.0, 
    "Y": 10.0 
}
PKA_C_TERM: float = 3.55
PKAS_ALL: dict[str, float] = PKAS_BASIC.copy() 
PKAS_ALL.update(PKAS_ACIDIC)
ACID_BASE_RES: list[str] = list(PKAS_ALL.keys())
MAX_TIME_SECONDS: int = 10


def handle_pI(seq_or_counts: str | dict[str, int]) -> float:
    counts: dict[str, int]
    if isinstance(seq_or_counts, str):
        counts = dict(map(lambda res: (res, seq_or_counts.count(res)), ACID_BASE_RES)) 
    else:
        counts = seq_or_counts
    return _pI(counts)

@timeout(MAX_TIME_SECONDS, "pI calculation timed out.")
def _pI(all_counts: dict[str, int]) -> float:
    counts = list(map(lambda res: all_counts[res], ACID_BASE_RES)) 
    N_basic: int = sum([all_counts[res] for res in BASIC_RES])
    to_optimize: func[[float], float] = lambda pH: _charge_at_pH(pH, N_basic, zip(counts, PKAS_ALL.values()))
    scipy_root: float = root_scalar(to_optimize, method="brenth", bracket=(0,14)).root
    return scipy_root

def _charge_at_pH(pH: float, num_basic_res: int, counts_and_pKAs: Iterator[tuple[int, float]]) -> float:
    free_protons: float = 0
    for count, pKA in counts_and_pKAs:
        proportion_protonated = 1 / (1 + 10 ** (pH - pKA))
        free_protons += count * (1 - proportion_protonated)
    # N terminus
    proportion_protonated = 1 / (1 + (10 ** (pH - PKA_N_TERM)))
    free_protons += (1 - proportion_protonated)
    # C terminus
    proportion_protonated = 1 / (1 + (10 ** (pH - PKA_C_TERM)))
    free_protons += (1 - proportion_protonated)
    # Finally, the charge
    # add 1 to num_basic_res because I forgot about N terminus
    charge: float = (1 + num_basic_res) - free_protons
    return charge