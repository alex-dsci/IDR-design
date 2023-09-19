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
KAS_ALL: list = list(map(lambda x: 10 ** (-x), PKAS_ALL.values()))
MAX_TIME_SECONDS: int = 10

@timeout(MAX_TIME_SECONDS, "pI calculation timed out.")
def get_pI(seq_or_counts: str | dict[str, int]) -> float:
    counts: list[int]
    N_basic: int
    if isinstance(seq_or_counts, str):
        counts = list(map(lambda res: seq_or_counts.count(res), ACID_BASE_RES)) 
        N_basic: int = sum([seq_or_counts.count(res) for res in BASIC_RES])
    else:
        counts = list(map(lambda res: seq_or_counts[res], ACID_BASE_RES)) 
        N_basic: int = sum([seq_or_counts[res] for res in BASIC_RES])
    to_optimize: func[[float], float] = lambda pH: charge_at_pH(pH, N_basic, zip(counts, KAS_ALL))
    scipy_root: float = root_scalar(to_optimize, method="brenth", bracket=(0,14)).root
    return scipy_root

def charge_at_pH(pH: float, num_basic_res: int, counts_and_KAs: Iterator[tuple[int, float]]) -> float:
    free_protons: float = 0
    for count, Ka in counts_and_KAs:
        proportion_protonated = 1 / (1 + Ka * (10 ** pH))
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