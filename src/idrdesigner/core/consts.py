"""Literal constants for the project."""
from typing import Final

AMINOACIDS: Final[list[str]] = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
]
PKAS_BASIC: Final[dict[str, float]] = {"K": 10.0, "R": 12.0, "H": 5.98}
BASIC_RES: Final[list[str]] = list(PKAS_BASIC.keys())
PKA_N_TERM: Final[float] = 7.5
PKAS_ACIDIC: Final[dict[str, float]] = {
    "D": 4.05,
    "E": 4.45,
    "C": 9.0,
    "Y": 10.0,
}
PKA_C_TERM: Final[float] = 3.55
PKAS_ALL: Final[dict[str, float]] = PKAS_BASIC.copy()
PKAS_ALL.update(PKAS_ACIDIC)
ACID_BASE_RES: Final[list[str]] = list(PKAS_ALL.keys())
BINARY_CHARGE: Final[dict[str, int]] = {"D": -1, "E": -1, "K": 1, "R": 1}
CHARGED_RES: Final[list[str]] = list(BINARY_CHARGE.keys())
