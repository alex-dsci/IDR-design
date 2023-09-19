from idr_design.feature_calculators.sub_features.isoelectric._other_isoelectric_methods import _halley_pI, _scipy_suite_pI
from idr_design.feature_calculators.sub_features.isoelectric._old_isoelectric import isoelectric_point as old_isoelectric_point
import idr_design.feature_calculators.sub_features.isoelectric.main as main
import os, pytest

from typing import Callable as func
from time import time

path_to_this_file = os.path.dirname(os.path.realpath(__file__))

TEST_TOLERANCE = 10 ** (-11)
CONVERGE_TOLERANCE: float = 10 ** (-13)
SCIPY_METHODS: dict[str, func[[dict[str, int]], float]] = {
    "brentq": lambda counts: _scipy_suite_pI(counts, method="brentq", bracket=(0,14)),
    "bisect": lambda counts: _scipy_suite_pI(counts, method="bisect", bracket=(0,14)),
    "brenth": lambda counts: _scipy_suite_pI(counts, method="brenth", bracket=(0,14)),
    "ridder": lambda counts: _scipy_suite_pI(counts, method="ridder", bracket=(0,14)),
    "toms748": lambda counts: _scipy_suite_pI(counts, method="toms748", bracket=(0,14)),
    "secant": lambda counts: _scipy_suite_pI(counts, guess=True, method="secant"),
    "newton": lambda counts: _scipy_suite_pI(counts, guess=True, method="newton"),
    "halley": lambda counts: _scipy_suite_pI(counts, guess=True, method="halley")
}

@pytest.mark.slow
class Test:
    feature_lookup_column: int = 88
    fasta_ids: list[str]
    sequences: list[str]
    fasta_lookup_sequences: dict[str, str]
    with open(f"{path_to_this_file}/yeast_proteome_clean.fasta", "r") as fastaf, open(f"{path_to_this_file}/230918 old code data - yeast proteome.csv") as resultf:
        lines: list[str] = list(map(lambda line: line.strip("\n"),fastaf.readlines()))
        fasta_ids, sequences = lines[::2], lines[1::2]
        fasta_lookup_sequences = dict(zip(fasta_ids, sequences))
    # These test assumes the old bisect code produces the correct result.
    # To write an actual pI test requires a manual pI calculation, which I am not going to waste my time on.
    @pytest.mark.slow_but_run_anyway
    @pytest.mark.parametrize(("fasta_id"), fasta_ids)
    def test_main(self, fasta_id: str):
        counts: dict[str, int] = dict(map(lambda x: (x, self.fasta_lookup_sequences[fasta_id].count(x)), "DEHCYKR"))
        old_val: float = main.get_pI(counts) 
        new_val: float = old_isoelectric_point(counts, CONVERGE_TOLERANCE)
        assert abs(new_val - old_val) < TEST_TOLERANCE 
    def test_my_halley(self):
        old_method_clock: float = 0
        halley_clock: float = 0
        for seq in self.sequences:
            counts: dict[str, int] = dict(map(lambda x: (x, seq.count(x)), "DEHCYKR"))
            old_method_clock -= time()
            old_val: float = old_isoelectric_point(counts, CONVERGE_TOLERANCE) 
            old_method_clock += time()
            halley_clock -= time()
            new_val: float = _halley_pI(counts, CONVERGE_TOLERANCE)
            halley_clock += time()
            assert abs(new_val - old_val) < TEST_TOLERANCE
        print()
        print(f"Time halleys (mine): {halley_clock / len(self.sequences)}")
        print(f"Time bisect (old code): {old_method_clock / len(self.sequences)}")
    def _test_generic_scipy(self, method_name: str):
        old_method_clock: float = 0
        new_method_clock: float = 0
        new_method: func[[dict[str, int]], float] = SCIPY_METHODS[method_name]
        for seq in self.sequences:
            counts: dict[str, int] = dict(map(lambda x: (x, seq.count(x)), "DEHCYKR"))
            old_method_clock -= time()
            old_val: float = old_isoelectric_point(counts, CONVERGE_TOLERANCE) 
            old_method_clock += time()
            new_method_clock -= time()
            new_val: float = new_method(counts) 
            new_method_clock += time()
            assert abs(new_val - old_val) < TEST_TOLERANCE
        print()
        print(f"Time {method_name} (scipy): {new_method_clock / len(self.sequences)}")
    def test_brentq(self):
        self._test_generic_scipy("brentq")
    def test_bisect(self):
        self._test_generic_scipy("bisect")
    def test_brenth(self):
        self._test_generic_scipy("brenth")
    def test_ridder(self):
        self._test_generic_scipy("ridder")
    def test_toms(self):
        self._test_generic_scipy("toms748")    
    def test_newton(self):
        self._test_generic_scipy("newton")
    def test_secant(self):
        self._test_generic_scipy("secant")
    def test_scipy_halley(self):
        self._test_generic_scipy("halley")
    