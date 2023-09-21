import os
from idr_design.feature_calculators.main import SequenceFeatureCalculator as FeatCalc
import pytest
from math import log
from itertools import product

path_to_this_file = os.path.dirname(os.path.realpath(__file__))
FLOAT_COMPARISON_TOLERANCE = 10 ** (-12)

class TestFeatCalc:
    feature_calculator: FeatCalc = FeatCalc()
    fasta_ids: list[str]
    fasta_lookup_sequences: dict[str, str]
    fasta_lookup_results: dict[str, dict[str, float]]
    features: list[str]
    with open(f"{path_to_this_file}/../yeast_proteome_clean.fasta", "r") as fastaf, open(f"{path_to_this_file}/230918 old code data - yeast proteome.csv", "r") as resultf:
        lines: list[str] = list(map(lambda line: line.strip("\n"),fastaf.readlines()))
        fasta_ids, sequences = lines[::2], lines[1::2]
        fasta_lookup_sequences = dict(zip(fasta_ids, sequences))
        features = resultf.readline().strip("\n").split(",")[1:]
        fasta_lookup_results = {}
        for row in map(lambda line: line.strip("\n"),resultf.readlines()):
            fasta_id: str = row.split(",")[0]
            results: list[float] = list(map(float,row.split(",")[1:]))
            results_dict: dict[str, float] = dict(zip(features, results))
            fasta_lookup_results[fasta_id] = results_dict
    nice_fasta_ids: list[str] = fasta_ids.copy()
    nice_fasta_ids.remove(">P32874")
    nice_fasta_ids.remove(">P89113")
    @pytest.mark.parametrize(("fasta_id"), fasta_ids)
    def test_individual_calcs(self, fasta_id: str):
        # old code did it in units of log base 20, which gets normalized out later
        self.fasta_lookup_results[fasta_id]["complexity"] *= log(20)
        seq: str = self.fasta_lookup_sequences[fasta_id] 
        for feature in self.features:
            expected: float = self.fasta_lookup_results[fasta_id][feature]
            match fasta_id, feature:
                # test scd - old code made a mistake that shows up here
                case ">P32874", "SCD":
                    assert self.feature_calculator[feature](seq) == 2.401181835670923 
                # test kappa - throws error
                case ">P89113", "my_kappa":
                    with pytest.raises(ValueError):
                        self.feature_calculator[feature](seq)
                case _:
                    assert abs(self.feature_calculator[feature](seq) - expected) < FLOAT_COMPARISON_TOLERANCE
    # Prevent code from compiling forever
    skip_after: int = 1000
    @pytest.mark.parametrize(("fasta_id"), nice_fasta_ids[:skip_after])
    def test_run_feats(self, fasta_id: str):
        # old code did it in units of log base 20, which gets normalized out later
        self.fasta_lookup_results[fasta_id]["complexity"] *= log(20)
        seq: str = self.fasta_lookup_sequences[fasta_id]
        calc: list[float] = self.feature_calculator.run_feats(seq)
        for i, feature in enumerate(self.feature_calculator.supported_features):
            expected: float = self.fasta_lookup_results[fasta_id][feature]
            assert abs(calc[i] - expected) < FLOAT_COMPARISON_TOLERANCE
    @pytest.mark.parametrize(("fasta_id"), fasta_ids[:skip_after])
    def test_run_feats_skip_failures(self, fasta_id: str):
        seq: str = self.fasta_lookup_sequences[fasta_id]
        calc: list[float | None] = self.feature_calculator.run_feats_skip_failures(seq)
        for i, feature in enumerate(self.feature_calculator.supported_features):
            expected: float = self.fasta_lookup_results[fasta_id][feature]
            result: float | None = calc[i]
            match fasta_id, feature:
                # test scd - old code made a mistake that shows up here
                case ">P32874", "SCD":
                    assert result is not None
                    assert result == 2.401181835670923 
                # test kappa - throws error
                case ">P89113", "my_kappa":
                    assert result is None
                case _:
                    assert result is not None
                    assert abs(result - expected) < FLOAT_COMPARISON_TOLERANCE
    @pytest.mark.slow
    def test_run_feats_multiple_seqs(self):
        massive_result: dict[str, dict[str, float | None]] = \
            self.feature_calculator.run_feats_mult_seqs_skip_fail(self.sequences)
        for fasta_id, feature in product(self.fasta_ids, self.feature_calculator.supported_features):
            seq: str = self.fasta_lookup_sequences[fasta_id]
            expected: float = self.fasta_lookup_results[fasta_id][feature]
            result: float | None = massive_result[seq][feature]
            match fasta_id, feature:
                # test scd - old code made a mistake that shows up here
                case ">P32874", "SCD":
                    assert result is not None
                    assert result == 2.401181835670923
                # test kappa - throws error
                case ">P89113", "my_kappa":
                    assert result is None
                case _:
                    assert result is not None
                    assert abs(result - expected) < FLOAT_COMPARISON_TOLERANCE


        