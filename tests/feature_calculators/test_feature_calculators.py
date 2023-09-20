import os
from idr_design.feature_calculators.main import SequenceFeatureCalculator as Calculator
import pytest
from math import log

path_to_this_file = os.path.dirname(os.path.realpath(__file__))
FLOAT_COMPARISON_TOLERANCE = 10 ** (-12)



class Test:
    feature_calculator: Calculator = Calculator()
    fasta_ids: list[str]
    fasta_lookup_sequences: dict[str, str]
    fasta_lookup_results: dict[str, dict[str, float]]
    features: list[str]
    with open(f"{path_to_this_file}/yeast_proteome_clean.fasta", "r") as fastaf, open(f"{path_to_this_file}/230918 old code data - yeast proteome.csv", "r") as resultf:
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
    
    @pytest.mark.parametrize(("fasta_id"), fasta_ids)
    def test_calc(self, fasta_id: str):
        self.fasta_lookup_results[fasta_id]["complexity"] *= log(20)
        seq: str = self.fasta_lookup_sequences[fasta_id] 
        for feature in self.features:
            expected: float = self.fasta_lookup_results[fasta_id][feature]
            match fasta_id, feature:
                # test scd - old code made a mistake that shows up here
                case ">P32874", "SCD":
                    continue 
                # test kappa - throws error
                case ">P89113", "my_kappa":
                    with pytest.raises(ValueError):
                        self.feature_calculator[feature](seq)
                case _:
                    assert abs(self.feature_calculator[feature](seq) - expected) <= FLOAT_COMPARISON_TOLERANCE
    
        