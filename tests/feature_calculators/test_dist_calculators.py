import os
from idr_design.feature_calculators.main import SequenceFeatureCalculator as FeatCalc, DistanceCalculator as DistCalc
import pytest
from math import log
from pandas import DataFrame, Series, read_csv
from itertools import product

path_to_this_file = os.path.dirname(os.path.realpath(__file__))
FLOAT_COMPARISON_TOLERANCE = 10 ** (-11)

@pytest.mark.slow
class TestDistCalc:
    feature_calculator: FeatCalc = FeatCalc()
    distance_calculator = DistCalc(f"{path_to_this_file}/../yeast_proteome_clean.fasta", feature_calculator)
    fasta_ids: list[str]
    fasta_lookup_sequences: dict[str, str]
    fasta_lookup_results: DataFrame
    with open(f"{path_to_this_file}/../yeast_proteome_clean.fasta", "r") as fastaf:
        lines: list[str] = list(map(lambda line: line.strip("\n"),fastaf.readlines()))
    # WAY TOO SLOW TO INCLUDE ALL FASTA IDS
    fasta_ids, sequences = lines[::2], lines[1::2]
    fasta_lookup_sequences = dict(zip(fasta_ids, sequences))
    # I hate this too but there are duplicates and pytest doesn't like user defined __init__'s 
    # All this misc. bs is just removing the duplicates
    fasta_lookup_results = read_csv(f"{path_to_this_file}/230918 old code data - yeast proteome.csv", index_col=0)
    # Prevent code from compiling forever
    skip_after: int = 30
    @pytest.mark.parametrize(("i","feature"), enumerate(feature_calculator.supported_features))
    def test_proteome_variance(self, i: int, feature: str):
        var_list: Series = self._get_var_list()
        assert abs(self.distance_calculator.proteome_variance[i] - var_list[feature]) < FLOAT_COMPARISON_TOLERANCE
    @pytest.mark.parametrize(("fasta_a", "fasta_b"), product(fasta_ids[:skip_after], fasta_ids[-skip_after:]))
    def test_distance(self, fasta_a: str, fasta_b: str):
        var_list: Series = self._get_var_list() 
        try:
            feats_a = self.feature_calculator.run_feats(self.fasta_lookup_sequences[fasta_a])
            feats_b = self.feature_calculator.run_feats(self.fasta_lookup_sequences[fasta_b])
        except:
            return
        diff_sqr: list[float] = list(map(lambda x, y: (x - y) * (x - y), feats_a, feats_b))
        assert abs(self.distance_calculator.sqr_distance(feats_a, feats_b) - sum(map(lambda tup: diff_sqr[tup[0]] / var_list[tup[1]],enumerate(self.feature_calculator.supported_features)))) < FLOAT_COMPARISON_TOLERANCE
    def _get_var_list(self) -> Series:
        processed_results: DataFrame = self.fasta_lookup_results
        # sorting out some peculiar behaviour from the old cod
        processed_results["SCD"][">P32874"] = 2.401181835670923
        processed_results["my_kappa"][">P89113"] = None
        # removing duplicates!!! variance calculation was suffering
        processed_results = processed_results\
            .rename(index=self.fasta_lookup_sequences)\
            .reset_index()\
            .drop_duplicates(subset="index", keep="first")\
            .set_index("index")
        # Strangely this doesn't work if you put it above with the others
        processed_results["complexity"] *= log(20)
        return processed_results.var()

        