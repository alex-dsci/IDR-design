from idr_design.feature_calculators.sub_features.kappa import my_kappa
from idr_design.feature_calculators.sub_features.omega import my_omega
from idr_design.feature_calculators.sub_features.SCD import sequence_charge_decoration
from idr_design.feature_calculators.sub_features.isoelectric.main import handle_pI
from idr_design.feature_calculators.sub_features.single_pass_features import SinglePassCalculator
from typing import Callable as func, Any, Iterator
from itertools import product
from numpy import var
import os, json
path_to_this_file = os.path.dirname(os.path.realpath(__file__))

# Use this class like:
# seq = ...
# calc = SequenceFeatureCalculator()
# pI = calc["pI"](seq)
FeatCalcHandler = func[[dict[str, int] | str], float] \
    | func[[str], float] 
class SequenceFeatureCalculator(dict[str, FeatCalcHandler]):
    supported_features: list[str]
    def __init__(self):
        super().__init__(self)
        with open(f"{path_to_this_file}/all_features.json", "r") as f:
            features: dict[str, dict[str, Any]] = json.load(f)["other"]
        for feature in features:
            match feature:
                case "my_kappa":
                    self["my_kappa"] = my_kappa
                case "my_omega":
                    self["my_omega"] = my_omega
                case "SCD":
                    self["SCD"] = sequence_charge_decoration
                case "isoelectric_point":
                    self["isoelectric_point"] = handle_pI
                case _:
                    raise KeyError("Bad all_features.json input (initializing SequenceFeatureCalculator)", feature)
        for feature, calc in SinglePassCalculator().items():
            self[feature] = calc
        self.supported_features = list(self.keys())
    def run_feats(self, seq: str) -> list[float]:
        return list(map(lambda feat: self[feat](seq), self.supported_features))
    def run_feats_skip_failures(self, seq: str) -> list[float | None]:
        result: list[float | None] = []
        for feat in self.supported_features:
            try:
                value: float = self[feat](seq)
                result.append(value)
            except:
                result.append(None)
        return result
    def run_feats_multiple_seqs(self, seqs: list[str]) -> dict[str, dict[str, float]]:
        grid_feats: dict[str, dict[str, float]] = dict(map(lambda x: (x, {}), seqs))
        for seq in seqs:
            values: list[float] = self.run_feats(seq)
            grid_feats[seq].update(zip(self.supported_features, values))
        return grid_feats
    def run_feats_mult_seqs_skip_fail(self, seqs: list[str]) -> dict[str, dict[str, float | None]]:
        grid_feats: dict[str, dict[str, float | None]] = dict(map(lambda x: (x, {}), seqs))
        for seq in seqs:
            values: list[float | None] = self.run_feats_skip_failures(seq)
            grid_feats[seq].update(zip(self.supported_features, values))
        return grid_feats
    
FeatureVector = list[float]

class DistanceCalculator:
    proteome_variance: list[float]
    proteome_path: str
    def __init__(self, proteome_path: str, calculator: SequenceFeatureCalculator, skip_failures: bool = True):
        self.proteome_path = proteome_path
        with open(proteome_path, "r") as file:
            lines: list[str] = list(map(lambda line: line.strip("\n"),file.readlines()))
        seqs: list[str] = lines[1::2]
        assert len(seqs) > 1
        if skip_failures:
            proteome_data_dirty: list[dict[str, float | None]] = list(calculator.run_feats_mult_seqs_skip_fail(seqs).values())
            for feat in calculator.supported_features:
                sum_sqrd: float = 0
                sum_feat: float = 0
                n: int = 0
                for protein_data in proteome_data_dirty:
                    dirty_value: float | None = protein_data[feat]
                    if isinstance(dirty_value, float):
                        sum_feat += dirty_value
                        sum_sqrd += dirty_value * dirty_value
                        n += 1
                assert n > 1 
                variance: float = (sum_sqrd - sum_feat * sum_feat) / (n - 1)
                assert variance != 0
                self.proteome_variance.append(variance)
        else:
            proteome_data_clean: list[dict[str, float]] = list(calculator.run_feats_multiple_seqs(seqs).values())
            for feat in calculator.supported_features:
                sum_sqrd: float = 0
                sum_feat: float = 0
                for protein_data in proteome_data_clean:
                    value: float = protein_data[feat]
                    sum_feat += value
                    sum_sqrd += value * value
                variance: float = (sum_sqrd - sum_feat * sum_feat) / (len(seqs) - 1)
                assert variance != 0 
                self.proteome_variance.append(variance)
    def sqr_distance(self, vec_a: FeatureVector, vec_b: FeatureVector) -> float:
        return sum(map(lambda a, b, var: (a - b) * (a - b) / var, vec_a, vec_b, self.proteome_variance))