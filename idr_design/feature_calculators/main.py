from idr_design.feature_calculators.sub_features.kappa import my_kappa
from idr_design.feature_calculators.sub_features.omega import my_omega
from idr_design.feature_calculators.sub_features.SCD import sequence_charge_decoration
from idr_design.feature_calculators.sub_features.isoelectric.main import handle_pI
from idr_design.feature_calculators.sub_features.single_pass_features import SinglePassCalculator
from typing import Callable as func, Any
from pandas import Series, DataFrame, read_csv
import os, json, re
path_to_this_file = os.path.dirname(os.path.realpath(__file__))
DEFAULT_PROTEOME_PATH: str = f"{path_to_this_file}/input_data/yeast_proteome_variance.csv"

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
    def run_feats_multiple_seqs(self, seqs: list[str] | Series) -> dict[str, dict[str, float]]:
        grid_feats: dict[str, dict[str, float]] = dict(map(lambda x: (x, {}), seqs))
        for seq in seqs:
            values: list[float] = self.run_feats(seq)
            grid_feats[seq].update(zip(self.supported_features, values))
        return grid_feats
    def run_feats_mult_seqs_skip_fail(self, seqs: list[str] | Series) -> dict[str, dict[str, float | None]]:
        grid_feats: dict[str, dict[str, float | None]] = dict(map(lambda x: (x, {}), seqs))
        for seq in seqs:
            values: list[float | None] = self.run_feats_skip_failures(seq)
            grid_feats[seq].update(zip(self.supported_features, values))
        return grid_feats
    
FeatureVector = list[float] | Series

class DistanceCalculator:
    proteome_variance: list[float]
    proteome_path: str
    def __init__(self, calculator: SequenceFeatureCalculator, proteome_path: str = DEFAULT_PROTEOME_PATH, skip_failures: bool = True):
        self.proteome_path = proteome_path
        self.proteome_variance = []
        if re.search(r".fasta$", proteome_path):
            self._init_with_fasta(calculator, proteome_path, skip_failures)
        elif re.search(r".csv$", proteome_path):
            self._init_with_csv(calculator, proteome_path)
        else:
            raise ValueError("Not a valid path!", proteome_path)
    def _init_with_fasta(self, calculator: SequenceFeatureCalculator, proteome_path: str, skip_failures: bool):
        with open(proteome_path, "r") as file:
            lines: list[str] = list(map(lambda line: line.strip("\n"),file.readlines()))
        seqs: list[str] = lines[1::2]
        assert len(seqs) > 1
        proteome_data: dict[str, dict[str, float | None]] | dict[str, dict[str, float]]
        if skip_failures:
            proteome_data = calculator.run_feats_mult_seqs_skip_fail(seqs)
        else:
            proteome_data = calculator.run_feats_multiple_seqs(seqs)
        proteome_df: DataFrame = DataFrame(proteome_data.values())
        var_list: Series = proteome_df.var()
        for feat in calculator.supported_features:
            self.proteome_variance.append(var_list[feat])
    def _init_with_csv(self, calculator: SequenceFeatureCalculator, proteome_variance_file: str):
        loaded_var_data: Series = read_csv(proteome_variance_file, index_col=0)['0']
        for feat in calculator.supported_features:
            self.proteome_variance.append(loaded_var_data[feat])
    def sqr_distance(self, vec_a: FeatureVector, vec_b: FeatureVector) -> float:
        diff: Series = Series(vec_a) - vec_b
        return sum(diff * diff / self.proteome_variance)
    def sqr_distance_many_to_one(self, df: DataFrame, target: FeatureVector) -> Series:
        return df.apply(lambda feat_vec: self.sqr_distance(feat_vec, target), axis=1)
