from idr_design.feature_calculators.sub_features.kappa import my_kappa
from idr_design.feature_calculators.sub_features.omega import my_omega
from idr_design.feature_calculators.sub_features.SCD import sequence_charge_decoration
from idr_design.feature_calculators.sub_features.isoelectric.main import handle_pI
from idr_design.feature_calculators.sub_features.single_pass_features import SinglePassCalculator
from typing import Callable as func, Any
import os, json

path_to_this_file = os.path.dirname(os.path.realpath(__file__))

# Use this class like:
# seq = ...
# calc = SequenceFeatureCalculator()
# pI = calc["pI"](seq)
class SequenceFeatureCalculator(dict[str, func[[str | dict[str, int]], float] | func[[str], float]]):
    def __init__(self):
        super().__init__(self)
        with open(f"{path_to_this_file}/all_features.json", "r") as f:
            features: dict[str, dict[str, Any]] = json.load(f)["other"]
        for feature in features:
            match feature:
                case "kappa":
                    self["my_kappa"] = my_kappa
                case "omega":
                    self["my_omega"] = my_omega
                case "SCD":
                    self["SCD"] = sequence_charge_decoration
                case "isoelectric_point":
                    self["isoelectric_point"] = handle_pI
                case _:
                    raise KeyError("Bad all_features.json input (initializing SequenceFeatureCalculator)", feature)
        temp: SinglePassCalculator = SinglePassCalculator()
        for feature in temp.keys():
            self[feature] = temp[feature]
