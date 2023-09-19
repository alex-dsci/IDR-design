from collections.abc import Callable as func
from typing import Any
import json, re, os
from math import log1p, lgamma
from idr_design.constants import AA_STRING

def _sum_scores_counts(scores: dict[str, float], pattern_counts: dict[str, int]) -> float:
    sum: float = 0
    for pattern, score in scores.items():
        sum += pattern_counts[pattern] * score
    return sum

def _sum_scores_seq(scores: dict[str, float], seq: str) -> float:
    sum: float = 0
    for pattern, score in scores.items():
        sum += len(re.findall(pattern, seq)) * score
    return sum

def _complexity(counts: list[int]) -> float:
    sum: int = 0
    log_gamma_sum: float = 0
    for count in counts:
        sum += count
        log_gamma_sum += lgamma(1 + count)
    return (lgamma(1 + sum) - log_gamma_sum) / sum

def handle_sum_or_count(pattern_or_scores: str | dict[str, float], seq_or_counts: str | dict[str, int]) -> float:
    if isinstance(seq_or_counts, str) and isinstance(pattern_or_scores, str):
        return len(re.findall(pattern_or_scores, seq_or_counts))
    if not isinstance(seq_or_counts, str) and isinstance(pattern_or_scores, str):
        return seq_or_counts[pattern_or_scores]
    if isinstance(seq_or_counts, str) and not isinstance(pattern_or_scores, str):
        return _sum_scores_seq(pattern_or_scores, seq_or_counts)
    if not isinstance(seq_or_counts, str) and not isinstance(pattern_or_scores, str):
        return _sum_scores_counts(pattern_or_scores, seq_or_counts)
    raise Exception("Unreachable code!", pattern_or_scores, seq_or_counts)

def handle_average(pattern_or_scores: str | dict[str, float], seq_or_counts: str | dict[str, int]) -> float:
    count: int = 0
    if isinstance(seq_or_counts, str):
        count = len(seq_or_counts)
    else:
        count = sum(seq_or_counts.values())
    if count == 0:
        raise ZeroDivisionError("Offending arguments: seq_or_counts", seq_or_counts)
    score: float = handle_sum_or_count(pattern_or_scores, seq_or_counts)
    return score / count

def handle_log1p_ratio(residue_scores_num: str | dict[str, float], residue_scores_denom: str | dict[str, float], seq_or_counts: str | dict[str, int]) -> float:
    if not isinstance(residue_scores_num, str):
        negative_scores = [pair for pair in residue_scores_num.items() if pair[1] < 0]
        if len(negative_scores) > 0:
            raise ValueError("Negative scores inputted. Offending arguments: residue_scores_num, negative_scores", residue_scores_num, negative_scores)
    if not isinstance(residue_scores_denom, str):
        negative_scores = [pair for pair in residue_scores_denom.items() if pair[1] < 0]
        if len(negative_scores) > 0:
            raise ValueError("Negative scores inputted. Offending arguments: residue_scores_denom, negative_scores", residue_scores_denom, negative_scores)
    num: float = handle_sum_or_count(residue_scores_num, seq_or_counts)
    denom: float = handle_sum_or_count(residue_scores_denom, seq_or_counts)
    return log1p(num) - log1p(denom)

def handle_length_calc(pattern: str, seq: str | dict[str, int], include_first: bool = False) -> int:
    if not isinstance(seq, str):
        raise ValueError("Length calculations take strings. Offending arguments: seq", seq)
    sum: int = 0
    for occurrence in re.finditer(pattern, seq):
        length_minus_first = occurrence.span()[1] - occurrence.span()[0] + int(include_first) - 1
        sum += length_minus_first
    return sum

def handle_complexity(seq_or_counts: str | dict[str, int]) -> float:
    counts: list[int]
    if isinstance(seq_or_counts, str):
        counts = [seq_or_counts.count(res) for res in AA_STRING]
    else:
        counts = list(seq_or_counts.values())
    return _complexity(counts)

class SinglePassCalculator(dict[str, func[[str | dict[str, int]], float]]):

    def __init__(self) -> None:
        super().__init__(self)
        path_to_this_file = os.path.dirname(os.path.realpath(__file__))
        with open(f"{path_to_this_file}/../all_features.json", "r") as f:
            features: dict[str, dict[str, Any]] = json.load(f)["single_pass"]
        for feature, calculation in features.items():
            match calculation["type"]:
                case "count":
                    self._init_count(feature, calculation)
                case "sum":
                    self._init_sum(feature, calculation)
                case "average":
                    self._init_average(feature, calculation)
                case "log_ratio":
                    self._init_log1p_ratio(feature, calculation)
                case "length":
                    self._init_length(feature, calculation)
                case "complexity":
                    self._init_complexity()
                case _:
                    raise KeyError("Offending arguments: calculation", calculation)

    def _init_count(self, feature: str, calculation: dict[str, Any]) -> None:
        pattern: str = ""
        if "pattern" in calculation.keys():
            pattern = calculation["pattern"]
        else:
            raise KeyError("Offending arguments: calculation", calculation)
        self[feature] = lambda seq_or_counts: handle_sum_or_count(pattern, seq_or_counts)

    def _init_sum(self, feature: str, calculation: dict[str, Any]) -> None:
        scores: dict[str, float] | str = {}
        if "score_mapping" in calculation.keys(): 
            scores = calculation["score_mapping"]
        else:
            raise KeyError("Offending arguments: calculation", calculation)
        self[feature] = lambda seq_or_counts: handle_sum_or_count(scores, seq_or_counts)

    def _init_average(self, feature: str, calculation: dict[str, Any]) -> None:
        pattern_or_scores: dict[str, float] | str = {}
        if "pattern" in calculation.keys() and "score_mapping" not in calculation.keys():
            pattern_or_scores = calculation["pattern"]
        elif "pattern" not in calculation.keys() and "score_mapping" in calculation.keys(): 
            pattern_or_scores = calculation["score_mapping"]
        else:
            raise KeyError("Offending arguments: calculation", calculation) 
        self[feature] = lambda seq_or_counts: handle_average(pattern_or_scores, seq_or_counts)

    def _init_log1p_ratio(self, feature: str, calculation: dict[str, Any]) -> None:
        pattern_or_scores_A: dict[str, float] | str = {}
        pattern_or_scores_B: dict[str, float] | str = {}
        if "numerator_pattern" in calculation.keys() and "denominator_pattern" in calculation.keys():
            pattern_or_scores_A = calculation["numerator_pattern"]
            pattern_or_scores_B = calculation["denominator_pattern"]
        else:
            raise KeyError("Offending arguments: calculation", calculation) 
        self[feature] = lambda seq_or_counts: handle_log1p_ratio(pattern_or_scores_A, pattern_or_scores_B, seq_or_counts)

    def _init_length(self, feature: str, calculation: dict[str, Any]) -> None:
        if "pattern" in calculation.keys():
            pattern = calculation["pattern"]
        else:
            raise KeyError("Offending arguments: calculation", calculation)
        self[feature] = lambda seq: handle_length_calc(pattern, seq)

    def _init_complexity(self) -> None:
        self["complexity"] = handle_complexity
    