import os
from itertools import product
import idr_design.feature_calculators.sub_features.single_pass_features as CB
from idr_design.feature_calculators.sub_features.single_pass_features import SinglePassCalculator as CBF
import random, re, pytest
from math import log1p, log

path_to_this_file = os.path.dirname(os.path.realpath(__file__))
ALPHABET_STRING = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
EXAMPLE_REGEX_ADDON = "{3,}"
SIZE = 1000
SEED = 2023
REGEX_MATCH_ANY = "."
REGEX_MATCH_NONE = "@"
FLOAT_COMPARISON_TOLERANCE = 10 ** (-14)

class BigExample:
    scores_A: dict[str, float]
    scores_B: dict[str, float]
    abscore_A: dict[str, float]
    abscore_B: dict[str, float]
    counts: dict[str, int]
    regex_pattern: str
    sequence: str
    regex_count: int
    regex_length: int
    dot_product_A: float
    dot_product_B: float
    dot_abs_A: float
    dot_abs_B: float
    def __init__(self, size: int, seed: int) -> None:
        random.seed(seed)
        self._init_inputs(size)
        self._calculate_test_results()
    def _init_inputs(self, size: int) -> None:
        self.scores_A = {}
        self.scores_B = {}
        self.abscore_A = {}
        self.abscore_B = {}
        self.counts = {}
        temp_sequence: list[str] = []
        looking_for_negatives_A: bool = True
        looking_for_negatives_B: bool = True
        while looking_for_negatives_A or looking_for_negatives_B or len(temp_sequence) == 0:
            looking_for_negatives_A = True
            looking_for_negatives_B = True
            temp_sequence = []
            self.regex_pattern = "["
            for char in ALPHABET_STRING:
                self.scores_A[char] = random.uniform(-1,1)
                if looking_for_negatives_A and self.scores_A[char] < 0:
                    looking_for_negatives_A = False
                self.abscore_A[char] = abs(self.scores_A[char])
                self.scores_B[char] = random.uniform(-1,1)
                if looking_for_negatives_B and self.scores_B[char] < 0:
                    looking_for_negatives_B = False
                self.abscore_B[char] = abs(self.scores_B[char])
                self.counts[char] = random.randint(0,1) * random.randint(1,size)
                if self.counts[char] > 0 and (len(self.regex_pattern) == 1 or random.randint(0, 2) > 1):
                    self.regex_pattern += char
                temp_sequence += [char] * self.counts[char]
        self.regex_pattern += f"]{EXAMPLE_REGEX_ADDON}"
        random.shuffle(temp_sequence)
        self.sequence = "".join(temp_sequence)
    def _calculate_test_results(self) -> None:
        self.regex_count = 0
        self.regex_length = 0
        for occurrence in re.finditer(self.regex_pattern, self.sequence):
            self.regex_count += 1
            self.regex_length += occurrence.span()[1] - occurrence.span()[0] - 1
        self.dot_product_A = 0
        self.dot_product_B = 0
        self.dot_abs_A = 0
        self.dot_abs_B = 0
        for char in ALPHABET_STRING:
            self.dot_product_A += self.scores_A[char] * self.counts[char]
            self.dot_product_B += self.scores_B[char] * self.counts[char]
            self.dot_abs_A += abs(self.scores_A[char]) * self.counts[char]
            self.dot_abs_B += abs(self.scores_B[char]) * self.counts[char]
    def __str__(self) -> str:
        return f"test_composition_based BigExample object - SIZE: {SIZE}, SEED: {SEED}\n" + \
            f"scores_A:\n{self.scores_A}\n" + \
            f"scores_B:\n{self.scores_B}\n" + \
            f"abscore_A:\n{self.abscore_A}\n" + \
            f"abscore_B:\n{self.abscore_B}\n" + \
            f"counts:\n{self.counts}\n" + \
            f"sequence:\n{self.sequence}\n" + \
            f"regex_pattern:\n{self.regex_pattern}\n" + \
            f"regex_counts:\n{self.regex_count}\n" + \
            f"regex_length:\n{self.regex_length}\n" + \
            f"dot_product_A:\n{self.dot_product_A}\n" + \
            f"dot_product_B:\n{self.dot_product_B}\n" + \
            f"dot_abs_A:\n{self.dot_abs_A}\n" + \
            f"dot_abs_B:\n{self.dot_abs_B}\n"

class TestProcedures:
    small_example = {
        "scores": {"A": 1, "B": -1, "C": 0.5},
        "sequence": "AABCBCBCBCCAB",
        "counts": {"A": 3, "B": 5, "C": 5},
        "dot": (1 * 3 + -1 * 5 + 0.5 * 5)
    }
    example = BigExample(SIZE, SEED)
    with open(f"{path_to_this_file}/SPCB_BigExample.txt", "w") as f:
        print(example, file=f)
    def test_sum_scores_counts(self):
        assert CB._sum_scores_counts(self.small_example["scores"], self.small_example["counts"]) == self.small_example["dot"]
        assert CB._sum_scores_counts(self.example.scores_A, self.example.counts) == self.example.dot_product_A
        assert CB._sum_scores_counts(self.example.scores_B, self.example.counts) == self.example.dot_product_B
    def test_sum_scores_seq(self):
        assert CB._sum_scores_seq(self.small_example["scores"], self.small_example["sequence"]) == self.small_example["dot"]
        assert CB._sum_scores_seq(self.example.scores_A, self.example.sequence) == self.example.dot_product_A
        assert CB._sum_scores_seq(self.example.scores_B, self.example.sequence) == self.example.dot_product_B
    def test_handle_sum_or_count(self):
        assert CB.handle_sum_or_count("BC", self.small_example["sequence"]) == 4
        assert CB.handle_sum_or_count("(BC){2}", self.small_example["sequence"]) == 2
        assert CB.handle_sum_or_count("(BC){3}", self.small_example["sequence"]) == 1
        assert CB.handle_sum_or_count("(BC){5}", self.small_example["sequence"]) == 0
        assert CB.handle_sum_or_count("(BC)+", self.small_example["sequence"]) == 1
        assert CB.handle_sum_or_count("BC", {"BC": len(re.findall("BC", self.small_example["sequence"]))}) == 4 
        assert CB.handle_sum_or_count(self.example.scores_A, self.example.counts) == self.example.dot_product_A
        assert CB.handle_sum_or_count(self.example.scores_A, self.example.sequence) == self.example.dot_product_A
        assert CB.handle_sum_or_count(self.example.scores_B, self.example.counts) == self.example.dot_product_B
        assert CB.handle_sum_or_count(self.example.scores_B, self.example.sequence) == self.example.dot_product_B
        for char in ALPHABET_STRING:
            assert CB.handle_sum_or_count(char, self.example.counts) == self.example.counts[char]
            assert CB.handle_sum_or_count(char, self.example.sequence) == self.example.counts[char]
        assert CB.handle_sum_or_count(self.example.regex_pattern, self.example.sequence) == self.example.regex_count
        assert CB.handle_sum_or_count(self.example.regex_pattern, {self.example.regex_pattern: self.example.regex_count}) == self.example.regex_count
        assert CB.handle_sum_or_count(REGEX_MATCH_NONE, self.example.sequence) == 0
        assert CB.handle_sum_or_count(REGEX_MATCH_ANY, self.example.sequence) == len(self.example.sequence)
    def test_handle_average(self):
        assert CB.handle_average(self.example.scores_A, self.example.counts) == self.example.dot_product_A / len(self.example.sequence)
        assert CB.handle_average(self.example.scores_A, self.example.sequence) == self.example.dot_product_A / len(self.example.sequence)
        assert CB.handle_average(self.example.scores_B, self.example.counts) == self.example.dot_product_B / len(self.example.sequence)
        assert CB.handle_average(self.example.scores_B, self.example.sequence) == self.example.dot_product_B / len(self.example.sequence)
        for char in ALPHABET_STRING:
            assert CB.handle_average(char, self.example.counts) == self.example.counts[char] / len(self.example.sequence) 
            assert CB.handle_average(char, self.example.sequence) == self.example.counts[char] / len(self.example.sequence)
        assert CB.handle_average(self.example.regex_pattern, self.example.sequence) == self.example.regex_count / len(self.example.sequence)
        assert CB.handle_average(REGEX_MATCH_NONE, self.example.sequence) == 0
        assert CB.handle_average(REGEX_MATCH_ANY, self.example.sequence) == 1
        with pytest.raises(ZeroDivisionError) as e:
            CB.handle_average(REGEX_MATCH_ANY, "")
        assert e.value.args[0] == "Offending arguments: seq_or_counts"
    def test_handle_log1p_ratio(self):
        assert CB.handle_log1p_ratio(self.example.abscore_A, self.example.abscore_B, self.example.counts) == log1p(self.example.dot_abs_A) - log1p(self.example.dot_abs_B)
        assert CB.handle_log1p_ratio(self.example.abscore_A, self.example.abscore_B, self.example.sequence) == log1p(self.example.dot_abs_A) - log1p(self.example.dot_abs_B)
        assert CB.handle_log1p_ratio(self.example.regex_pattern, self.example.abscore_A, self.example.sequence) == log1p(self.example.regex_count) - log1p(self.example.dot_abs_A)
        for char_A, char_B in product(ALPHABET_STRING, ALPHABET_STRING):
            assert CB.handle_log1p_ratio(char_A, char_B, self.example.sequence) == log1p(self.example.counts[char_A]) - log1p(self.example.counts[char_B])
        assert CB.handle_log1p_ratio(self.example.abscore_A, self.example.regex_pattern, self.example.sequence) ==  log1p(self.example.dot_abs_A) - log1p(self.example.regex_count)
        assert CB.handle_log1p_ratio(self.example.regex_pattern, self.example.abscore_A, self.example.sequence) ==  log1p(self.example.regex_count) - log1p(self.example.dot_abs_A)
        assert CB.handle_log1p_ratio(self.example.abscore_A, self.example.abscore_A, self.example.sequence) == 0
        assert CB.handle_log1p_ratio(self.example.regex_pattern, self.example.regex_pattern, self.example.sequence) == 0
        with pytest.raises(ValueError) as e:
            CB.handle_log1p_ratio(self.example.scores_A, self.example.abscore_B, self.example.counts)
        assert e.value.args[0] == "Negative scores inputted. Offending arguments: residue_scores_num, negative_scores"
        with pytest.raises(ValueError) as e:
            CB.handle_log1p_ratio(self.example.abscore_A, self.example.scores_B, self.example.counts)
        assert e.value.args[0] == "Negative scores inputted. Offending arguments: residue_scores_denom, negative_scores"
    def test_handle_length_calc(self):
        assert CB.handle_length_calc("BC+", self.small_example["sequence"]) == 9 - 4
        assert CB.handle_length_calc("BC+", self.small_example["sequence"], True) == 9
        assert CB.handle_length_calc("(BC)+", self.small_example["sequence"]) == 8 - 1
        assert CB.handle_length_calc("(BC)+", self.small_example["sequence"], True) == 8
        assert CB.handle_length_calc("(BC){3}", self.small_example["sequence"]) == 6 - 1
        assert CB.handle_length_calc("(BC){3}", self.small_example["sequence"], True) == 6
        assert CB.handle_length_calc("(BC){2}", self.small_example["sequence"]) == 8 - 2
        assert CB.handle_length_calc("(BC){2}", self.small_example["sequence"], True) == 8
        assert CB.handle_length_calc(self.example.regex_pattern, self.example.sequence) == self.example.regex_length
        with pytest.raises(ValueError) as e:
            CB.handle_length_calc(self.example.regex_pattern, self.example.counts) 
        assert e.value.args[0] == "Length calculations take strings. Offending arguments: seq"
        assert CB.handle_length_calc(REGEX_MATCH_ANY, self.example.sequence) == 0
        assert CB.handle_length_calc(REGEX_MATCH_ANY, self.example.sequence, True) == len(self.example.sequence)
        assert CB.handle_length_calc(REGEX_MATCH_NONE, self.example.sequence) == 0
 
class TestCBF:
    feature_calculator = CBF()
    fasta_ids: list[str]
    fasta_lookup_sequences: dict[str, str]
    fasta_lookup_results: dict[str, dict[str, float]]
    features: list[str]
    with open(f"{path_to_this_file}/../yeast_proteome_clean.fasta", "r") as fastaf, open(f"{path_to_this_file}/../230918 old code SPCB feats - yeast proteome.csv", "r") as resultf:
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
    def test_cbf_against_old_program(self, fasta_id: str):
        seq: str = self.fasta_lookup_sequences[fasta_id]
        for feature in self.features:
            expected: float = self.fasta_lookup_results[fasta_id][feature]
            if feature == "complexity":
                expected *= log(20)
            assert abs(self.feature_calculator[feature](seq) - expected) <= FLOAT_COMPARISON_TOLERANCE
    

    
            

