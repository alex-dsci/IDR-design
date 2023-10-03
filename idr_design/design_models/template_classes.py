import random
from idr_design.constants import AA_STRING
from idr_design.feature_calculators.main import DistanceCalculator as DistCalc, SequenceFeatureCalculator as FeatCalc
from abc import abstractmethod, ABC
from sys import stdout
from typing import TextIO, Iterator
from pandas import Series
from time import time
from math import sqrt

TERMINAL_LENGTH: int | None = None
DEFAULT_PRECISION: float = 10 ** (-4)
class SequenceDesigner(ABC):
    feature_calculator: FeatCalc 
    distance_calculator: DistCalc
    seed: str | None
    log: TextIO | None
    def __init__(self, seed: str | None = None) -> None:
        self.feature_calculator = FeatCalc()
        self.distance_calculator = DistCalc(self.feature_calculator)
        self.seed = seed
    def design_similar(self, query: str | int, target: str, verbose: bool = False) -> list[str]:
        if verbose:
            self.log = stdout
        else:
            self.log = None                
        queries: list[str]
        if isinstance(query, int):
            queries = self._get_random_seqs(target, query)
        else:
            queries = [query]
        output: list[str] = []
        for seq in queries:
            output.append(self.search_similar(seq, target))
        return output
    def _get_random_seqs(self, target: str, n: int) -> list[str]:
        if self.seed is not None:
            random.seed(self.seed + target)
        output: list[str] = []
        for _ in range(n):
            new_seq: str = "".join([random.choice(AA_STRING) for _ in range(len(target))])
            while True:
                if None in self.feature_calculator.run_feats_skip_failures(new_seq):
                    new_seq: str = "".join([random.choice(AA_STRING) for _ in range(len(target))])
                    continue
                break
            output.append(new_seq)
        return output
    def _print_progress(self, current_guess: str, sqr_dist: float, time: float,  max_length: int | None = TERMINAL_LENGTH, one_line: bool = True):
        if self.log is None:
            return
        output: str
        if max_length is None or len(current_guess) < max_length:
            output = current_guess
        else:
            output = current_guess[:max_length] + "..."
        if one_line:
            print(output, sqrt(sqr_dist), time, end="\r", file=self.log)
        else:
            print(output, sqrt(sqr_dist), time, file=self.log)
    @abstractmethod
    def search_similar(self, seq: str, target: str) -> str:
        pass

class IterativeGuessModel(SequenceDesigner, ABC):
    precision: float
    query_seq: str
    query_feats: Series
    target_feats: Series
    def search_similar(self, query: str, target: str, precision: float = DEFAULT_PRECISION) -> str:
        self.precision = precision
        self.query_seq = query
        self.query_feats = Series(self.feature_calculator.run_feats(self.query_seq), index=self.feature_calculator.supported_features)
        self.target_feats = Series(self.feature_calculator.run_feats(target), index=self.feature_calculator.supported_features)
        # It doesn't matter what step_size initially is, I just need the loop to run at least once.
        # Typically you would do this with numpy.inf but I don't have numpy imported
        step_size: float = self.precision + 1
        while step_size > self.precision:
            t = time()
            next_seqs: Iterator[str] = self.next_round_seqs()
            next_seq: str = self.query_seq
            next_feats: Series = self.query_feats
            next_sqd: float = self.distance_calculator.sqr_distance(self.query_feats, self.target_feats)
            for guess in next_seqs:
                try:
                    guess_feats: Series = Series(self.feature_calculator.run_feats(guess), index=self.feature_calculator.supported_features)
                except KeyboardInterrupt as interrupt:
                    raise interrupt
                except:
                    continue
                guess_sqd: float = self.distance_calculator.sqr_distance(guess_feats, self.target_feats)
                if guess_sqd < next_sqd:
                    next_seq = guess
                    next_feats = guess_feats
                    next_sqd = guess_sqd
            step_size = self.distance_calculator.sqr_distance(next_feats, self.query_feats)
            self.query_seq = next_seq
            self.query_feats = next_feats
            self._print_progress(self.query_seq, next_sqd, time() - t) 
        return self.query_seq
    @abstractmethod
    def next_round_seqs(self) -> Iterator[str]:
        pass