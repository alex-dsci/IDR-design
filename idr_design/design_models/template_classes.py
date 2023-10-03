import random
from idr_design.constants import AA_STRING
from idr_design.feature_calculators.main import DistanceCalculator as DistCalc, SequenceFeatureCalculator as FeatCalc
from idr_design.design_models.logger import ProgressLogger
from abc import abstractmethod, ABC
from sys import stdout
from typing import Iterator, Literal, Union
from pandas import Series
from time import time
from math import sqrt
from sys import stdout

TERMINAL_LENGTH: int | None = 100 
DEFAULT_PRECISION: float = 10 ** (-4)

PROGRESS_COLUMNS = ["seq", "dist_to_target", "avg_round_time"]
TERMINAL_DISPLAY = ProgressLogger(
    file=stdout,
    col_names=PROGRESS_COLUMNS, 
    display_mode=True,
    max_lens=[100, 20, 20]
)
class SequenceDesigner(ABC):
    feature_calculator: FeatCalc 
    distance_calculator: DistCalc
    seed: str | None
    log: ProgressLogger | None
    verbose: bool
    def __init__(self, seed: str | None = None, log: ProgressLogger | None = TERMINAL_DISPLAY) -> None:
        self.feature_calculator = FeatCalc()
        self.distance_calculator = DistCalc(self.feature_calculator)
        self.seed = seed
        self.log = log
    def design_similar(self, query: str | int, target: str, verbose: bool = False) -> list[str]:
        self.verbose = verbose               
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
    @abstractmethod
    def search_similar(self, seq: str, target: str) -> str:
        pass

class IterativeGuessModel(SequenceDesigner, ABC):
    precision: float
    query_seq: str
    query_feats: Series
    target_feats: Series
    logged_time: Union[Literal["avg"], Literal["round"]]
    def search_similar(self, query: str, target: str, precision: float = DEFAULT_PRECISION, logged_time: Union[Literal["avg"], Literal["round"]] = "avg") -> str:
        self.precision = precision
        self.query_seq = query
        self.query_feats = Series(self.feature_calculator.run_feats(self.query_seq), index=self.feature_calculator.supported_features)
        self.target_feats = Series(self.feature_calculator.run_feats(target), index=self.feature_calculator.supported_features)
        self.logged_time = logged_time
        # It doesn't matter what step_size initially is, I just need the loop to run at least once.
        # Typically you would do this with numpy.inf but I don't have numpy imported
        step_size: float = self.precision + 1
        count_iters: int = 0
        total_t: float = time()
        if self.verbose and self.log is not None:
            self.log.write_header()
        while step_size > self.precision:
            count_iters += 1
            round_t: float = time()
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
            if self.verbose and self.log is not None:
                if self.logged_time == "avg":
                    t = (time() - total_t) / count_iters
                else:
                    t = time() - round_t
                self.log.write_data([
                    next_seq,
                    str(sqrt(next_sqd)),
                    str(t)
                ])
        if self.verbose and self.log is not None and self.log.display_mode:
            self.log.print("\n")
        return self.query_seq
    @abstractmethod
    def next_round_seqs(self) -> Iterator[str]:
        pass