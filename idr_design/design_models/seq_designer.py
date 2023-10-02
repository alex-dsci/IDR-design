import random
from idr_design.constants import AA_STRING
from idr_design.feature_calculators.main import DistanceCalculator as DistCalc, SequenceFeatureCalculator as FeatCalc
from abc import abstractmethod, ABC
from sys import stdout

MAX_TIME = 10
TERMINAL_LENGTH = 100
DEFAULT_PRECISION = 10 ** (-4)
class SequenceDesigner(ABC):
    feature_calculator: FeatCalc 
    distance_calculator: DistCalc
    seed: str | None
    def __init__(self, seed: str | None = None) -> None:
        self.feature_calculator = FeatCalc()
        self.distance_calculator = DistCalc(self.feature_calculator)
        self.seed = seed
    def design_similar(self, query: str | int, target: str) -> list[str]:
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
    def _print_progress(self, current_guess: str, dist: float, time: float,  max_length: int | None = TERMINAL_LENGTH, one_line: bool = True, file=stdout):
        output: str
        if max_length is None or len(current_guess) < max_length:
            output = current_guess
        else:
            output = current_guess[:max_length] + "..."
        if one_line:
            print(output, dist, time, end="\r", file=file)
        else:
            print(output, dist, time, file=file)
    @abstractmethod
    def search_similar(self, seq: str, target: str) -> str:
        pass