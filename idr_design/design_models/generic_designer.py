import random
from idr_design.constants import AA_STRING
from idr_design.feature_calculators.main import DistanceCalculator as DistCalc, SequenceFeatureCalculator as FeatCalc
from idr_design.design_models.logger import ProgressLogger
from abc import abstractmethod, ABC
from sys import stdout
from sys import stdout

TERMINAL_LENGTH: int | None = 100 
DEFAULT_PRECISION: float = 10 ** (-4)

TERMINAL_DISPLAY = ProgressLogger(
    file=stdout,
    col_names=["seq", "dist_to_target", "time"], 
    display_mode=False,
    max_lens=[100, 20, 20]
)
class SequenceDesigner(ABC):
    feature_calculator: FeatCalc 
    distance_calculator: DistCalc
    seed: str | None
    log: ProgressLogger | None
    verbose: bool
    def __init__(self, distance_calculator: DistCalc | None = None, seed: str | None = None, log: ProgressLogger | None = TERMINAL_DISPLAY) -> None:
        self.feature_calculator = FeatCalc()
        if distance_calculator is not None:
            self.distance_calculator = distance_calculator
        else:
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


