import random
from idr_design.constants import AA_STRING
from idr_design.feature_calculators.main import DistanceCalculator as DistCalc, SequenceFeatureCalculator as FeatCalc
from abc import abstractmethod, ABC

MAX_TIME = 10
DEFAULT_PRECISION = 10 ** (-4)
class SequenceDesigner(ABC):
    feature_calculator: FeatCalc 
    distance_calculator: DistCalc
    seed: int | None
    def __init__(self, seed: int | None = None) -> None:
        self.feature_calculator = FeatCalc()
        self.distance_calculator = DistCalc(self.feature_calculator)
        self.seed = seed
    def design_similar(self, query: str | int, target: str) -> list[str]:
        queries: list[str]
        if isinstance(query, int):
            queries = self._get_random_seqs(len(target), query)
        else:
            queries = [query]
        output: list[str] = []
        for seq in queries:
            output.append(self.search_similar(seq, target))
        return output
    def _get_random_seqs(self, len: int, n: int) -> list[str]:
        if self.seed is not None:
            random.seed(self.seed)
        output: list[str] = []
        for _ in range(n):
            new_seq: str = "".join([random.choice(AA_STRING) for _ in range(len)])
            while True:
                if None in self.feature_calculator.run_feats_skip_failures(new_seq):
                    new_seq: str = "".join([random.choice(AA_STRING) for _ in range(len)])
                    continue
                break
            output.append(new_seq)
        return output
    @abstractmethod
    def search_similar(self, seq: str, target: str) -> str:
        pass