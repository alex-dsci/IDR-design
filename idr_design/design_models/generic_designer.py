import random
from idr_design.constants import AA_STRING
from idr_design.feature_calculators.main import DistanceCalculator as DistCalc, SequenceFeatureCalculator as FeatCalc
from idr_design.design_models.progress_logger import PrintDesignProgress, DEV_NULL
from abc import abstractmethod, ABC
from pandas import Series, concat, DataFrame
from time import time
from typing import cast

TERMINAL_LENGTH: int | None = 100 
DEFAULT_PRECISION: float = 10 ** (-4)

class SequenceDesigner(ABC):
    feature_calculator: FeatCalc 
    distance_calculator: DistCalc
    seed: str | None
    log: PrintDesignProgress
    job_name: str
    job_num: int
    def __init__(self, distance_calculator: DistCalc | None = None, seed: str | None = None, log: PrintDesignProgress = DEV_NULL) -> None:
        self.feature_calculator = FeatCalc()
        if distance_calculator is not None:
            self.distance_calculator = distance_calculator
        else:
            self.distance_calculator = DistCalc(self.feature_calculator)
        self.seed = seed
        self.log = log
    def design_similar(self, query: str | int, target: str, verbose: bool = False, job_name: str = "Design") -> Series: # type Series[str]
        assert job_name.strip() != ""
        self.job_name = job_name
        orig_log: PrintDesignProgress = self.log
        if not verbose:
            self.log = DEV_NULL              
        qs: Series
        if isinstance(query, int):
            qs = self._get_random_seqs(target, query)
        else:
            qs = Series((query))
        queries = cast("Series[str]", qs)
        self.log.enter_design_similar(job_name=self.job_name, tgt_seq=target, num_designs=len(queries), algorithm=str(self.__class__.__name__))
        output: DataFrame = DataFrame(columns=("time", "seq"))
        for i, seq in enumerate(queries):
            self.job_num = i
            t: float = time()
            designed_seq: str = self.search_similar(seq, target)
            t = time() - t
            output = concat((output, DataFrame({"time": [t], "seq": [designed_seq]})))
        self.log.exit_design_similar(job_name=self.job_name, query_seqs=queries, final_seqs=output["seq"], times=output["time"])
        self.log = orig_log
        return output["seq"]
    def _get_random_seqs(self, target: str, n: int) -> Series: # type Series[str]
        if self.seed is not None:
            random.seed(self.seed + target)
        output = cast("Series[str]", Series())
        for _ in range(n):
            new_seq: str = "".join([random.choice(AA_STRING) for _ in range(len(target))])
            while True:
                if None in self.feature_calculator.run_feats_skip_failures(new_seq):
                    new_seq: str = "".join([random.choice(AA_STRING) for _ in range(len(target))])
                    continue
                break
            output = concat((output, Series((new_seq))))
        return Series(output)
    @abstractmethod
    def search_similar(self, seq: str, target: str) -> str:
        pass


