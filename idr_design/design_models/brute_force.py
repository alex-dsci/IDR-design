import random
from itertools import product
from idr_design.constants import AA_STRING
from idr_design.feature_calculators.main import DistanceCalculator as DistCalc, SequenceFeatureCalculator as FeatCalc
from pandas import DataFrame, Series
from idr_design.timeout_decorator import timeout

MAX_TIME = 10
DEFAULT_PRECISION = 10 ** (-4)
class BruteForce:
    feature_calculator: FeatCalc 
    distance_calculator: DistCalc
    def __init__(self) -> None:
        self.feature_calculator = FeatCalc()
        self.distance_calculator = DistCalc(self.feature_calculator)
    def design_similar(self, query: str | int, target: str) -> list[str]:
        queries: list[str]
        if isinstance(query, int):
            queries = self._get_random_seqs(len(target), query)
        else:
            queries = [query]
        output: list[str] = []
        for seq in queries:
            output.append(self._search_similar(seq, target))
        return output
    @timeout(MAX_TIME, "Search similar timed out!")
    def _search_similar(self, query: str, target: str, precision: float = DEFAULT_PRECISION) -> str:
        target_feats: list[float] = self.feature_calculator.run_feats(target)
        next_seqs: Series = self._get_next_seqs(query)
        dict_next_feats: dict[str, dict[str, float | None]] = self.feature_calculator.run_feats_mult_seqs_skip_fail(next_seqs)
        df_next_feats: DataFrame = DataFrame(dict_next_feats.values(), index=self.feature_calculator.supported_features)
        failed_rows = df_next_feats.index[df_next_feats.isna().any(axis=1)]
        clean_next_seqs: Series = next_seqs.drop(index=failed_rows)
        clean_next_feats: DataFrame = DataFrame(df_next_feats.dropna(),index=clean_next_seqs)
        assert len(clean_next_feats) > 0
        next_dists: Series = self.distance_calculator.sqr_distance_many_to_one(clean_next_feats, target_feats)
        next_place: int = int(next_dists.argmin())
        next_seq: str = next_seqs[next_place]
        step_size: float = self.distance_calculator.sqr_distance(df_next_feats.iloc(next_place), df_next_feats.iloc(0))
        while step_size > precision:
            next_seqs: Series = self._get_next_seqs(next_seq)
            dict_next_feats: dict[str, dict[str, float | None]] = self.feature_calculator.run_feats_mult_seqs_skip_fail(next_seqs)
            df_next_feats: DataFrame = DataFrame(dict_next_feats.values(), index=self.feature_calculator.supported_features)
            failed_rows = df_next_feats.index[df_next_feats.isna().any(axis=1)]
            clean_next_seqs: Series = next_seqs.drop(index=failed_rows)
            clean_next_feats: DataFrame = DataFrame(df_next_feats.dropna(),index=clean_next_seqs)
            next_dists: Series = self.distance_calculator.sqr_distance_many_to_one(clean_next_feats, target_feats)
            next_place: int = int(next_dists.argmin())
            step_size: float = self.distance_calculator.sqr_distance(df_next_feats.iloc(next_place), df_next_feats.iloc(0)) 
            next_seq: str = next_seqs[next_place]
        return next_seq
    def _get_random_seqs(self, len: int, n: int) -> list[str]:
        output: list[str] = []
        for _ in range(n):
            output.append("".join([random.choice(AA_STRING) for _ in range(len)]))
        return output
    def _get_next_seqs(self, query: str) -> Series:
        output: list[str] = [query]
        moves = product(range(len(query)), AA_STRING)
        for i, res in moves:
            new_seq: str = query[:i] + res + query[i+1:]
            output.append(new_seq)
        return Series(output)