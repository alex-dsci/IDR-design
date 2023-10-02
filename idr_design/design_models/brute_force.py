from idr_design.design_models.seq_designer import SequenceDesigner
from idr_design.constants import AA_STRING
from pandas import DataFrame, Series
from itertools import product
from time import time

DEFAULT_PRECISION = 10 ** (-4)
class BruteForce(SequenceDesigner):
    def search_similar(self, query: str, target: str, precision: float = DEFAULT_PRECISION) -> str:
        def get_all_next_seqs(query: str) -> Series:
            output: list[str] = [query]
            moves = product(range(len(query)), AA_STRING)
            for i, res in moves:
                if query[i] == res:
                    continue
                new_seq: str = query[:i] + res + query[i+1:]
                output.append(new_seq)
            return Series(output)
        target_feats: Series = Series(self.feature_calculator.run_feats(target), index=self.feature_calculator.supported_features)
        next_seqs: Series = get_all_next_seqs(query)
        dict_next_feats: dict[str, dict[str, float | None]] = self.feature_calculator.run_feats_mult_seqs_skip_fail(next_seqs)
        df_next_feats: DataFrame = DataFrame(dict_next_feats.values())
        failed_rows = df_next_feats.index[df_next_feats.isna().any(axis=1)]
        clean_next_seqs: Series = next_seqs.drop(index=failed_rows)
        clean_next_feats: DataFrame = df_next_feats.dropna()
        next_dists: Series = self.distance_calculator.sqr_distance_many_to_one(clean_next_feats, target_feats)
        next_place: int = int(next_dists.argmin())
        next_seq: str = clean_next_seqs[next_place]
        step_size: float = self.distance_calculator.sqr_distance(clean_next_feats.iloc[next_place], clean_next_feats.iloc[0])
        while step_size > precision:
            t = time()
            next_seqs = get_all_next_seqs(next_seq)
            dict_next_feats = self.feature_calculator.run_feats_mult_seqs_skip_fail(next_seqs)
            df_next_feats = DataFrame(dict_next_feats.values())
            failed_rows = df_next_feats.index[df_next_feats.isna().any(axis=1)]
            clean_next_seqs = next_seqs.drop(index=failed_rows)
            clean_next_feats = df_next_feats.dropna()
            next_dists = self.distance_calculator.sqr_distance_many_to_one(clean_next_feats, target_feats)
            next_place = int(next_dists.argmin())
            assert next_place >= 0
            next_seq = clean_next_seqs[next_place]
            step_size = self.distance_calculator.sqr_distance(clean_next_feats.iloc[next_place], clean_next_feats.iloc[0])
            print(next_seq, next_dists.min(), time() - t, end="\r") 
        return next_seq
    