from idr_design.design_models.seq_designer import SequenceDesigner
from idr_design.constants import AA_STRING
from pandas import DataFrame, Series
from itertools import product
from typing import Iterator
import random
from time import time

DEFAULT_PRECISION: float = 10 ** (-4)
GOOD_SAMPLE_SIZE = 2
REST_SAMPLE_SIZE = 6 
class RandMultiChange(SequenceDesigner):
    good_sample_size: int
    rest_sample_size: int
    def __init__(self, seed: int | None = None, good_size: int = GOOD_SAMPLE_SIZE, rest_size: int = REST_SAMPLE_SIZE) -> None:
        super().__init__(seed)
        self.good_sample_size = good_size
        self.rest_sample_size = rest_size
    def apply_move(self, query: str, move: list[tuple[int, str]]) -> str:
        result: str = query
        for i, res in move:
            result = result[:i] + res + result[i+1:]
        return result
    def apply_moves(self, query: str, moves: list[list[tuple[int, str]]]) -> Iterator[str]:
        for move in moves:
            yield self.apply_move(query, move)
    def search_similar(self, query: str, target: str, precision: float = DEFAULT_PRECISION) -> str:
        if self.seed is not None:
            random.seed(str(self.seed) + target)
        def seq_generator(query: str, moves: list[tuple[int, str]]) -> Iterator[tuple[str, int, str] | None]:
            for i, res in moves:
                if query[i] == res:
                    continue
                new_seq: str = query[:i] + res + query[i+1:]
                yield (new_seq, i, res)
            yield None
        target_feats: Series = Series(self.feature_calculator.run_feats(target), index=self.feature_calculator.supported_features)
        next_seq: str = query
        next_feats: Series = Series(self.feature_calculator.run_feats(query), index=self.feature_calculator.supported_features)
        next_dist: float = self.distance_calculator.sqr_distance(next_feats, target_feats)
        one_point_moves: list[tuple[int, str]] = list(product(range(len(next_seq)), AA_STRING))
        random.shuffle(one_point_moves)
        next_one_pt_muts: Iterator[tuple[str, int, str] | None] = seq_generator(next_seq, one_point_moves)
        move_data: dict[int, tuple[str, float]] = {}
        def stop_sampling(move_data: dict[int, tuple[str, float]]) -> bool:
            good_sample: list[int] = [key for key, data in move_data.items() if data[1] > precision]
            return len(good_sample) > self.good_sample_size
        while not stop_sampling(move_data):
            guess_or_none = next(next_one_pt_muts)
            if guess_or_none is None:
                break
            guess, i, res = guess_or_none
            guess_feats: Series = Series(self.feature_calculator.run_feats_skip_failures(guess), index=self.feature_calculator.supported_features)
            if None in guess_feats.values:
                continue
            guess_dist: float = self.distance_calculator.sqr_distance(guess_feats, target_feats)
            if guess_dist < next_dist:
                if i in move_data.keys():
                    if guess_dist > move_data[i][1]:
                        continue
                move_data.update({i: (res, self.distance_calculator.sqr_distance(guess_feats, next_feats))})
        N_pt_moves: list[list[tuple[int, str]]] = [[]]
        good_sample: list[int] = [key for key, data in move_data.items() if data[1] > precision]
        rest_sample: list[int] = [key for key, data in move_data.items() if data[1] <= precision]
        sample = good_sample + random.sample(rest_sample, min(self.rest_sample_size, len(rest_sample)))
        for pos in sample:
            res: str = move_data[pos][0]
            adjoin_next_move: list[list[tuple[int, str]]] = list(map(lambda x: x + [(pos, res)], N_pt_moves))
            N_pt_moves += adjoin_next_move
        next_seqs: Series = Series(list(self.apply_moves(next_seq, N_pt_moves)))
        dict_next_feats: dict[str, dict[str, float | None]] = self.feature_calculator.run_feats_mult_seqs_skip_fail(next_seqs)
        df_next_feats: DataFrame = DataFrame(dict_next_feats.values())
        failed_rows = df_next_feats.index[df_next_feats.isna().any(axis=1)]
        clean_next_seqs: Series = next_seqs.drop(index=failed_rows)
        clean_next_feats: DataFrame = df_next_feats.dropna()
        next_dists: Series = self.distance_calculator.sqr_distance_many_to_one(clean_next_feats, target_feats)
        next_place: int = int(next_dists.argmin())
        next_seq: str = clean_next_seqs.iloc[next_place]
        step_size: float = self.distance_calculator.sqr_distance(clean_next_feats.iloc[next_place], clean_next_feats.iloc[0])
        while step_size > precision:
            t = time()
            next_feats: Series = Series(self.feature_calculator.run_feats_skip_failures(next_seq), index=self.feature_calculator.supported_features)
            next_dist: float = self.distance_calculator.sqr_distance(next_feats, target_feats)
            one_point_moves: list[tuple[int, str]] = list(product(range(len(next_seq)), AA_STRING))
            random.shuffle(one_point_moves)
            next_one_pt_muts: Iterator[tuple[str, int, str] | None] = seq_generator(next_seq, one_point_moves)
            move_data: dict[int, tuple[str, float]] = {}
            while not stop_sampling(move_data):
                guess_or_none = next(next_one_pt_muts)
                if guess_or_none is None:
                    break
                guess, i, res = guess_or_none
                guess_feats: Series = Series(self.feature_calculator.run_feats_skip_failures(guess), index=self.feature_calculator.supported_features)
                if None in guess_feats.values:
                    continue
                guess_dist: float = self.distance_calculator.sqr_distance(guess_feats, target_feats)
                if guess_dist < next_dist:
                    if i in move_data.keys():
                        if guess_dist > move_data[i][1]:
                            continue
                    move_data.update({i: (res, self.distance_calculator.sqr_distance(guess_feats, next_feats))})
            N_pt_moves: list[list[tuple[int, str]]] = [[]]
            good_sample: list[int] = [key for key, data in move_data.items() if data[1] > precision]
            rest_sample: list[int] = [key for key, data in move_data.items() if data[1] <= precision]
            sample = good_sample + random.sample(rest_sample, min(self.rest_sample_size, len(rest_sample)))
            for pos in sample:
                res: str = move_data[pos][0]
                adjoin_next_move: list[list[tuple[int, str]]] = list(map(lambda x: x + [(pos, res)], N_pt_moves))
                N_pt_moves += adjoin_next_move
            next_seqs: Series = Series(list(self.apply_moves(next_seq, N_pt_moves)))
            dict_next_feats: dict[str, dict[str, float | None]] = self.feature_calculator.run_feats_mult_seqs_skip_fail(next_seqs)
            df_next_feats: DataFrame = DataFrame(dict_next_feats.values())
            failed_rows = df_next_feats.index[df_next_feats.isna().any(axis=1)]
            clean_next_seqs: Series = next_seqs.drop(index=failed_rows)
            clean_next_feats: DataFrame = df_next_feats.dropna()
            next_dists: Series = self.distance_calculator.sqr_distance_many_to_one(clean_next_feats, target_feats)
            next_place: int = int(next_dists.argmin())
            next_seq: str = clean_next_seqs.iloc[next_place]
            step_size: float = self.distance_calculator.sqr_distance(clean_next_feats.iloc[next_place], clean_next_feats.iloc[0])  
            self._print_progress(next_seq, next_dists.min(), time() - t) 
        return next_seq
    