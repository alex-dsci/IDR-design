from idr_design.design_models.logger import ProgressLogger
from idr_design.design_models.template_classes import TERMINAL_DISPLAY, IterativeGuessModel 
from idr_design.constants import AA_STRING
from pandas import Series
from itertools import product
import random

GOOD_SAMPLE_SIZE = 2
REST_SAMPLE_SIZE = 14
class RandMultiChange(IterativeGuessModel):
    good_sample_size: int
    rest_sample_size: int
    def __init__(self, seed: str | None = None, log: ProgressLogger | None = TERMINAL_DISPLAY, good_size: int = GOOD_SAMPLE_SIZE, rest_size: int = REST_SAMPLE_SIZE) -> None:
        super().__init__(seed, log)
        self.good_sample_size = good_size
        self.rest_sample_size = rest_size
    def next_round_seqs(self) -> Series:
        query_dist: float = self.distance_calculator.sqr_distance(self.query_feats, self.target_feats)
        one_point_moves: list[tuple[int, str]] = list(product(range(len(self.query_seq)), AA_STRING))
        random.shuffle(one_point_moves)
        good_moves: dict[int, tuple[str, float]] = {}
        negative_delta_moves: dict[int, tuple[str, float]] = {}
        for i, res in one_point_moves:
            if len(good_moves) >= self.good_sample_size:
                break
            if self.query_seq[i] == res:
                continue
            guess: str = self.query_seq[:i] + res + self.query_seq[i+1:]
            guess_feats: Series = Series(self.feature_calculator.run_feats_skip_failures(guess), index=self.feature_calculator.supported_features)
            if None in guess_feats.values:
                continue
            guess_dist: float = self.distance_calculator.sqr_distance(guess_feats, self.target_feats)
            if guess_dist < query_dist:
                step_size: float = self.distance_calculator.sqr_distance(guess_feats, self.query_feats)
                if step_size > self.precision:
                    if i in good_moves.keys() and guess_dist > good_moves[i][1]:
                        continue
                    good_moves.update({i: (res, step_size)})
                else:
                    if i not in negative_delta_moves.keys() and len(negative_delta_moves) >= self.rest_sample_size:
                        continue
                    if i in negative_delta_moves.keys() and guess_dist > negative_delta_moves[i][1]:
                        continue
                    negative_delta_moves.update({i: (res, step_size)}) 
        result: list[str] = [self.query_seq]
        for i, data in good_moves.items():
            res: str = data[0]
            result_plus_move: list[str] = list(map(lambda x: x[:i] + res + x[i+1:], result))
            result += result_plus_move
        for i, data in negative_delta_moves.items():
            res: str = data[0]
            result_plus_move: list[str] = list(map(lambda x: x[:i] + res + x[i+1:], result))
            result += result_plus_move
        return Series(result)
    