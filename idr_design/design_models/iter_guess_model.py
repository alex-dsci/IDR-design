import random
from idr_design.constants import AA_STRING
from itertools import product
from idr_design.design_models.logger import ProgressLogger
from idr_design.design_models.generic_designer import DEFAULT_PRECISION, TERMINAL_DISPLAY, SequenceDesigner


from pandas import Series


from abc import ABC, abstractmethod
from math import sqrt
from time import time
from typing import Iterator, Literal, Union


class IterativeGuessModel(SequenceDesigner, ABC):
    precision: float
    query_seq: str
    query_feats: Series
    target_feats: Series
    LogTimeOptions = Union[Literal["avg"], Literal["round"], Literal["total"]]
    logged_time: LogTimeOptions = "total"
    def search_similar(self, query: str, target: str, precision: float = DEFAULT_PRECISION) -> str:
        self.precision = precision
        self.query_seq = query
        self.query_feats = Series(self.feature_calculator.run_feats(self.query_seq), index=self.feature_calculator.supported_features)
        self.target_feats = Series(self.feature_calculator.run_feats(target), index=self.feature_calculator.supported_features)
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
                elif self.logged_time == "round":
                    t = time() - round_t
                else:
                    t = time() - total_t
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


class BruteForce(IterativeGuessModel):
    def next_round_seqs(self) -> Iterator[str]:
        moves = product(range(len(self.query_seq)), AA_STRING)
        for i, res in moves:
            if self.query_seq[i] == res:
                continue
            new_seq: str = self.query_seq[:i] + res + self.query_seq[i+1:]
            yield new_seq


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