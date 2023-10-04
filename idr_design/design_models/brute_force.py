from idr_design.design_models.template_classes import IterativeGuessModel 
from idr_design.constants import AA_STRING
from pandas import Series
from itertools import product
from typing import Iterator

class BruteForce(IterativeGuessModel):
    def next_round_seqs(self) -> Iterator[str]:
        moves = product(range(len(self.query_seq)), AA_STRING)
        for i, res in moves:
            if self.query_seq[i] == res:
                continue
            new_seq: str = self.query_seq[:i] + res + self.query_seq[i+1:]
            yield new_seq
        
    