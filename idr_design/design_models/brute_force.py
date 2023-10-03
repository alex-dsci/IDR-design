from idr_design.design_models.template_classes import IterativeGuessModel
from idr_design.constants import AA_STRING
from pandas import Series
from itertools import product

class BruteForce(IterativeGuessModel):
    def next_round_seqs(self) -> Series:
        output: list[str] = [self.query_seq]
        moves = product(range(len(self.query_seq)), AA_STRING)
        for i, res in moves:
            if self.query_seq[i] == res:
                continue
            new_seq: str = self.query_seq[:i] + res + self.query_seq[i+1:]
            output.append(new_seq)
        return Series(output)
    