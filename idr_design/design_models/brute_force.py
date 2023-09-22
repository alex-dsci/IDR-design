from idr_design.design_models.iter_guess_model import IterativeGuessModel
from idr_design.constants import AA_STRING
from itertools import product
from pandas import Series

class BruteForce(IterativeGuessModel):
    def __init__(self) -> None:
        super().__init__()
    def _get_next_seqs(self, query: str) -> Series:
        output: list[str] = [query]
        moves = product(range(len(query)), AA_STRING)
        for i, res in moves:
            if query[i] == res:
                continue
            new_seq: str = query[:i] + res + query[i+1:]
            output.append(new_seq)
        return Series(output)