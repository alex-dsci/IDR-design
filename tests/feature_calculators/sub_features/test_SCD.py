import os
from idr_design.feature_calculators.sub_features.SCD import sequence_charge_decoration
import pytest

path_to_this_file = os.path.dirname(os.path.realpath(__file__))
TEST_TOLERANCE = 10 ** (-14)

class Test:
    feature_lookup_column: int = 92
    fasta_ids: list[str]
    fasta_lookup_sequences: dict[str, str]
    fasta_lookup_results: dict[str, float]
    with open(f"{path_to_this_file}/../../yeast_proteome_clean.fasta", "r") as fastaf, open(f"{path_to_this_file}/../230918 old code data - yeast proteome.csv", "r") as resultf:
        lines: list[str] = list(map(lambda line: line.strip("\n"),fastaf.readlines()))
        fasta_ids, sequences = lines[::2], lines[1::2]
        fasta_lookup_sequences = dict(zip(fasta_ids, sequences))
        fasta_lookup_results = {}
        _ = resultf.readline()
        for row in map(lambda line: line.strip("\n"),resultf.readlines()):
            fasta_id: str = row.split(",")[0]
            result: float = float(row.split(",")[feature_lookup_column])
            fasta_lookup_results[fasta_id] = result
    
    @pytest.mark.parametrize(("fasta_id"), fasta_ids)
    def test_main(self, fasta_id: str):
        # So P32874 fails because the old code forgets about the first residue, which is usually M...
        # I assert my dominance on this one.
        if fasta_id == ">P32874":
            return
        seq: str = self.fasta_lookup_sequences[fasta_id]
        expected: float = self.fasta_lookup_results[fasta_id]
        calc: float = sequence_charge_decoration(seq) 
        assert abs(calc - expected) < TEST_TOLERANCE 

