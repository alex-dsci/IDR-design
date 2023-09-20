import os
from idr_design.feature_calculators.sub_features.kappa import my_kappa
import pytest, re

path_to_this_file = os.path.dirname(os.path.realpath(__file__))
TEST_TOLERANCE = 10 ** (-13)

@pytest.mark.slow
class Test:
    feature_lookup_column: int = 93
    fasta_ids: list[str]
    fasta_lookup_sequences: dict[str, str]
    fasta_lookup_results: dict[str, float]
    with open(f"{path_to_this_file}/yeast_proteome_clean.fasta", "r") as fastaf, open(f"{path_to_this_file}/230918 old code data - yeast proteome.csv", "r") as resultf:
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
        seq: str = self.fasta_lookup_sequences[fasta_id]
        # Extremely necessary. P89113 has less than 2 charged residues.
        if len(re.sub("[^DERK]","",seq)) < 2:
            with pytest.raises(ValueError) as e:
                my_kappa(seq)
            assert e.value.args[0] == "Can't calculate kappa on something with less than two charges!"
            return
        expected: float = self.fasta_lookup_results[fasta_id]
        calc: float = my_kappa(seq) 
        assert abs(calc - expected) < TEST_TOLERANCE 

