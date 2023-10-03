from idr_design.design_models.brute_force import BruteForce
from idr_design.design_models.rand_mch import RandMultiChange
from idr_design.design_models.template_classes import SequenceDesigner
from itertools import product
import pytest, os
from time import time
from math import sqrt

path_to_this_file = os.path.dirname(os.path.realpath(__file__))
SEED = "2022"
PRINT_PROGRESS = True
@pytest.mark.slow
class TestDesigners:
    brute_force: BruteForce = BruteForce(SEED)
    sample_multipt: RandMultiChange = RandMultiChange(seed=SEED)
    sfc = brute_force.feature_calculator
    dc = brute_force.distance_calculator
    small_example = [
        # For testing failure handling, typically small sequences raise an error on the kappa calculation
        ("KRTAE", 50),
        # A0A090N8E9_M7YB41_IDR_1 (short sequence)
        ("PDEAPAWALKADDATGAKDPGQSSSGSAKKADAP", 10), 
        # 8GLV (huge sequence)
        ("MREIVHIQGGQCGNQIGAKFWEVVSDEHGIDPTGTYHGDSDLQLERINVYFNEATGGRYVPRAILMDLEPGTMDSVRSGPYGQIFRPDNFVFGQTGAGNNWAKGHYTEGAELIDSVLDVVRKEAESCDCLQGFQVCHSLGGGTGSGMGTLLISKIREEYPDRMMLTFSVVPSPKVSDTVVEPYNATLSVHQLVENADECMVLDNEALYDICFRTLKLTTPTFGDLNHLISAVMSGITCCLRFPGQLNADLRKLAVNLIPFPRLHFFMVGFTPLTSRGSQQYRALTVPELTQQMWDAKNMMCAADPRHGRYLTASALFRGRMSTKEVDEQMLNVQNKNSSYFVEWIPNNVKSSVCDIPPKGLKMSATFIGNSTAIQEMFKRVSEQFTAMFRRKAFLHWYTGEGMDEMEFTEAESNMNDLVSEYQQYQDASAEEEGEFEGEEEEA", 1)
    ]
    # BRUTE FORCE RUNS UNBEARABLY SLOW
    @pytest.mark.skip
    @pytest.mark.parametrize(("i", "model"), product(
            range(len(small_example)),
            [sample_multipt, brute_force]
        ))
    def test_small(self, i: int, model: SequenceDesigner):
        print()
        print(model)
        seq, n = self.small_example[i]
        print(seq)
        t = time()
        results = model.design_similar(n, seq)
        t = time() - t
        for result in results:
            result_feats = self.sfc.run_feats(result)
            target_feats = self.sfc.run_feats(seq)
            dist = sqrt(self.dc.sqr_distance(result_feats, target_feats))
            print(result, dist)
        print("Average time:", t / n)
    fasta_ids: list[str]
    fasta_lookup_sequences: dict[str, str]
    with open(f"{path_to_this_file}/../yeast_proteome_clean.fasta", "r") as fastaf:
        lines: list[str] = list(map(lambda line: line.strip("\n"),fastaf.readlines()))
    # WAY TOO SLOW TO INCLUDE ALL FASTA IDS
    fasta_ids, sequences = lines[::2], lines[1::2]
    fasta_lookup_sequences = dict(zip(fasta_ids, sequences))
    # I hate this too but there are duplicates and pytest doesn't like user defined __init__'s 
    # Prevent code from compiling forever
    skip_after: int = 100
    @pytest.mark.parametrize(("fasta_id", "model"), product(
            # [fasta_ids[3]],
            fasta_ids[:skip_after],
            # [sample_multipt, brute_force]
            # [brute_force, sample_multipt]
            [sample_multipt]
            # [brute_force]
        ))
    def test_large(self, fasta_id: str, model: SequenceDesigner):
        print()
        print(model)
        seq = self.fasta_lookup_sequences[fasta_id]
        print(fasta_id)
        t = time()
        result = model.design_similar(1, seq, verbose=PRINT_PROGRESS)[0]
        t = time() - t
        result_feats = self.sfc.run_feats(result)
        target_feats = self.sfc.run_feats(seq)
        dist = sqrt(self.dc.sqr_distance(result_feats, target_feats))
        if PRINT_PROGRESS:
            print()
        print("Final design:")
        print(result)
        print("Dist, time")
        print(dist, t)
    
    
        