from idr_design.design_models.brute_force import BruteForce
from idr_design.design_models.rand_mch import RandMultiChange
from idr_design.design_models.template_classes import SequenceDesigner
from idr_design.design_models.logger import ProgressLogger
from itertools import product
import pytest, os
from time import time
from math import sqrt

path_to_this_file = os.path.dirname(os.path.realpath(__file__))
SEED = "2022"

@pytest.mark.slow
class TestIterativeGuessModels:
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
    # Brute force runs really slow on this test (and overall)
    @pytest.mark.skip
    @pytest.mark.parametrize(("i", "model"), product(
            range(len(small_example)),
            [sample_multipt, brute_force]
        ))
    def test_display(self, i: int, model: SequenceDesigner):
        print()
        print(model)
        seq, n = self.small_example[i]
        print(seq)
        t = time()
        results = model.design_similar(n, seq, verbose=True)
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
    fasta_ids, sequences = lines[::2], lines[1::2]
    fasta_lookup_sequences = dict(zip(fasta_ids, sequences))
    # I hate this too but there are duplicates and pytest doesn't like user defined __init__'s
    with open(f"{path_to_this_file}/iter_guess_output.txt", "w"):
        pass
    # Prevent code from compiling forever, can't include all fasta ids
    skip_after: int = 100
    @pytest.mark.parametrize(("fasta_id", "model"), product(
            # [fasta_ids[3]],
            fasta_ids[:skip_after],
            # [sample_multipt, brute_force]
            # [brute_force, sample_multipt]
            [sample_multipt]
            # [brute_force]
        ))
    def test_print(self, fasta_id: str, model: SequenceDesigner):
        with open(f"{path_to_this_file}/iter_guess_output.txt", "a") as f:
            log = ProgressLogger(
                file=f,
                display_mode=False,
                col_names= [f"{fasta_id}|PROGRESS", "dist_to_target", "round_time"] 
            )
            if isinstance(model, RandMultiChange):
                model.log = log
                model.logged_time = "round"
            # print(model, file=f)
            seq = self.fasta_lookup_sequences[fasta_id]
            t = time()
            result = model.design_similar(1, seq, verbose=True)[0]
            t = time() - t
            result_feats = self.sfc.run_feats(result)
            target_feats = self.sfc.run_feats(seq)
            dist = sqrt(self.dc.sqr_distance(result_feats, target_feats))
            print(f"{fasta_id}|RESULT", file=f)
            print(result, file=f)
            print("Dist, time:", file=f)
            print(f"{dist}, {t}", file=f)
    
    
        