from idr_design.design_models.iter_guess_model import BruteForce
from idr_design.design_models.iter_guess_model import RandMultiChange
from idr_design.design_models.iter_guess_model import IterativeGuessModel
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
    # Usually want to test print. Comment this out if you want to see a small example.
    @pytest.mark.skip
    @pytest.mark.parametrize(("i", "model"), product(
            range(len(small_example)),
            [sample_multipt, brute_force]
        ))
    def test_display(self, i: int, model: IterativeGuessModel):
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
    fasta_ids: list[str] = []
    fasta_lookup_sequences: dict[str, str]
    with open(f"{path_to_this_file}/../disprot_idrs_clean.fasta", "r") as fastaf:
        lines: list[str] = list(map(lambda line: line.strip("\n"),fastaf.readlines()))
    terrible_fasta_ids, sequences = lines[::2], lines[1::2]
    fasta_lookup_sequences = dict(zip(terrible_fasta_ids, sequences))
    # Prevent code from compiling forever, can't include all fasta ids
    skip_after: int = 50
    n = 30
    admissible_length = range(15,40)
    for id, seq in fasta_lookup_sequences.items():
        if len(fasta_ids) >= skip_after:
            break
        if len(seq) not in admissible_length:
            continue
        try:
            sfc.run_feats(seq)
            fasta_ids.append(id)
        except KeyboardInterrupt as interrupt:
            raise interrupt
        except:
            continue
    # I hate this too but there are duplicates and pytest doesn't like user defined __init__'s

    # clear file
    outfiles = ["brute_force_output.txt", "rand_mch_output.txt"]
    for outfile in outfiles:
        with open(f"{path_to_this_file}/{outfile}", "w"):
            pass
    @pytest.mark.parametrize(("fasta_id", "model"), product(
            # [fasta_ids[3]],
            fasta_ids,
            [sample_multipt, brute_force]
            # [brute_force, sample_multipt]
            # [sample_multipt]
            # [brute_force]
        ))
    def test_print(self, fasta_id: str, model: IterativeGuessModel):
        if isinstance(model, BruteForce):
            outfile = self.outfiles[0]
        elif isinstance(model, RandMultiChange):
            outfile = self.outfiles[1]
        else:
            raise ValueError(model)
        with open(f"{path_to_this_file}/{outfile}", "a") as f:
            log = ProgressLogger(
                file=f,
                display_mode=False,
                col_names= [f"{fasta_id}|PROGRESS", "dist_to_target", "time"] 
            )
            model.log = log
            model.logged_time = "total"
            # print(model, file=f)
            seq = self.fasta_lookup_sequences[fasta_id]
            t = time()
            results = model.design_similar(self.n, seq, verbose=True)
            t = time() - t
            print(f"{fasta_id}|RESULT", file=f)
            for result in results:
                result_feats = self.sfc.run_feats(result)
                target_feats = self.sfc.run_feats(seq)
                dist = sqrt(self.dc.sqr_distance(result_feats, target_feats))
                print(result, file=f)
                print(f"Dist: {dist}", file=f)
            print("Average time:", t / self.n, file=f)
            
            
            
    
    
        