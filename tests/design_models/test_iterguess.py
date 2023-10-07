from idr_design.design_models.iter_guess_model import BruteForce
from idr_design.design_models.iter_guess_model import RandMultiChange
from idr_design.design_models.iter_guess_model import IterativeGuessModel
from idr_design.design_models.progress_logger import DisplayToStdout, LogToFile, LogToCSV
from itertools import product
import pytest, os

path_to_this_file = os.path.dirname(os.path.realpath(__file__))
SEED = "2022"

class TestIterativeGuessModels:
    brute_force: BruteForce = BruteForce(seed=SEED)
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
    @pytest.mark.slow
    @pytest.mark.parametrize(("i", "model"), product(
            range(len(small_example)),
            [sample_multipt, brute_force]
        ))
    def test_display(self, i: int, model: IterativeGuessModel):
        print()
        model.log = DisplayToStdout()
        seq = self.small_example[i][0]
        model.design_similar(self.small_example[i][1], seq, job_name=str(i), verbose=True) 
    fasta_ids: list[str] = []
    fasta_lookup_sequences: dict[str, str]
    with open(f"{path_to_this_file}/../disprot_idrs_clean.fasta", "r") as fastaf:
        lines: list[str] = list(map(lambda line: line.strip("\n"),fastaf.readlines()))
    terrible_fasta_ids, sequences = lines[::2], lines[1::2]
    fasta_lookup_sequences = dict(zip(terrible_fasta_ids, sequences))
    # Prevent code from compiling forever, can't include all fasta ids
    skip_after: int = 10
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
    # I hate this too but pytest doesn't like user defined __init__'s
    @pytest.mark.overwrites
    @pytest.mark.parametrize(("fasta_id", "model"), product(
            # [fasta_ids[3]],
            fasta_ids,
            # [sample_multipt, brute_force]
            [brute_force, sample_multipt]
            # [sample_multipt]
            # [brute_force]
        ))
    def test_to_file(self, fasta_id: str, model: IterativeGuessModel):
        if isinstance(model, RandMultiChange):
            name: str = "rand_mch_output.txt"
        elif isinstance(model, BruteForce):
            name: str = "brute_force_output.txt"
        else:
            raise ValueError(model)
        path: str = f"{path_to_this_file}/{name}"
        if fasta_id == self.fasta_ids[0]:
            with open(path, "w"):
                pass
        with open(path, "a") as f:
            model.log = LogToFile(file=f)
            seq = self.fasta_lookup_sequences[fasta_id]
            model.design_similar(self.n, seq, job_name=fasta_id[1:], verbose=True)
    @pytest.mark.overwrites
    @pytest.mark.parametrize(("fasta_id", "model"), product(
            # [fasta_ids[3]],
            fasta_ids,
            # [sample_multipt, brute_force]
            [brute_force, sample_multipt]
            # [sample_multipt]
            # [brute_force]
        ))
    def test_to_csv(self, fasta_id: str, model: IterativeGuessModel):
        if isinstance(model, RandMultiChange):
            name: str = "rand_mch_output"
        elif isinstance(model, BruteForce):
            name: str = "brute_force_output"
        else:
            raise ValueError(model)
        path: str = f"{path_to_this_file}/{name}"
        model.log = LogToCSV(path=path)
        seq = self.fasta_lookup_sequences[fasta_id]
        model.design_similar(self.n, seq, job_name=fasta_id[1:], verbose=True)
    
            
    
    
        