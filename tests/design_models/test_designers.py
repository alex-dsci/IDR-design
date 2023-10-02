from idr_design.design_models.brute_force import BruteForce
from idr_design.design_models.rand_mpm import RandMPM
from idr_design.design_models.seq_designer import SequenceDesigner
from itertools import product
import pytest
from time import time

SEED = 2023
SMALL_TEST = "KRTAE"
A0A090N8E9_M7YB41_IDR_1 = "PDEAPAWALKADDATGAKDPGQSSSGSAKKADAP"
UNI_8GLV = "MREIVHIQGGQCGNQIGAKFWEVVSDEHGIDPTGTYHGDSDLQLERINVYFNEATGGRYVPRAILMDLEPGTMDSVRSGPYGQIFRPDNFVFGQTGAGNNWAKGHYTEGAELIDSVLDVVRKEAESCDCLQGFQVCHSLGGGTGSGMGTLLISKIREEYPDRMMLTFSVVPSPKVSDTVVEPYNATLSVHQLVENADECMVLDNEALYDICFRTLKLTTPTFGDLNHLISAVMSGITCCLRFPGQLNADLRKLAVNLIPFPRLHFFMVGFTPLTSRGSQQYRALTVPELTQQMWDAKNMMCAADPRHGRYLTASALFRGRMSTKEVDEQMLNVQNKNSSYFVEWIPNNVKSSVCDIPPKGLKMSATFIGNSTAIQEMFKRVSEQFTAMFRRKAFLHWYTGEGMDEMEFTEAESNMNDLVSEYQQYQDASAEEEGEFEGEEEEA"
P00004_P00004_IDR_1 = "MGDVEKGKKIFVQKCAQCHTVEKGGKHKTGPNLHGLFGRKTGQAPGFTYTDANKNKGITWKEETLMEYLENPKKYIPGTKMIFAGIKKKTEREDLIAYLKKATN"
class Test:
    brute_force: BruteForce = BruteForce(SEED)
    sfc = brute_force.feature_calculator
    dc = brute_force.distance_calculator
    sample_multipt: RandMPM = RandMPM(SEED)
    @pytest.mark.parametrize(("model", "seq"), product(
            # [brute_force, sample_multipt],
            [sample_multipt],
            [SMALL_TEST, A0A090N8E9_M7YB41_IDR_1, P00004_P00004_IDR_1]
            # [SMALL_TEST, A0A090N8E9_M7YB41_IDR_1, P00004_P00004_IDR_1, UNI_8GLV]
        ))
    def test(self, model: SequenceDesigner, seq: str):
        t = time()
        print()
        result = model.design_similar(1, seq)[0]
        feats = self.sfc.run_feats(result)
        featsb = self.sfc.run_feats(seq)
        dist = self.dc.sqr_distance(feats, featsb)
        print(seq)
        print(result)
        print(dist, time() - t)
        print()
    
        