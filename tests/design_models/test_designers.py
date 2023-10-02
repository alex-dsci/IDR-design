from idr_design.design_models.brute_force import BruteForce
from idr_design.design_models.rand_mch import RandMultiChange
from idr_design.design_models.seq_designer import SequenceDesigner
from itertools import product
import pytest
from time import time

SEED = 2022
SMALL_TEST = "KRTAE"
A0A090N8E9_M7YB41_IDR_1 = "PDEAPAWALKADDATGAKDPGQSSSGSAKKADAP"
UNI_8GLV = "MREIVHIQGGQCGNQIGAKFWEVVSDEHGIDPTGTYHGDSDLQLERINVYFNEATGGRYVPRAILMDLEPGTMDSVRSGPYGQIFRPDNFVFGQTGAGNNWAKGHYTEGAELIDSVLDVVRKEAESCDCLQGFQVCHSLGGGTGSGMGTLLISKIREEYPDRMMLTFSVVPSPKVSDTVVEPYNATLSVHQLVENADECMVLDNEALYDICFRTLKLTTPTFGDLNHLISAVMSGITCCLRFPGQLNADLRKLAVNLIPFPRLHFFMVGFTPLTSRGSQQYRALTVPELTQQMWDAKNMMCAADPRHGRYLTASALFRGRMSTKEVDEQMLNVQNKNSSYFVEWIPNNVKSSVCDIPPKGLKMSATFIGNSTAIQEMFKRVSEQFTAMFRRKAFLHWYTGEGMDEMEFTEAESNMNDLVSEYQQYQDASAEEEGEFEGEEEEA"
P00004_P00004_IDR_1 = "MGDVEKGKKIFVQKCAQCHTVEKGGKHKTGPNLHGLFGRKTGQAPGFTYTDANKNKGITWKEETLMEYLENPKKYIPGTKMIFAGIKKKTEREDLIAYLKKATN"
class Test:
    brute_force: BruteForce = BruteForce(SEED)
    sfc = brute_force.feature_calculator
    dc = brute_force.distance_calculator
    sample_multipt: RandMultiChange = RandMultiChange(SEED)
    @pytest.mark.parametrize(("seq", "model"), product(
            # [SMALL_TEST, A0A090N8E9_M7YB41_IDR_1],
            # [UNI_8GLV],
            [SMALL_TEST, A0A090N8E9_M7YB41_IDR_1, P00004_P00004_IDR_1, UNI_8GLV],

            [sample_multipt, brute_force]
            # [sample_multipt]
        ))
    def test(self, seq: str, model: SequenceDesigner):
        t = time()
        print()
        print(model)
        result = model.design_similar(1, seq)[0]
        result_feats = self.sfc.run_feats(result)
        target_feats = self.sfc.run_feats(seq)
        dist = self.dc.sqr_distance(result_feats, target_feats)
        print()
        print(seq)
        print(result, dist, time() - t)
    
        