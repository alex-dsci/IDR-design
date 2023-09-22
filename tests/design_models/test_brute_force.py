from idr_design.design_models.brute_force import BruteForce

SEQ = "MTIAPITGTIKRRVIMDIVLGFSLGGVMASYWWWGFHMDKINKREKFYAELAERKKQEN"
class Test:
    seq_gen: BruteForce = BruteForce()
    sfc = seq_gen.feature_calculator
    dc = seq_gen.distance_calculator
    def test_runs(self):
        result = self.seq_gen.design_similar(1, SEQ, 2023)[0]
        feats = self.sfc.run_feats(result)
        featsb = self.sfc.run_feats(SEQ)
        dist = self.dc.sqr_distance(feats, featsb)
        print()
        print(SEQ)
        print(result)
        print(dist)
        print()