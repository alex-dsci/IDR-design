from numpy import sqrt, power
from scipy.stats import nbinom

AA_CHARGE = {
    "D":-1,
    "E":-1,
    "K":1,
    "R":1
}
def _actual_neighbours(seq: str, blob: int) -> float:
    last_position: int | None = None
    last_charge: int | None = None
    count: int = 0
    for i in range(len(seq)):
        if (seq[i] in AA_CHARGE.keys()):
            if last_position is None:
                last_position = i
                last_charge = AA_CHARGE[seq[i]]
                continue
            if i - last_position <= blob and AA_CHARGE[seq[i]] == last_charge:
                count += 1
            last_position = i
            last_charge = AA_CHARGE[seq[i]]
    return count

# Given one charged residue, the next blob residues are charged in an iid. manner.
# That means the probability that you pick another charged residue within
# the next blob residues is a negative binomial with parameter => pro/charge proportion
# Then you take out the case where the two charges are different, and voila!
def _exp_neighbours(count_neg: int, count_pos: int, length: int, blob: int) -> float:
    count_charged: int = count_neg + count_pos
    proportion_charged: float = count_charged / length
    assert 0 < proportion_charged < 1
    prob_next_charge_in_blob: float = proportion_charged * sum(power(1-proportion_charged, range(blob)))
    prob_charges_are_diff: float = 2 * count_neg * count_pos / (count_charged ** 2)
    prob_final: float = prob_next_charge_in_blob * (1 - prob_charges_are_diff)
    return prob_final * count_charged
def _sd_neighbours(count_neg: int, count_pos: int, length: int, blob: int) -> float:
    count_charged: int = count_neg + count_pos
    proportion_charged: float = count_charged / length
    assert 0 < proportion_charged < 1
    prob_next_charge_in_blob: float = proportion_charged * sum(power(1-proportion_charged, range(blob)))
    prob_charges_are_diff: float = 2 * count_neg * count_pos / (count_charged ** 2)
    prob_final: float = prob_next_charge_in_blob * (1 - prob_charges_are_diff)
    return sqrt(prob_final * (1 - prob_final) * count_charged)

# This code calculates the "z-score" of the number of neighbouring same-charge interactions
def my_kappa(seq: str, blob: int = 5) -> float:
    count_pos: int = len([r for r in seq if r in AA_CHARGE and AA_CHARGE[r] > 0])
    count_neg: int = len([r for r in seq if r in AA_CHARGE and AA_CHARGE[r] < 0])
    if count_pos + count_neg < 2:
        raise ValueError("Can't calculate kappa on something with less than two charges!", seq)
    return (_actual_neighbours(seq, blob) - _exp_neighbours(count_pos, count_neg, len(seq), blob)) \
        / _sd_neighbours(count_pos, count_neg, len(seq),blob)