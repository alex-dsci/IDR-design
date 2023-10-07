from numpy import sqrt, power
from scipy.stats import nbinom

PRO_OR_CHARGED = "DERKP"
def _actual_neighbours(seq: str, blob: int) -> float:
    first_position: int | None = None
    last_position: int | None = None
    count: int = 0
    for i in range(len(seq)):
        if (seq[i] in PRO_OR_CHARGED):
            if last_position is None:
                first_position = last_position = i
                continue
            if i - last_position <= blob:
                count += 1
            last_position = i
    # What I find strange about this circularization process 
    # (comparing the first and last procharge res as if they are arranged in a circle)
    # is that it doesn't get done in the old kappa algorithm, which screws up testing unless we make
    # this very arbitrary choice to do circularization here and not there...
    assert first_position is not None
    assert last_position is not None
    if first_position + len(seq) - last_position <= blob:
        count += 1
    return count

# Given one pro/charge residue, the next blob residues are pro/charge in an iid. manner.
# That means the probability that you pick another pro/charge residue within
# the next blob residues is a negative binomial with parameter => pro/charge proportion
def _exp_neighbours(count_procharge: int, length: int, blob: int) -> float:
    proportion_procharge: float = count_procharge / length
    assert 0 < proportion_procharge < 1
    prob_next_proch_in_blob: float = proportion_procharge * sum(power(1-proportion_procharge, range(blob)))
    return prob_next_proch_in_blob * count_procharge
def _sd_neighbours(count_procharge: int, length: int, blob: int) -> float:
    proportion_procharge: float = count_procharge / length
    assert 0 < proportion_procharge < 1
    prob_next_proch_in_blob: float = proportion_procharge * sum(power(1-proportion_procharge, range(blob)))
    return sqrt(prob_next_proch_in_blob * (1 - prob_next_proch_in_blob) * count_procharge)

# This code calculates the "z-score" of the number of neighbouring pro/charge interactions
def my_omega(seq: str, blob: int = 5) -> float:
    count_procharge: int = len([r for r in seq if r in PRO_OR_CHARGED])
    if count_procharge < 2:
        raise ValueError("Can't calculate omega on something with less than two procharges!", seq)
    return (_actual_neighbours(seq, blob) - _exp_neighbours(count_procharge, len(seq), blob)) \
        / _sd_neighbours(count_procharge,len(seq),blob)
