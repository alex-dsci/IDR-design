from math import sqrt
from scipy.stats import nbinom

PRO_OR_CHARGED = "DERKP"
def _exp_neighbours(count_procharge: int, length: int, blob: int) -> float:
    assert count_procharge > 0
    proportion_procharge: float = count_procharge / length
    prob: float = float(nbinom.cdf(k=blob-1,n=1,p=proportion_procharge)) 
    return prob * count_procharge
def _var_neighbours(count_procharge: int, length: int, blob: int) -> float:
    assert count_procharge > 0
    proportion_procharge: float = count_procharge / length
    prob: float = float(nbinom.cdf(k=blob-1,n=1,p=proportion_procharge)) 
    return prob * (1 - prob) * count_procharge
def _actual_neighbours(seq: str, blob: int) -> float:
    first_position: int | None = None
    last_position: int | None = None
    count: int = 0
    for i in range(len(seq)):
        if (seq[i] in PRO_OR_CHARGED):
            if last_position is None:
                first_position = last_position = i
                count += 1
                continue
            if i - last_position <= blob:
                count += 1
            last_position = i
    assert first_position is not None
    assert last_position is not None
    if first_position + len(seq) - last_position <= blob:
        count += 1
    return count

# This code calculates the "z-score" of the number of neighbouring pro/charge interactions
def my_omega(seq: str, blob: int = 5) -> float:
    count_procharge: int = len([r for r in seq if r in PRO_OR_CHARGED])
    if count_procharge < 2:
        raise ValueError("Can't calculate omega on something with less than two procharges!", seq)
    return (_actual_neighbours(seq, blob) - _exp_neighbours(count_procharge, len(seq), blob)) \
        / sqrt(_var_neighbours(count_procharge,len(seq),blob))
