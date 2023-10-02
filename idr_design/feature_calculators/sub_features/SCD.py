from math import sqrt

def sequence_charge_decoration(seq: str) -> float:
    CHARGES: dict[str, int] = {
        "D": -1,
        "E": -1,
        "R": 1,
        "K": 1
    }
    scd: float = 0
    charged_res_positions: list[int] = []
    charges: list[int] = []
    for i in range(len(seq)):
        if seq[i] in CHARGES.keys():
            charged_res_positions.append(i)
            charges.append(CHARGES[seq[i]])
    for i in range(len(charged_res_positions)):
        for j in range(i):
            scd += charges[i] * charges[j] * sqrt(charged_res_positions[i] - charged_res_positions[j])
    return scd / len(seq)
