def isoelectric_point(freq, converge_thres)	:  ###based on An algorithm for isoelectric point estimation David L. Tabb
                                ###and the bisection method for finding 0s, similar to python SeqUtils	
    ph = 0.0
    #find the limits
    lowerph = 0.0
    while (charge_at_PH(freq,ph) > 0.0):
        lowerph=ph 
        ph+=1.0
        # print(f"updating pH: {ph}")
    #print (str(lowerph) + "is the lower limit on iso point")
    upperph = 1.0 + lowerph
    # print(f"updating pH: {ph}")
    #refine the value using bisection
    while (upperph - lowerph > converge_thres):
        ph = (upperph+ lowerph)/2.0
        # print(f"updating pH: {ph}")
        ch=charge_at_PH(freq,float(ph))
        if (ch > 0.0):
            lowerph=ph
        else:
            upperph=ph
            
    return ph

                
def charge_at_PH(freq,ph):
    # t = time()
    ch = 0.0
    pKaapos={ 'K': 10.0, 'R': 12.0, 'H': 5.98, }
    pKaaneg={ 'D': 4.05, 'E': 4.45, 'C': 9.0, 'Y': 10.0  }
    pKN = 7.5 
    pKC = 3.55
    cr = 10**(pKN-ph)
    ch +=  cr/(cr + 1.0)
    for aa in pKaapos.keys():
        cr = 10**(pKaapos[aa]-ph)
        ch += freq[aa]*cr/(cr + 1.0)
    cr = 10**(ph-pKC)
    ch -=  cr/(cr + 1.0)
    for aa in pKaaneg.keys():
        cr = 10**(ph-pKaaneg[aa])
        ch -= freq[aa]*cr/(cr + 1.0)
    # print("time", time() - t)
    return ch	