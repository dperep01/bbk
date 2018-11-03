import numpy as np

CONTRACTS=dict([(l,i+1) for i, l in enumerate(['F','G','H','J','K','M', 'N', 'Q', 'U', 'V', 'X', 'Z'])])

def calc_TT(cc):
    c0 = cc[0]
    TT = []
    for c in cc:
        TT.append(calc_T(c0, c))
    return np.asarray(TT)
          

def calc_T(c1, c2):
    y1 = c1[:-1]
    y2 = c2[:-1]
    l1 = c1[-1]
    l2 = c2[-1]
    y = int(y2) - int(y1)
    assert((y>0) or CONTRACTS[l2]>=CONTRACTS[l1])
    if y==0:
        return CONTRACTS[l2] - CONTRACTS[l1]
    else:
        return y*2*len(CONTRACTS) - CONTRACTS[l1] - (len(CONTRACTS) - CONTRACTS[l2])
