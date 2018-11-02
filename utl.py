CONTRACTS=dict([(l,i+1) for i, l in enumerate(['F','G','H','J','K','M', 'N', 'Q', 'U', 'V', 'X', 'Z'])])

def calc_T(c1, c2, y):
    assert((y>0) or CONTRACTS[c2]>=CONTRACTS[c1])
    if y==0:
        return CONTRACTS[c2] - CONTRACTS[c1]
    else:
        return y*2*len(CONTRACTS) - CONTRACTS[c1] - (len(CONTRACTS) - CONTRACTS[c2])
