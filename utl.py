import numpy as np
import pandas as pd

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

def calc_expiry(mkt='SCF.csv', save=False):
#   pass in file name with raw Reuters data for individual contracts in one csv file 
    fn = mkt
    df = pd.read_csv(fn, parse_dates=['Trade Date'], infer_datetime_format=True)
    df = df[['RIC', 'Trade Date', 'Universal Close Price']].dropna()
    df.rename(columns={'Trade Date':'Date', 'Universal Close Price':'Close'}, inplace=True)

    expiry = pd.DataFrame()
    for n, g in df.groupby('RIC'):
        g = g.sort_values('Date')
        g['diff'] = g.Date.diff().shift(-1)
        exp = g[(g['diff'].dt.days > 365) | (g['diff'].isnull())].copy().rename(columns={'Date':'Exp'})
        exp['CRIC'] = exp.Exp.dt.year.astype(str) + exp.RIC.str[3:4]
        exp = exp[['RIC', 'CRIC', 'Exp']]
        expiry = expiry.append(exp)

    if save:
        df.to_csv(mkt+'-with-expiry.csv', index=False)
    return expiry

#import utl
#idf = pd.read_csv('SCF-with-expiry.csv', index_col=1)
#mx = pd.DataFrame(columns=['Date']+list(range(12)))
#i = 0
#for n,g in df.groupby('Date'):
#    v = g.sort_values('CRIC')
#    v['TT'] = utl.calc_TT(v.CRIC.tolist())
#    v = v.set_index('TT').T
#    v['Date'] = n
#    mx = mx.append(v['Close':'Close'])
#print(mx)
