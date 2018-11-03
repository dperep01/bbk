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

def gen_expiry(mkt='SCF', save=False):
#   pass in file name with raw Reuters data for individual contracts in one csv file 
    fn = mkt+'.csv'
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
        fn = mkt+'-expiry.csv'
        df.to_csv(fn, index=False)
        print('Saved expiry data to %s'%fn)
    else:
        return expiry

def add_expiry(mkt='SCF', save=False):
    fn = mkt+'.csv'
    df = pd.read_csv(fn, parse_dates=['Trade Date'], infer_datetime_format=True)
    df = df[['RIC', 'Trade Date', 'Universal Close Price']].dropna()
    df.rename(columns={'Trade Date':'Date', 'Universal Close Price':'Close'}, inplace=True)

    expiry = gen_expiry(mkt)
    df['Exp'] = np.nan; df['CRIC'] = np.nan

    for index, row in expiry.sort_values('Exp', ascending=False).iterrows():
        idx = (df.RIC==row.RIC) & (df.Date <= row.Exp)
        df.loc[idx, ['Exp', 'CRIC']] = [row.Exp, row.CRIC]
    if save:
        fn = mkt+'-with-expiry.csv'
        df.to_csv(fn, index=False)
        print('Saved data with added expiry to %s'%fn)
    else:
        return df

def gen_fc_3d(mkt, save=False):
    df = pd.read_csv(mkt+'-with-expiry.csv', index_col=1)
    mx = pd.DataFrame(columns=['Date']+list(range(12)))

    for n,g in df.groupby('Date'):
        v = g.sort_values('CRIC')
        v['TT'] = calc_TT(v.CRIC.tolist())
        v = v.set_index('TT').T
        v['Date'] = n
        mx = mx.append(v['Close':'Close'])
    if save:
        fn = mkt+'-fc-3d.csv'
        mx.to_csv(fn, index=False)
        print('Saved 3d forward curve to %s'%fn)
    else:
        return mx
