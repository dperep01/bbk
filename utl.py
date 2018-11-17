import numpy as np
import pandas as pd
import datetime as dt
import scipy.stats as stats
import math

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

def gen_expiry(mkt='SCF', save=False, two=False):
#   pass in file name with raw Reuters data for individual contracts in one csv file 
    fn = 'data/'+ mkt+'.csv'
    df = pd.read_csv(fn, parse_dates=['Trade Date'], infer_datetime_format=True)
    df = df[['RIC', 'Trade Date', 'Universal Close Price']].dropna()
    df.rename(columns={'Trade Date':'Date', 'Universal Close Price':'Close'}, inplace=True)
    last_date = df.Date.max()

    expiry = pd.DataFrame()
    for n, g in df.groupby('RIC'):
        g = g.sort_values('Date')
        g['diff'] = g.Date.diff().shift(-1)
        exp = g[(g['diff'].dt.days > 365) | (g['diff'].isnull())].copy().rename(columns={'Date':'Exp'})
        exp['CRIC'] = exp.Exp.dt.year.astype(str) + exp.RIC.str.extract('([A-Z]+)')[0].str[-1]
        exp = exp[['RIC', 'CRIC', 'Exp']]
        expiry = expiry.append(exp[exp.Exp!=last_date])

    years = df.RIC.str.extract('(\d+)')[0].unique().astype(str)
    two_dig = df.RIC.str.extract('(\d+)')[0].str.len().max()>1

    contracts = df.RIC.str.extract('([A-Z]+)')[0].str[-1].unique()
    next_year = dt.datetime.now().year+1
    prev_year = dt.datetime.now().year-1
    for c in contracts:
        prev_year_c = str(prev_year)[-2:] if str(prev_year)[-2:] in years else str(prev_year)[-1:]
        next_year_c = str(next_year)[-2:] if str(next_year)[-2:] in years else str(next_year)[-1:]
        date = df[df.RIC==mkt+c+prev_year_c].sort_values('Date')['Date'].max()
        expiry = expiry.append(pd.DataFrame(data=[[mkt+c+next_year_c, str(next_year)+c, dt.datetime(next_year, date.month, date.day)]], columns=['RIC', 'CRIC', 'Exp']))

    if save:
        fn = 'data/'+mkt+'-expiry.csv'
        expiry.to_csv(fn, index=False)
        print('Saved expiry data to %s'%fn)
    else:
        return expiry

def add_expiry(mkt='SCF', save=False):
    fn = 'data/'+mkt+'.csv'
    df = pd.read_csv(fn, parse_dates=['Trade Date'], infer_datetime_format=True)
    df = df[['RIC', 'Trade Date', 'Universal Close Price']].dropna()
    df.rename(columns={'Trade Date':'Date', 'Universal Close Price':'Close'}, inplace=True)

#    expiry = gen_expiry(mkt)
    expiry = pd.read_csv('data/'+mkt+'-expiry.csv', parse_dates=['Exp'], infer_datetime_format=True)
    df['Exp'] = np.nan; df['CRIC'] = np.nan

    for index, row in expiry.sort_values('Exp', ascending=False).iterrows():
        idx = (df.RIC==row.RIC) & (df.Date <= row.Exp)
        df.loc[idx, ['Exp', 'CRIC']] = [row.Exp, row.CRIC]
    if save:
        fn = 'data/'+mkt+'-with-expiry.csv'
        df.to_csv(fn, index=False)
        print('Saved data with added expiry to %s'%fn)
    else:
        return df

def gen_fc_3d(mkt, save=False):
    df = pd.read_csv('data/'+mkt+'-with-expiry.csv', index_col=1)
    maturity = [str(e) for e in range(12)]
    mx = pd.DataFrame(columns=['Date']+maturity)

    for n,g in df.groupby('Date'):
        v = g.sort_values('CRIC')
        max_mat = [str(e) for e in (range(len(v)) if len(v)<=len(maturity) else maturity)]
        v['TT'] = [str(e) for e in calc_TT(v.CRIC.tolist())]
        v = v.set_index('TT').T
        v['Date'] = n
#        if n=='2017-01-17': raise
        mx = mx.append(v['Close':'Close'][['Date']+max_mat])
    if save:
        fn = 'data/'+mkt+'-fc-3d.csv'
        mx.to_csv(fn, index=False)
        print('Saved 3d forward curve to %s'%fn)
    else:
        return mx

def rCI(r, size, alpha=.05, method='pearson'):
    assert(method in['pearson', 'spearman'])
    if method == 'pearson':
        r_z = np.arctanh(r)
        se = 1/np.sqrt(size-3)
        z = stats.norm.ppf(1-alpha/2)
        lower_z, upper_z = r_z-z*se, r_z+z*se
        lower, upper = np.tanh((lower_z, upper_z))
    elif method == 'spearman':
        stderr = 1.0 / math.sqrt(size - 3)
        delta = 1.96 * stderr
        lower = math.tanh(math.atanh(r) - delta)
        upper = math.tanh(math.atanh(r) + delta)
    return r, lower, upper
