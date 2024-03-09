'''
This cell of code is taken from David Duarte's answer on StackOverflow 
https://stackoverflow.com/questions/61289020/fast-implied-volatility-calculation-in-python and tweaked very slightly
'''

from scipy.stats import norm
import numpy as np
import yfinance as yf
import pandas as pd
N = norm.cdf

def bs_call(S, K, T, r, vol):
    d1 = (np.log(S/K) + (r + 0.5*vol**2)*T) / (vol*np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    return S * norm.cdf(d1) - np.exp(-r * T) * K * norm.cdf(d2)

def bs_put(S, K, T, r, vol):
    d1 = (np.log(S/K) + (r + 0.5*vol**2)*T) / (vol*np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    return np.exp(-r * T) * K * norm.cdf(-d2) - S * norm.cdf(-d1)
    
def bs_vega(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)

def find_vol_call(target_value, S, K, T, r, *args):
    MAX_ITERATIONS = 10000
    PRECISION = 1.0e-5
    sigma = 0.5
    for i in range(0, MAX_ITERATIONS):
        price = bs_call(S, K, T, r, sigma)
        vega = bs_vega(S, K, T, r, sigma)
        diff = target_value - price  # our root
        if (abs(diff) < PRECISION):
            return sigma
        sigma = sigma + diff/vega # f(x) / f'(x)
    return sigma # value wasn't found, return best guess so far

def find_vol_put(target_value, S, K, T, r, *args):
    MAX_ITERATIONS = 10000
    PRECISION = 1.0e-5
    sigma = 0.5
    for i in range(0, MAX_ITERATIONS):
        price = bs_put(S, K, T, r, sigma)
        vega = bs_vega(S, K, T, r, sigma)
        diff = target_value - price  # our root
        if (abs(diff) < PRECISION):
            return sigma
        sigma = sigma + diff/vega # f(x) / f'(x)
    return sigma # value wasn't found, return best guess so far

def get_delayed_options_data(date):
    spx = yf.Ticker('^SPX')
    ndx = yf.Ticker('^NDX')

    spx_calls = spx.option_chain(date.isoformat()).calls
    spx_puts = spx.option_chain(date.isoformat()).puts

    ndx_calls = ndx.option_chain(date.isoformat()).calls
    ndx_puts = ndx.option_chain(date.isoformat()).puts

    spx_calls['midprice'] = spx_calls.apply(lambda x: (x['bid'] + x['ask'])/2, axis=1)
    ndx_calls['midprice'] = ndx_calls.apply(lambda x: (x['bid'] + x['ask'])/2, axis=1)
    spx_puts['midprice'] = spx_puts.apply(lambda x: (x['bid'] + x['ask'])/2, axis=1)
    ndx_puts['midprice'] = ndx_puts.apply(lambda x: (x['bid'] + x['ask'])/2, axis=1)

    ndx_calls = ndx_calls[['strike', 'bid', 'midprice', 'ask', 'impliedVolatility', 'inTheMoney']]
    spx_calls = spx_calls[['strike', 'bid', 'midprice', 'ask', 'impliedVolatility', 'inTheMoney']]
    ndx_puts = ndx_puts[['strike', 'bid', 'midprice', 'ask', 'impliedVolatility', 'inTheMoney']]
    spx_puts = spx_puts[['strike', 'bid', 'midprice', 'ask', 'impliedVolatility', 'inTheMoney']]

    ndx_calls['type'] = ndx_calls.apply(lambda x: 'call', axis=1)
    ndx_puts['type'] = ndx_puts.apply(lambda x: 'put', axis=1)
    spx_calls['type'] = spx_calls.apply(lambda x: 'call', axis=1)
    spx_puts['type'] = spx_puts.apply(lambda x: 'put', axis=1)


    ndx_options = pd.concat([ndx_calls, ndx_puts], ignore_index=True)
    spx_options = pd.concat([spx_calls, spx_puts], ignore_index=True)

    ndx_options = ndx_options[(ndx_options['midprice'].isna() == False) & (ndx_options['midprice']>0)]

    spx_options = spx_options[(spx_options['midprice'].isna() == False) & (spx_options['midprice']>0)]

    ndx_itm = ndx_options[ndx_options['inTheMoney']==True].sort_values(by=['strike']).reset_index(drop=True)
    ndx_otm = ndx_options[ndx_options['inTheMoney']==False].sort_values(by=['strike']).reset_index(drop=True)

    spx_itm = spx_options[spx_options['inTheMoney']==True].sort_values(by=['strike']).reset_index(drop=True)
    spx_otm = spx_options[spx_options['inTheMoney']==False].sort_values(by=['strike']).reset_index(drop=True)

    return ndx_itm, ndx_otm, spx_itm, spx_otm