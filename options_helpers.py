'''
The functions related to computing implied volatility are sourced from: 
https://www.codearmo.com/blog/implied-volatility-european-call-python
'''

from scipy.stats import norm
import numpy as np
import yfinance as yf
import pandas as pd
N = norm.cdf

N_prime = norm.pdf
N = norm.cdf


def black_scholes_call(S, K, T, r, sigma):
    '''

    :param S: Asset price
    :param K: Strike price
    :param T: Time to maturity
    :param r: risk-free rate (treasury bills)
    :param sigma: volatility
    :return: call price
    '''

    ###standard black-scholes formula
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call = S * N(d1) -  N(d2)* K * np.exp(-r * T)
    return call

def black_scholes_put(S, K, T, r, sigma):
    '''

    :param S: Asset price
    :param K: Strike price
    :param T: Time to maturity
    :param r: risk-free rate (treasury bills)
    :param sigma: volatility
    :return: call price
    '''

    ###standard black-scholes formula
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    put = -S * N(-d1) +  N(-d2)* K * np.exp(-r * T)
    return put


def vega(S, K, T, r, sigma):
    '''

    :param S: Asset price
    :param K: Strike price
    :param T: Time to Maturity
    :param r: risk-free rate (treasury bills)
    :param sigma: volatility
    :return: partial derivative w.r.t volatility
    '''

    ### calculating d1 from black scholes
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))

    #see hull derivatives chapter on greeks for reference
    vega = S * N_prime(d1) * np.sqrt(T)
    return vega

def implied_volatility_put(P, S, K, T, r, tol=0.0001,
                            max_iterations=10000):
    '''

    :param P: Observed put price
    :param S: Asset price
    :param K: Strike Price
    :param T: Time to Maturity
    :param r: riskfree rate
    :param tol: error tolerance in result
    :param max_iterations: max iterations to update vol
    :return: implied volatility in percent
    '''


    ### assigning initial volatility estimate for input in Newton_rap procedure
    sigma = 0.3
    
    for i in range(max_iterations):

        ### calculate difference between blackscholes price and market price with
        ### iteratively updated volality estimate
        diff = black_scholes_put(S, K, T, r, sigma) - P

        ###break if difference is less than specified tolerance level
        if abs(diff) < tol:
            print(f'found on {i}th iteration')
            print(f'difference is equal to {diff}')
            break

        ### use newton rapshon to update the estimate
        sigma = sigma - diff / vega(S, K, T, r, sigma)

    return sigma


def implied_volatility_call(C, S, K, T, r, tol=0.0001,
                            max_iterations=10000):
    '''

    :param C: Observed call price
    :param S: Asset price
    :param K: Strike Price
    :param T: Time to Maturity
    :param r: riskfree rate
    :param tol: error tolerance in result
    :param max_iterations: max iterations to update vol
    :return: implied volatility in percent
    '''


    ### assigning initial volatility estimate for input in Newton_rap procedure
    sigma = 0.3
    
    for i in range(max_iterations):

        ### calculate difference between blackscholes price and market price with
        ### iteratively updated volality estimate
        diff = black_scholes_call(S, K, T, r, sigma) - C

        ###break if difference is less than specified tolerance level
        if abs(diff) < tol:
            print(f'found on {i}th iteration')
            print(f'difference is equal to {diff}')
            break

        ### use newton rapshon to update the estimate
        sigma = sigma - diff / vega(S, K, T, r, sigma)

    return sigma

    
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