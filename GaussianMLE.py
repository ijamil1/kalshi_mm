from statistics import variance
from statsmodels.base.model import GenericLikelihoodModel
from scipy import stats
import numpy as np
from datetime import date, datetime, timedelta

'''
Assuming dS = u * S * dt + sigma * S * dW where S is the value of the NASDAQ-100 or SP500 and dW is standard Brownian motion
=> dS/S = u * dt + sigma * dW ~ Normal (mean = u * dt, variance = sigma^2 * dt)

Data: daily returns <=> dS/S and setting  dt = 1 so that daily returns ~ Normal (mean = u,  variance = sigma^2)
'''

class GaussianMLE(GenericLikelihoodModel):
    def __init__(self, endog, exog=None, **kwds):
        if exog is None:
            exog = np.zeros_like(endog)
            
        super(GaussianMLE, self).__init__(endog, exog, **kwds) 
    
    def calc_likelihood(self, endog, mu_ = 0, sigma_ = 1):
        return stats.norm.pdf(endog, loc= mu_, scale = sigma_)
    
    def nloglikeobs(self, params):
        mu = params[0]
        sigma = params[1]

        return -np.log(self.calc_likelihood(self.endog, mu_ = mu, sigma_= sigma))
    
    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        if start_params is None:
            mu_start = self.endog.mean()
            sigma_start = np.std(self.endog)
            start_params = np.array([mu_start, sigma_start])
            
        return super(GaussianMLE, self).fit(start_params=start_params,
                                                    maxiter=maxiter, maxfun=maxfun, **kwds)

def fit_data(endog):
    model = GaussianMLE(endog)
    res = model.fit()
    return res.params

def simulate(cur_idx_val, mean, std):
    res_list = []
    cur_time = datetime.utcnow()
    timediff = round((cur_time - datetime.now()).seconds/3600)
    dt = (datetime(cur_time.year, cur_time.month, cur_time.day, hour=16+timediff) - cur_time).seconds/(6.5*60*60)
    mu = mean * dt
    variance = (std**2) * dt

    for i in range(10000):
        til_eod_idx_ret = mu + (variance**(1/2)) * stats.norm.rvs()
        res_list.append(cur_idx_val * (1+til_eod_idx_ret))
    res_list.sort()
    return res_list