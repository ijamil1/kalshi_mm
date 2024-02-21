from statsmodels.base.model import GenericLikelihoodModel
from scipy import stats
import numpy as np
from datetime import date, datetime, timedelta
import bisect

'''
Assuming dS = u * S * dt + sigma * S * dW where S is the value of the NASDAQ-100 or SP500 and dW is standard Brownian motion
=> dS/S = u * dt + sigma * dW ~ Normal (mean = u * dt, variance = sigma^2 * dt) 

This model implies that dG = (u - 0.5 * sigma ^2) * dt + sigma * dW ~ Normal (mean = (u - 0.5 * sigma ^2) * dt, variance = sigma^2 * dt)
where G = ln (S) so dG = d ln(S) => log returns ~ Normal (mean = (u - 0.5 * sigma ^2) * dt, variance = sigma^2 * dt)
so log (S_T/S_0) ~ Normal (mean = (u - 0.5 * sigma ^2) * T, variance = sigma^2 * T)

Let S_0 = opening index value and S_T = eod index value and T = 1 st

log (S_T/S_0) ~ Normal (mean = (u - 0.5 * sigma ^2), variance = sigma^2)
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

def simulate_hist(cur_idx_val, mean, std):
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

def simulate(S0, mean, std, dt):
  cur_h = dt.hour
  cur_m = dt.minute
  num_mins = 60*(16 - (cur_h+1)) + 60 - cur_m
  dt = num_mins/(60*6.5)
  est_eod_vals = np.exp(np.log(S0) + np.random.normal(loc = mean * dt, scale = std * np.sqrt(dt), size=1000000))
  est_eod_vals.sort()
  return est_eod_vals

def in_interval_prob(inc_lb, inc_ub, samples):
  lb_insert_pt = bisect.bisect_left(samples, inc_lb) #index at which inc_lb > all values at lower indices => inc_lb > index + 1 values from samples
  ub_insert_pt = bisect.bisect_left(samples, inc_ub) #index at which inc_ub > all values at lower indices => inc_ub > index + 1 values from samples
  lb_cdf = (lb_insert_pt+1)/len(samples)
  ub_cdf = (ub_insert_pt+1)/len(samples)
  return ub_cdf-lb_cdf

def get_mu(mean, std):
  #log (S_T/S_0) ~ Normal (mean = (u - 0.5 * sigma ^2), variance = sigma^2)
  return mean + 0.5 * std**2


