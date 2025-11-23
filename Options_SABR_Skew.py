#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

from scipy.optimize import least_squares
from scipy.stats import norm

from Options_Math_Helpers import *


# In[2]:


class SABRSkew(OptionsMathHelpers):
    
    def __init__(self):
        # constructor 
        pass
        
    def _beta_term(self, beta, exponent=1):
        partial_term = 1 - beta
        return partial_term ** exponent

    def _strike_fwd_term(self, fwd, strike, exponent=1):
        partial_term = fwd * strike
        return partial_term ** exponent        
    
    def _log_term(self, fwd, strike, exponent=1):
        partial_term = np.log(fwd / strike)
        return partial_term ** exponent
            
    def _z(self, alpha, beta, nu, fwd, strike):
        exponent = self._beta_term(beta, 1) / 2
        z1 = nu / alpha
        z2 = self._strike_fwd_term(fwd, strike, exponent)
        z3 = self._log_term(fwd, strike)
        return z1 * z2 * z3
        
    def _xz(self, rho, z, eps=1e-14):
        sqrt_term = 1 - 2*rho*z + z*z
        sqrt_term = np.clip(sqrt_term, 0, None)  # clip lower bound at 0
        
        xz_numerator = np.sqrt(sqrt_term) + z - rho
        xz_denominator = 1 - rho
        
        # Clip numerator and denominator to avoid log(0) or negative
        xz_numerator = np.clip(xz_numerator, eps, None)
        xz_denominator = np.clip(xz_denominator, eps, None)

        return np.log(xz_numerator / xz_denominator)

    def sabr_vol_formula(self, fwd=None, 
                               strike=None, 
                               time=None, 
                               alpha=None, # 0.0 to 5.0
                               beta=None, # 0.0 to 1.0
                               rho=None, # -1.0 to 1.0
                               nu=None, # 0.0 to 5.0
                               **kwargs): 
        
        denom_sum  = 1
        denom_sum += self._beta_term(beta, 2) * self._log_term(fwd, strike, 2) / 24
        denom_sum += self._beta_term(beta, 4) * self._log_term(fwd, strike, 4) / 1920
    
        exponent1   = self._beta_term(beta)
        exponent2   = exponent1 / 2
        
        scalar_sum  = (self._beta_term(beta, 2) / 24) * (alpha * alpha / self._strike_fwd_term(fwd, strike, exponent1))
        scalar_sum += 0.25 * rho * beta * nu * alpha / self._strike_fwd_term(fwd, strike, exponent2)
        scalar_sum += (2 - 3*rho*rho) * nu * nu / 24
    
        z = self._z(alpha, beta, nu, fwd, strike) 
        
        sabr1 = alpha / self._strike_fwd_term(fwd, strike, exponent2)
        sabr2 = 1 / denom_sum
        sabr3 = z / self._xz(rho, z)
        sabr4 = 1 + scalar_sum * time
    
        sabr_vol = sabr1 * sabr2 * sabr4  # not sabr3
    
        # create a boolean mask where fwd is NOT close to strike
        mask = np.isclose(fwd, strike, rtol=1e-12, atol=1e-14)
        
        # apply z/xz factor only for non-ATM points
        sabr_vol = np.where(mask, sabr_vol, sabr_vol * sabr3)
    
        return sabr_vol

    def sabr_vol(self, fwd=None, 
                       strike=None, 
                       time=None, 
                       alpha=0.5, # 0.0 to 5.0
                       beta=1.0, # 0.0 to 1.0
                       rho=0.0, # -1.0 to 1.0
                       nu=1.0, # 0.0 to 5.0
                       **kwargs): 
        fwd, strike, time, alpha, beta, rho, nu = self.to_arrays(fwd, strike, time, alpha, beta, rho, nu) 
        sabr_vol = self.sabr_vol_formula(fwd=fwd, 
                                         strike=strike, 
                                         time=time, 
                                         alpha=alpha,
                                         beta=beta,
                                         rho=rho,
                                         nu=nu) 
        return sabr_vol
        

    def sabr_alpha(self, atm_vol, fwd, beta):
        return atm_vol * (fwd ** self._beta_term(beta))


    def _sabr_vol_weighted(self, params, fwd, strike, time, target_vols, beta, weights):
        alpha, rho, nu = params
      
        # compute model vols vectorized
        model_vols = self.sabr_vol_formula(fwd=fwd, 
                                           strike=strike, 
                                           time=time, 
                                           alpha=alpha,
                                           beta=beta,
                                           rho=rho,
                                           nu=nu)
        
        vol_errors = model_vols - target_vols    
        return weights * vol_errors
        
    def calibrate_sabr_weighted(self,
                                fwd=None, strike=None, time=None, 
                                target_vols=None, 
                                beta=1.0,
                                weighting='vega',
                                weights=None,
                                weight_eps=1e-8,
                                initial_guess=(0.2, 0.0, 0.5),
                                bounds=([1e-8, -0.999, 1e-8], [5.0, 0.999, 5.0]),
                                **lsq_kwargs):
        """
        weighting: 'vega' | 'sqrt' | 'norm' | 'price'
          - 'vega': residual = vega * (model_vol - market_vol)  <-- common, approx price-space
          - 'sqrt': sqrt(vega) * vol_error (less extreme weights)
          - 'norm' : vegas normalized to [0,1] * vol_error
          - 'price': uses exact Black price residuals (most accurate)
        """
        fwd, strike, time, target_vols, beta, weights, weight_eps, initial_guess = self.to_arrays(fwd, strike, time, target_vols, 
                                                                                                  beta, weights, weight_eps, initial_guess) 
        
        if weighting in ('vega', 'sqrt', 'norm'):
            # avoid zero vegas
            weights = np.maximum(weights, weight_eps)
    
            if weighting == 'vega':
                weights = weights
            elif weighting == 'sqrt':
                weights = np.sqrt(weights)
            else:  # 'norm'
                weights = weights / np.max(weights)
        elif vega_weighting == 'price':
            pass
            # see commented section below
        else:
            raise ValueError("Unknown vega_weighting: choose 'vega', 'sqrt', 'norm' or 'price'")
                
        obj_fn = lambda initial_guess: self._sabr_vol_weighted(initial_guess, fwd, strike, time, 
                                                               target_vols, beta, weights)
    
        result = least_squares(obj_fn, initial_guess, bounds=bounds, **lsq_kwargs)
        alpha, rho, nu = result.x
    
        return alpha, rho, nu, result
    
     #       # more exact: convert vols -> Black prices and return price residuals
      #      # Price approximation: ∆Price ≈ vega * ∆σ, but you can compute full Black price if desired.
       #     # Here we'll use exact Black price differences (call price; sign doesn't matter for LS).
        #    model_prices = np.array([black_price(F, K, T, mv) for K, mv in zip(strikes, model_vols)])
         #   market_prices = np.array([black_price(F, K, T, mv) for K, mv in zip(strikes, market_vols)])
          #  return model_prices - market_prices
    
 

