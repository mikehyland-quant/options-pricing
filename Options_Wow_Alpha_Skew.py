#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

from scipy.optimize import least_squares
from scipy.stats import norm

from Options_Math_Helpers import *
from Options_Math_Black_Scholes import *

ombs = OptionsMathBlackScholes()


# In[1]:


class WowAlphaSkew(OptionsMathHelpers):
    
    def __init__(self):
        # constructor 
        pass
        
    def wow_alpha_vol(self, fwd=None, strike=None, 
                      vol_atm=None, wow=None, convexity=None, **kwargs):

        log_moneyness = np.log(fwd / strike)      # this is the alpha of "wow alpha"
        
        hi_vol = vol_atm * (1 + wow - (log_moneyness * convexity))
        lo_vol = vol_atm * (1 - wow - (log_moneyness * convexity))
  
        return hi_vol, lo_vol

    
    def wow_alpha_value(self, opt_type=None, fwd=None, strike=None, time=None, rate=0.0,
                        vol_atm=None, wow=None, convexity=None, **kwargs):
        fwd, strike, time, rate, vol_atm, wow, convexity = self.to_arrays(fwd, strike, time, rate, vol_atm, wow, convexity) 
        opt_type = self.option_type(opt_type, strike)
        
        hi_vol, lo_vol = self.wow_alpha_vol(fwd=fwd, strike=strike, 
                                            vol_atm=vol_atm, wow=wow, convexity=convexity)

        hi_d0 = ombs.bs_d0_formula(fwd=fwd, strike=strike, time=time, vol=hi_vol)
        lo_d0 = ombs.bs_d0_formula(fwd=fwd, strike=strike, time=time, vol=lo_vol)

        hi_d1 = ombs.bs_d1_formula(vol=hi_vol, time=time, d0=hi_d0)
        lo_d1 = ombs.bs_d1_formula(vol=lo_vol, time=time, d0=lo_d0)

        hi_d2 = ombs.bs_d2_formula(vol=hi_vol, time=time, d0=hi_d0)
        lo_d2 = ombs.bs_d2_formula(vol=lo_vol, time=time, d0=lo_d0)
        
        hi_vol_value = ombs.bs_option_value_formula(opt_type=opt_type, fwd=fwd, strike=strike, time=time, d1=hi_d1, d2=hi_d2)
        lo_vol_value = ombs.bs_option_value_formula(opt_type=opt_type, fwd=fwd, strike=strike, time=time, d1=lo_d1, d2=lo_d2)
        
        return (hi_vol_value + lo_vol_value) / 2


    def _wow_alpha_value_weighted(self, params, opt_type, fwd, strike, time, vol_atm, target_value, weight):
        wow, convexity = params
      
        # compute model vols vectorized
        model_value = self.wow_alpha_value(opt_type=opt_type,
                                            fwd=fwd, 
                                            strike=strike, 
                                            time=time, 
                                            vol_atm=vol_atm,
                                            wow=wow,
                                            convexity=convexity)
        
        value_error = model_value - target_value   
        return weight * value_error

 
    def calibrate_wow_alpha_weighted(self,
                                     opt_type=None,
                                     fwd=None, strike=None, time=None, 
                                     vol_atm=None,
                                     target_value=None, 
                                     weighting='vega',
                                     weight=None,
                                     weight_eps=1e-8,
                                     initial_guess=(0.2, 0.0),
                                     bounds=([-10.0, -10.0], [10.0, 10.0]),
                                     **lsq_kwargs):
        """
        weighting: 'vega' | 'sqrt' | 'norm' | 'price'
          - 'vega': residual = vega * (model_vol - market_vol)  <-- common, approx price-space
          - 'sqrt': sqrt(vega) * vol_error (less extreme weights)
          - 'norm' : vegas normalized to [0,1] * vol_error
          - 'price': uses exact Black price residuals (most accurate)
        """
        fwd, strike, time, target_value, weight, weight_eps, initial_guess = self.to_arrays(fwd, strike, time, target_value, 
                                                                                              weight, weight_eps, initial_guess) 

        if weighting in ('vega', 'sqrt', 'norm'):
            # avoid zero vegas
            weight = np.maximum(weight, weight_eps)
    
            if weighting == 'vega':
                weight = weight
            elif weighting == 'sqrt':
                weight = np.sqrt(weight)
            else:  # 'norm'
                weight = weight / np.max(weight)
        elif vega_weighting == 'price':
            pass
            # see commented section below
        else:
            raise ValueError("Unknown vega_weighting: choose 'vega', 'sqrt', 'norm' or 'price'")
                
        obj_fn = lambda initial_guess: self._wow_alpha_value_weighted(initial_guess, opt_type, fwd, strike, time, 
                                                                      vol_atm, target_value, weight)

        result = least_squares(obj_fn, initial_guess, bounds=bounds, **lsq_kwargs)
        wow, convexity = result.x
    
        return wow, convexity, result
    
     #       # more exact: convert vols -> Black prices and return price residuals
      #      # Price approximation: ∆Price ≈ vega * ∆σ, but you can compute full Black price if desired.
       #     # Here we'll use exact Black price differences (call price; sign doesn't matter for LS).
        #    model_prices = np.array([black_price(F, K, T, mv) for K, mv in zip(strikes, model_vols)])
         #   market_prices = np.array([black_price(F, K, T, mv) for K, mv in zip(strikes, market_vols)])
          #  return model_prices - market_prices
    
 

