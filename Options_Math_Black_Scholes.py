#!/usr/bin/env python
# coding: utf-8

# '''
# Preface:
#     The classic Black Scholes formula as implemented below assumes:
#         1) The amount of time that is used for the determining the distribution of the underlying value 
#         at expiry is the same amount of time that should be used for discounting the forward price of 
#         the option.  Frequently these two amounts of time are different as options rarely settle at expiry
#         but instead settle at expiry plus a few days.
#         2) The interest rate used to project the underlying value at expiry based on the underlying spot 
#         value is the same as the interest rate that should be used for discounting the forward price of
#         the option.  In crytpo markets in 2025, for example, these two rates are very different.
#     To overcome these shortcomings, the following implementation is recommended:
#         1) Set the interest rate in the Black Scholes formula to zero.
#         2) Calculate the underlying value at expiry prior to calling the formula.  Enter the underlying
#         value at epxiry as the undelrying_value variable# Assistant
#         3) Calculate the amount of time to expiration of the option prior to calling the formual.  Set
#         the time variable equal to this amount.
#         4) Discount the formulas's resulting forward price of the option after using the formula below
#         using the Time_Value_Money Class and choices for rate and time that are specific to the discounting
#         situation (time to payment rather than time to expiry and rate for discounting rather than forward
#         rate of underlying).
#     The term "value" is used rather than "price" to emphasize that formulas below can be used for rates, too
# 
# Parameters:
#     c_or_p (string): acceptable names for 'call' or 'put' (see c_or_p function below)
#     u (float): see note above, value of underlying
#     k (float): strike value expressed in same units as underlying_value
#     v (float): volatility expressed as a percentage of underlying value (should be the same value regardless of 
#         using forward or spot underlying price)
#     t (float): see note above, fraction of a year (use Daycount_Math prior to calling functions below)
#     r (float): see note above, continuously compounded rate used to (i) discount forward option 
#         value to present value and (ii) create underlying value at expiry (if spot underlying value is used
#         as the input) 
#     d0 (float): the amount that is directly in the middle of d1 and d2 - 
#         similar to z-score of u relative to k based on v * sqrt(t)
#     d1 (float): the amount equal to the d1 term in the Black Scholes formula
#     d2 (float): the amount equal to the d2 term in the Black Scholes formula
#     opt_value (float): the value of the option in the same units as the units of the underlying
# 
# Results:
#     d0 : same as in parameters
#     d1 : same as in parameters
#     d2 : same as in parameters
#     option_value : value of option in same units as underlying value
#     delta : change in option_value for a one unit change in underlying value (first order greek)
#     vega : change in option_value for a one percent absolute change in volatility (first order greek)
#     theta : change in option_value for a one day change in time (first order greek)
#     rho : change in option_value for a one percent absolute change in rate (first order greek)
#     gamma : change in delta for a one unit change in underlying value (second order greek)
#     volga : change in vega for a one percent absolute value change in volatility (second order greek)
#     speed : (third order greek)
#     ultima : (third order greek)
#     vanna : (cross greek)
#     charm : (cross greek)
#     veta : (cross greek)
#     zoma : (cross greek)
#     color : (cross greek)
# '''   

# In[1]:


import numpy as np 
from scipy.stats import norm
from scipy.optimize import brentq

from Options_Math_Helpers import *


# In[3]:


class OptionsMathBlackScholes(OptionsMathHelpers):

    
    def __init__(self):
        # constructor 
        pass

    
    # -----------------------
    # Classic Black-Scholes Formulas
    #      Formulas just calculate answers 
    #      Formulas should be good for use with single inputs or arrays
    #      All data should be checked before using formulas as formulas assume data is consistent and clean
    #      (self, opt_type=None, fwd=None, strike=None, vol=None, time=None, r='r', d0=None, d1=None, d2=None, diy=365.0, **kwargs):
    # -----------------------
        
    def bs_d0_formula(self, fwd=None, strike=None, vol=None, time=None, rate=0.0, **kwargs): 
        # Prevent division by zero or invalid sqrt(time)
        time =  np.maximum(time, 1e-12)
        vol = np.maximum(vol, 1e-12)
        vol_sqrt_time = vol * np.sqrt(time)  
        log_moneyness = np.log(fwd / strike)
        growth_rate   = rate * time         
        return (log_moneyness + growth_rate) / vol_sqrt_time

    
    def bs_d1_formula(self, vol=None, time=None, d0=None, **kwargs):
        return d0 + (vol * np.sqrt(time) / 2.0)

    
    def bs_d2_formula(self, vol=None, time=None, d0=None, **kwargs):
        return d0 - (vol * np.sqrt(time) / 2.0)

    
    def bs_option_value_formula(self, opt_type=None, fwd=None, strike=None, time=None, rate=0.0, d1=None, d2=None, **kwargs):
        scalar       = np.asarray(np.where(opt_type == 'CALL', 1, -1), dtype=float)
        fwd_term     =  scalar * norm.cdf(d1 * scalar) *  fwd 
        strike_term  = -scalar * norm.cdf(d2 * scalar) * (strike * self.pv_factor(time=time, rate=rate))
        return fwd_term + strike_term         
        

    def bs_delta_formula(self, opt_type=None, d1=None, **kwargs):
        call_delta = norm.cdf(d1)
        put_delta  = call_delta - 1  # N(d1) − 1 = −N(−d1)
        return np.where(opt_type == 'CALL', call_delta, put_delta)

    
    def bs_vega_formula(self, fwd=None, time=None, d1=None, **kwargs):
        return fwd * np.sqrt(time) * norm.pdf(d1) * 0.01

    
    def bs_theta_formula(self, opt_type=None, fwd=None, strike=None, vol=None, time=None, rate=0.0, d1=None, d2=None, diy=365.0, **kwargs):
        scalar       = np.asarray(np.where(opt_type == 'CALL', 1, -1), dtype=float)
        fwd_term     = -1 * fwd * vol * norm.pdf(d1) / (2 * np.sqrt(time))
        strike_term  = -1 * strike * rate * norm.cdf(d2 * scalar) * self.pv_factor(time=time, rate=rate) * scalar         
        return (fwd_term + strike_term) / diy
        

    def bs_rho_formula(self, opt_type=None, strike=None, time=None, rate=0.0, d2=None, **kwargs):
        scalar = np.asarray(np.where(opt_type == 'CALL', 1, -1), dtype=float)
        return strike * time * norm.cdf(d2 * scalar) * self.pv_factor(time=time, rate=rate) * scalar * 0.01
        

    def bs_gamma_formula(self, fwd=None, vol=None, time=None, d1=None, **kwargs):
        denominator = fwd * vol * np.sqrt(time)
        return norm.pdf(d1) / denominator  


    def bs_volga_formula(self, vol=None, d1=None, d2=None, vega=None, **kwargs):
        numerator = vega * d1 * d2
        return numerator / vol

    
    # -----------------------
    # Option formulas with data checks
    #      Formulas can be called with multiple input combinations (with or without d-terms, etc.)
    #      Formulas convert inputs to NumPy arrays for elementwise math
    #      Formulas should also work with non-array inputs
    #      (self, opt_type=None, fwd=None, strike=None, vol=None, time=None, rate=0.0, d0=None, d1=None, d2=None, diy=365.0, **kwargs):
    # -----------------------        
        
    def bs_d0(self, fwd=None, strike=None, vol=None, time=None, rate=0.0, **kwargs):
        fwd, strike, time, vol, rate = self.to_arrays(fwd, strike, time, vol, rate)
        return self.bs_d0_formula(fwd=fwd, strike=strike, time=time, vol=vol, rate=rate)
        

    def bs_d1(self, fwd=None, strike=None, vol=None, time=None, rate=0.0, d0=None, **kwargs):        
        fwd, strike, time, vol, rate, d0 = self.to_arrays(fwd, strike, time, vol, rate, d0)
        if np.all(np.isnan(d0)): # need to calc d0
            d0 = self.bs_d0_formula(fwd=fwd, strike=strike, time=time, vol=vol, rate=rate) 
        return self.bs_d1_formula(d0=d0, vol=vol, time=time)

    
    def bs_d2(self, fwd=None, strike=None, vol=None, time=None, rate=0.0, d0=None, **kwargs):  
        fwd, strike, time, vol, rate, d0 = self.to_arrays(fwd, strike, time, vol, rate, d0)
        if np.all(np.isnan(d0)): # need to calc d0
            d0 = self.bs_d0_formula(fwd=fwd, strike=strike, time=time, vol=vol, rate=rate)            
        return self.bs_d2_formula(d0=d0, vol=vol, time=time)

       
    def bs_option_value(self, opt_type=None, fwd=None, strike=None, vol=None, time=None, rate=0.0, 
                        d0=None, d1=None, d2=None, **kwargs):
        fwd, strike, vol, time, rate, d0, d1, d2 = self.to_arrays(fwd, strike, vol, time, rate, d0, d1, d2)          
        if np.all(np.isnan(d1)) or np.all(np.isnan(d2)): # we need to calc d1 and/or d2
            if np.all(np.isnan(d0)):  # we need to calc d0 then d1 and d2
                d0 = self.bs_d0_formula(fwd=fwd, strike=strike, time=time, vol=vol, rate=rate)
            if np.all(np.isnan(d1)):                    
                d1 = self.bs_d1_formula(d0=d0, vol=vol, time=time)
            if np.all(np.isnan(d2)):
                d2 = self.bs_d2_formula(d0=d0, vol=vol, time=time)      
        opt_type = self.option_type(opt_type, strike)
        return self.bs_option_value_formula(opt_type=opt_type, fwd=fwd, strike=strike, time=time, rate=rate, d1=d1, d2=d2)   

    
    # -----------------------
    # Greeks (first order)
    # -----------------------
    def bs_delta(self, opt_type=None, fwd=None, strike=None, vol=None, time=None, rate=0.0, d0=None, d1=None, **kwargs):   
        fwd, strike, vol, time, rate, d0, d1 = self.to_arrays(fwd, strike, vol, time, rate, d0, d1)          
        if np.all(np.isnan(d1)):  # need to calc d1
            if np.all(np.isnan(d0)): # need to calc d0
                d0 = self.bs_d0_formula(fwd=fwd, strike=strike, time=time, vol=vol, rate=rate)
            d1 = self.bs_d1_formula(d0=d0, vol=vol, time=time)
        opt_type = self.option_type(opt_type, d1)
        return self.bs_delta_formula(opt_type=opt_type, d1=d1)
        

    def bs_vega(self, fwd=None, strike=None, vol=None, time=None, rate=0.0, d0=None, d1=None, **kwargs):  
        fwd, strike, vol, time, rate, d0, d1 = self.to_arrays(fwd, strike, vol, time, rate, d0, d1)  
        if np.all(np.isnan(d1)): # need to calc d1
            if np.all(np.isnan(d0)): # need to calc d0
                d0 = self.bs_d0_formula(fwd=fwd, strike=strike, time=time, vol=vol, rate=rate)
            d1 = self.bs_d1_formula(d0=d0, vol=vol, time=time)
        return self.bs_vega_formula(fwd=fwd, time=time, d1=d1)

    
    def bs_theta(self, opt_type=None, fwd=None, strike=None, vol=None, time=None, rate=0.0, 
                 d0=None, d1=None, d2=None, diy=365.0, **kwargs):
        fwd, strike, time, vol, rate, d0, d1, d2, diy = self.to_arrays(fwd, strike, time, vol, rate, d0, d1, d2, diy)  
        if np.all(np.isnan(d1)) or np.all(np.isnan(d2)): # we need to calc d1 and/or d2
            if np.all(np.isnan(d0)):  # we need to calc d0 then d1 and d2
                d0 = self.bs_d0_formula(fwd=fwd, strike=strike, time=time, vol=vol, rate=rate)
            if np.all(np.isnan(d1)):                    
                d1 = self.bs_d1_formula(d0=d0, vol=vol, time=time)
            if np.all(np.isnan(d2)):
                d2 = self.bs_d2_formula(d0=d0, vol=vol, time=time)      
        opt_type = self.option_type(opt_type, strike) 
        return self.bs_theta_formula(opt_type=opt_type, fwd=fwd, strike=strike, vol=vol, time=time, rate=rate, d1=d1, d2=d2, diy=diy)


    def bs_rho(self, opt_type=None, fwd=None, strike=None, vol=None, time=None, rate=0.0, d0=None, d2=None, **kwargs):
        fwd, strike, time, vol, rate, d0, d2 = self.to_arrays(fwd, strike, time, vol, rate, d0, d2)  
        if np.all(np.isnan(d2)): # need to calc d2
            if np.all(np.isnan(d0)): # need to calc d0
                d0 = self.bs_d0_formula(fwd=fwd, strike=strike, time=time, vol=vol, rate=rate)
            d2 = self.bs_d2_formula(d0=d0, vol=vol, time=time)  
        opt_type = self.option_type(opt_type, strike)       
        return self.bs_rho_formula(opt_type=opt_type, strike=strike, time=time, rate=rate, d2=d2) 

    
    # -----------------------
    # Greeks (second order)
    # -----------------------
    def bs_gamma(self, fwd=None, strike=None, vol=None, time=None, rate=0.0, d0=None, d1=None, **kwargs):        
        fwd, strike, time, vol, rate, d0, d1 = self.to_arrays(fwd, strike, time, vol, rate, d0, d1)  
        if np.all(np.isnan(d1)):
            if np.all(np.isnan(d0)):
                d0 = self.bs_d0_formula(fwd=fwd, strike=strike, time=time, vol=vol, rate=rate)
            d1 = self.bs_d1_formula(d0=d0, vol=vol, time=time)         
        return self.bs_gamma_formula(fwd=fwd, vol=vol, time=time, d1=d1)

        
    def bs_volga(self, fwd=None, strike=None, vol=None, time=None, rate=0.0, 
                 d0=None, d1 = -100.0, d2=None, vega=-100.0, **kwargs):
        fwd, strike, time, vol, rate, d0, d1, d2, vega = self.to_arrays(fwd, strike, time, vol, rate, d0, d1, d2, vega)  
        if np.all(np.isnan(d1)) or np.all(np.isnan(d2)): # we need to calc d1 and/or d2
            if np.all(np.isnan(d0)):  # we need to calc d0 then d1 and d2
                d0 = self.bs_d0_formula(fwd=fwd, strike=strike, time=time, vol=vol, rate=rate)
            if np.all(np.isnan(d1)):                    
                d1 = self.bs_d1_formula(d0=d0, vol=vol, time=time)
            if np.all(np.isnan(d2)):
                d2 = self.bs_d2_formula(d0=d0, vol=vol, time=time)  
        if np.all(vega == None):
            vega = self.bs_vega_formula(fwd=fwd, time=time, d1=d1)
        return self.bs_volga_formula(vol=vol, d1=d1, d2=d2, vega=vega)


    # -----------------------
    # Greeks (third order)
    # -----------------------
    def bs_speed():
        pass
        #pending

    
    def bs_ultima():
        pass
        #pending        


    # -----------------------
    # Greeks (cross)
    # -----------------------
    def bs_vanna():
        pass
        #pending


    def bs_charm():
        pass
        #pending


    def bs_veta():
        pass
        #pending


    def bs_zomma():
        pass
        #pending


    def bs_color():
        pass
        #pending

    # -----------------------
    # Combinations of greeks
    # ----------------------- 
    def bs_first_order_greeks(self, opt_type=None, fwd=None, strike=None, vol=None, time=None, rate=0.0, 
                              d0=None, d1=None, d2=None, diy=365.0, **kwargs):
        fwd, strike, time, vol, rate, d0, d1, d2, diy = self.to_arrays(fwd, strike, time, vol, rate, d0, d1, d2, diy) 
        if np.all(np.isnan(d1)) or np.all(np.isnan(d2)): # we need to calc d1 and/or d2
            if np.all(np.isnan(d0)):  # we need to calc d0 then d1 and d2
                d0 = self.bs_d0_formula(fwd=fwd, strike=strike, time=time, vol=vol, rate=rate)
            if np.all(np.isnan(d1)):                    
                d1 = self.bs_d1_formula(d0=d0, vol=vol, time=time)
            if np.all(np.isnan(d2)):
                d2 = self.bs_d2_formula(d0=d0, vol=vol, time=time)  
        opt_type = self.option_type(opt_type, strike)  
        delta = self.bs_delta_formula(opt_type=opt_type, d1=d1)
        vega  = self.bs_vega_formula(fwd=fwd, time=time, d1=d1)
        theta = self.bs_theta_formula(opt_type=opt_type, fwd=fwd, strike=strike, vol=vol, time=time, rate=rate, d1=d1, d2=d2, diy=diy)
        rho   = self.bs_rho_formula(opt_type=opt_type, strike=strike, time=time, rate=rate, d2=d2)
        return (delta, vega, theta, rho)

    
    def bs_second_order_greeks():
        pass


    def bs_third_order_greeks():
        pass


    def bs_cross_greeks():
        pass
                

    def bs_all_greeks():
        pass

        
    # -----------------------
    # Volatility backsolvers
    # ----------------------- 

    def bs_vol_solver_newton(self, opt_value=None, opt_type=None, fwd=None, strike=None, time=None, rate=0.0,
                                  initial_guess=0.5, tolerance=1e-10, max_iterations=100):
        """
        Vectorized Newton-Raphson implied volatility solver.
        Rows that fail to converge will return np.nan, but the rest continue.
        """
        # Convert inputs to arrays - NOTE initial_guess GOES TO vol
        fwd, strike, time, rate, vol, opt_value = self.to_arrays(fwd, strike, time, rate, initial_guess, opt_value)
        opt_type = self.option_type(opt_type, opt_value)
    
        # Initialize  -  assumption is that opt_value and strike will have to be array like to begin with
        if fwd.shape != opt_value.shape:
            fwd = np.full_like(opt_value, fwd, dtype=float)
        if time.shape != opt_value.shape:
            time =  np.full_like(opt_value, time, dtype=float)
        if rate.shape != opt_value.shape:
            rate = np.full_like(opt_value, rate, dtype=float)
        if vol.shape != opt_value.shape:
            vol = np.full_like(opt_value, vol, dtype=float)
            
        converged = np.zeros_like(opt_value, dtype=bool)
    
        for i in range(max_iterations):
            
            # Skip already converged elements
            mask = ~converged
    
            if not np.any(mask):  # All done
                break
    
            # Protect against very small v by taking the max of the vol and 1e-8
            vol[mask] = np.maximum(vol[mask], 1e-8)  
    
            d0 = self.bs_d0_formula(fwd=fwd[mask], strike=strike[mask], vol=vol[mask], time=time[mask], rate=rate[mask])
            d1 = self.bs_d1_formula(vol=vol[mask], time=time[mask], d0=d0)
            d2 = self.bs_d2_formula(vol=vol[mask], time=time[mask], d0=d0)
            opt_value_est =  self.bs_option_value_formula(opt_type=opt_type[mask], fwd=fwd[mask],
                                                          strike=strike[mask], time=time[mask], rate=rate[mask], d1=d1, d2=d2)
    
            diff = opt_value_est - opt_value[mask]
            
            vega = self.bs_vega_formula(fwd=fwd[mask], time=time[mask], d1=d1) 
    
            # Newton-Raphson step with small factor to improve stability
            safe_step = np.where(np.abs(vega) > 1e-8, 0.01 * diff / vega, 0.0)
            vol[mask] = vol[mask] - safe_step
    
            # Update convergence mask
            converged[mask] = np.abs(diff) < tolerance
    
        # Assign np.nan to non-converged elements
        vol[~converged] = np.nan
        
        return vol.astype(float)



#     def bs_vol_solver_brentq(self, opt_value=None, opt_type=None, fwd=None, strike=None, time=None, rate=0.0):
#         """
#         Solve for implied volatility given a market price.
#         """
#         
#         fwd, strike, time, rate, opt_value = self.to_arrays(fwd, strike, time, rate, opt_value)  
#         opt_type = self.option_type(opt_type, u)
# 
#         lo_vol_bound = np.full(opt_value.shape, 1e-6)
#         hi_vol_bound = np.full(opt_value.shape, 10.0)     
#         
#         # Define the function whose root we want to find
#         def objective(v):
#             d0 = self.bs_d0_formula(fwd=fwd, strike=strike, vol=vol, time=time, rate=rate)
#             d1 = self.bs_d1_formula(vol=vol, time=time, d0=d0)
#             d2 = self.bs_d2_formula(vol=vol, time=time, d0=d0)
#             return self.bs_option_value_formula(opt_type=opt_type, fwd=fwd, strike=strike, vol=vol, time=time, rate=rate) - opt_value
#     
#         # Use brentq to find sigma between lo_vol_bound and hi_vol_bound
#         try:
#             return brentq(objective, lo_vol_bound, hi_vol_bound)
#         except ValueError:
#             return np.nan  # If no solution in the range
# 
