#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import pytz

from datetime import datetime, timedelta

from Options_Math_Helpers import *


# In[2]:


class OptionsMathAlgebra(OptionsMathHelpers):

    
    def __init__(self):
        # constructor 
        pass


    # -----------------------
    # Intrinsic value; extrinsic/time value = option value - intrinsic value
    # ----------------------- 
    def intrinsic_value_both(self, spot=None, strike_pv=None):
        # Calcs intrinsic values for options
        # Subtract intrinsic values from options prices to get time values
        # Calc the present value of the strike values prior to calling this function
        
        spot, strike_pv = self.to_arrays(spot, strike_pv)
        
        call_intrinsic = np.maximum(spot - strike_pv, 0.0)
        put_intrinsic  = np.maximum(strike_pv - spot, 0.0)
        
        return call_intrinsic, put_intrinsic


    def intrinsic_value(self, opt_type=None, spot=None, strike_pv=None):
        
        call_intrinsic, put_intrinsic = self.intrinsic_value_both(spot=spot, strike_pv=strike_pv)      
        
        opt_type = self.option_type(opt_type, spot)
        
        intrinsic = np.where(opt_type == 'CALL', call_intrinsic, put_intrinsic)

        return intrinsic

        
    # ----------------------- 
    # Put/Call parity formulas
    # -----------------------         


    def put_call_parity_underlying(self, price_c=None, price_p=None, strike_pv=None):       
        price_c, price_p, strike_pv = self.to_arrays(price_c, price_p, strike_pv)
        return strike_pv + (price_c - price_p)

    
    def put_call_parity_call(self, spot=None, price_p=None, strike_pv=None):   
        spot, price_p, strike_pv = self.to_arrays(spot, price_p, strike_pv)
        return (spot - strike_pv) + price_p


    def put_call_parity_put(self, u=None, price_c=None, strike_pv=None):
        spot, price_c, strike_pv = self.to_arrays(spot, price_c, strike_pv)
        return (strike_pv - spot) + price_c        
  

