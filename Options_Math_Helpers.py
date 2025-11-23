#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 

from Options_Math_Helpers import *


# In[2]:


class OptionsMathHelpers():

    
    def __init__(self):
        # constructor 
        pass

    # -----------------------
    # Helpers
    # -----------------------
    def pv_factor(self, time=1, rate=0.0, **kwargs):
        # Compute time value of money - eventually replace with function from TVM Class
        return np.exp(-rate * time) #if interest_rate == 0 then pv_factor will equal one

        
    def to_arrays(self, *args, dtype=float):
        # Convert all inputs to NumPy arrays of dtype=d_type.
        # Raises ValueError if any array contains NaN values.
        # Returns a tuple of arrays in the same order as the inputs.
        arrays = [np.asarray(arg, dtype=dtype) for arg in args]
            
        # Only check NaN for numeric dtypes
        #for i, arr in enumerate(arrays):
        #    if np.issubdtype(arr.dtype, np.number) and np.any(np.isnan(arr)):
        #        raise ValueError(f"Array {i+1} contains NaN values")

        return tuple(arrays)   # Note output is a tuple!!!


    def option_type(self, opt_type, array_to_mimic):        
        # Convert array_to_mimic so we can broadcast to its shape if needed
        array_to_mimic, = self.to_arrays(array_to_mimic)
        
        # If scalar indicator, broadcast to array_to_mimic's shape

        if np.isscalar(opt_type):
            opt_type = np.full_like(array_to_mimic, opt_type, dtype=object)

        # Convert to object array and uppercase elementwise
        opt_type, = self.to_arrays(opt_type, dtype=object)
        opt_type = np.char.upper(opt_type.astype(str))

        call_names = ['C', 'CALL']
        put_names  = ['P', 'PUT']
        valid_names = call_names + put_names
        
        # Validate input
        if not np.all(np.isin(opt_type, valid_names)):
            raise ValueError("opt_type must contain only values associated with 'call' or 'put'")

        return np.where(np.isin(opt_type, call_names), 'CALL', 'PUT')


    def to_option_grid_df(self, df):
  
        call_df = df[df['option_type'] == 'CALL'].copy()
        put_df  = df[df['option_type'] == 'PUT'].copy()

        grid_df = pd.merge(call_df, put_df, 
                           on=['years_to_expiry', 'strike', 'expiration_time_nyc'], 
                           suffixes=['_c', '_p'])

        grid_df = grid_df.sort_values(by=['years_to_expiry', 'strike'])
        
        return grid_df

