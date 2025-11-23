#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datetime import date, timedelta
import numpy as np
import pandas as pd
import xlwings as xw

from Options_Math_Helpers import *

omh = OptionsMathHelpers()


# In[25]:


# fully vectorized and ready to go
class TimeStuff():
    
            
    def __init__(self):
        # constructor 
        pass


    def utc_to_local(self, time, zone='America/New_York'):
        time = time.dt.tz_localize('UTC')
        return time.dt.tz_convert(zone)
    
    
    def diff_in_time(self, start_date, end_date, date_format, time_unit):
        d_format = f'datetime64[{date_format}]'
        start_date, end_date = omh.to_arrays(start_date, end_date, dtype=d_format)        
        return (end_date - start_date) / np.timedelta64(1, time_unit)
    
    
    def accrual_days_actual(self, start_date, end_date):   
        start_date = start_date.astype(str)  # this removes timezone
        end_date = end_date.astype(str)      # this removes timezone
        return self.diff_in_time(start_date, end_date, 'D', 'D')
        
   
    def accrual_days_thirty(self, start_date, end_date):
        start_date = start_date.astype(str)  # this removes timezone
        end_date = end_date.astype(str)      # this removes timezone
        start_date, end_date = omh.to_arrays(start_date, end_date, dtype='datetime64[D]') 
        
        # Extract years, months, days as integers
        y1 = start_date.astype('datetime64[Y]').astype(int)
        y2 = end_date.astype('datetime64[Y]').astype(int)
    
        m1 = start_date.astype('datetime64[M]').astype(int)
        m2 = end_date.astype('datetime64[M]').astype(int)
    
        d1 = start_date.astype('datetime64[D]').astype(int)
        d2 = end_date.astype('datetime64[D]').astype(int)
    
        # Adjust day-of-month for 30/360 ISDA rules
        d1 = np.where(d1 % 31 == 30, 30, d1 % 31 + 1)
        d2 = np.where((d2 % 31 + 1 == 31) & (d1 > 29), 30, d2 % 31 + 1)
    
        # Compute 30/360 days
        days = 360 * (y2 - y1) + 30 * ((m2 - m1) % 12) + (d2 - d1)

        return days


# In[ ]:


# not yet vectorized
def accrual_period_act_act(self, start_date, end_date):
    start_date, end_date = omh.to_arrays(start_date, end_date, dtype='datetime64[D]')      
    
    answer = 0.0
    
    if start_date <= end_date:

        rolling_date = start_date
        while rolling_date < end_date:
            
            rolling_year = rolling_date.astype('datetime64[Y]')
            year_end = (rolling_date.astype('datetime64[Y]') + np.timedelta64(1, 'Y')).astype('datetime64[D]') \
                                                                                          - np.timedelta64(1, 'D')
            current_end = min(year_end, end_date)

            days_in_period = (current_end - rolling_date).days + 1
            days_in_year = 366 if self.is_leap_year(rolling_year) else 365
            
            answer = answer + (days_in_period / days_in_year)

            current_date = current_end + np.timedelta64(1, 'D')

    return answer


def is_leap_year(self, year):
    # Return True if year is a leap year
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


# In[26]:


"""
Rules:
    The formulas below use the ISDA 2006 rules
    https://www.isda.org/a/smMDE/Blackline-2000-v-2006-ISDA-Definitions.pdf
    Section 4.16
"""


# General datetime stuff
# 
# pandas
#     import pandas as pd
# 
#     pd.Timestamp.now()
#     
#     [array or df_column] = pd.to_datetime([array or df_column], unit='x')
#     no units for anything bigger than seconds
#     's' for seconds
#     'ms' for milliseconds
# 
#     strftime()	datetime → string	Format a datetime object into a string
#     strptime()	string → datetime	Parse a string into a datetime object
#             Common Format Codes
#             Code	Meaning	Example
#             %Y	Year (4 digits)	2025
#             %y	Year (2 digits)	25
#             %m	Month (01–12)	11
#             %d	Day (01–31)	10
#             %H	Hour (00–23)	14
#             %I	Hour (01–12, 12-hour clock)	02
#             %p	AM/PM	PM
#             %M	Minute (00–59)	45
#             %S	Second (00–59)	30
#             %f	Microsecond (000000–999999)	123456
#             %a	Weekday (abbrev)	Mon
#             %A	Weekday (full)	Monday
#             %b	Month name (abbrev)	Nov
#             %B	Month name (full)	November
#             %Z	Timezone	UTC
#             %z	UTC offset	+0000
# 
#     Goal	Expression	Result type
#     Difference between two timestamps	t2 - t1	Timedelta
#     Difference between two columns	df['end'] - df['start']	Timedelta column
#     Time between rows	df['timestamp'].diff()	Timedelta column
#     Convert to hours/minutes	diff / pd.Timedelta(hours=1)	float
# 
# 
# numpy
#     import numpy as np
#         no need to import anything else if staying in numpy
#         numpy.datetime64 → stores timestamps (dates + time)
#         numpy.timedelta64 → stores durations (differences between timestamps)
# 
#     np.datetime64('now')
#     
#     [array or df_column] = np.datetime64(array or df_column, 'x')
#                             Unit code	Meaning	        Example literal
#                             'Y'	        Year	        np.datetime64('2025', 'Y')
#                             'M'	        Month	        np.datetime64('2025-11', 'M')
#                             'W'	        Week	        np.datetime64('2025-11-10', 'W')
#                             'D'	        Day	            np.datetime64('2025-11-10', 'D')
#                             'h'	        Hour	        np.datetime64('2025-11-10T12', 'h')
#                             'm'	        Minute	        np.datetime64('2025-11-10T12:34', 'm')
#                             's'	        Second	        np.datetime64('2025-11-10T12:34:56', 's')
#                             'ms'        Millisecond	    np.datetime64(1731245696123, 'ms')
# 
#     time differences
#         t1 = np.datetime64('2025-11-10T12:00:00')
#         t2 = np.datetime64('2025-11-10T14:30:00')
#         
#         delta = t2 - t1 # The result is a numpy.timedelta64 object — a duration, not a timestamp.
#         print(delta)
#         
#         hours = delta / np.timedelta64(1, 'h')
#         minutes = delta / np.timedelta64(1, 'm')
#         seconds = delta / np.timedelta64(1, 's')
# 
#     adding/subtracting time
#         t = np.datetime64('2025-11-10T12:00:00')        
#        
#         t_minus_30m = t - np.timedelta64(30, 'm')  # Subtract 30 minutes
#         t_plus_5h = t + np.timedelta64(5, 'h')  # Add 5 hours
# 
# 
# 
# Timestamps to dates
#     Library	Method	                    Result
#     NumPy	dt.astype('datetime64[D]')	numpy.datetime64 at day precision
#     Pandas	df['col'].dt.date	        Python date
#     Pandas	df['col'].dt.floor('D')	    datetime64[ns] with time 00:00:00
# 
# 
