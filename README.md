This code performs options math including Black Scholes pricing and greeks, volatility back-solving, and parameterized skew generation.

# Options_Math_Helpers
This is the Parent class to all of the other Options_Math classes.  It contains administrative commands that help format data.

# Options_Math_Algebra
This code covers intrinsic value and put-call parity.

# Options_Math_Black_Scholes
This code has all of the classic Black-Scholes formulas for pricing and greeks (first, second and some third order).  The code also includes a Newton-Raphson implied volatility solver and a yet to be implemented BrentQ implied volatility solver.  There is a long comment at the start of the module that encourages use of a zero interest rate, an forward price (instead of spot), and discounting of the option price outside the Black-Scholes formula

# Options_SABR_Skew
This code has all of the formulas needed to compute skew under the SABR approach.  It is best for interest rate derivatives.  There is also code to solve for the SABR parameters that best fit the market volatility data.

# Options_Wow_Alpha_Skew
This code has all of the formulas needed to compute skew under the SIG's old wow-alpha approach.  It is best for equities and fits certain crypto options surfaces very well also.  Like the SABR skew approach above, there is code to solve for the wow-=alpha parameters that best fit the market volatility data.





# enrich-options-prices
This code enriches the market prices of a grid of options to non-arbitrage levels.

Options markets are frequently incomplete in that market prices (bids and/or asks) are not shown for each available strike.  Additionally, it is often the case that the prices that are shown violate option arbitrage conditions (for example, when market bids  for options are so conservative that they imply negative time value).  This poor and/or lack of information can make it difficult to calibrate option volatility surfaces.

The code below uses simple “no arbitrage” rules for options pricing to create enriched option prices that reflect the highest bid and lowest ask prices for each option given the market prices available. 

An option trader acting as a price taker should only transact at prices within the enriched price boundaries.  Acting on prices outside the enriched price boundaries (i.e., hitting a bid below the boundary or lifting an ask above the boundary) will create arbitrage opportunities for market makers.

Note that this code only works for:
•	European options 
•	American options with:
   o An underlying asset that generates no intermittent cashflows (i.e. dividends and/or interest payments), and
   o Strike prices that are not so far out of the money as to be potentially exercised early

The “no arbitrage” rules being used to enrich the time values and option prices are as follows:
•	The minimum time value for an option bid should be zero 
•	For a call option and a put option at the same strike:
   o The enriched time value of each option’s bid should be the maximum of the call option’s bid time value and the put option’s bid time value
   o The enriched time value of each option’s ask should be the minimum of the call option’s ask time value and the put option’s ask time value
•	For option prices at strikes below the underlying asset price at expiry:
   o  The minimum time value bid should be the maximum time value bid of all options with lower strike prices
   o	The maximum time value ask should be the minimum time value ask of all options with strikes greater than the option strike but less than the underlying asset price at expiry
•	For option prices at strikes above the underlying asset price at expiry:
   o	The minimum time value bid should be the maximum time value bid of all options with higher strike prices
   o	The maximum time value ask should be the minimum time value ask of all options with strikes less than the option strike but greater than the underlying asset price at expiry
  
The program below takes as inputs to the formula:
•	Market prices (bids and asks) for options: 
   o	At the same expiry time 
   o	But at different strike prices
   o	Represented as a dataframe with columns as follows:
      	Bid price for call options
      	Ask price for call options
      	Strike price
      	Strike price present value (computed and stored prior to use in this code; this code does not calculate present values of strike prices) **
      	Bid price for put options
      	Ask price for put options
•	The spot price of the underlying asset

The program breaks each option price into a dataframes of:
•	Intrinsic values 
   o	For calls: max(0, underlying asset price – present value of strike price)
   o	For puts:  max(0, present value of strike price – underlying asset price)
•	Time values (option prices – intrinsic values)

The time values are then enriched via the “no arbitrage” rules above and stored in a dataframe (same structure as initial input dataframe).

Finally, the enriched time values are combined with the original intrinsic values to create enriched options prices (bid and asks) in an output dataframe of the same structure as the initial input dataframe. 

** The present value of the strike prices must be used rather than the strike prices themselves.  This prevents the prices of deep in the money call and put spreads from being equal to the difference in the strike prices (which would be too much).  Instead, the deep in the money spreads are correctly priced at levels that reflect the present value of the strike price differences.

