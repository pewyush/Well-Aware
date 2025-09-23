import numpy as np
import scipy.stats as si

def black_scholes(S, K, T, r, sigma, option="call"):
    """
    Black-Scholes option pricing model.
    S : spot price
    K : strike price
    T : time to maturity (years)
    r : risk-free rate
    sigma : volatility
    option : "call" or "put"
    """
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    if option == "call":
        return (S * si.norm.cdf(d1, 0.0, 1.0) -
                K * np.exp(-r*T) * si.norm.cdf(d2, 0.0, 1.0))
    elif option == "put":
        return (K * np.exp(-r*T) * si.norm.cdf(-d2, 0.0, 1.0) -
                S * si.norm.cdf(-d1, 0.0, 1.0))
