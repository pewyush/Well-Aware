import numpy as np

def monte_carlo_simulation(S0, mu, sigma, T, n_simulations=1000, n_steps=252):
    """
    Monte Carlo simulation for stock prices.
    S0 : initial stock price
    mu : expected return
    sigma : volatility
    T : time horizon (in years)
    n_simulations : number of paths
    n_steps : number of steps per year
    """
    dt = T / n_steps
    prices = np.zeros((n_steps, n_simulations))
    prices[0] = S0

    for t in range(1, n_steps):
        rand = np.random.standard_normal(n_simulations)
        prices[t] = prices[t-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*rand)

    return prices
