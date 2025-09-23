import numpy as np

def binomial_tree_pricing(S, K, T, r, sigma, N=100, option="call"):
    """
    Binomial Tree option pricing model.
    S : spot price
    K : strike price
    T : time to maturity (years)
    r : risk-free rate
    sigma : volatility
    N : number of steps
    option : "call" or "put"
    """
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))    # up factor
    d = 1 / u                          # down factor
    p = (np.exp(r*dt) - d) / (u - d)   # risk-neutral probability

    # stock prices at maturity
    ST = np.array([S * (u**j) * (d**(N-j)) for j in range(N+1)])

    # option payoff at maturity
    if option == "call":
        payoff = np.maximum(ST - K, 0)
    else:
        payoff = np.maximum(K - ST, 0)

    # backward induction
    for i in range(N-1, -1, -1):
        payoff = np.exp(-r*dt) * (p * payoff[1:] + (1-p) * payoff[:-1])

    return payoff[0]
