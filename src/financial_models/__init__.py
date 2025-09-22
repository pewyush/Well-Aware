from .monte_carlo import monte_carlo_simulation
from .black_scholas import black_scholes
from .binomial_tree_pricing import binomial_tree_pricing


MODELS = {
    "Monte Carlo": monte_carlo_simulation,
    "Black-Scholes": black_scholes,
    "Binomial Tree": binomial_tree_pricing,
}

