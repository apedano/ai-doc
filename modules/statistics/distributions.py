import sympy as sp


def normal(x, mu, sigma):
    return (
        1 / (sp.sqrt(2 * sp.pi) * sigma)
        * sp.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
    )