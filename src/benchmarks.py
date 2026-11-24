"""Temporary benchmark functions.
"""
import numpy as np


def sphere(x):
    return np.sum(x ** 2)


def rastrigin(x):
    return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))


def ackley(x):
    n = len(x)
    return (-20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / n))
            - np.exp(np.sum(np.cos(2 * np.pi * x)) / n) + 20 + np.e)


FUNCTIONS = {
    "sphere": (sphere, (-5.12, 5.12)),
    "rastrigin": (rastrigin, (-5.12, 5.12)),
    "ackley": (ackley, (-32.768, 32.768)),
}