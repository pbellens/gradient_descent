import numpy as np
from collections import namedtuple

Iteration = namedtuple('Iteration', ['i', 'step', 'result'])

def gradient_descent(gradient, start, learn_rate, n_iter = 1e3, tolerance = 1e-6):
    yield Iteration(i = 0, step = 0, result = start)

    result = start
    for i in range(1, n_iter + 1):
        step = learn_rate * np.array(gradient(result))
        if np.all(np.abs(step) <= tolerance):
            break

        result -= step

        yield Iteration(i = i, step = step, result = result)

    return result
