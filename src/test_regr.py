import gd
from functools import *
import numpy as np

def residuals(betas, x, y):
    X = np.full((x.size, 2), 1.0)
    X[:,1] = x

    return y - np.matmul(X, betas) 

def ssr_grad(betas, x, y):
    r = residuals(betas, x, y)
    return [-2 * r.sum(), -2 * np.dot(r, x)]

def ssr_loss(betas, x, y):
    r = residuals(betas, x, y)
    return np.dot(r, r)

if __name__ == "__main__":
    x = np.linspace(0.0, 16.0, 8)
    y = 1 + 2 * x

    search = gd.gradient_descent(
            gradient = lambda b : ssr_grad(b, x, y), 
            start = [0.1, 3.4], 
            learn_rate = 0.0001, 
            n_iter = 100000)
    for it in search:
        print(f'{it.i} {it.step} {it.result}')
