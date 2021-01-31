import gd
from functools import *

if __name__ == "__main__":
    f = lambda x : x * x
    g = lambda x : 2 * x

    print('descending')
    r = reduce(
            lambda v, e : f'{v}\n{e.i} {e.step} {e.result}',
            gd.gradient_descent(g, 4, 0.8, 1000), '')
    print(f'{r}')

