from typing import List
import numpy as np
import parameters as p

def single_extension(x: int, x_idx: int, r: List[int], bitwidth: int):
    v = np.array([int(char) for char in np.binary_repr(x_idx, width=bitwidth)])
    if not len(v) == len(r):
        raise ValueError("Single Extension: length check failed !!!")
    for idx, k in enumerate(v):
        x = x * ((1-k)*(1-r[idx]) + k*r[idx]) % p.prime
    return x
