from typing import List
import numpy as np
import parameters as p

def single_extension(x: int, x_idx: int, r: List[int], bitwise: int):
    v = np.array([int(char) for char in np.binary_repr(x_idx, width=bitwise)])
    if not len(v) == len(r):
        raise ValueError("Single Extension: length check failed !!!")
    for idx, k in enumerate(v):
        x = x * ((1-k)*(1-r[idx]) + k*r[idx]) % p.prime
    return x

def binary_to_decimal(binary_list):
    decimal_number = 0
    length = len(binary_list)
    for i in range(length):
        decimal_number += binary_list[i] * (2 ** (length - 1 - i))
    return decimal_number