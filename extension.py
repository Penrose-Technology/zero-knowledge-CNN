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

def mod_inverse(a, m):
    # Extended Euclidean algorithm to find the modular inverse of a under modulo m
    m0, x0, x1 = m, 0, 1
    if m == 1:
        return 0
    while a > 1:
        q = a // m
        m, a = a % m, m
        x0, x1 = x1 - q * x0, x0
    if x1 < 0:
        x1 += m0
    return x1

def gaussian_elimination(A, b, mod):
    n = len(A)
    
    # Augmented matrix [A | b]
    aug_matrix = [[A[i][j] for j in range(n)] + [b[i]] for i in range(n)]
    
    for i in range(n):
        # Partial pivoting
        max_row = max(range(i, n), key=lambda r: abs(aug_matrix[r][i]))
        aug_matrix[i], aug_matrix[max_row] = aug_matrix[max_row], aug_matrix[i]
        
        # Make the diagonal element 1
        inv = mod_inverse(aug_matrix[i][i], mod)
        for j in range(i, n + 1):
            aug_matrix[i][j] = (aug_matrix[i][j] * inv) % mod
        
        # Eliminate the lower elements
        for j in range(n):
            if i != j:
                ratio = aug_matrix[j][i]
                for k in range(i, n + 1):
                    aug_matrix[j][k] = (aug_matrix[j][k] - ratio * aug_matrix[i][k]) % mod
    
    # Extract solution vector
    solution = [row[-1] for row in aug_matrix]
    return solution

if __name__ == "__main__":
    mod = p.prime
    A = [[3, 4], [2, 1]]
    b = [1, 3]
    
    solution = gaussian_elimination(A, b, mod)
    print("Solution:", solution)
