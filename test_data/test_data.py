import parameters as p
import numpy as np
from sympy.ntheory import n_order

def n_order_verification():
    print(f"order is {n_order(p.n_root, p.prime)}, prime is {p.prime}")
    return n_order(p.n_root, p.prime) == p.prime-1

def calculate_omega():
    return int(p.n_root**((p.prime-1)//p.N) % p.prime)

# Generate ntt vector for a N-degree polynomial.    
def N_roots_of_unit(omega, inverse=False):
    A = np.ones(p.N, dtype='int32')
    for i in range(1, p.N):
        A[i] = A[i-1]*omega % p.prime
    #print(A)
    if inverse:
        for i in range(1, (p.N>>1)):
            A[i], A[p.N-i] = A[p.N-i], A[i]
        #print(A)
    return A

# Generate ntt matrix F(x,y) = omega^(xy); x,y in [0, N-1]
def ntt_mat(omega):
    F = np.ones((p.N,p.N), dtype='int32')
    for i in range(1, p.N):
        F[i][1] = F[i-1][1]*omega % p.prime 
        for j in range(1, p.N):
            F[i][j] = F[i][j-1]*F[i][1] % p.prime 
    return F

# Generate n-th roots
def ntt_root_init():
    omega = calculate_omega()
    A = N_roots_of_unit(omega)
    return A

# Generate n-th inverse roots
def intt_root_init():
    omega = calculate_omega()
    A = N_roots_of_unit(omega, inverse=True)
    return A

def init(omega_test=False):
    c = np.array([int(x) for x in range(p.N)])
    if omega_test:
        r1 = np.zeros(p.bitwise)
        r1[p.bitwise-1] = 1
    else:
        r1 = np.random.randint(p.prime, size=p.bitwise)
    r2 = np.random.randint(p.prime, size=p.bitwise)
    A = ntt_root_init()
    inv_A = intt_root_init()
    return c, r1, r2, A, inv_A

def main():
    n_order_verification()
    omega = calculate_omega()
    print(f"polynomial degree is {p.N}, primitive root omega is {omega}")
    c, r1, r2, A, inv_A = init(omega_test=True)
    print(f"ntt roots of unit vector is \n{A}")
    print(f"intt roots of unit vector is \n{inv_A}")
    print(f"input vector is \n{c}")
    print(f"random vector r1 is \n{r1} \nrandom vector r2 is \n{r2}")
    #F = ntt_mat(omega)
    #print(f"ntt roots of unit matrix is \n{F}")


if __name__ == "__main__":
    main()