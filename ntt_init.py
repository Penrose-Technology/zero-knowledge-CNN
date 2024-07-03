from networkx import omega
import numpy as np
import parameters as p
from sympy.ntheory import n_order
import test_data as T

# Generate ntt matrix F(x,y) = omega^(xy); x,y in [0, N-1]
def ntt_mat(omega):
    F = np.ones((p.N,p.N), dtype='int32')
    for i in range(1, p.N):
        F[i][1] = F[i-1][1]*omega % p.prime 
        for j in range(1, p.N):
            F[i][j] = F[i][j-1]*F[i][1] % p.prime 
    return F

# Generate the multilinear extension A_F(r,y) of ntt matrix F(x,y)       
def Initilization(A, r):
    A_F = np.ones(p.N, dtype='int32')
    for i in range(p.bitwise):
        #inverse loop
        for j in range(2**(i+1)-1, -1, -1):   
            A_F[j] = A_F[j % (2**i)] * ((1-r[i])+r[i]*A[j*(p.N>>(i+1))]) % p.prime
    return A_F

def test_A_F(F, r):
    A_F = np.zeros(p.N, dtype='int32')
    for j in range(p.N):
        for i in range(p.N):
            v = np.array([int(char) for char in np.binary_repr(i, width=p.bitwise)])
            for idx,k in enumerate(v):
                F[i][j] = F[i][j] * ((1-k)*(1-r[idx]) + k*r[idx]) % p.prime
            A_F[j] = (A_F[j] + F[i][j]) % p.prime
    return A_F

def main():
    _, r1, _, A, inv_A = T.init(omega_test=True)
    r = r1
    # All value should be equal(before extension = after extention) if we check the point F(1,y) 
    A_F = Initilization(A, r)
    print(f"our result is \n{A_F}")
    #A_F_t = test_A_F(F, r)
    #print(f"reference result is \n{A_F_t}")

    try:
        if not (A_F==A).all():
            raise ValueError('test failed, not equal !!!')
    except ValueError as e:
        print(str(e))

if __name__ == "__main__":
    main()

