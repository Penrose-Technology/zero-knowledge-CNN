import sys
import argparse
import numpy as np
from sympy import ntt, intt
import parameters as p
from test_data import test_data as T
import ntt_init

class Prover:
    def __init__(self):
        pass

    # g: The original polynomial which prover wants to prove.
    # r: Random value choosed by verifier. (random check point at each iteration.)
    # A_F: Storing value for the rest unextended bits in each iteration.
    # g_t: Storing sumation(s(0), s(1), s(2)) at x=0, x=1, x=2 at each iteration.
    def sumcheck(self, A_F, r, width=p.bitwise):
        sum = A_F.sum()%p.prime
        g_t = np.zeros((width,3), dtype='int32')
        for i in range(width):
            for b in range(2**(width-i-1)):
                for t in range(3):
                    g_t[i,t] = ( g_t[i,t] + A_F[b]*(1 - t) + A_F[b+2**(width-i-1)]*t ) % p.prime
                A_F[b] = ( A_F[b]*(1 - r[i]) + A_F[b+2**(width-i-1)]*r[i] ) % p.prime       
        return sum, g_t, A_F
    
    # Sumcheck for NTT transformation: s = c*F
    def sumcheck_ntt(self, A_c, A_F, r, inverse=False, width=p.bitwise):
        if inverse:
            sum = (A_c*A_F).sum()*p.N_inv % p.prime
        else:
            sum = (A_c*A_F).sum() % p.prime 

        g_t = np.zeros((width,3), dtype='int32')
        for i in range(width):
            for b in range(2**(width-i-1)):
                for t in range(3):
                    F_t = ( A_F[b]*(1 - t) + A_F[b+2**(width-i-1)]*t ) % p.prime 
                    c_t = ( A_c[b]*(1 - t) + A_c[b+2**(width-i-1)]*t ) % p.prime
                    g_t[i,t] = ( g_t[i,t] + F_t*c_t ) % p.prime
                A_F[b] = ( A_F[b]*(1 - r[i]) + A_F[b+2**(width-i-1)]*r[i] ) % p.prime
                A_c[b] = ( A_c[b]*(1 - r[i]) + A_c[b+2**(width-i-1)]*r[i] ) % p.prime

        if inverse:
           g_t = p.N_inv*g_t % p.prime

        return sum, g_t


class Verifier:
    def __init__(self):
        pass

    # Solve linear equation y = k+ bx.
    #   e.g.    1 1       k       g(1)
    #         (     ) * (   ) = (      )
    #           1 2       b       g(2)
    def solve_linear(self, a, r):
        coefficient = np.array([[1,1],[1,2]])
        dependcy = np.array([a[1],a[2]])
        x = np.linalg.solve(coefficient, dependcy)
        s = x[0] + x[1]*r
        return x%p.prime, s%p.prime
    
    # Solve quadratic equation y = c + bx + ax^2
    #   e.g.    1 0 0         c       g(0)
    #         ( 1 1 1   ) * ( b ) = ( g(1) )
    #           1 2 2^2       a       g(2)
    def solve_quadratic(self, a, r):
        coefficient = np.array([[1,0,0],[1,1,1],[1,2,4]])
        dependcy = np.array([a[0],a[1],a[2]])
        x = np.linalg.solve(coefficient, dependcy)
        s = x[0] + x[1]*r + x[2]*(r**2)
        return x%p.prime, s%p.prime

    # Verifier independly calculate s(r) with 3 points from prover (e.g. s(0), s(1), s(2)).
    # Note that s(x) is a linear equation or quadratic equation.
    # For the first iteration, check H ?= s_1 (0) + s_1 (1)
    # For next 2 ~ (l-1) iterations, check s_(i) (r_(i)) ?= s_(i+1) (0) + s_(i+1) (1)
    def sum_verify(self, g_t, r, width=p.bitwise, ntt_flag=False):
        s = np.zeros(width, dtype='int32')
        if ntt_flag:
            for i in range(width):
                _, s[i] = self.solve_quadratic(g_t[i], r[i])
        else:
            for i in range(width):
                _, s[i] = self.solve_linear(g_t[i], r[i])
        return s

    # For the last(l) iteration, verifier needs to calculate the summation of s'_l (r) by his own,
    # then check s'_l (r) ?= s_l (r_l)
    def multi_ext(self, g, r, width=p.bitwise):
        gv = np.copy(g)
        for i in range(2**width):
            v = np.array([int(char) for char in np.binary_repr(i, width=width)])
            for idx,k in enumerate(v):
                gv[i] = (gv[i] * ((1-k)*(1-r[idx]) + k*r[idx])) % p.prime
        return gv.sum()%p.prime
    
    
def sumcheck_verify(g, r):
    # Proving Process
    P = Prover()
    g_p = g.copy()
    sum, g_t, _ = P.sumcheck(g_p, r)
    print(f"prover sends initial sumation: \n {sum}")
    print(f"prover sends g_t: \n {g_t}")

    # Verification Process
    V = Verifier()
    s = V.sum_verify(g_t, r)
    last = V.multi_ext(g, r)
    print(f"verifier output: \n{s}")
    print(f"verifier last random oracle query: \n{last}")

    try:
        if not sum == (g_t[0][0] + g_t[0][1]) % p.prime:
            raise ValueError('first round error !!!')
        for i in range(p.bitwise-1):
            if not s[i] == (g_t[i+1][0] + g_t[i+1][1]) % p.prime:
                raise ValueError(f'{i+1} round error !!!')
        if not last == s[p.bitwise-1]:
            raise ValueError('last round error !!!')
    except ValueError as e:
        print(str(e))    

def sumcheck_ntt_verify(c, A, r1, r2, inverse=False):

    A_F = ntt_init.Initilization(A, r1)    
    c_p = c.copy()
    A_F_p = A_F.copy()

    # Proving Process
    P = Prover()    
    sum, g_t = P.sumcheck_ntt(c_p, A_F_p, r2, inverse=inverse)
    print(f"prover sends initial sumation: \n {sum}")
    print(f"prover sends g_t: \n {g_t}")
    if inverse:
        ref = np.array(intt(c, prime=p.prime))
    else:
        ref = np.array(ntt(c, prime=p.prime))

    # Verification Process
    V = Verifier()
    s = V.sum_verify(g_t, r2, ntt_flag=True)
    # Verifier calculates extended(A_F)*extended(c) on his own
    # TODO
    last = V.multi_ext(c, r2)*V.multi_ext(A_F, r2) % p.prime
    if inverse:
        last = last*p.N_inv % p.prime
    print(f"verifier output: \n{s}")
    print(f"verifier last random oracle query: \n{last}")

    # Extended(ntt/intt_output) at r1 should equal to the original summation of c*A_F 
    out_ext_r1 = V.multi_ext(ref, r1)
    print(f"ntt/intt output result at r1 is {out_ext_r1}, original summation is {sum}")

    try:
        if not sum == (g_t[0][0] + g_t[0][1]) % p.prime:
            raise ValueError('first round error !!!')
        for i in range(p.bitwise-1):
            if not s[i] == (g_t[i+1][0] + g_t[i+1][1]) % p.prime:
                raise ValueError(f'{i+1} round error !!!')
        if not last == s[p.bitwise-1]:
            raise ValueError('last round error !!!')
        if not out_ext_r1 == sum:
            raise ValueError('ntt/intt result error !!!')
    except ValueError as e:
        print(str(e))
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='specify sumcheck operation')
    parser.add_argument("--operation", '-o', choices=['ntt', 'intt', 'single'], default='single')
    args = parser.parse_args()

    # initialize test data
    c, r1, r2, A, inv_A = T.init()

    if args.operation == 'ntt':
        sumcheck_ntt_verify(c, A, r1, r2)
    elif args.operation == 'intt':
        sumcheck_ntt_verify(c, inv_A, r1, r2, inverse=True)
    else:
        sumcheck_verify(c, r2)


if __name__ == "__main__":
    main()
    