import numpy as np
from pkg_resources import invalid_marker
import parameters as p
import test_data as T
import ntt_init
from sumcheck import Prover, Verifier
from sympy import ntt, intt

import sumcheck

def _ntt_run(A, c, r1, r2, inverse=False):
    #initialize ntt multilinear extension A_F(r,y)
    A_F = ntt_init.Initilization(A, r1)
    #print(f"A_F=\n{A_F}\n")
    #run sumcheck for ntt transformation
    P = Prover()
    sum, g_t, A_s = P.sumcheck_ntt(c, A_F, r2, inverse=inverse)
    return sum, g_t


def ntt_run(c, r1, r2, A):
    print(f"r1=\n{r1} \nr2=\n{r2} \nA=\n{A} \nc=\n{c}")
    sum, g_t = _ntt_run(A, c, r1, r2)
    # proof check

    # ntt evluation
    out_ntt = np.array(ntt(c, prime=p.prime))
    print(f"ref_ntt=\n{out_ntt}")
    return out_ntt, sum, g_t

def intt_run(c, r1, r2, inv_A):
    print(f"r1=\n{r1} \nr2=\n{r2} \ninv_A=\n{inv_A} \nc=\n{c}")
    P = Prover()
    sum, g_t = _ntt_run(inv_A, c, r1, r2, inverse=True)
    # proof check
    #sumcheck.sumcheck_ntt_verify(inv_A, c, r2, inverse=True)

    # intt evluation
    out_intt = np.array(intt(c, prime=p.prime))
    print(f"ref_intt=\n{out_intt}")
    return out_intt, sum, g_t    

def main():
    c, r1, r2, A, inv_A = T.init()
    ntt_run(c, r1, r2, A)
    #intt_run(c, r1, r2, inv_A)
    #a=np.array(intt(np.array(ntt(c, prime=p.prime)), prime=p.prime))
    #print(a)

if __name__ == "__main__":
    main()

    
