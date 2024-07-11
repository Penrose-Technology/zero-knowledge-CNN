import numpy as np
import parameters as p
import sys
import random
from sympy import ntt, intt
from typing import List, Dict, Tuple
import argparse
from circuit import circuit as C
import wire_prediction_ext as Wire
import value_prediction_ext as V
import interactive_proof as I
import hadmard_product as h
import sumcheck as S
from test_data import test_data as T
import ntt_init

class ConVolution:
    def __init__(self):
        pass

    def con2d(self, a, w):
        A = np.mat(a)
        W = np.mat(w)
        U = np.mat(np.zeros((p.con_size-p.kernel_size+1, p.con_size-p.kernel_size+1), dtype='int32'))
        rows, cols = A.shape
        for i in range(rows-1):
            for j in range(cols-1):
                U[i,j] = np.multiply(A[i:(i+p.kernel_size),j:(j+p.kernel_size)], W).sum()
                #print(np.multiply(A[i:(i+kernel_size),j:(j+kernel_size)], W))
        return U

    def trim(self, raw):
        raw = raw[(p.con_size*(p.kernel_size-1)+p.kernel_size-2):(p.con_size**2)][::-1]
        result = raw.reshape((p.out_size,p.con_size))[:,0:p.out_size]
        return result

    def shape(self, a, w):
        w_bar = np.zeros_like(a)
        a_bar = a.flatten()[::-1]
        w_bar[0:p.kernel_size,0:p.kernel_size] = w
        w_bar = w_bar.flatten()
        return a_bar, w_bar

    def con1d(self, a, w):
        a_bar, w_bar = self.shape(a, w)
        u_bar = np.convolve(a_bar, w_bar)
        return self.trim(u_bar), a_bar, w_bar

    def ntt_intt(self, a_bar, w_bar):
        #np.array(ntt(a_bar, prime=self.prime))
        #np.array(ntt(w_bar, prime=self.prime))
        prod = np.array(ntt(a_bar, prime=p.prime))*np.array(ntt(w_bar, prime=p.prime))
        u_bar = np.array(intt(prod, prime=p.prime))
        return self.trim(u_bar)
    


def convolution_prover(X_i_2d: np.ndarray, w_i_2d: np.ndarray, zeta: Tuple[List[int], List[int]], \
                       r: Tuple[Tuple[List[int], List[int]], Tuple[List[int], List[int]], List[int]], mu: List[List[int]]):

    con = ConVolution()
    Pro = S.Prover()

    # Convert 2D matrix to 1D sequence
    X_i_1d, w_i_1d = con.shape(X_i_2d, w_i_2d)
    # Calculation
    gate_in_left_h = ntt(X_i_1d, prime=p.prime)
    gate_in_right_h = ntt(w_i_1d, prime=p.prime)

    # Run fft sumcheck:
    (r1, r2, r_out_h) = r
    (r1_X_i, r1_w_i) = r1
    (r2_X_i, r2_w_i) = r2
    (zeta_X_i, zeta_w_i) = zeta
    zeta_ext_X_i = ntt_init.Initilization(zeta_X_i, r1_X_i)
    zeta_ext_w_i = ntt_init.Initilization(zeta_w_i, r1_w_i)
    X_i_sum, X_i_g_t = Pro.sumcheck_ntt(X_i_1d, zeta_ext_X_i, r2_X_i)
    w_i_sum, w_i_g_t = Pro.sumcheck_ntt(w_i_1d, zeta_ext_w_i, r2_w_i)
    print(f"X_i_g_t is \n{X_i_g_t}, \n w_i_g_t is \n{w_i_g_t}")

    # Run hadmard product gkr
    gate_list_h = h.hardmard_product()
    input_data_h = [[left, right] for left, right in zip(gate_in_left_h, gate_in_right_h)]
    r1_X_i.insert(0, 0); r1_w_i.insert(0, 0)
    r_h = [[[2*i%p.prime for i in r1_X_i], [(2*i+1)%p.prime for i in r1_w_i]]]
    print(f"r_h is {r_h}")
    proof_h, output_data_h = I.generate_proof(input_data_h, gate_list_h, r_h, r_out_h, mu, only_multi=True)
    print(f"proof_h is {proof_h}")

    proof = (X_i_g_t, w_i_g_t)

    return proof

def ntt_verifier(Verifier: S.Verifier, ntt_i_g_t: np.ndarray, r2: np.ndarray):
    
    ntt_s = Verifier.sum_verify(ntt_i_g_t, r2, ntt_flag=True)
    
    try:
        if not len(ntt_s) == len(ntt_i_g_t):
            raise ValueError('Length Check Error !!!')

        #if not sum_i == (ntt_i_g_t[0][0] + ntt_i_g_t[0][1]) % p.prime:
        #    raise ValueError('Value Check Error !!! First Round Error !!!')
        for j in range(len(ntt_s)-1):
            if not ntt_s[j] == (ntt_i_g_t[j+1][0] + ntt_i_g_t[j+1][1]) % p.prime:
                raise ValueError(f'Value Check Error !!! {j+1} Round Error !!!')
        # Don't check the last extension
    except ValueError as e:
        print(f"NTT Sumcheck ERROR !!! {e}") 
        sys.exit(1)
    
    return ntt_s[-1]

def convolution_verifier(proof, r2: np.ndarray):

    Ver = S.Verifier()

    X_i_g_t, w_i_g_t = proof

    W_hadmard_in_left = ntt_verifier(Ver, X_i_g_t, r2)
    W_hadmard_in_right = ntt_verifier(Ver, w_i_g_t, r2)
    
    #I.Verifier(proof_h, vk, r_h, r_h_out, mu, output_data_h, input_data_h, only_multi=True)

def main():
    # initialize test data
    _, r1_X_i, r2_X_i, zeta_X_i, _ = T.init()
    _, r1_w_i, r2_w_i, zeta_w_i, _ = T.init()
    X_i_2d = np.random.randint(0, p.prime, size=(16, 16))    
    w_i_2d = np.array([[1,2], [3,4]], dtype='int32')

    r1 = (r1_X_i.tolist(), r1_w_i.tolist())
    print(f"r1_X_i is {r1_X_i}")
    r2 = (r2_X_i.tolist(), r2_w_i.tolist())
    zeta = (zeta_X_i.tolist(), zeta_w_i.tolist())
    r_out_h = [random.randint(0, p.prime) for _ in range(p.bitwise)]
    r = (r1, r2, r_out_h)

    #vk = I.set_up(gate_list, r_h, r_h_out, mu, only_multi=True)

    proof = convolution_prover(X_i_2d, w_i_2d, zeta, r, mu=[])

    #convolution_verifier(proof, r2)


if __name__ == '__main__':
    main()

      