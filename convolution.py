import numpy as np
import parameters as p
import sys
import random
from sympy import ntt, intt
from typing import List, Tuple
import argparse
from utilities import gadget as g
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
                U[i,j] = np.multiply(A[i:(i+p.kernel_size),j:(j+p.kernel_size)], W).sum() % p.prime
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
    


def convolution_prover(Pro: S.Prover, Ver: S.Verifier, \
                       X_i_1d: np.ndarray, w_i_1d: np.ndarray, zeta: List[int], zeta_inv: List[int],  \
                       r: Tuple[List[int], List[int], List[int], List[int]], mu: List[List[int]]):

    ####################################################
    # Run ntt sumcheck
    ####################################################
    # ntt calculation
    gate_in_left_h = ntt(X_i_1d, prime=p.prime)
    gate_in_right_h = ntt(w_i_1d, prime=p.prime)

    # Parse random check points:
    (r_hntt, r_ntt, r_hintt, r_intt) = r

    zeta_ext_X = ntt_init.Initilization(zeta, r_hntt)
    zeta_ext_w = zeta_ext_X.copy()
    F_zeta_ext = Ver.multi_ext(zeta_ext_X, r_ntt)

    X_i_sum, X_i_g_t = Pro.sumcheck_ntt(X_i_1d, zeta_ext_X, r_ntt)
    w_i_sum, w_i_g_t = Pro.sumcheck_ntt(w_i_1d, zeta_ext_w, r_ntt)

    # ntt extension at r_hntt
    F_X_ntt_in_ext = Ver.multi_ext(gate_in_left_h, r_hntt)
    F_w_ntt_in_ext = Ver.multi_ext(gate_in_right_h, r_hntt)

    assert (X_i_sum == F_X_ntt_in_ext and w_i_sum == F_w_ntt_in_ext),   \
        f'NTT Check Error! NTT input sumcheck: \n X_i_sum = {X_i_sum}, F_X_ntt_in_ext = {F_X_ntt_in_ext} \n \
            w_i_sum = {w_i_sum}, F_w_ntt_in_ext = {F_w_ntt_in_ext}'
    
    
    ####################################################
    # Run hadmard product gkr
    ####################################################
    gate_list_h = h.hardmard_product()
    input_data_h = [[left, right] for left, right in zip(gate_in_left_h, gate_in_right_h)]
    # Use the same random point (r_hntt) at input layer
    r_h_in = [[r_hntt, []]]
    # Use the same random point (r_hintt) at outnput layer
    r_h_out = r_intt
    proof_h, output_data_h = I.generate_proof(input_data_h, gate_list_h, r_h_in, r_h_out, mu, only_multi=True)
    #print(f"proof_h is {proof_h}")
    #print(f"output_data_h is: \n {output_data_h}")

    F_W_in_ext, _, _, _ = proof_h
    assert (F_W_in_ext[0] == F_X_ntt_in_ext  and  F_W_in_ext[1] == F_w_ntt_in_ext),   \
        f'Random Point Check Error! Hadmard input layer: \n F_W_in_ext_left = {F_W_in_ext[0]}, F_X_ntt_in_ext = {F_X_ntt_in_ext} \n  \
            F_W_in_ext_right = {F_W_in_ext[1]}, F_w_ntt_in_ext = {F_w_ntt_in_ext}'


    ####################################################
    # Run intt sumcheck
    ####################################################
    # intt calculation
    X_o_1d = intt(output_data_h, prime=p.prime)
    zeta_inv_ext_X = ntt_init.Initilization(zeta_inv, r_hintt)
    
    F_W_out_ext = Ver.multi_ext(output_data_h, r_intt, width=len(r_intt))
    F_zeta_inv_ext = Ver.multi_ext(zeta_inv_ext_X, r_intt, width=len(r_intt))
    #F_intt_in_ext = (F_W_out_ext*F_zeta_inv_ext*p.N_inv) % p.prime
    #print(f"debug:: F_intt_in_ext is {F_intt_in_ext}")

    X_o_sum, X_o_g_t = Pro.sumcheck_ntt(output_data_h, zeta_inv_ext_X, r_intt, inverse=True)

    # intt extension at r_hintt
    F_X_intt_out_ext = Ver.multi_ext(X_o_1d, r_hintt, width=len(r_hintt))

    assert (X_o_sum == F_X_intt_out_ext),  \
        f'INTT Check Error! INTT input sumcheck: \n X_o_sum = {X_o_sum}, F_X_intt_out_ext = {F_X_intt_out_ext}'

    extension_values = (F_W_out_ext, F_zeta_ext, F_zeta_inv_ext)
    proof = (X_i_g_t, w_i_g_t, proof_h, X_o_g_t, extension_values)

    return proof, X_o_1d








def convolution_verifier(Ver: S.Verifier, vk, proof, in_ext, out_ext, r):

    # Parse proof
    X_i_g_t, w_i_g_t, proof_h, X_o_g_t, extension_values = proof
    # F_W_in_ext used for ntt sum
    F_W_in_ext, _, _, _ = proof_h
    # F_W_out_ext and F_zeta_inv_ext used for intt last extension
    (F_W_out_ext, F_zeta_ext, F_zeta_inv_ext) = extension_values

    # Parse random check points
    (r_hntt, r_ntt, r_hintt, r_intt) = r

    ####################################################
    # Run intt sumcheck
    ####################################################
    X_o_s = Ver.sum_verify(X_o_g_t, r_intt, ntt_flag=True)
    X_o_sum = out_ext
    # Final output check,
    # Verifier does this check:
    # hadmard output(intt input): F_W_out(u*);  intt matrix: F_zeta_inv(z*, u*);    
    # intt_out_ext calculated by Verifier (extension at random point (z*, u*)),
    # Relationship should hold: F_W_out(u*) * F_zeta_inv(z*, u*) *N_inv = intt_out_ext
    X_o_ext = (F_W_out_ext*F_zeta_inv_ext*p.N_inv) % p.prime
    g.sumcheck_verification(X_o_s, X_o_sum, X_o_g_t, X_o_ext, tag="X_o INTT")

    
    ####################################################
    # Run hadmard product gkr
    ####################################################
    # Use the same random point (r_hntt) at input layer
    r_h_in = [[r_hntt, []]]
    I.Verifier(proof_h, vk, r_h_in, [], F_W_out_ext, only_multi=True)


    ####################################################
    # Run intt sumcheck
    ####################################################
    # X_i_1d check
    X_i_s = Ver.sum_verify(X_i_g_t, r_ntt, ntt_flag=True)
    X_i_sum = F_W_in_ext[0]
    X_i_ext = (in_ext[0]*F_zeta_ext) % p.prime
    g.sumcheck_verification(X_i_s, X_i_sum, X_i_g_t, X_i_ext, tag="X_i NTT")

    # w_i_1d check
    w_i_s = Ver.sum_verify(w_i_g_t, r_ntt, ntt_flag=True)
    w_i_sum = F_W_in_ext[1]
    w_i_ext = (in_ext[1]*F_zeta_ext) % p.prime
    g.sumcheck_verification(w_i_s, w_i_sum, w_i_g_t, w_i_ext, tag="w_i NTT")   

    return 0



def main():
    # initialize test data
    _, _, _, zeta, zeta_inv = T.init()
    con = ConVolution()
    Ver = S.Verifier()
    Pro = S.Prover()

    X_i_2d = np.random.randint(0, p.prime, size=(p.con_size, p.con_size))    
    w_i_2d = np.array([[1,2], [3,4]], dtype='int32')

    # Generate random points
    r_hntt = [random.randint(0, p.prime) for _ in range(p.bitwise)]
    r_hintt = [random.randint(0, p.prime) for _ in range(p.bitwise)]
    r_ntt = [random.randint(0, p.prime) for _ in range(p.bitwise)]
    r_intt = [random.randint(0, p.prime) for _ in range(p.bitwise)]
    r = (r_hntt, r_ntt, r_hintt, r_intt)

    # Convert 2D matrix to 1D sequence
    X_i_1d, w_i_1d = con.shape(X_i_2d, w_i_2d)

    # Input data extension at a random point
    in_X_ext = Ver.multi_ext(X_i_1d, r_ntt, width=len(r_ntt))
    in_w_ext = Ver.multi_ext(w_i_1d, r_ntt, width=len(r_ntt))
    in_ext = (in_X_ext, in_w_ext)
    
    # Setup
    vk = I.set_up(gate_list=h.hardmard_product(), r=[[r_hntt, []]], r_out=r_intt, mu=[], only_multi=True)

    # Proving and Calculation
    proof, X_o_1d = convolution_prover(Pro, Ver, X_i_1d, w_i_1d, zeta.tolist(), zeta_inv.tolist(), r, mu=[])

    # Output data extension at a random point
    out_ext = Ver.multi_ext(X_o_1d, r_hintt, width=len(r_hintt))

    # Verification
    convolution_verifier(Ver, vk, proof, in_ext, out_ext, r)

    # Convert output 1D sequence to 2D matrix 
    X_o_2d = con.trim(np.array(X_o_1d))

    # Reference output
    X_o_2d_ref = con.con2d(X_i_2d, w_i_2d)
    print(f"X_o_2d is: \n {X_o_2d}")
    print(f"X_o_2d_ref is: \n {X_o_2d_ref}")


if __name__ == '__main__':
    main()

      