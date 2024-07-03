import sys
import numpy as np
import parameters as p
from typing import List, Dict, Tuple
import argparse
from circuit import circuit as C
import wire_prediction_ext as Wire
import value_prediction_ext as V
import sumcheck as S

Ver = S.Verifier()
Pro = S.Prover()

def generate_proof(input_data, gate_list: List[List[str]], r: List[List[List[int]]], r_f: List[int]):
    cir = C.circuit(np.array(input_data), gate_list)
    map, final_out = cir.circuit_imp()
    cir.print_map(map)

    # Get random check point of F_W
    r_b: List[List[int]] = []
    r_c: List[List[int]] = []
    for r_i in r:
        r_b.append(r_i[1])
        r_c.append(r_i[2])
    r_b.append(r_f)
    r_c.append(r_f)

    # Get gate value F_W
    F_W = C.get_F_W(map, final_out)

    # Get multilinear extension of F_W at a random point
    F_W_b_ext = V.get_F_W_ext(F_W, r_b)
    F_W_c_ext = V.get_F_W_ext(F_W, r_c)
    #print(f"F_W_b_ext is {F_W_b_ext}")
    #print(f"F_W_c_ext is {F_W_c_ext}")

    # Get wire prediction F_add and F_multi
    F_add = C.get_F_gate('ADD', gate_list)
    F_multi = C.get_F_gate('MULTI', gate_list)

    # Get g_t(0), g_t(1), g_t(2) and sumation for f(b, c)
    # f(b, c) = add(a*, b, c)*W_add(b, c) + multi(a*, b, c)*W_multi(b, c)
    F_sum, F_ext_g_t = Wire.get_F_gate_ext_g_t(F_add, F_multi, r, F_W)

    # Assemble proof
    F_W_b_proof = F_W_b_ext
    F_W_c_proof = F_W_c_ext
    F_proof = (F_sum, F_ext_g_t)

    proof = (F_W_b_proof, F_W_c_proof, F_proof)

    return proof


def get_s(g_t_i: np.ndarray, r_i: List[int]):

    s_i = Ver.sum_verify(g_t_i, r_i, width=len(r_i), ntt_flag=True)

    print(f"verifier output, s is: \n{s_i}")
    return s_i

def linear_check(s_i: np.ndarray, sum_i: int, g_t_i: List[List[int]], ext_i: int):

    try:
        if not len(s_i) == len(g_t_i):
            raise ValueError('Length Check Error !!!')

        if not sum_i == (g_t_i[0][0] + g_t_i[0][1]) % p.prime:
            raise ValueError('Value Check Error !!! First Round Error !!!')
        for j in range(len(s_i)-1):
            if not s_i[j] == (g_t_i[j+1][0] + g_t_i[j+1][1]) % p.prime:
                raise ValueError(f'Value Check Error !!! {j+1} Round Error !!!')
        if not s_i[len(s_i)-1] == ext_i:
            raise ValueError('Value Check Error !!! Last Round Error !!!')
    except ValueError as e:
        print(str(e)) 
        sys.exit(1)

    return 0


def verification_i(sum_i: int, g_t_i: List[List[int]], ext_i: int, r_i: List[int]):

    s_i = get_s(np.array(g_t_i), r_i)

    linear_check(s_i, sum_i, g_t_i, ext_i)

    #print(f"extension at {r_i} is {ext_i}")
    return 0

# Verifier should evaluate add(a*, b*, c*) and multi(a*, b*, c*) on his own.
def set_up(gate_list: List[List[str]], r: List[List[List[int]]]):

    # Get wire prediction F_add and F_multi
    F_add = C.get_F_gate('ADD', gate_list)
    F_multi = C.get_F_gate('MULTI', gate_list)

    # Extension at a random point
    F_add_ext = Wire.get_F_gate_ext('ADD', F_add, r)
    F_multi_ext = Wire.get_F_gate_ext('MULTI', F_multi, r)

    vk = (F_add_ext, F_multi_ext)

    return vk

def Verifier(proof, vk, r: List[List[List[int]]], r_out: List[int], W_out: List[int]):

    # Parse proof
    F_W_b_proof, F_W_c_proof, F_proof = proof

    # Verifier receives W(b*), W(c*) from Prover.
    F_W_b_ext = F_W_b_proof
    F_W_c_ext = F_W_c_proof
    print(f"F_W_b_ext is {F_W_b_ext}, F_W_c_ext is {F_W_c_ext}")

    # Verifier receives sumcheck proof of Sum = Sigma_f(b, c) from Prover.
    (F_sum, F_ext_g_t) = F_proof

    # Parse vk
    # Verifier calculates add(a*, b*, c*) and multi(a*, b*, c*).
    (F_add_ext, F_multi_ext) = vk

    # Get random check point (b*, c*).
    r_bc = [(r[i][1] + r[i][2]) for i in range(len(r))]

    # Evaluate total layer number
    d = len(F_sum)-1

    # Output layer check
    # Verifier evaluates W_out(z*) on his own from output layer (d-th layer).
    F_W_out_ext = Ver.multi_ext(W_out, r_out, width=len(r_out))
    print(f"F_W_out_ext is {F_W_out_ext}")
    F_ext = (  F_add_ext[d]*(F_W_b_ext[d] + F_W_c_ext[d]) + F_multi_ext[d]*(F_W_b_ext[d] * F_W_c_ext[d])  ) % p.prime
    verification_i(F_W_out_ext, F_ext_g_t[d], F_ext, r_bc[d])
    print(f">>Output Layer Check Pass......")

    if d == 0:
        # Circuit only has one layer
        pass

    else:
        # Circuit at least has two layers
        for i in range(d-1,-1,-1):
            # Verifier evaluates f(b*, c*) for the last round of sumcheck, with
            # W(b*), W(c*) received from Prover and, 
            # add(a*, b*, c*), multi(a*, b*, c*) calculated by Verifier.
            # f(b*, c*) = add(a*, b*, c*)*(W(b*) + W(c*)) + multi(a*, b*, c*)*(W(b*) * W(c*))
            F_ext = (  F_add_ext[i]*(F_W_b_ext[i] + F_W_c_ext[i]) + F_multi_ext[i]*(F_W_b_ext[i] * F_W_c_ext[i])  ) % p.prime
            print(f"F_ext is {F_ext}")

            # Run sumcheck protocol
            verification_i(F_sum[i], F_ext_g_t[i], F_ext, r_bc[i])
            print(f"F_sum is {F_sum[i]}, F_ext_g_t is {F_ext_g_t[i]}")






def main():
    # circuit inilization
    input_data = np.array(
                 [[1,3],
                  [2,2],
                  [4,5],
                  [3,2]])
    gate_list = [['add', 'multi', 'add', 'add'],
                 ['add', 'multi']]
                 #['add']]

    final_out = [8, 45]
    
    #cir = C.circuit(input_data, gate_list)
    #map, final_out = cir.circuit_imp()
    #cir.print_map(map)
    #print(f"The circuit final output is {final_out}")

    # get random value
    r = [[[40, 41], [51, 52, 53], [61, 62, 63]],
         [[70], [80, 81], [90, 91]]]
    #r = [[[40, 41], [51, 52, 53], [61, 62, 63]],
    #     [[70], [40, 41], [40, 41]]]

    r_f = [70]

    # set up
    vk = set_up(gate_list, r)
    
    # Proving
    proof = generate_proof(input_data, gate_list, r, r_f)
    
    # Verification
    """
    for i, r_i in enumerate(r_w):
        width = len(r_i)

        # check F_W
        s = verification(F_W_ext_g_t[i], r_i)

        try:
            if not F_W_sum[i] == (F_W_ext_g_t[i][0][0] + F_W_ext_g_t[i][0][1]) % p.prime:
                raise ValueError('add: first round error !!!')
            for j in range(width-1):
                if not s[j] == (F_W_ext_g_t[i][j+1][0] + F_W_ext_g_t[i][j+1][1]) % p.prime:
                    raise ValueError(f'add: {j+1} round error !!!')
            if not F_W_ext[i] == s[width-1]:
                raise ValueError('add: last round error !!!')
        except ValueError as e:
            print(str(e)) 
    """

    """
    for i, layer in enumerate(gate_list):

        width = len(r[i][1]+r[i][2])

        #check add gate
        s = verification(F_add_ext_g_t[i], r[i][1]+r[i][2])

        try:
            if not F_add_sum[i] == (F_add_ext_g_t[i][0][0] + F_add_ext_g_t[i][0][1]) % p.prime:
                raise ValueError('add: first round error !!!')
            for j in range(width-1):
                if not s[j] == (F_add_ext_g_t[i][j+1][0] + F_add_ext_g_t[i][j+1][1]) % p.prime:
                    raise ValueError(f'add: {j+1} round error !!!')
            if not F_add_ext[i] == s[width-1]:
                raise ValueError('add: last round error !!!')
        except ValueError as e:
            print(str(e))  
        
        #check multi gate 
        s = verification(F_multi_ext_g_t[i], r[i][1]+r[i][2])

        try:
            if not F_multi_sum[i] == (F_multi_ext_g_t[i][0][0] + F_multi_ext_g_t[i][0][1]) % p.prime:
                raise ValueError('multi: first round error !!!')
            for j in range(width-1):
                if not s[j] == (F_multi_ext_g_t[i][j+1][0] + F_multi_ext_g_t[i][j+1][1]) % p.prime:
                    raise ValueError(f'multi: {j+1} round error !!!')
            if not F_multi_ext[i] == s[width-1]:
                raise ValueError('multi: last round error !!!')
        except ValueError as e:
            print(str(e)) 
    """

    # interactive proof
    Verifier(proof, vk, r, r_f, final_out)


if __name__ == '__main__':
    main()