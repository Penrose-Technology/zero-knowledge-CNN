import sys
import numpy as np
import parameters as p
from typing import List, Dict, Tuple
import argparse
from circuit import circuit as C
import wire_prediction_ext as Wire
import value_prediction_ext as V
import sumcheck as S
from utilities import gadget as g


Ver = S.Verifier()
Pro = S.Prover()

@g.timer
def generate_proof(input_data, gate_list: List[List[str]], r: List[List[List[int]]], r_out: List[int], mu: List[List[int]], only_add=False, only_multi=False):
    cir = C.circuit(np.array(input_data), gate_list)
    map, output_data = cir.circuit_imp()
    #cir.print_map(map)

    # Get gate value F_W
    F_W = C.get_F_W(map, output_data)
    F_W_l, F_W_r = C.get_F_W_in_separate(input_data)

    # Get multilinear extension: W_i(b*^(i)) at b*^(i), W_i(c*^(i)) at c*^(i)

    F_W_b_ext: List[int] = []
    F_W_c_ext: List[int] = []

    for idx, layer_i in enumerate(F_W[:-1]):
        if idx == 0:
            input_bitwidth = int(np.log2(len(F_W_l)))
            F_W_in_ext = V._get_F_W_i_ext_input(F_W_l, F_W_r, input_bitwidth, r[idx][0])
        else:
            input_bitwidth = int(np.log2(len(layer_i)))
            F_W_b_ext_i = V._get_F_W_i_ext(layer_i, input_bitwidth, r[idx][0])
            F_W_c_ext_i = V._get_F_W_i_ext(layer_i, input_bitwidth, r[idx][1])
            F_W_b_ext.append(F_W_b_ext_i)
            F_W_c_ext.append(F_W_c_ext_i)
    #print(f"debug:: F_W_in_ext is {F_W_in_ext}")


    # Get wire prediction F_add and F_multi
    F_add = C.get_F_gate('ADD', gate_list)
    F_multi = C.get_F_gate('MULTI', gate_list)

    # Get g_t(0), g_t(1), g_t(2) and sumation for f_i(b*^(i), c*^(i)).
    # Add_i = mu_0*add_i(b*^(i+1), b*^(i), c*^(i)) + mu_1*add_i(c*^(i+1), b*^(i), c*^(i)), 
    # Multi_i =  mu_0*multi_i(b*^(i+1), b*^(i), c*^(i)) + mu_1*multi_i(c*^(i+1), b*^(i), c*^(i)),
    # 
    # Relationship holds:
    # f_i(b*^(i), c*^(i)) = Sum { Add_i*(W_i(b*^(i)) + W_i(c*^(i))) + Multi_i*(W_i(b*^(i)) * W_i(c*^(i))) }

    layer_i_size = []
    layer_i_add = []
    layer_i_multi = []

    F_sum: List[int] = []
    F_ext_g_t: List[np.ndarray] = []

    # Parse Wire Prediction
    if only_add:
        for i, (key, value) in enumerate(F_add.items()):
            idx = i // 2
            if key == f'The {idx}-th Layer Input Size':
                layer_i_size.append(value)
            if key == f'The {idx}-th Layer ADD-Gate':
                layer_i_add.append(value)
    elif only_multi:
        for i, (key, value) in enumerate(F_multi.items()):
            idx = i // 2
            if key == f'The {idx}-th Layer Input Size':
                layer_i_size.append(value)
            if key == f'The {idx}-th Layer MULTI-Gate':
                layer_i_multi.append(value)
    else:
        for i, (key, value) in enumerate(F_add.items()):
            idx = i // 2
            #if key == f'The {idx}-th Layer Input Size':
            #    layer_i_size.append(value)
            if key == f'The {idx}-th Layer ADD-Gate':
                layer_i_add.append(value)
        for i, (key, value) in enumerate(F_multi.items()):
            idx = i // 2
            if key == f'The {idx}-th Layer Input Size':
                layer_i_size.append(value)
            if key == f'The {idx}-th Layer MULTI-Gate':
                layer_i_multi.append(value)

    d = len(gate_list)

    if d == 1:
        # Circuit only has one layer
        if only_add:
            # add_i(g, b*^(d), c*^(d))*(W_i(b*^(d)) + W_i(c*^(d)))
            sum_i, g_t_i = Wire.get_F_gate_ext_g_t_input(layer_i_size[d-1], layer_i_add[d-1], {}, r_out, r[d-1][0], 1, F_W_l, F_W_r, only_add=True)
        elif only_multi:
            # multi_i(g, b*^(d), c*^(d))*(W_i(b*^(d)) * W_i(c*^(d)))
            sum_i, g_t_i = Wire.get_F_gate_ext_g_t_input(layer_i_size[d-1], {}, layer_i_multi[d-1], r_out, r[d-1][0], 1, F_W_l, F_W_r, only_multi=True)
        else:
            # add_i(g, b*^(d), c*^(d))*(W_i(b*^(d)) + W_i(c*^(d))) + multi_i(g, b*^(d), c*^(d))*(W_i(b*^(d)) * W_i(c*^(d)))
            sum_i, g_t_i = Wire.get_F_gate_ext_g_t_input(layer_i_size[d-1], layer_i_add[d-1], layer_i_multi[d-1], r_out, r[d-1][0], 1, F_W_l, F_W_r)
    else:
        # The first d-1 layer
        for i in range(d-1):
            if only_add:
                if i == 0:
                    # mu_0*[add_0(b*^(1), g*)*(W_l(g*) + W_r(g*))]
                    sum_i_left, g_t_i_left = Wire.get_F_gate_ext_g_t_input(layer_i_size[i], layer_i_add[i], {}, r[i+1][0], r[i][0], mu[i][0], F_W_l, F_W_r, only_add=True)
                    # mu_1*[add_0(c*^(1), g*)*(W_l(g*) + W_r(g*))]
                    sum_i_right, g_t_i_right = Wire.get_F_gate_ext_g_t_input(layer_i_size[i], layer_i_add[i], {}, r[i+1][1], r[i][0], mu[i][1], F_W_l, F_W_r, only_add=True) 
                else:
                    # mu_0*[add_i(b*^(i+1), b*^(i), c*^(i))*(W_i(b*^(i)) + W_i(c*^(i)))]
                    sum_i_left, g_t_i_left = Wire.get_F_gate_ext_g_t(layer_i_size[i], layer_i_add[i], {}, r[i+1][0], r[i][0], r[i][1], mu[i][0], F_W[i], only_add=True)
                    # mu_1*[add_i(c*^(i+1), b*^(i), c*^(i))*(W_i(b*^(i)) + W_i(c*^(i)))]
                    sum_i_right, g_t_i_right = Wire.get_F_gate_ext_g_t(layer_i_size[i], layer_i_add[i], {}, r[i+1][1], r[i][0], r[i][1], mu[i][1], F_W[i], only_add=True)
            elif only_multi:
                if i == 0:
                    # mu_0*[multi_0(b*^(1), g*)*(W_l(g*) * W_r(g*))]
                    sum_i_left, g_t_i_left = Wire.get_F_gate_ext_g_t_input(layer_i_size[i], {}, layer_i_multi[i], r[i+1][0], r[i][0], mu[i][0], F_W_l, F_W_r, only_multi=True)
                    # mu_1*[multi_0(c*^(1), g*)*(W_l(g*) * W_r(g*))]
                    sum_i_right, g_t_i_right = Wire.get_F_gate_ext_g_t_input(layer_i_size[i], {}, layer_i_multi[i], r[i+1][1], r[i][0], mu[i][1], F_W_l, F_W_r, only_multi=True)  
                else:
                    # mu_0*[multi_i(b*^(i+1), b*^(i), c*^(i))*(W_i(b*^(i)) * W_i(c*^(i)))]
                    sum_i_left, g_t_i_left = Wire.get_F_gate_ext_g_t(layer_i_size[i], {}, layer_i_multi[i], r[i+1][0], r[i][0], r[i][1], mu[i][0], F_W[i], only_multi=True)
                    # mu_1*[multi_i(c*^(i+1), b*^(i), c*^(i))*(W_i(b*^(i)) * W_i(c*^(i)))]
                    sum_i_right, g_t_i_right = Wire.get_F_gate_ext_g_t(layer_i_size[i], {}, layer_i_multi[i], r[i+1][1], r[i][0], r[i][1], mu[i][1], F_W[i], only_multi=True)                           
            else:
                if i == 0:
                    # mu_0*[add_0(b*^(1), g*)*(W_l(g*) + W_r(g*)) + multi_0(b*^(1), g*)*(W_l(g*) * W_r(g*))]
                    sum_i_left, g_t_i_left = Wire.get_F_gate_ext_g_t_input(layer_i_size[i], layer_i_add[i], layer_i_multi[i], \
                                                                        r[i+1][0], r[i][0], mu[i][0], F_W_l, F_W_r)
                    # mu_1*[add_0(c*^(1), g*)*(W_l(g*) + W_r(g*)) + multi_0(c*^(1), g*)*(W_l(g*) * W_r(g*))]
                    sum_i_right, g_t_i_right = Wire.get_F_gate_ext_g_t_input(layer_i_size[i], layer_i_add[i], layer_i_multi[i], \
                                                                        r[i+1][1], r[i][0], mu[i][1], F_W_l, F_W_r)  
                else:
                    # mu_0*[add_i(b*^(i+1), b*^(i), c*^(i))*(W_i(b*^(i)) + W_i(c*^(i))) + multi_i(b*^(i+1), b*^(i), c*^(i))*(W_i(b*^(i)) * W_i(c*^(i)))]
                    sum_i_left, g_t_i_left = Wire.get_F_gate_ext_g_t(layer_i_size[i], layer_i_add[i], layer_i_multi[i], \
                                                                    r[i+1][0], r[i][0], r[i][1], mu[i][0], F_W[i])
                    # mu_1*[add_i(c*^(i+1), b*^(i), c*^(i))*(W_i(b*^(i)) + W_i(c*^(i))) + multi_i(c*^(i+1), b*^(i), c*^(i))*(W_i(b*^(i)) * W_i(c*^(i)))]
                    sum_i_right, g_t_i_right = Wire.get_F_gate_ext_g_t(layer_i_size[i], layer_i_add[i], layer_i_multi[i], \
                                                                    r[i+1][1], r[i][0], r[i][1], mu[i][1], F_W[i])        
            
            sum_i = (sum_i_left + sum_i_right) % p.prime
            g_t_i = (g_t_i_left + g_t_i_right) % p.prime
            F_sum.append(sum_i)
            F_ext_g_t.append(g_t_i)
        
        # Output layer
        if only_add:
            # add_i(g, b*^(d), c*^(d))*(W_i(b*^(d)) + W_i(c*^(d)))
            sum_i, g_t_i = Wire.get_F_gate_ext_g_t(layer_i_size[d-1], layer_i_add[d-1], {}, r_out, r[d-1][0], r[d-1][1], 1, F_W[d-1], only_add=True)
        elif only_multi:
            # multi_i(g, b*^(d), c*^(d))*(W_i(b*^(d)) * W_i(c*^(d)))
            sum_i, g_t_i = Wire.get_F_gate_ext_g_t(layer_i_size[d-1], {}, layer_i_multi[d-1], r_out, r[d-1][0], r[d-1][1], 1, F_W[d-1], only_multi=True)
        else:
            # add_i(g, b*^(d), c*^(d))*(W_i(b*^(d)) + W_i(c*^(d))) + multi_i(g, b*^(d), c*^(d))*(W_i(b*^(d)) * W_i(c*^(d)))
            sum_i, g_t_i = Wire.get_F_gate_ext_g_t(layer_i_size[d-1], layer_i_add[d-1], layer_i_multi[d-1], \
                                                r_out, r[d-1][0], r[d-1][1], 1, F_W[d-1])


    F_sum.append(sum_i)
    F_ext_g_t.append(g_t_i)

    # Assemble proof
    F_W_in_proof = F_W_in_ext
    F_W_b_proof = F_W_b_ext
    F_W_c_proof = F_W_c_ext
    F_proof = (F_sum, F_ext_g_t)

    proof = (F_W_in_proof, F_W_b_proof, F_W_c_proof, F_proof)

    return proof, output_data


def get_s(g_t_i: np.ndarray, r_i: List[int], cubic=False):

    s_i = Ver.sum_verify(g_t_i, r_i, width=len(r_i), ntt_flag=True, cubic=cubic)

    print(f"verifier output, s is: \n{s_i}")
    return s_i


def verification_i(sum_i: int, g_t_i: List[List[int]], ext_i: int, r_i: List[int], cubic=False, tag=None):

    s_i = get_s(np.array(g_t_i), r_i, cubic=cubic)
    # print(f"extension at {r_i} is {ext_i}")

    g.sumcheck_verification(s_i, sum_i, g_t_i, ext_i, tag=tag)

    return 0

# Verifier should evaluate Add_i and Multi_i on his own.
@g.timer
def set_up(gate_list: List[List[str]], r: List[List[List[int]]], r_out: List[int], mu: List[List[int]], only_add=False, only_multi=False):

    # Get wire prediction F_add and F_multi
    F_add = C.get_F_gate('ADD', gate_list)
    F_multi = C.get_F_gate('MULTI', gate_list)

    F_add_ext = []
    F_multi_ext = []

    d = len(gate_list)

    # Extension at a random point
    # Verifier precomputes: Add_i = mu_0*add_i(b*^(i+1), b*^(i), c*^(i)) + mu_1*add_i(c*^(i+1), b*^(i), c*^(i))
    if not only_multi:
        for i, (key, value) in enumerate(F_add.items()):
            idx = i // 2
            if key == f'The {idx}-th Layer Input Size':
                layer_i_size = value
            else:
                layer_i_gate = value
                # Circuit only has one layer
                if d == 1:
                    F_add_ext_i = Wire._get_F_gate_i_ext_input(layer_i_size, layer_i_gate, r_out, r[d-1][0])
                else:
                    # Output layer
                    if key == f'The {d-1}-th Layer ADD-Gate':
                        F_add_ext_i = Wire._get_F_gate_i_ext(layer_i_size, layer_i_gate, r_out, r[d-1][0], r[d-1][1])
                    # The first layer
                    elif key == f'The 0-th Layer ADD-Gate':
                        left_add_ext = Wire._get_F_gate_i_ext_input(layer_i_size, layer_i_gate, r[idx+1][0], r[idx][0])
                        right_add_ext = Wire._get_F_gate_i_ext_input(layer_i_size, layer_i_gate, r[idx+1][1], r[idx][0])
                        F_add_ext_i = (mu[idx][0]*left_add_ext + mu[idx][1]*right_add_ext) % p.prime
                    # The rest d-2 layers
                    elif key == f'The {idx}-th Layer ADD-Gate':
                        left_add_ext = Wire._get_F_gate_i_ext(layer_i_size, layer_i_gate, r[idx+1][0], r[idx][0], r[idx][1])
                        right_add_ext = Wire._get_F_gate_i_ext(layer_i_size, layer_i_gate, r[idx+1][1], r[idx][0], r[idx][1])
                        F_add_ext_i = (mu[idx][0]*left_add_ext + mu[idx][1]*right_add_ext) % p.prime
                F_add_ext.append(F_add_ext_i)

    # Verifier precomputes: Multi_i =  mu_0*multi_i(b*^(i+1), b*^(i), c*^(i)) + mu_1*multi_i(c*^(i+1), b*^(i), c*^(i))
    if not only_add:
        for i, (key, value) in enumerate(F_multi.items()):
            idx = i // 2
            if key == f'The {idx}-th Layer Input Size':
                layer_i_size = value
            else:
                layer_i_gate = value
                # Circuit only has one layer
                if d == 1:
                    F_multi_ext_i = Wire._get_F_gate_i_ext_input(layer_i_size, layer_i_gate, r_out, r[d-1][0])
                else:
                    # Output layer
                    if key == f'The {d-1}-th Layer MULTI-Gate':
                        F_multi_ext_i = Wire._get_F_gate_i_ext(layer_i_size, layer_i_gate, r_out, r[d-1][0], r[d-1][1])
                    # The first layer
                    elif key == f'The 0-th Layer MULTI-Gate':
                        left_multi_ext = Wire._get_F_gate_i_ext_input(layer_i_size, layer_i_gate, r[idx+1][0], r[idx][0])
                        right_multi_ext = Wire._get_F_gate_i_ext_input(layer_i_size, layer_i_gate, r[idx+1][1], r[idx][0])
                        F_multi_ext_i = (mu[idx][0]*left_multi_ext + mu[idx][1]*right_multi_ext) % p.prime
                    # The rest d-2 layer
                    elif key == f'The {idx}-th Layer MULTI-Gate':
                        left_multi_ext = Wire._get_F_gate_i_ext(layer_i_size, layer_i_gate, r[idx+1][0], r[idx][0], r[idx][1])
                        right_multi_ext = Wire._get_F_gate_i_ext(layer_i_size, layer_i_gate, r[idx+1][1], r[idx][0], r[idx][1])
                        F_multi_ext_i = (mu[idx][0]*left_multi_ext + mu[idx][1]*right_multi_ext) % p.prime
                F_multi_ext.append(F_multi_ext_i)

    vk = (F_add_ext, F_multi_ext)
    print(f"debug:: F_add_ext is {F_add_ext}, F_multi_ext is {F_multi_ext}")
    return vk

@g.timer
def Verifier(proof, vk, r: List[List[List[int]]], mu: List[List[int]], F_W_out_ext: int, only_add=False, only_multi=False):

    # Parse proof
    F_W_in_proof, F_W_b_proof, F_W_c_proof, F_proof = proof

    # Verifier receives W(b*), W(c*) from Prover.
    F_W_in_ext = F_W_in_proof
    F_W_b_ext = F_W_b_proof
    F_W_c_ext = F_W_c_proof
    #print(f"F_W_in_ext is {F_W_in_ext}, F_W_b_ext is {F_W_b_ext}, F_W_c_ext is {F_W_c_ext}")

    # Verifier receives sumcheck proof of Sum = Sigma_f(b, c) from Prover.
    _, F_ext_g_t = F_proof

    # Parse vk
    # Verifier calculates add(a*, b*, c*) and multi(a*, b*, c*).
    (F_add_ext, F_multi_ext) = vk

    # Get random check point (b*, c*).
    r_bc = [(r[i][0] + r[i][1]) for i in range(len(r))]

    # Evaluate total layer number
    if only_add:
        d = len(F_add_ext)
    else:
        d = len(F_multi_ext)

    # Output layer check
    # Verifier evaluates W_out(z*) on his own from output layer (d-th layer).
    #F_W_out_ext = Ver.multi_ext(W_out, r_out, width=len(r_out))
    #print(f"F_W_out_ext is {F_W_out_ext}")

    if d == 1:
        if only_add:
            F_ext = F_add_ext[d-1]*(F_W_in_ext[0] + F_W_in_ext[1]) % p.prime
        elif only_multi:
            F_ext = F_multi_ext[d-1]*(F_W_in_ext[0] * F_W_in_ext[1]) % p.prime
        else:
            F_ext = (  F_add_ext[d-1]*(F_W_in_ext[0] + F_W_in_ext[1]) + F_multi_ext[d-1]*(F_W_in_ext[0] * F_W_in_ext[1])  ) % p.prime
        print(f"debug:: F_ext is {F_ext}")
        verification_i(F_W_out_ext, F_ext_g_t[d-1], F_ext, r[d-1][0], cubic=True, tag='GKR Output(0) Layer')
    else:
        if only_add:
            F_ext = F_add_ext[d-1]*(F_W_b_ext[d-2] + F_W_c_ext[d-2]) % p.prime
        elif only_multi:
            F_ext = F_multi_ext[d-1]*(F_W_b_ext[d-2] * F_W_c_ext[d-2]) % p.prime
        else:
            F_ext = (  F_add_ext[d-1]*(F_W_b_ext[d-2] + F_W_c_ext[d-2]) + F_multi_ext[d-1]*(F_W_b_ext[d-2] * F_W_c_ext[d-2])  ) % p.prime
        verification_i(F_W_out_ext, F_ext_g_t[d-1], F_ext, r_bc[d-1], tag=f'GKR Output({d-1}) Layer')

    if d == 1:
        # Circuit only has one layer.
        pass

    else:
        # Circuit at least has two layers.
        # Verifier evaluates f_i(b*^(i), c*^(i)) for the i-th layer, with
        # W_i(b*^(i)), W_i(c*^(i)) received from Prover and, 
        # Add_i = mu_0*add_i(b*^(i+1), b*^(i), c*^(i)) + mu_1*add_i(c*^(i+1), b*^(i), c*^(i)), 
        # Multi_i =  mu_0*multi_i(b*^(i+1), b*^(i), c*^(i)) + mu_1*multi_i(c*^(i+1), b*^(i), c*^(i)), are calculated by Verifier.
        # 
        # Relationship holds:
        # f_i(b*^(i), c*^(i)) = Sum { Add_i*(W_i(b*^(i)) + W_i(c*^(i))) + Multi_i*(W_i(b*^(i)) * W_i(c*^(i))) }
        for i in range(d-2,-1,-1):
            # In the input layer, Verifier needs to calculate W_in(b*), W_in(c*) on his own.
            if i == 0:
                #F_W_l, F_W_r = C.get_F_W_in_separate(input_data)
                #print(f"debug::: F_W_l is {F_W_l}, F_W_r is {F_W_r}")
                #F_W_l_input_ext = Ver.multi_ext(F_W_l, r[0][0], width=len(r[0][0]))
                #F_W_r_input_ext = Ver.multi_ext(F_W_r, r[0][0], width=len(r[0][0]))
                if only_add:
                    F_ext_i = F_add_ext[i]*(F_W_in_ext[0] + F_W_in_ext[1]) % p.prime
                elif only_multi:
                    F_ext_i = F_multi_ext[i]*(F_W_in_ext[0] * F_W_in_ext[1]) % p.prime
                else:
                    F_ext_i = (  F_add_ext[i]*(F_W_in_ext[0] + F_W_in_ext[1]) + F_multi_ext[i]*(F_W_in_ext[0] * F_W_in_ext[1])  ) % p.prime
            # In the rest layers, Verifier receives W_i(b*), W_i(c*) from Prover.
            else:
                if only_add:
                    F_ext_i = F_add_ext[i]*(F_W_b_ext[i-1] + F_W_c_ext[i-1]) % p.prime
                elif only_multi:
                    F_ext_i = F_multi_ext[i]*(F_W_b_ext[i-1] * F_W_c_ext[i-1]) % p.prime
                else:
                    F_ext_i = (  F_add_ext[i]*(F_W_b_ext[i-1] + F_W_c_ext[i-1]) + F_multi_ext[i]*(F_W_b_ext[i-1] * F_W_c_ext[i-1])  ) % p.prime

            # Sumation of i-th layer,
            # Relationship holds: 
            # f_i(b*^(i), c*^(i)) = mu_0*W_i+1(b*^(i+1)) + mu_1*W_i+1(c*^(i+1))
            F_sum_i = (mu[i][0]*F_W_b_ext[i] + mu[i][1]*F_W_c_ext[i]) % p.prime

            # Run sumcheck protocol
            #print(f"debug:::F_ext_i is {F_ext_i}, \n F_sum_i is {F_sum_i}, \n F_ext_g_t is {F_ext_g_t[i]}")
            if i == 0:
                verification_i(F_sum_i, F_ext_g_t[i], F_ext_i, r[0][0], cubic=True, tag=f'GKR {i}-th Layer')
            else:
                verification_i(F_sum_i, F_ext_g_t[i], F_ext_i, r_bc[i], tag=f'GKR {i}-th Layer')
            

def circuit_precheck(input_data: np.ndarray, gate_list: List[List[str]], r: List[List[List[int]]], mu: List[List[int]]):
    try:
        if not len(input_data) == len(gate_list[0]):
            raise ValueError("input data shape doesn't match the input layer !!!")
        if not len(gate_list) == len(r):
            raise ValueError("random check points shape doesn't match the circuit structure !!!")
        if not len(mu) == len(gate_list)-1:
            raise ValueError("shape of mu doesn't match the circuit structure !!!")
    except ValueError as e:
        print(f"CIRCUIT ERROR!!! {e}")
        sys.exit(-1)



def main():
    # circuit inilization
    input_data = np.array(
                 [[1,3],
                  [2,2],
                  [4,5],
                  [3,2]])
    
    gate_list = [['add', 'multi', 'add', 'add'],
                 ['add', 'multi']]

    #final_out = [8, 45]
    
    r = [[[52, 53], []],
         [[40, 41], [120, 121]]]

    r_f = [70]

    mu = [[152, 398]]

    # circuit check
    circuit_precheck(input_data, gate_list, r, mu)

    # set up
    vk = set_up(gate_list, r, r_f, mu)
    
    # Proving
    proof, final_out = generate_proof(input_data, gate_list, r, r_f, mu)
    print(f"debug:: proof is {proof}")
    
    # Verification
    # Verifier evaluates W_out(z*) on his own from output layer (d-th layer).
    W_out_ext = Ver.multi_ext(final_out, r_f, width=len(r_f))
    print(f"W_out_ext is {W_out_ext}")
    Verifier(proof, vk, r, mu, W_out_ext)


if __name__ == '__main__':
    main()