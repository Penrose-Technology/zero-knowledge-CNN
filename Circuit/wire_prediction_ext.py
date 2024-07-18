import numpy as np
from typing import List, Dict, Tuple
import argparse
import circuit as C
import parameters as p
import sumcheck as S
from calculation import single_extension


# def get_F_gate_ext_t(F_gate_i, r): 
#     if not len(F_gate_i.shape) == 3:
#         raise ValueError("Error input polynomial dimention!!!")
#     if not ( (2**len(r[0])==F_gate_i.shape[0]) or (2**len(r[1])==F_gate_i.shape[1]) or (2**len(r[2])==F_gate_i.shape[2]) ):
#         raise ValueError("Error random size!!!")
#     a_size, b_size, c_size = F_gate_i.shape
#     a_bitwise, b_bitwise, c_bitwise = int(np.log2(a_size)), int(np.log2(b_size)), int(np.log2(c_size))

#     F_gate_a_ext = np.zeros((a_size), dtype='int32')
#     F_gate_b_ext = np.zeros((a_size, b_size), dtype='int32')
#     F_gate_c_ext = np.copy(F_gate_i)
#     for a_i in range(a_size):
#         for b_i in range(b_size):
#             for c_i in range(c_size):
#                 F_gate_c_ext[a_i][b_i][c_i] = single_extension(F_gate_c_ext[a_i][b_i][c_i], c_i, r[2], c_bitwise)
#             # store c direction after extension, b direction before extension
#             F_gate_b_ext[a_i][b_i] = F_gate_c_ext[a_i][b_i].sum()%p.prime

#             F_gate_b_ext[a_i][b_i] = single_extension(F_gate_b_ext[a_i][b_i], b_i, r[1], b_bitwise)   
#         # store b direction after extension, a direction before extension
#         F_gate_a_ext[a_i] = F_gate_b_ext[a_i].sum()%p.prime

#         F_gate_a_ext[a_i] = single_extension(F_gate_a_ext[a_i], a_i, r[0], a_bitwise)
#         out = F_gate_a_ext.sum()%p.prime

#     #print(f"F_gate_c_ext shape is {F_gate_c_ext.shape}, F_gate_c_ext is \n{F_gate_c_ext}")
#     #print(f"F_gate_b_ext shape is {F_gate_b_ext.shape}, F_gate_b_ext is \n{F_gate_b_ext}")
#     #print(f"F_gate_a_ext shape is {F_gate_a_ext.shape}, F_gate_a_ext is \n{F_gate_a_ext}")
#     #print(f"---output is {out}")

#     return out

def _get_F_gate_i_ext(layer_i_size: Dict[str, int], \
                      layer_i_gate: Dict[Tuple[int, int, int], int], \
                      r_a: List[int], r_b: List[int], r_c: List[int]):

    F_gate_c_i_ext: Dict[Tuple[int, int], int] = {}
    F_gate_b_i_ext: Dict[int, int] = {}

    c_bitwidth = int(np.log2(layer_i_size['c']))
    b_bitwidth = int(np.log2(layer_i_size['b']))
    a_bitwidth = int(np.log2(layer_i_size['a']))

    # inilization
    for key, value in layer_i_gate.items():
        a_i, b_i, _ = key
        F_gate_c_i_ext[(a_i, b_i)] = 0
        F_gate_b_i_ext[a_i] = 0
    F_gate_a_i_ext = 0
    
    # multilinear extension in (a, b, c) direction.
    for key, value in layer_i_gate.items():
        a_i, b_i, c_i = key
        c_i_extend = single_extension(value, c_i, r_c, c_bitwidth)
        F_gate_c_i_ext[(a_i, b_i)] = (F_gate_c_i_ext[(a_i, b_i)] + c_i_extend) % p.prime

    for key, value in F_gate_c_i_ext.items():
        a_i, b_i = key
        b_i_extend = single_extension(value, b_i, r_b, b_bitwidth)
        F_gate_b_i_ext[a_i] = (F_gate_b_i_ext[a_i] + b_i_extend) % p.prime

    for key, value in F_gate_b_i_ext.items():
        a_i = key
        a_i_extend = single_extension(value, a_i, r_a, a_bitwidth)
        F_gate_a_i_ext = (F_gate_a_i_ext + a_i_extend) % p.prime

    return F_gate_a_i_ext

def _get_F_gate_i_ext_input(layer_i_size: Dict[str, int], \
                            layer_i_gate: Dict[Tuple[int, int], int], \
                            r_a: List[int], r_b: List[int]):

    F_gate_b_i_ext: Dict[int, int] = {}

    b_bitwidth = int(np.log2(layer_i_size['b']))
    a_bitwidth = int(np.log2(layer_i_size['a']))

    # inilization
    for key, value in layer_i_gate.items():
        a_i,  _ = key
        F_gate_b_i_ext[a_i] = 0
    F_gate_a_i_ext = 0
    
    # multilinear extension in (a, b) direction.
    for key, value in layer_i_gate.items():
        a_i, b_i = key
        b_i_extend = single_extension(value, b_i, r_b, b_bitwidth)
        F_gate_b_i_ext[a_i] = (F_gate_b_i_ext[a_i] + b_i_extend) % p.prime

    for key, value in F_gate_b_i_ext.items():
        a_i = key
        a_i_extend = single_extension(value, a_i, r_a, a_bitwidth)
        F_gate_a_i_ext = (F_gate_a_i_ext + a_i_extend) % p.prime

    return F_gate_a_i_ext     


def get_F_gate_ext(gate_type: str, F_gate: Dict[str, Dict], r: List[List[List[int]]], extension_check=False):

    if not ((gate_type == 'ADD') or (gate_type == 'MULTI')):
        raise ValueError("Error Gate Type !!!")
    
    out = []

    for i, (key, value) in enumerate(F_gate.items()):

        idx = i // 2

        if idx == 0:
            if key == f'The {idx}-th Layer Input Size':
                layer_i_size = value
                b_bitwidth = int(np.log2(layer_i_size['b']))
                a_bitwidth = int(np.log2(layer_i_size['a']))

            if key == f'The {idx}-th Layer {gate_type}-Gate':
                layer_i_gate = value
                if extension_check:
                    for key, value in layer_i_gate.items():
                        #get original value (before extension)
                        a_o_i, b_o_i = key
                        b_o_i_v = [int(char) for char in np.binary_repr(b_o_i, width=b_bitwidth)]
                        a_o_i_v = [int(char) for char in np.binary_repr(a_o_i, width=a_bitwidth)]

                        # Points at {0, 1}^n should be equal before and after extension.
                        if not _get_F_gate_i_ext_input(layer_i_size, layer_i_gate, a_o_i_v, b_o_i_v) == value:
                            raise ValueError("Multilinear Extension Error ! Value unequal !!!")
                    return [0] 
        else:
            if key == f'The {idx}-th Layer Input Size':
                layer_i_size = value
                c_bitwidth = int(np.log2(layer_i_size['c']))
                b_bitwidth = int(np.log2(layer_i_size['b']))
                a_bitwidth = int(np.log2(layer_i_size['a']))

            if key == f'The {idx}-th Layer {gate_type}-Gate':
                layer_i_gate = value
                if extension_check:
                    for key, value in layer_i_gate.items():
                        #get original value (before extension)
                        a_o_i, b_o_i, c_o_i = key
                        c_o_i_v = [int(char) for char in np.binary_repr(c_o_i, width=c_bitwidth)]
                        b_o_i_v = [int(char) for char in np.binary_repr(b_o_i, width=b_bitwidth)]
                        a_o_i_v = [int(char) for char in np.binary_repr(a_o_i, width=a_bitwidth)]

                        # Points at {0, 1}^n should be equal before and after extension.
                        if not _get_F_gate_i_ext(layer_i_size, layer_i_gate, a_o_i_v, b_o_i_v, c_o_i_v) == value:
                            raise ValueError("Multilinear Extension Error ! Value unequal !!!")
                    return [0]            
                else:
                    out.append(_get_F_gate_i_ext(layer_i_size, layer_i_gate, r[idx][0], r[idx][1], r[idx][2]))

    return out


def _get_F_gate_ext_g_t(P:S.Prover, \
                        gate_type: str, \
                        layer_i_size: Dict[str, int], \
                        layer_i_gate: Dict[Tuple[int, int, int], int], \
                        r_a: List[int], r_b: List[int], r_c: List[int], \
                        mu: int, \
                        F_W_i: List[int]):
    
    if not ((gate_type == 'ADD') or (gate_type == 'MULTI')):
        raise ValueError("Error Gate Type !!!")

    c_bitwidth = int(np.log2(layer_i_size['c']))
    b_bitwidth = int(np.log2(layer_i_size['b']))
    a_bitwidth = int(np.log2(layer_i_size['a']))         
    bc_bitwidth = b_bitwidth + c_bitwidth
    r_bc = r_b + r_c
    A_F_i = np.zeros((2**bc_bitwidth), dtype='int32')
    #print(f"A_F is {A_F}, shape is {np.shape(A_F)}")

    # Initalize F(a, b, c) from dictionary and precompute F(a*, b, c)
    for key, init_value in layer_i_gate.items():
        a_o_i, b_o_i, c_o_i = key
        init_idx = b_o_i*2**(c_bitwidth) + c_o_i
        unextended_value = (init_value*mu) % p.prime
        A_F_i[init_idx] = (A_F_i[init_idx] + single_extension(unextended_value, a_o_i, r_a, bitwidth=a_bitwidth) ) % p.prime
    #print(f"A_F_i is {A_F_i}")      

    F_W_gate_i = np.zeros_like(A_F_i)
    # Evaluation
    # W_add(b, c) = W(b) + W(c)
    # W_multi(b, c) = W(b) * W(c)
    for n in range(2**b_bitwidth):
        for m in range(2**c_bitwidth):
            if gate_type == 'ADD':
                F_W_gate_i[(n<<c_bitwidth) + m] = (F_W_i[n] + F_W_i[m]) % p.prime
            else:
                F_W_gate_i[(n<<c_bitwidth) + m] = (F_W_i[n] * F_W_i[m]) % p.prime

    # Extension
    # f(b, c) = add(a*, b, c)*W_add(b, c) + multi(a*, b, c)*W_multi(b, c)
    sum_i, g_t_i = P.sumcheck_ntt(F_W_gate_i, A_F_i, r_bc, inverse=False, width=bc_bitwidth)
    #print(f"g_t is {g_t}, sum is {sum}")
    return sum_i, g_t_i

#########################
def _get_F_gate_ext_g_t_input(P:S.Prover, \
                              gate_type: str, \
                              layer_in_size: Dict[str, int], \
                              layer_in_gate: Dict[Tuple[int, int], int], \
                              r_a: List[int], r_b: List[int],  \
                              mu: int, \
                              F_W_left_in: List[int], F_W_right_in: List[int]):
    
    if not ((gate_type == 'ADD') or (gate_type == 'MULTI')):
        raise ValueError("Error Gate Type !!!")

    # Input layer, b_bitwidth==a_bitwidth
    a_bitwidth = int(np.log2(layer_in_size['a']))         
    A_F_in = np.zeros((2**a_bitwidth), dtype='int32')
    #print(f"A_F is {A_F}, shape is {np.shape(A_F)}")

    # Initalize F(a, b) from dictionary and precompute F(a*, b)
    for key, init_value in layer_in_gate.items():
        a_o_i,  _ = key
        init_idx = a_o_i
        unextended_value = (init_value*mu) % p.prime
        A_F_in[init_idx] = (A_F_in[init_idx] + single_extension(unextended_value, a_o_i, r_a, bitwidth=a_bitwidth) ) % p.prime
    #print(f"A_F_i is {A_F_i}")      

    # Evaluation
    # W_in_add(b) = W_in_left(b) + W_in_right(b)
    # W_in_multi(b) = W_in_left(b) * W_in_right(b)
    if gate_type == 'ADD':
        F_W_l = (np.array(F_W_left_in) + np.array(F_W_right_in)) % p.prime
        #F_W_r = np.ones_like(F_W_l)
        sum_in, g_t_in = P.sumcheck_ntt(A_F_in, F_W_l, r_b, width=a_bitwidth, cubic=True)
    else:
        F_W_l = np.array(F_W_left_in)
        F_W_r = np.array(F_W_right_in)
        sum_in, g_t_in = P.sumcheck_cubic(A_F_in, F_W_l, F_W_r, r_b, width=a_bitwidth)

    # Extension
    # f(b) = add(a*, b)*W_add(b) + multi(a*, b)*W_multi(b)
    
    #print(f"g_t is {g_t}, sum is {sum}")
    return sum_in, g_t_in
#########################

#########################
def get_F_gate_ext_g_t_input(P: S.Prover, \
                             layer_in_size: Dict[str, int], \
                             layer_in_add: Dict[Tuple[int, int], int], \
                             layer_in_multi: Dict[Tuple[int, int], int], \
                             r_a: List[int], r_b: List[int], \
                             mu: int, \
                             F_W_left_in: List[int], F_W_right_in: List[int], \
                             only_add=False, only_multi=False):
    if only_add:
        sum_add_in, g_t_add_in = _get_F_gate_ext_g_t_input(P, 'ADD', layer_in_size, layer_in_add, r_a, r_b, mu, F_W_left_in, F_W_right_in)
        sum_multi_in = 0; g_t_multi_in = np.zeros_like(g_t_add_in)
    if only_multi:
        sum_multi_in, g_t_multi_in = _get_F_gate_ext_g_t_input(P, 'MULTI', layer_in_size, layer_in_multi, r_a, r_b, mu, F_W_left_in, F_W_right_in)
        sum_add_in = 0; g_t_add_in = np.zeros_like(g_t_multi_in)
    else:
        sum_add_in, g_t_add_in = _get_F_gate_ext_g_t_input(P, 'ADD', layer_in_size, layer_in_add, r_a, r_b, mu, F_W_left_in, F_W_right_in)
        sum_multi_in, g_t_multi_in = _get_F_gate_ext_g_t_input(P, 'MULTI', layer_in_size, layer_in_multi, r_a, r_b, mu, F_W_left_in, F_W_right_in)

    sum_in = (sum_add_in + sum_multi_in) % p.prime
    g_t_in = (g_t_add_in + g_t_multi_in) % p.prime

    return sum_in, g_t_in
#########################

def get_F_gate_ext_g_t( P: S.Prover, \
                        layer_i_size: Dict[str, int], \
                        layer_i_add: Dict[Tuple[int, int, int], int], \
                        layer_i_multi: Dict[Tuple[int, int, int], int], \
                        r_a: List[int], r_b: List[int], r_c: List[int], \
                        mu: int, \
                        F_W_i: List[int], \
                        only_add=False, only_multi=False):
    if only_add:
        sum_add_i, g_t_add_i = _get_F_gate_ext_g_t(P,'ADD', layer_i_size, layer_i_add, r_a, r_b, r_c, mu, F_W_i)
        sum_multi_i = 0; g_t_multi_i = np.zeros_like(g_t_add_i)
    if only_multi:
        sum_multi_i, g_t_multi_i = _get_F_gate_ext_g_t(P,'MULTI', layer_i_size, layer_i_multi, r_a, r_b, r_c, mu, F_W_i)
        sum_add_i = 0; g_t_add_i = np.zeros_like(g_t_multi_i)
    else:
        sum_add_i, g_t_add_i = _get_F_gate_ext_g_t(P,'ADD', layer_i_size, layer_i_add, r_a, r_b, r_c, mu, F_W_i)
        sum_multi_i, g_t_multi_i = _get_F_gate_ext_g_t(P, 'MULTI', layer_i_size, layer_i_multi, r_a, r_b, r_c, mu, F_W_i)

    sum_i = (sum_add_i + sum_multi_i) % p.prime
    g_t_i = (g_t_add_i + g_t_multi_i) % p.prime

    return sum_i, g_t_i

def main():
    parser = argparse.ArgumentParser(description='specify gate type')
    parser.add_argument("--operation", '-o', choices=['ADD', 'MULTI'], default='ADD')
    parser.add_argument("--g_t", '-g', action="store_true")
    parser.add_argument("--input", '-i', action="store_true")
    args = parser.parse_args()

    # circuit inilization
    input_data = np.array(
                 [[1,3],
                  [2,2],
                  [4,5],
                  [3,2]])
    gate_list = [['add', 'multi', 'add', 'add'],
                 ['add', 'multi']]
                 #['add']]
    
    cir = C.circuit(input_data, gate_list)
    map, final_out = cir.circuit_imp()
    cir.print_map(map)
    print(f"The circuit final output is {final_out}")

    # get random value
    r = [[[40, 41], [51, 52, 53], [61, 62, 63]],
         [[70], [80, 81], [90, 91]]]
    
    # extension
    F_gate = C.get_F_gate(args.operation, gate_list)
    print(f"F_gate is \n {F_gate}")
    F_W = C.get_F_W(map, final_out)
    print(f"F_W is: \n{F_W}")
    F_W_l, F_W_r = C.get_F_W_in_separate(input_data)
    out_1 = get_F_gate_ext(args.operation, F_gate, r)
    print(f"out_1 is: \n{out_1}")

    # extension check
    P = S.Prover()
    
    if args.g_t:
        get_F_gate_ext_g_t(P, layer_i_size={'a': 4, 'b': 8, 'c': 8}, \
                        layer_i_add={(0, 0, 1): 1, (2, 4, 5): 1, (3, 6, 7): 1}, \
                        layer_i_multi={(1, 2, 3): 1}, \
                        r_a=r[0][0], r_b=r[0][1], r_c=r[0][2], \
                        mu=0, \
                        F_W_i=F_W[0])
    elif args.input:
        get_F_gate_ext_g_t_input(P, layer_in_size={'a': 4, 'b': 8}, \
                        layer_in_add={(0, 0): 1, (2, 2): 1, (3, 3): 1}, \
                        layer_in_multi={(1, 1): 1}, \
                        r_a=r[0][0], r_b=[51, 52], \
                        mu=0, \
                        F_W_left_in=F_W_l, F_W_right_in=F_W_r)
    else:
        if get_F_gate_ext(args.operation, F_gate, r, extension_check=True) == [0]:
            print("Extension Check Pass !")

if __name__ == '__main__':
    main()
