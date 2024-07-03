import numpy as np
import extension
import parameters as p
from typing import List, Dict, Tuple
import argparse
from circuit import circuit as C
import sumcheck as S
from extension import single_extension


def get_F_gate_ext_t(F_gate_i, r): 
    if not len(F_gate_i.shape) == 3:
        raise ValueError("Error input polynomial dimention!!!")
    if not ( (2**len(r[0])==F_gate_i.shape[0]) or (2**len(r[1])==F_gate_i.shape[1]) or (2**len(r[2])==F_gate_i.shape[2]) ):
        raise ValueError("Error random size!!!")
    a_size, b_size, c_size = F_gate_i.shape
    a_bitwise, b_bitwise, c_bitwise = int(np.log2(a_size)), int(np.log2(b_size)), int(np.log2(c_size))

    F_gate_a_ext = np.zeros((a_size), dtype='int32')
    F_gate_b_ext = np.zeros((a_size, b_size), dtype='int32')
    F_gate_c_ext = np.copy(F_gate_i)
    for a_i in range(a_size):
        for b_i in range(b_size):
            for c_i in range(c_size):
                F_gate_c_ext[a_i][b_i][c_i] = single_extension(F_gate_c_ext[a_i][b_i][c_i], c_i, r[2], c_bitwise)
            # store c direction after extension, b direction before extension
            F_gate_b_ext[a_i][b_i] = F_gate_c_ext[a_i][b_i].sum()%p.prime

            F_gate_b_ext[a_i][b_i] = single_extension(F_gate_b_ext[a_i][b_i], b_i, r[1], b_bitwise)   
        # store b direction after extension, a direction before extension
        F_gate_a_ext[a_i] = F_gate_b_ext[a_i].sum()%p.prime

        F_gate_a_ext[a_i] = single_extension(F_gate_a_ext[a_i], a_i, r[0], a_bitwise)
        out = F_gate_a_ext.sum()%p.prime

    #print(f"F_gate_c_ext shape is {F_gate_c_ext.shape}, F_gate_c_ext is \n{F_gate_c_ext}")
    #print(f"F_gate_b_ext shape is {F_gate_b_ext.shape}, F_gate_b_ext is \n{F_gate_b_ext}")
    #print(f"F_gate_a_ext shape is {F_gate_a_ext.shape}, F_gate_a_ext is \n{F_gate_a_ext}")
    #print(f"---output is {out}")

    return out


def get_F_gate_i_ext(F_gate_i: Dict[Tuple[int, int, int], int], input_bitwise: Tuple[int, int, int], r: List[List[int]]): 

    F_gate_c_i_ext: Dict[Tuple[int, int], int] = {}
    F_gate_b_i_ext: Dict[int, int] = {}
    a_bitwise, b_bitwise, c_bitwise = input_bitwise

    # inilization
    for key, value in F_gate_i.items():
        a_i, b_i, _ = key
        F_gate_c_i_ext[(a_i, b_i)] = 0
        F_gate_b_i_ext[a_i] = 0
    F_gate_a_i_ext = 0
    
    # multilinear extension in (a, b, c) direction.
    for key, value in F_gate_i.items():
        a_i, b_i, c_i = key
        c_i_extend = single_extension(value, c_i, r[2], c_bitwise)
        F_gate_c_i_ext[(a_i, b_i)] = (F_gate_c_i_ext[(a_i, b_i)] + c_i_extend) % p.prime

    for key, value in F_gate_c_i_ext.items():
        a_i, b_i = key
        b_i_extend = single_extension(value, b_i, r[1], b_bitwise)
        F_gate_b_i_ext[a_i] = (F_gate_b_i_ext[a_i] + b_i_extend) % p.prime

    for key, value in F_gate_b_i_ext.items():
        a_i = key
        a_i_extend = single_extension(value, a_i, r[0], a_bitwise)
        F_gate_a_i_ext = (F_gate_a_i_ext + a_i_extend) % p.prime

    #print(f"F_gate_c_i_ext is {F_gate_c_i_ext}")
    #print(f"F_gate_b_i_ext is {F_gate_b_i_ext}")
    #print(f"F_gate_a_i_ext is {F_gate_a_i_ext}")

    return F_gate_a_i_ext


def get_F_gate_ext(gate_type: str, F_gate: Dict[str, Dict], r: List[List[List[int]]], extension_check=False):

    if not ((gate_type == 'ADD') or (gate_type == 'MULTI')):
        raise ValueError("Error Gate Type !!!")
    
    out = []

    for i, (key, value) in enumerate(F_gate.items()):

        idx = i // 2

        if key == f'The {idx}-th Layer Input Size':
            size = value
            c_bitwise = int(np.log2(size['c']))
            b_bitwise = int(np.log2(size['b']))
            a_bitwise = int(np.log2(size['a']))
            input_bitwise = (a_bitwise, b_bitwise, c_bitwise)

        if key == f'The {idx}-th Layer {gate_type}-Gate':
            F_gate_i = value
            if extension_check:
                for key, value in F_gate_i.items():
                    #get original value (before extension)
                    a_o_i, b_o_i, c_o_i = key
                    c_o_i_v = [int(char) for char in np.binary_repr(c_o_i, width=c_bitwise)]
                    b_o_i_v = [int(char) for char in np.binary_repr(b_o_i, width=b_bitwise)]
                    a_o_i_v = [int(char) for char in np.binary_repr(a_o_i, width=a_bitwise)]
                    r_o_i = [a_o_i_v, b_o_i_v, c_o_i_v]

                    # Points at {0, 1}^n should be equal before and after extension.
                    if not get_F_gate_i_ext(F_gate_i, input_bitwise, r_o_i) == value:
                        raise ValueError("Multilinear Extension Error ! Value unequal !!!")
                return [0]            
            else:
                out.append(get_F_gate_i_ext(F_gate_i, input_bitwise, r[idx]))

    return out



def _get_F_gate_ext_g_t(gate_type: str, F_gate: Dict[str, Dict], r: List[List[List[int]]]):

    if not ((gate_type == 'ADD') or (gate_type == 'MULTI')):
        raise ValueError("Error Gate Type !!!")

    A_F_out = []

    for i, (key, value) in enumerate(F_gate.items()):

        idx = i // 2

        if key == f'The {idx}-th Layer Input Size':
            size = value
            c_bitwidth = int(np.log2(size['c']))
            b_bitwidth = int(np.log2(size['b']))
            a_bitwidth = int(np.log2(size['a']))
            abc_bitwidth = a_bitwidth + b_bitwidth + c_bitwidth           
            bc_bitwidth = b_bitwidth + c_bitwidth
            A_F = np.zeros((2**abc_bitwidth), dtype='int32')
            #print(f"A_F is {A_F}, shape is {np.shape(A_F)}")
        
        if key == f'The {idx}-th Layer {gate_type}-Gate':
            F_gate_i = value

            # Initalize F(a, b, c) from dictionary
            for key, init_value in F_gate_i.items():
                a_o_i, b_o_i, c_o_i = key
                init_idx = a_o_i*2**(bc_bitwidth) + b_o_i*2**(c_bitwidth) + c_o_i
                A_F[init_idx] = init_value
            #print(f"A_F is {A_F}")

            # Precompute F(a*, b, c) 
            for a_i in range(a_bitwidth):
                for a_i_bits in range(2**(abc_bitwidth-a_i-1)):
                    A_F[a_i_bits] = ( A_F[a_i_bits]*(1 - r[idx][0][a_i]) + A_F[a_i_bits+2**(abc_bitwidth-a_i-1)]*r[idx][0][a_i] ) % p.prime

            A_F_i = A_F[:2**bc_bitwidth]
            A_F_out.append(A_F_i)
            #print(f"A_F_i is {A_F_i}, shape is {np.shape(A_F_i)}") 
    return A_F_out  

def get_F_gate_ext_g_t(F_gate_add: Dict[str, Dict], F_gate_multi: Dict[str, Dict], r: List[List[List[int]]], F_W: List[List[int]]):

    g_t = []
    sum = []

    if not len(F_gate_add) == len(F_gate_multi):
        raise ValueError("F_gate_add not equal to F_gate_multi !!!")
    
    A_F_add = _get_F_gate_ext_g_t('ADD', F_gate_add, r)
    A_F_multi = _get_F_gate_ext_g_t('MULTI', F_gate_multi, r)

    if not len(A_F_add) == len(A_F_multi):
        raise ValueError("A_F_add not equal to A_F_multi !!!")

    for i, (key, value) in enumerate(F_gate_add.items()):

        idx = i // 2

        if key == f'The {idx}-th Layer Input Size':
            size = value
            c_bitwidth = int(np.log2(size['c']))
            b_bitwidth = int(np.log2(size['b']))  
            bc_bitwidth = b_bitwidth + c_bitwidth
            r_bc = r[idx][1] + r[idx][2]
            
            F_W_add_i = np.zeros_like(A_F_add[idx])
            F_W_multi_i = np.zeros_like(A_F_multi[idx])
            g_t_i = np.zeros((bc_bitwidth, 3), dtype='int32')

            # Evaluation
            # W_add(b, c) = W(b) + W(c)
            # W_multi(b, c) = W(b) * W(c)
            for n in range(2**b_bitwidth):
                for m in range(2**c_bitwidth):
                    F_W_add_i[(n<<c_bitwidth) + m] = (F_W[idx][n] + F_W[idx][m]) % p.prime
                    F_W_multi_i[(n<<c_bitwidth) + m] = (F_W[idx][n] * F_W[idx][m]) % p.prime
            #print(f"F_W_add_i is {F_W_add_i}, \n F_W_multi_i is {F_W_multi_i}")

            # Extension
            # f(b, c) = add(a*, b, c)*W_add(b, c) + multi(a*, b, c)*W_multi(b, c)
            sum_i = (A_F_add[idx]*F_W_add_i + A_F_multi[idx]*F_W_multi_i).sum() % p.prime
            sum.append(sum_i)
            for bit in range(bc_bitwidth):
                for b in range(2**(bc_bitwidth-bit-1)):
                    for t in range(3):
                        A_F_add_t = ( A_F_add[idx][b]*(1 - t) + A_F_add[idx][b+2**(bc_bitwidth-bit-1)]*t ) % p.prime
                        A_F_multi_t = ( A_F_multi[idx][b]*(1 - t) + A_F_multi[idx][b+2**(bc_bitwidth-bit-1)]*t ) % p.prime
                        F_W_add_i_t = ( F_W_add_i[b]*(1 - t) + F_W_add_i[b+2**(bc_bitwidth-bit-1)]*t ) % p.prime
                        F_W_multi_i_t = ( F_W_multi_i[b]*(1 - t) + F_W_multi_i[b+2**(bc_bitwidth-bit-1)]*t ) % p.prime
                        g_t_i[bit, t] = ( g_t_i[bit, t] + A_F_add_t*F_W_add_i_t + A_F_multi_t*F_W_multi_i_t ) % p.prime

                    A_F_add[idx][b] = ( A_F_add[idx][b]*(1 - r_bc[bit]) + A_F_add[idx][b+2**(bc_bitwidth-bit-1)]*r_bc[bit] ) % p.prime
                    A_F_multi[idx][b] = ( A_F_multi[idx][b]*(1 - r_bc[bit]) + A_F_multi[idx][b+2**(bc_bitwidth-bit-1)]*r_bc[bit] ) % p.prime
                    F_W_add_i[b] = ( F_W_add_i[b]*(1 - r_bc[bit]) + F_W_add_i[b+2**(bc_bitwidth-bit-1)]*r_bc[bit] ) % p.prime
                    F_W_multi_i[b] = ( F_W_multi_i[b]*(1 - r_bc[bit]) + F_W_multi_i[b+2**(bc_bitwidth-bit-1)]*r_bc[bit] ) % p.prime
            g_t.append(g_t_i)

    #print(f"g_t is {g_t}, sum is {sum}")        
    return sum, g_t


def main():
    parser = argparse.ArgumentParser(description='specify gate type')
    parser.add_argument("--operation", '-o', choices=['ADD', 'MULTI'], default='ADD')
    parser.add_argument("--cali", '-c', action="store_true")
    parser.add_argument("--g_t", '-g', action="store_true")
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
    F_gate_multi = C.get_F_gate('MULTI', gate_list)
    F_W = C.get_F_W(map, final_out)
    print(f"F_W is: \n{F_W}")
    out_1 = get_F_gate_ext(args.operation, F_gate, r)
    print(f"out_1 is: \n{out_1}")

    # extension check
    if args.cali:
        if get_F_gate_ext(args.operation, F_gate, r, extension_check=True) == [0]:
            print("Extension Check Pass !")
    elif args.g_t:
        get_F_gate_ext_g_t(F_gate, F_gate_multi, r, F_W)

    #regular extension
    else:
        out_2 = []
        F_gate = C.get_F_gate_1(args.operation, gate_list)
        for i in range(len(gate_list)):
            out_2.append(get_F_gate_ext_t(F_gate[i], r[i]))
        print(f"out_2 is: \n{out_2}")
    
        if not out_1 == out_2:
            raise ValueError("Multilinear Extension Not Equal To Regular Extension !!!")

if __name__ == '__main__':
    main()
