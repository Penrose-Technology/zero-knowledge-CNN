import numpy as np
import parameters as p
from typing import List, Dict, Tuple
import argparse
from circuit import circuit as C
from extension import single_extension
import sumcheck as S

def get_F_W_i_ext(F_W_i: List[int], input_bitwise: int, r: List[int]):
    F_W_i_ext = np.array(F_W_i.copy())
    a_bitwise = input_bitwise
    for a_i, value in enumerate(F_W_i_ext):
        F_W_i_ext[a_i] = single_extension(value, a_i, r, a_bitwise)
    out = F_W_i_ext.sum()%p.prime
    return out

def get_F_W_ext(F_W: List[List[int]], r: List[List[int]], extension_check=False):
    
    out = []

    for idx, F_W_i in enumerate(F_W):
        a_bitwise = int(np.log2(len(F_W_i)))
        if extension_check:
            for i, value in enumerate(F_W_i):
                r_o_i = [int(char) for char in np.binary_repr(i, width=a_bitwise)]
                if not get_F_W_i_ext(F_W_i, a_bitwise, r_o_i) == value:
                    raise ValueError("Multilinear Extension Error ! Value unequal !!!")
            return [0]
        else:
            out.append(get_F_W_i_ext(F_W_i, a_bitwise, r[idx]))
    return out

def get_F_W_i_ext_g_t(F_W_i: List[int], r: List[int]):
    P = S.Prover()
    sum, g_t, _ = P.sumcheck(np.array(F_W_i), r, width=len(r))
    return sum, g_t

def get_F_W_ext_g_t(F_W: List[List[int]], r: List[List[int]]):
    g_t_out = []
    sum = []
    for idx, F_W_i in enumerate(F_W):
        sum_i, g_t_out_i = get_F_W_i_ext_g_t(F_W_i, r[idx])
        g_t_out.append(g_t_out_i)
        sum.append(sum_i)
    return sum, g_t_out



def main():
    parser = argparse.ArgumentParser(description='specify gate type')
    parser.add_argument("--cali", '-c', action="store_true")
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
    r = [[40, 41, 43], 
         [51, 52], 
         [61]]
    
    # extension
    F_W = C.get_F_W(map, final_out)
    print(f"F_W is {F_W}")
    #out_1 = get_F_W_ext(F_W, r)
    #print(out_1) 
    sum, g_t = get_F_W_ext_g_t(F_W, r)
    print(f"sum is {sum}, g_t is {g_t}")

    # extension check
    if args.cali:
        if get_F_W_ext(F_W, r, extension_check=True) == [0]:
            print("Extension Check Pass !")    

if __name__ == '__main__':
    main()