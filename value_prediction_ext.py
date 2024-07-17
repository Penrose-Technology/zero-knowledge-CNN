import numpy as np
import parameters as p
from typing import List, Dict, Tuple
import argparse
from circuit import circuit as C
from extension import single_extension
import sumcheck as S

def _get_F_W_i_ext(F_W_i: List[int], input_bitwidth: int, r: List[int]):
    F_W_i_ext = np.array(F_W_i.copy())
    a_bitwidth = input_bitwidth
    for a_i, value in enumerate(F_W_i_ext):
        F_W_i_ext[a_i] = single_extension(value, a_i, r, a_bitwidth)
    out = F_W_i_ext.sum()%p.prime
    return out

def _get_F_W_i_ext_input(F_W_left_in: List[int], F_W_right_in: List[int], input_bitwidth: int, r: List[int]):

    if not len(F_W_left_in) == len(F_W_right_in):
        raise ValueError('input length not equal !!!')
    
    F_W_left = np.array(F_W_left_in.copy())
    F_W_right = np.array(F_W_right_in.copy())
    a_bitwidth = input_bitwidth
    for a_i, value in enumerate(F_W_left):
        F_W_left[a_i] = single_extension(value, a_i, r, a_bitwidth)
    for a_i, value in enumerate(F_W_right):
        F_W_right[a_i] = single_extension(value, a_i, r, a_bitwidth)
    out = (F_W_left.sum()%p.prime, F_W_right.sum()%p.prime)
    return out

def get_F_W_ext(F_W: List[List[int]], F_W_left_in: List[int], F_W_right_in: List[int], r: List[List[int]], extension_check=False):
    
    out = []

    for idx, F_W_i in enumerate(F_W):
        a_bitwidth = int(np.log2(len(F_W_i)))
        if extension_check:
            if idx == 0:
                #print(f"debug::::F_W_left_in is {F_W_left_in}")
                for i, value in enumerate(F_W_left_in):
                    r_o_i = [int(char) for char in np.binary_repr(i, width=a_bitwidth)]
                    out = _get_F_W_i_ext_input(F_W_left_in, F_W_right_in, a_bitwidth, r_o_i)
                    #print(f"debug::::out is {out}, value is {value}")
                    if not out[0] == value:
                        raise ValueError("Multilinear Extension Error ! Value unequal !!!")
            else:
                for i, value in enumerate(F_W_i):
                    r_o_i = [int(char) for char in np.binary_repr(i, width=a_bitwidth)]
                    if not _get_F_W_i_ext(F_W_i, a_bitwidth, r_o_i) == value:
                        raise ValueError("Multilinear Extension Error ! Value unequal !!!")
            return [0]
        else:
            if idx == 0:
                out.append(_get_F_W_i_ext_input(F_W_left_in, F_W_right_in, a_bitwidth, r[idx]))
            else:
                out.append(_get_F_W_i_ext(F_W_i, a_bitwidth, r[idx]))
    return out





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
    F_W_l, F_W_r = C.get_F_W_in_separate(input_data)
    print(f"F_W is {F_W}")
    out = get_F_W_ext(F_W, F_W_l, F_W_r, r)
    print(f"out is {out}")

    # extension check
    if args.cali:
        if get_F_W_ext(F_W, F_W_l, F_W_r, r, extension_check=True) == [0]:
            print("Extension Check Pass !")    

if __name__ == '__main__':
    main()