import sys
import random
import numpy as np
import parameters as p
from typing import List, Dict, Tuple
from circuit import circuit as C
import interactive_proof as I
import sumcheck as S


def hardmard_product():
    # generate N multiplication gates
    gate_list = ['multi' for _ in range(p.N)]

    return [gate_list]

def import_input_data():
    # TODO
    input_data = np.random.randint(0, p.prime, size=(p.N, 2))

    r: List[List[int]] = []
    for i in range(2):
        r_i = [random.randint(0, p.prime) for _ in range(p.bitwise+1)]
        r.append(r_i)
    r_out = [random.randint(0, p.prime) for _ in range(p.bitwise)]

    # useless, only one layer
    mu = []

    return input_data, [r], r_out, mu


def main():
    gate_list = hardmard_product()

    input_data, _, r_f, mu = import_input_data()
    r = [[[634, 628, 751, 499, 160, 533, 611, 639], []]]
    #print(f"gate_list = \n{gate_list} \n input_data = \n{input_data} \n r is {r} \n r_out is {r_f}")
    cir = C.circuit(input_data, gate_list)
    map, output_data = cir.circuit_imp()
    cir.print_map(map)
    F_multi = C.get_F_gate('MULTI', gate_list)
    #print(F_multi)

    Ver = S.Verifier()


    # circuit check
    I.circuit_precheck(input_data, gate_list, r, mu)

    # set up
    vk = I.set_up(gate_list, r, r_f, mu, only_multi=True)
    
    # Proving
    proof, final_out = I.generate_proof(input_data, gate_list, r, r_f, mu, only_multi=True)
    print(f"debug::: proof is {proof}")
    
    # Verification
    # Verifier evaluates W_out(z*) on his own from output layer (d-th layer).
    W_out_ext = Ver.multi_ext(final_out, r_f, width=len(r_f))
    I.Verifier(proof, vk, r, mu, W_out_ext, only_multi=True)

if __name__ == '__main__':
    main()


