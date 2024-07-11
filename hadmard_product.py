import sys
import random
import numpy as np
import parameters as p
from typing import List, Dict, Tuple
from circuit import circuit as C
import interactive_proof as I

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

    input_data, r, r_f, mu = import_input_data()
    print(f"gate_list = \n{gate_list} \n input_data = \n{input_data} \n r is {r}")
    cir = C.circuit(input_data, gate_list)
    map, output_data = cir.circuit_imp()
    cir.print_map(map)
    F_multi = C.get_F_gate('MULTI', gate_list)
    print(F_multi)


    # circuit check
    I.circuit_precheck(input_data, gate_list, r, mu)

    # set up
    vk = I.set_up(gate_list, r, r_f, mu, only_multi=True)
    
    # Proving
    proof, final_out = I.generate_proof(input_data, gate_list, r, r_f, mu, only_multi=True)
    
    # Verification
    I.Verifier(proof, vk, r, r_f, mu, final_out, input_data, only_multi=True)

if __name__ == '__main__':
    main()


