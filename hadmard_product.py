import sys, os
import random
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
utilities_dir = os.path.join(current_dir, 'utilities')
circuit_dir = os.path.join(current_dir, 'Circuit')
sys.path.append(circuit_dir)
sys.path.append(utilities_dir)

from typing import List, Dict, Tuple
from utilities import parameters as p
from utilities import sumcheck as S
from Circuit import circuit as C
from Circuit import interactive_proof as I

def hardmard_product():
    # generate N multiplication gates
    gate_list = ['multi' for _ in range(p.N)]

    return [gate_list]

def import_input_data():
    # TODO
    input_data = np.random.randint(0, p.prime, size=(p.N, 2))

    r_out = [random.randint(0, p.prime) for _ in range(p.bitwise)]
    r = [random.randint(0, p.prime) for _ in range(p.bitwise)]

    # useless, only one layer
    mu = []

    return input_data, [[r, []]], r_out, mu


def main():
    gate_list = hardmard_product()

    input_data, r, r_f, mu = import_input_data()
    cir = C.circuit(input_data, gate_list)
    map, output_data = cir.circuit_imp()
    cir.print_map(map)

    Pro = S.Prover()
    Ver = S.Verifier()

    # circuit check
    I.circuit_precheck(input_data, gate_list, r, mu)

    # set up
    vk = I.set_up(gate_list, r, r_f, mu, only_multi=True)
    
    # Proving
    proof, final_out = I.generate_proof(Pro, input_data, gate_list, r, r_f, mu, only_multi=True)
    #print(f"debug:: proof is {proof}")
    
    # Verification
    # Verifier evaluates W_out(z*) on his own from output layer (d-th layer).
    W_out_ext = Ver.multi_ext(final_out, r_f, width=len(r_f))
    I.Verifier(Ver, proof, vk, r, mu, W_out_ext, only_multi=True)

if __name__ == '__main__':
    main()


