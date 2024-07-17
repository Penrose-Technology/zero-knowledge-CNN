from typing import List, Dict, Tuple
import numpy as np
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
import parameters as p

class gate:
    def __init__(self, name: str, left: int, right: int) -> None:
        self.__name = name
        self.__left = int(left)
        self.__right = int(right)

    def get_name(self):
        return self.__name
    
    def get_left(self):
        return self.__left
    
    def get_right(self):
        return self.__right

    def gate_imp(self):
        if self.__name == 'add':
            out = (self.__left + self.__right) % p.prime
        elif self.__name == 'multi':
            out = (self.__left * self.__right) % p.prime
        else:
            raise ValueError ("Illegal gate name!!!")
        return int(out)
    
class layer:
    def __init__(self, label: int, gate_v: List[gate]) -> None:
        self.__label = label
        self.__gate_v = gate_v

    def get_label(self):
        return self.__label
    
    def get_layer(self):
        return self.__gate_v
    
    def get_input(self):
        W = []
        for gate_i in self.__gate_v:
            W.extend([gate_i.get_left(), gate_i.get_right()])
        return W

    def print_layer(self):
        print(f"This is the {self.__label}-th layer")
        for i, gate_i in enumerate(self.__gate_v):
            print(f"Gate Label: {i}, Gate Type: {gate_i.get_name()}, Left Value: {gate_i.get_left()}, Right Value: {gate_i.get_right()}")


    
class circuit:
    def __init__(self, input_data: np.ndarray, gate_list: List[List[str]]) -> None:
        self.input_data = input_data
        self.gate_type = gate_list
        self.gate = gate
        self.layer = layer
        self.__d = len(gate_list)
        
    def circuit_imp(self):
        this_layer: List[gate] = []
        map: List[layer] = []
        final_out = []

        for i in range(self.__d):
            next_input_data = np.array([])
            this_layer.clear()

            if i == 0:
                this_data = self.input_data
            else:
                this_data = this_input_data
            
            for label, data in enumerate(this_data):
                gate_instance = self.gate(self.gate_type[i][label], data[0], data[1])
                this_layer.append(gate_instance)
                next = gate_instance.gate_imp()
                next_input_data = np.append(next_input_data, next)
                if i == self.__d-1:
                    final_out.append(next)

            if i == self.__d-1:
                this_input_data = next_input_data.copy()
            else:
                this_input_data = next_input_data.copy().reshape(-1, 2)
            
            layer_instance = self.layer(i, this_layer.copy())

            map.append(layer_instance)
        
        return map, final_out
    
    def print_map(self, map: List[layer]):
        for layer in map:
            layer.print_layer()


def get_F_gate_1(gate_type: str, gate_list: List[List[str]]):

    F_gate = []

    if gate_type == 'ADD':
        gate_type_name = 'add'
    elif gate_type == 'MULTI':
        gate_type_name = 'multi'
    else:
        raise ValueError("Error Gate Type !!!")
    
    for i, layer in enumerate(gate_list):
        F_gate_i = np.zeros( (len(layer), 2*len(layer), 2*len(layer)) , dtype='int32')
        print(f"F_gate_i shape is {F_gate_i.shape}")
        for j, gate in enumerate(layer):
            if gate == gate_type_name:
                F_gate_i[j][2*j][2*j+1] = 1
        F_gate.append(F_gate_i)
    return F_gate



def get_F_gate(gate_type: str, gate_list: List[List[str]]):

    F_gate = {}

    if gate_type == 'ADD':
        gate_type_name = 'add'
    elif gate_type == 'MULTI':
        gate_type_name = 'multi'
    else:
        raise ValueError("Error Gate Type !!!")

    for i, layer in enumerate(gate_list):
        F_gate_i = {}
        if i == 0:
            input_size = {'a': len(layer), 'b': len(layer)}
            F_gate[f'The {i}-th Layer Input Size'] = input_size
            for j, gate in enumerate(layer):
                if gate == gate_type_name:
                    F_gate_i[(j, j)] = 1
        else:
            input_size = {'a': len(layer), 'b': 2*len(layer), 'c': 2*len(layer)}
            F_gate[f'The {i}-th Layer Input Size'] = input_size
            for j, gate in enumerate(layer):
                if gate == gate_type_name:
                    F_gate_i[(j, 2*j, 2*j+1)] = 1
        F_gate[f'The {i}-th Layer {gate_type}-Gate'] = F_gate_i
    #print(f"F_gate len is {len(F_gate)}")

    return F_gate



def get_F_W(map: List[layer], final_out: List[int]):
    F_W: List[List[int]] = []
    for i, layer in enumerate(map):
        W = layer.get_input()
        F_W.append(W)
    F_W.append(final_out)
    return F_W

def get_F_W_in_separate(input_data):

    F_W_l = [i[0] for i in input_data]
    F_W_r = [i[1] for i in input_data]

    return F_W_l, F_W_r


def main():
    input_data = np.array(
                 [[1,3],
                  [2,2],
                  [4,5],
                  [3,2]])
    gate_list = [['multi', 'multi', 'add', 'multi'],
                 ['add', 'multi']]
                 #['add']]
    
    C = circuit(input_data, gate_list)
    map, final_out = C.circuit_imp()
    C.print_map(map)
    print(f"The circuit final output is {final_out}")

    F_W = get_F_W(map, final_out)
    print(f"F_W is \n {F_W}")

    F_W_l, F_W_r = get_F_W_in_separate(input_data)
    print(f"F_W_l is \n{F_W_l}, \n F_W_r is \n{F_W_r}")

    F_add = get_F_gate_1('ADD', gate_list)
    #print(f"F_add is \n {F_add}")

    F_multi = get_F_gate_1('MULTI', gate_list)
    #print(f"F_multi is \n {F_multi}")
  

if __name__ == '__main__':
    main()