import numpy as np
from sage.all import *
import matplotlib.pyplot as plt
import json
import os


def quant_conv_and_relu(data_size: int, kernel_size: int, group_length: int, lookup_ceil: int, lookup_floor: int):
    data = np.random.normal(loc=0, scale=1, size=data_size**2)
    data_scaled = data / np.max(np.abs(data))
    x_i = data_scaled.reshape(data_size, data_size)

    kernel = np.random.normal(loc=0, scale=1, size=kernel_size**2)
    kernel_scaled = kernel / np.max(np.abs(kernel))
    w_i = kernel_scaled.reshape(kernel_size, kernel_size)

    s_x_i, z_x_i = scalar_and_zero(x_i)
    s_w_i, z_w_i = scalar_and_zero(w_i)
    y_i = con2d(x_i, w_i)
    r_i = np.array([[i if i > 0 else 0 for i in row] for row in y_i])
    s_r_i, z_r_i = scalar_and_zero(r_i)
    assert z_r_i == 0, "zero point of relu is not 0!"

    x_i_q = np.round(x_i / s_x_i + z_x_i)
    w_i_q = np.round(w_i / s_w_i + z_w_i)
    r_i_q = np.round(r_i / s_r_i + z_r_i)
    m = (s_x_i * s_w_i) / s_r_i

    ## quantization in finite field
    p = 52435875175126190479447740508185965837690552500527637822603658699938581184513
    Fp = GF(p)
    bitwidth = 8
    int_scalar = 24
    m_Fp = int(m * (1 << int_scalar))

    # Arithmetic in Fp
    y_i_q_Fp = con2d(m_Fp * (np.array(x_i_q) - np.ones(x_i_q.shape) * z_x_i), 
                           np.array(w_i_q) - np.ones(w_i_q.shape) * z_w_i
                           )
    y_i_q_Fp = np.array(Matrix(Fp, y_i_q_Fp))

    # adjustment: truncate before clip in Fp
    r_i_q_Fp_1 = np.array([[(int(i) >> int_scalar) for i in row] for row in y_i_q_Fp])
    
    # use lookup table instead of clip
    r_i_q_Fp_1 = np.where(
        (r_i_q_Fp_1 > (1 << bitwidth) - 1) & (r_i_q_Fp_1 < ((1 << bitwidth) + lookup_ceil * (1 << bitwidth))),
        255,
        np.where(
            (r_i_q_Fp_1 > ((p >> int_scalar) - lookup_floor * (1 << bitwidth))) & (r_i_q_Fp_1 < (p >> int_scalar)),
            0,
            r_i_q_Fp_1
        )
    )

    quant_info = {
        'x_zero_point': int(z_x_i),
        'x_scalar': s_x_i,
        'w_zero_point': int(z_w_i),
        'w_scalar': s_w_i,
        'r_scalar': s_r_i,
        'bitwidth': bitwidth,
        'lookup_ceil': lookup_ceil,
        'lookup_floor': lookup_floor,
        'int_scalar': int_scalar >> 3,
        'kernel_size': (kernel_size, kernel_size),
        'input_size': (data_size, data_size),
        'group_length': group_length,
    }

    quant_input_data = {
        'x_i': x_i_q.astype(int).tolist(),
        'w_i': w_i_q.astype(int).tolist(),
    }

    quant_output_data_ref = {
        'r_i_q': r_i_q.astype(int).tolist(),
    }
  
    with open ('./else/data/info.json', 'w') as f:
        json.dump(quant_info, f, indent=4)
    with open ('./else/data/input_data.json', 'w') as f:
        json.dump(quant_input_data, f, indent=4)
    with open ('./else/data/output_data_ref.json', 'w') as f:
        json.dump(quant_output_data_ref, f, indent=4)

    err_code = os.system('cargo test conv_relu::test::tests::layer_test')
    if not err_code == 0:
        raise Exception("con_relu run failed!")
    
    with open('./else/data/output_data.json', 'r') as f:
        dic = json.load(f)
    quant_output_data = np.array(dic['r_i_q'])
    
    picture(r_i_q, quant_output_data)



def con2d(x_i: np.ndarray, w_i: np.ndarray):
    x_rows, x_cols = x_i.shape
    w_rows, w_cols = w_i.shape
    y_i = np.zeros((x_rows - w_rows + 1, x_cols - w_cols + 1))

    for i in range(x_rows - w_rows + 1):
        for j in range(x_cols - w_cols + 1):
            y_i[i,j] = (x_i[i:(i + w_rows), j:(j + w_cols)] * w_i).sum()
    return y_i

def scalar_and_zero(r: np.ndarray):
    s = (np.max(r) - np.min(r)) / 255
    z = round(255 - np.max(r) / s)
    return s, z

def picture(y1: np.ndarray, y2: np.ndarray):

    plt.figure(figsize=(16, 10)) 
    y1 = y1.reshape(-1)
    y2 = y2.reshape(-1)
    x = np.arange(len(y1))

    plt.subplot(1, 2, 1)
    plt.plot(x, y1, linestyle='-', label='reference', marker='o', color='r', linewidth=0.5)  
    plt.plot(x, y2, linestyle='-', label='zkcnn', marker='s', color='g', linewidth=0.5)  
    plt.title('Normal vs Snark')
    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.legend(loc="best")

    plt.subplot(1, 2, 2)
    plt.plot(x, (y1 - y2).reshape(-1), label='derivation', marker='', color='b', linewidth=0.5)
    plt.title('Derivation')
    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.legend(loc="best")
    plt.ylim(0, 25)
    plt.yticks(np.arange(0, 25, 5))

    plt.savefig('./else/figures/conv_relu.png')

if __name__ == "__main__":
    
    kernel_size = 3
    data_size = 64
    group_length = 1    # default 1
    lookup_ceil = 2     # default 2 (should be larger when kernel size increases)
    lookup_floor = 3    # default 3 (should be larger when kernel size increases)

    quant_conv_and_relu(data_size, kernel_size, group_length, lookup_ceil, lookup_floor)