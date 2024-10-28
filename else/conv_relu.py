import numpy as np
from sage.all import *
import matplotlib.pyplot as plt
import quant

data = np.random.normal(loc=0, scale=1, size=100)
data_scaled = data / np.max(np.abs(data))
data_reshape = data_scaled.reshape(10, 10)

kernel = np.random.normal(loc=0, scale=1, size=9)
kernel_scaled = kernel / np.max(np.abs(kernel))
kernel_reshape = kernel_scaled.reshape(3, 3)

def quant_conv_and_relu(x_i: np.ndarray, w_i: np.ndarray):
    s_x_i, z_x_i = quant.scalar_and_zero(x_i)
    s_w_i, z_w_i = quant.scalar_and_zero(w_i)
    y_i = quant.con2d(x_i, w_i)
    s_y_i, z_y_i = quant.scalar_and_zero(y_i)
    r_i = np.array([[i if i > 0 else 0 for i in row] for row in y_i])
    s_r_i, z_r_i = quant.scalar_and_zero(r_i)
    assert z_r_i == 0, "zero point of relu is not 0!"

    x_i_q = np.round(x_i / s_x_i + z_x_i)
    w_i_q = np.round(w_i / s_w_i + z_w_i)
    r_i_q = np.round(r_i / s_r_i + z_r_i)
    m = (s_x_i * s_w_i) / s_r_i # use s_r_i instead of s_y_i according to relu defination.

    ## quantization in finite field
    p = 18446744073709551557
    Fp = GF(p)
    int_scalar = 10 ** 6
    offset = (p - 1) >> 1
    m_Fp = Fp(int(m * int_scalar))

    # Arithmetic in Fp
    x_i_q_Fp = Matrix(Fp, x_i_q)
    w_i_q_Fp = Matrix(Fp, w_i_q)
    z_x_i_Fp = Matrix(Fp, np.ones(x_i_q.shape) * z_x_i)
    z_w_i_Fp = Matrix(Fp, np.ones(w_i_q.shape) * z_w_i)
    q_x_z_w = quant.con2d(np.array(x_i_q_Fp), np.array(z_w_i_Fp))
    q_w_z_x = quant.con2d(np.array(z_x_i_Fp), np.array(w_i_q_Fp))
    z_x_z_w = quant.con2d(np.array(z_x_i_Fp), np.array(z_w_i_Fp))
    y_i_q_Fp = quant.con2d(np.array(x_i_q_Fp), np.array(w_i_q_Fp)) - q_x_z_w - q_w_z_x + z_x_z_w
    y_i_q_Fp = np.array(Matrix(Fp, y_i_q_Fp))
    #y_i_q_Fp = y_i_q_Fp * m_Fp + offset # z_r_i is zero, no need to add it.
    y_i_q_Fp = y_i_q_Fp * m_Fp # z_r_i is zero, no need to add it.
    
    # clip before truncate in Fp
    #r_i_q_Fp = np.array([[quant.clip(i, offset, 255 * int_scalar + offset) for i in row] for row in y_i_q_Fp])
    #r_i_q_Fp = np.array([[int(i - offset) // int_scalar for i in row] for row in r_i_q_Fp])
    #quant.picture(r_i_q, r_i_q_Fp)

    # adjustment: truncate before clip in Fp
    r_i_q_Fp_1 = np.array([[int(i) // int_scalar for i in row] for row in y_i_q_Fp])
    # use lookup table instead of clip
    r_i_q_Fp_1 = np.where(
        (r_i_q_Fp_1 > 255) & (r_i_q_Fp_1 < (256 + 2 * (1 << 8))),
        255,
        np.where(
            (r_i_q_Fp_1 > (p // int_scalar - 3 * (1 << 8))) & (r_i_q_Fp_1 < (p // int_scalar)),
            0,
            r_i_q_Fp_1
        )
    )

    quant.picture(r_i_q, r_i_q_Fp_1)
    plt.savefig('figures/conv_relu.png')



if __name__ == "__main__":
    quant_conv_and_relu(data_reshape, kernel_reshape)