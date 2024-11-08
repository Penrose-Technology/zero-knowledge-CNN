import numpy as np
from sage.all import *
from fractions import Fraction
import matplotlib.pyplot as plt

data = np.random.normal(loc=0, scale=1, size=100)
data_scaled = data / np.max(np.abs(data))
data_reshape = data_scaled.reshape(10, 10)

kernel = np.random.normal(loc=0, scale=1, size=9)
kernel_scaled = kernel / np.max(np.abs(kernel))
kernel_reshape = kernel_scaled.reshape(3, 3)

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

def scalar_approximation(r: float):
    max_denominator = 255
    fraction_approx = Fraction(r).limit_denominator(max_denominator)
    return fraction_approx.numerator, fraction_approx.denominator

def clip(x, min_value, max_value):
    if x < min_value:
        return min_value
    elif x > max_value:
        return max_value
    else:
        return x
    
def inverse_int_scalar_Fp(Fp, int_scalar: int, a: np.ndarray, b: np.ndarray):
    c = np.array(Matrix(Fp, np.zeros_like(a)))
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            assert a[i][j] == b[i][j] * int_scalar, "value error!"
            if a[i][j] == 0:
                c[i][j] = Fp(0)
            else:
                c[i][j] = Fp(b[i][j]) * (Fp(a[i][j]) ** -1)
    return c

def quan_conv(x_i: np.ndarray, w_i: np.ndarray):
    s_x_i, z_x_i = scalar_and_zero(x_i)
    s_w_i, z_w_i = scalar_and_zero(w_i)
    y_i = con2d(x_i, w_i)
    s_y_i, z_y_i = scalar_and_zero(y_i)


    ## quantization with real scalar
    x_i_q = np.round(x_i / s_x_i + z_x_i)
    w_i_q = np.round(w_i / s_w_i + z_w_i)
    m = (s_x_i * s_w_i) / s_y_i
    q_x_z_w = con2d(x_i_q, np.ones(w_i_q.shape) * z_w_i)
    q_w_z_x = con2d(np.ones(x_i_q.shape) * z_x_i, w_i_q)
    z_x_z_w = con2d(np.ones(x_i_q.shape) * z_x_i, np.ones(w_i_q.shape) * z_w_i)
    y_i_q = con2d(x_i_q, w_i_q) - q_x_z_w - q_w_z_x + z_x_z_w
    y_i_q = y_i_q * m + z_y_i
    y_i_q = y_i_q.clip(0, 255)
    y_i_ref = s_y_i * (y_i_q - z_y_i)
    #print(y_i_q)


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
    z_y_i_Fp = np.array(Matrix(Fp, np.ones(y_i_q.shape) * z_y_i *int_scalar))
    q_x_z_w = con2d(np.array(x_i_q_Fp), np.array(z_w_i_Fp))
    q_w_z_x = con2d(np.array(z_x_i_Fp), np.array(w_i_q_Fp))
    z_x_z_w = con2d(np.array(z_x_i_Fp), np.array(z_w_i_Fp))
    y_i_q_Fp = con2d(np.array(x_i_q_Fp), np.array(w_i_q_Fp)) - q_x_z_w - q_w_z_x + z_x_z_w
    y_i_q_Fp = np.array(Matrix(Fp, y_i_q_Fp))
    y_i_q_Fp = y_i_q_Fp * m_Fp + z_y_i_Fp
    y_i_q_Fp = y_i_q_Fp + offset
    # clip and truncate in Fp
    y_i_q_Fp_clip = np.array([[clip(i, offset, 255 * int_scalar + offset) for i in row] for row in y_i_q_Fp])
    y_i_q_Fp_truncate = np.array([[int(i - offset) // int_scalar for i in row] for row in y_i_q_Fp_clip])
    #print(f"y_i_q_Fp after clip is {y_i_q_Fp_clip - offset}")
    #print(f"y_i_q_Fp after truncate is {y_i_q_Fp_truncate}")
    #print(f"y_i_q_Fp before truncate is {y_i_q_Fp_truncate * int_scalar}")
    #print(f"{inverse_int_scalar_Fp(Fp, int_scalar, y_i_q_Fp_truncate * int_scalar, y_i_q_Fp_truncate)}")
    picture(round(y_i_q), y_i_q_Fp_truncate)
    plt.savefig('figures/conv.png')


def picture(y1: np.ndarray, y2: np.ndarray):

    plt.figure(figsize=(10, 6)) 
    x = [i for i in range(0, len(y1)**2)]

    plt.subplot(1, 2, 1)
    plt.plot(x, y1.reshape(-1), label='y1', marker='o', color='r')  
    plt.plot(x, y2.reshape(-1), label='y2', marker='s', color='g')  
    plt.title('Comparison of Two ndarrays')
    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.legend(loc="best")

    plt.subplot(1, 2, 2)
    plt.plot(x, (y1 - y2).reshape(-1), label='derivation', marker='o', color='r')
    plt.title('Derivation')
    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.legend(loc="best")

    


if __name__ == "__main__":
    quan_conv(data_reshape, kernel_reshape)