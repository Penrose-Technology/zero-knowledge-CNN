from sympy import primitive_root

bitwise = 8
prime = 3*2**8 + 1
#n_root = primitive_root(prime)
n_root = 11
N = 2**bitwise #256
N_inv = N**(prime-2) % prime

con_size = 16
kernel_size = 2
padding_size = 0
stride = 1
out_size = con_size + 2*padding_size - kernel_size + 1