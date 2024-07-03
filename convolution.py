import numpy as np
import parameters as p
from sympy import ntt, intt

class ConVolution:
    def __init__(self):
        pass

    def con2d(self, a, w):
        A = np.mat(a)
        W = np.mat(w)
        U = np.mat(np.zeros((p.con_size-p.kernel_size+1, p.con_size-p.kernel_size+1), dtype='int32'))
        rows, cols = A.shape
        for i in range(rows-1):
            for j in range(cols-1):
                U[i,j] = np.multiply(A[i:(i+p.kernel_size),j:(j+p.kernel_size)], W).sum()
                #print(np.multiply(A[i:(i+kernel_size),j:(j+kernel_size)], W))
        return U

    def trim(self, raw):
        raw = raw[(p.con_size*(p.kernel_size-1)+p.kernel_size-2):(p.con_size**2)][::-1]
        result = raw.reshape((p.out_size,p.con_size))[:,0:p.out_size]
        return result

    def shape(self, a, w):
        w_bar = np.zeros_like(a)
        a_bar = a.flatten()[::-1]
        w_bar[0:p.kernel_size,0:p.kernel_size] = w
        w_bar = w_bar.flatten()
        return a_bar, w_bar

    def con1d(self, a, w):
        a_bar, w_bar = self.shape(a, w)
        u_bar = np.convolve(a_bar, w_bar)
        return self.trim(u_bar), a_bar, w_bar

    def ntt_intt(self, a_bar, w_bar):
        #np.array(ntt(a_bar, prime=self.prime))
        #np.array(ntt(w_bar, prime=self.prime))
        prod = np.array(ntt(a_bar, prime=p.prime))*np.array(ntt(w_bar, prime=p.prime))
        u_bar = np.array(intt(prod, prime=p.prime))
        return self.trim(u_bar)

      