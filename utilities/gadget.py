import sys
import numpy as np
import parameters as p
from typing import List
import time
from functools import wraps

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{func.__name__} Running Time: {elapsed_time:.6f} secs")
        return result
    return wrapper


def sumcheck_verification(s_i: np.ndarray, sum_i: int, g_t_i: List[List[int]], ext_i: int, tag=None):

    try:
        if not len(s_i) == len(g_t_i):
            raise ValueError(f'{tag}-Length Check Error !!!')

        if not sum_i == (g_t_i[0][0] + g_t_i[0][1]) % p.prime:
            raise ValueError(f'{tag}-Value Check Error !!! First(0) Round Error !!!')
        for j in range(len(s_i)-1):
            if not s_i[j] == (g_t_i[j+1][0] + g_t_i[j+1][1]) % p.prime:
                raise ValueError(f'{tag}-Value Check Error !!! {j+1} Round Error !!!')
        if not s_i[len(s_i)-1] == ext_i:
            raise ValueError(f'{tag}-Value Check Error !!! Last Round({len(s_i)-1}) Error !!!')
    except ValueError as e:
        print(str(e)) 
        sys.exit(1)

    print(f"{tag} - sumcheck_verification pass......")

    return 0