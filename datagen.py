import scipy as sp
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
import io


def test():
    fh = io.BytesIO()
    outfile = "rmat4.4x4.lb.mtx"

    # A = lil_matrix((4, 4))
    # A = A.tocsr()
    A = csr_matrix([[1.0, 2.5, 0, 0], [0, 0, 3.7, 0], [4.3, 0, 5.2, 0], [0, 1.9, 0, 0.8]])
    sp.io.mmwrite(fh, A, comment='\n A matrix for SpMV\n', precision=3)
    print(fh.getvalue().decode('utf-8'))

    with open(outfile, 'w+') as f:
        f.write(fh.getvalue().decode('utf-8'))



from collections.abc import Iterable
import numpy as np

import random
import math
import bisect  # Not too sure what this does...

from typing import overload


PRIMES = [2, 3, 5, 7]

# Global variables to store the prime lookup table (will generate a larger list after defining the `primes()` function)
PRIMES = [2, 3, 5, 7]
MAX_K = len(PRIMES)  # The max prime index (1-indexed)
MAX_N = 10  # The max value for which all primes <= N are contained in the lookup table

def _calculate_modulus(x: Iterable, N):
    """
    Calculate the smallest p that works for given input vector X
    """
    min_p = max(x)
    m = 1
    p = m*N + 1
    while p < min_p:
        p = m*N + 1
        m += 1

    return p

def is_prime(n):
    """ Primality test using the Miller-Rabin algorithm """
    if n < 2:
        return False
    for i in range(2, int(n**0.5)+1):
        if n % i == 0:
            return False
    return True


def calculate_modulus(x: Iterable, N):
    """ TODO: I have a sneaking suspicion that this could be better
    """

    m = int(np.ceil(np.max(x) / N))  # The smallest m such that modulus > max(x)
    while not is_prime(m * N + 1):
        m += 1
    modulus = m * N + 1
    # m = (modulus - 1) // N  # Not sure what this line was meant to do

    return modulus, m


def get_exponents(N):
    for i in range (N):
        for j in range (N):
            yield i*j



def primitive_root(p):
    """
    Find the primitive root of a prime number p
    TODO: Check if this is correct
    """
    if p == 2:
        return 1
    p1 = 2
    p2 = (p-1) // p1

    while True:
        g = random.randint(2, p-1)
        if not (pow(g, (p-1)//p1, p) == 1):
            if not (pow(g, (p-1)//p2, p) == 1):
                return g


def unique_twiddle_factors(omega_N, N, p):
    unique_exp = np.unique([i*j for i in range(N) for j in range(N)])
    print(f"unique:\n{unique_exp}")
    # return np.unique([pow(omega, x, p) for x in unique_exp])  # TODO: Would need to cast x to just builtin int
    return np.unique(np.power(omega_N, unique_exp) % p)


def calculate_twiddle_ntt(x, N):
    """
    param root: Primitive root of unity for the modulus p
    """

    p, m = calculate_modulus(x, N)

    root = primitive_root(p)  # TODO: Implement this primitive_root
    
    omega_N = pow(root, m)
    
    unique_exp = np.unique([i*j for i in range(N) for j in range(N)])

    twiddle_factors = np.unique(np.power(omega_N, unique_exp) % p)
    
    return twiddle_factors

# @overload  # TODO: Figure out how to use this
def calculate_twiddle_ntt(root, m, N, p):
    """
    param root: Primitive root of unity for the modulus p
    """
    
    omega = pow(root, m)
    
    unique_exp = np.unique([i*j for i in range(N) for j in range(N)])

    twiddle_factors = np.unique([pow(omega, x, p) for x in unique_exp])
    
    return twiddle_factors

def calculate_twiddle_ftt(x, N):
    p = calculate_modulus(x, N)
    twiddle_factors = []
    for i in range(N):
        twiddle_factors.append([pow(2, i*j, p) for j in range(N)])
    return twiddle_factors


def calculate_twiddle_ntt(x, N):
    p = calculate_modulus([2, 3, 5, 7], N)
    twiddle_factors = []
    for i in range(N):
        twiddle_factors.append([pow(2, i*j, p) for j in range(N)])
    return twiddle_factors


