# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
import numpy as np
cimport numpy as np
from libc.math cimport log

from .utils import int_encode

def LZ76(seq, base=2):
    """
    Compute the LZ76 factorization of a sequence.

    Parameters
    ----------
    seq : list or np.ndarray
        Input sequence of hashable items.
    base : int, optional
        Logarithm base for entropy (default is 2).

    Returns
    -------
    N : np.ndarray
        Indices where new phrases begin.
    L : np.ndarray
        Lempel-Ziv complexity estimates at each phrase.
    """
    cdef np.ndarray[np.int32_t, ndim=1] l2
    l2, _ = int_encode(seq)
    cdef Py_ssize_t n = l2.shape[0]
    cdef list N = []
    cdef list Cn = []
    cdef dict codeDict = {}
    cdef Py_ssize_t i = 0, j
    cdef int cn = 0
    cdef bytes s

    while i < n:
        sequence = l2[:i]
        for j in range(i+1, n+1):
            s = l2[i:j].tobytes()
            if s not in codeDict:
                cn += 1
                N.append(j)
                Cn.append(cn)
                i = j
                codeDict[s] = True
                break
        if j == n:
            break
    if i < n:
        cn += 1
        N.append(n)
        Cn.append(cn)
    N_arr = np.array(N, dtype=np.int32)
    Cn_arr = np.array(Cn, dtype=np.int32)
    L_arr = (Cn_arr + 1) * np.log(Cn_arr) / N_arr
    return N_arr, L_arr / np.log(base)

cdef inline int fastfind(np.int32_t[:] small, np.int32_t[:] big):
    """Return index if small is found in big, -1 otherwise."""
    cdef Py_ssize_t i, j, m, n
    n = big.shape[0]
    m = small.shape[0]
    if m == 0:
        return 0
    if m > n:
        return -1
    for i in range(n - m + 1):
        for j in range(m):
            if big[i + j] != small[j]:
                break
        else:
            return i
    return -1

def ZL77(seq, base=2):
    """
    Compute the ZL77 factorization of a sequence.

    Parameters
    ----------
    seq : list or np.ndarray
        Input sequence of hashable items.
    base : int, optional
        Logarithm base for entropy (default is 2).

    Returns
    -------
    N : np.ndarray
        Indices where new phrases begin.
    L : np.ndarray
        Ziv-Lempel complexity estimates at each phrase.
    """
    cdef np.ndarray[np.int32_t, ndim=1] l2
    l2, _ = int_encode(seq)
    cdef Py_ssize_t n = l2.shape[0]
    cdef list N = []
    cdef list Cn = []
    cdef Py_ssize_t i = 0, j
    cdef int cn = 0
    while i < n:
        sequence = l2[:i]
        for j in range(i+1, n+1):
            s = l2[i:j]
            if fastfind(s, sequence) == -1:
                cn += 1
                N.append(j)
                Cn.append(cn)
                i = j
                break
        if j == n:
            break
    if i < n:
        cn += 1
        N.append(n)
        Cn.append(cn)
    N_arr = np.array(N, dtype=np.int32)
    Cn_arr = np.array(Cn, dtype=np.int32)
    L_arr = (Cn_arr + 1) * np.log(Cn_arr) / N_arr
    return N_arr, L_arr / np.log(base)

