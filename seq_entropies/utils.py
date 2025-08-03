import numpy as np
from collections import Counter
from entropy_estimators import Entropy 

def int_encode(items):
    """
    Encode a sequence of hashable items as integers.

    Parameters
    ----------
    items : list
        List or sequence of hashable items.

    Returns
    -------
    encoded_array : np.ndarray
        Array of integers representing items (always np.int32).
    unique_items : dict
        Dictionary mapping original items to their integer codes.
    """
    unique_items = {item: idx for idx, item in enumerate(sorted(set(items)))}
    encoded_array = np.array([unique_items[item] for item in items], dtype=np.int32)
    return encoded_array, unique_items

def embed_seq(sequence, window=2, padding=False):
    """
    Embed a sequence into overlapping subsequences (n-grams).

    Parameters
    ----------
    sequence : list or np.ndarray
        Sequence of items.
    window : int, optional
        Length of each n-gram (default is 2).
    padding : bool, optional
        Whether to add padding to the left of the sequence

    Returns
    -------
    output : list of tuples
        List of window-length n-grams.
    """
    sint, _ = int_encode(sequence)
    if padding:
    	sint =np.array( [-1]*(window-1) + list(sint) + [-1])
    N = len(sequence)
    output = [tuple(sint[i:i+window]) for i in range(N-window+1)]
    return output

def block_entropy(sequence, window=2, method="MLE", base=2):
    """
    Compute block entropy for a given sequence and block size.

    Parameters
    ----------
    sequence : list or np.ndarray
        Input sequence.
    window : int, optional
        Block size (default=2).
    method : str, optional
        Estimation method for entropy (default="MLE").
    base : int, optional
        Logarithm base (default=2).

    Returns
    -------
    float
        Block entropy.
    """
    if window == 0:
        return np.log(len(set(sequence))) / np.log(base)
    elif window > 1:
        counts = list(Counter(embed_seq(sequence, window)).values())
    else:
        counts = list(Counter(sequence).values())
    return Entropy(counts, method=method, base=base)

def block_cond_entropy(sequence, window=2, method="MLE", base=2):
    """
    Compute block conditional entropy H(block_size) - H(block_size-1).

    Parameters
    ----------
    sequence : list or np.ndarray
        Input sequence.
    window : int, optional
        Block size (default=2).
    method : str, optional
        Estimation method for entropy (default="MLE").
    base : int, optional
        Logarithm base (default=2).

    Returns
    -------
    float
        Conditional block entropy.
    """
    if window < 1:
        return 0.0
    counts1 = list(Counter(embed_seq(sequence, window)).values())
    counts0 = list(Counter(embed_seq(sequence, window-1)).values())
    return Entropy(counts1, method=method, base=base) - \
           Entropy(counts0, method=method, base=base)

