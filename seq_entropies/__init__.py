"""
seq_entropies: Sequence entropy and complexity estimators.

This package provides efficient routines for calculating Lempel-Ziv
complexities (LZ76, LZ77) and related sequence entropy statistics.
"""

from .utils import int_encode, embed_seq,block_entropy,block_cond_entropy
from .lz import LZ76, ZL77

