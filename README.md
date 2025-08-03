# seq-entropies

**Fast sequence entropy and Lempel-Ziv complexity estimators (LZ76, LZ77, etc.)**

`seq_entropies` provides efficient Python and Cython implementations for Lempel-Ziv factorization (LZ76, ZL77), block entropy, and conditional entropy, designed for symbolic or numerical sequence analysis.

---

## Features

- **LZ76 and ZL77**: Fast, Cython-based estimators of Lempel-Ziv complexity.
- **Block Entropy**: Maximum likelihood (and custom) estimation of block entropy for sequences.
- **Conditional Entropy**: Block-based conditional entropy estimates.
- **Utility functions**: Sequence embedding and integer encoding.

---

## Requirements

- [nltk](https://www.nltk.org/)
- numpy
- [entropy_estimators](https://github.com/fermosc24/entropy-estimators) (must be installed manually, see below)

---

## Installation

### 1. Install Required Dependency

Before installing this package, install `entropy_estimators` from GitHub:

```bash
pip install git+https://github.com/fermosc24/entropy-estimators.git
```

### 2. Install seq_entropies

Clone the repository (or copy the files), then run:

```bash
git clone https://github.com/fermosc24/seq-entropies.git
cd seq-entropies
pip install -e .
```

Requirements:  
- Python 3.7+  
- numpy  
- Cython (for build)  

---

## Public API

- `int_encode(items)`
  - **Purpose:** Encodes a sequence of hashable items as a numpy array of integers, returning both the encoded array and a mapping dictionary.
  - **Usage:** `encoded, mapping = int_encode(['a', 'b', 'a'])`

- `embed_seq(sequence, window=2)`
  - **Purpose:** Returns overlapping n-grams (tuples) from the sequence using a sliding window.
  - **Usage:** `ngrams = embed_seq(['a', 'b', 'c', 'd'], window=2)`

- `block_entropy(sequence, window=2, method="ML", base=2)`
  - **Purpose:** Estimates the block entropy of the sequence for a given block size (window).
  - **Usage:** `H = block_entropy(['a', 'b', 'a'], window=2)`

- `block_cond_entropy(sequence, window=2, method="ML", base=2)`
  - **Purpose:** Computes the conditional entropy: H(window) - H(window-1) for the sequence.
  - **Usage:** `C = block_cond_entropy(['a', 'b', 'a'], window=2)`

- `LZ76(sequence, base=2)`
  - **Purpose:** Computes the LZ76 factorization and complexity of the sequence using a fast Cython implementation.
  - **Returns:** Tuple of indices and complexity values.
  - **Usage:** `N, L = LZ76(['a', 'b', 'a', 'b'])`

- `ZL77(sequence, base=2)`
  - **Purpose:** Computes the LZ77 factorization and complexity of the sequence using a fast Cython implementation.
  - **Returns:** Tuple of indices and complexity values.
  - **Usage:** `N, L = ZL77(['a', 'b', 'a', 'b'])`

---

## Usage

```python
import seq_entropies as se

# Example symbolic sequence
seq = ['a', 'b', 'a', 'b', 'c', 'a', 'b']

# Integer encoding and embedding
encoded, mapping = se.int_encode(seq)
print("Integer encoding:", encoded)
print("Mapping:", mapping)

ngrams = se.embed_seq(seq, window=3)
print("3-grams:", ngrams)

# Lempel-Ziv Complexity (LZ76 & ZL77)
N_lz76, L_lz76 = se.LZ76(seq)
print("LZ76:", N_lz76, L_lz76)

N_zl77, L_zl77 = se.ZL77(seq)
print("ZL77:", N_zl77, L_zl77)

# Block entropy (default ML estimator)
H2 = se.block_entropy(seq, window=2,method="MLE")
print("Block entropy (window=2):", H2)

# Block conditional entropy (default ML estimator)
C2 = se.block_cond_entropy(seq, window=2,method="MLE")
print("Block conditional entropy (window=2):", C2)


print("Block entropy with CWJ estimator:",
      se.block_entropy(seq, window=2, method="CWJ"))
```
---

## References

- J. M. Amigó, J. Szczepański, E. Wajnryb, and M. V. Sánchez‑Vives. *Estimating the Entropy Rate of Spike Trains via Lempel‑Ziv Complexity*. Neural Computation, **16**(4): 717–736, 2004. [https://doi.org/10.1162/089976604322860677](https://doi.org/10.1162/089976604322860677)

- T. M. Cover & J. A. Thomas. *Elements of Information Theory*, 2nd Edition. Wiley‑Interscience, 2006. [https://doi.org/10.1002/0471200611](https://doi.org/10.1002/0471200611)

- A. Lempel & J. Ziv. *On the Complexity of Finite Sequences*. IEEE Transactions on Information Theory, **22**(1): 75–81, 1976. [https://doi.org/10.1109/TIT.1976.1055501](https://doi.org/10.1109/TIT.1976.1055501)

- A. Lesne, J.-L. Blanc, and L. Pezard. *Entropy Estimation of Very Short Symbolic Sequences*. Physical Review E, **79**(4 Pt 2): 046208, April 2009. [https://doi.org/10.1103/PhysRevE.79.046208](https://doi.org/10.1103/PhysRevE.79.046208)

- J. Ziv & A. Lempel. *A Universal Algorithm for Sequential Data Compression*. IEEE Transactions on Information Theory, **23**(3): 337–343, 1977. [https://doi.org/10.1109/TIT.1977.1055714](https://doi.org/10.1109/TIT.1977.1055714)


---

## License

MIT License

---

**Developed by Fermín Moscoso del Prado Martín**
