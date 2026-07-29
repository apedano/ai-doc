# Cross-attention: a numeric example

This example builds the key/value attention operator from scratch, treating $X$ and $Q$ as **two separate inputs** to the layer. $X$ is projected into keys and values; $Q$ is supplied directly, with no projection tying it back to $X$.


```python
import numpy as np
from mpmath import sqrt
import scipy as sp

# import scipy as sp
```

## Setup

**Memory / source sequence** $X$ — 3 tokens, $d_{model}=2$:

$$X = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}$$

**Projections applied only to $X$:**

$$W_K = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} \qquad W_V = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}$$

**Queries** — given directly as a layer input, *not* derived from $X$. Here we have 2 queries (the number of queries need not match the number of memory tokens):

$$Q = \begin{bmatrix} 1 & 1 \\ 0 & 2 \end{bmatrix}$$

$Q$ only needs to share dimensionality $d_k = 2$ with $K$ — nothing else constrains it.




```python
X = np.array([[1, 0],
               [0, 1],[1, 1]])
W_K=np.array([[0, 1],
               [1, 0]])
W_V=np.array([[1, 1],
               [0, 1]])
Q=np.array([[1, 1],
               [0, 2]])
```

## Step 1 — Project X into K and V

$$K = XW_K = \begin{bmatrix}0&1\\1&0\\1&1\end{bmatrix} \qquad V = XW_V = \begin{bmatrix}1&1\\0&1\\1&2\end{bmatrix}$$






```python
K=np.matmul(X,W_K)
print(f"K={K}")
V=np.matmul(X,W_V)
print(f"V={V}")
```

    K=[[0 1]
     [1 0]
     [1 1]]
    V=[[1 1]
     [0 1]
     [1 2]]
    

## Step 2 — Scores for each query

Scaled by $\sqrt{d_k} = \sqrt{2} \approx 1.414$.

**Query 1** ($[1,1]$): dot with $K_1, K_2, K_3 \to 1,\ 1,\ 2$ → scaled $[0.707,\ 0.707,\ 1.414]$

**Query 2** ($[0,2]$): dot with $K_1, K_2, K_3 \to 2,\ 0,\ 2$ → scaled $[1.414,\ 0,\ 1.414]$




```python
scores_mpf=np.matmul(Q,np.transpose(K))/(sqrt(2))
print(f"scores={scores_mpf}")
# 2. Convert mpf matrix to a standard NumPy float64 array
scores = np.array(scores_mpf, dtype=np.float64)
print(f"scores={scores}")
```

    scores=[[mpf('0.70710678118654746') mpf('0.70710678118654746')
      mpf('1.4142135623730949')]
     [mpf('1.4142135623730949') mpf('0.0') mpf('1.4142135623730949')]]
    scores=[[0.70710678 0.70710678 1.41421356]
     [1.41421356 0.         1.41421356]]
    

## Step 3 — Softmax (row-wise)

**Query 1:** $e^{0.707}=2.028,\ e^{0.707}=2.028,\ e^{1.414}=4.114$, sum $=8.170$

$$a_{1,:} = [0.248,\ 0.248,\ 0.504]$$

**Query 2:** $e^{1.414}=4.114,\ e^{0}=1,\ e^{1.414}=4.114$, sum $=9.228$

$$a_{2,:} = [0.446,\ 0.108,\ 0.446]$$




```python

A=sp.special.softmax(scores,axis=1)
print(f"a={A}")
```

    a=[[0.24825508 0.24825508 0.50348984]
     [0.44580827 0.10838345 0.44580827]]
    

## Step 4 — Output $Y = AV$

$$Y_1 = 0.248[1,1] + 0.248[0,1] + 0.504[1,2] = [0.752,\ 1.504]$$

$$Y_2 = 0.446[1,1] + 0.108[0,1] + 0.446[1,2] = [0.892,\ 1.446]$$

$$Y = \begin{bmatrix} 0.752 & 1.504 \\ 0.892 & 1.446 \end{bmatrix}$$

The output for the first query vector $Q_{1,:} \rightarrow Y_{1,:}$



```python
Y=np.matmul(A,V)
print(f"Y={Y}")
```

    Y=[[0.75174492 1.50348984]
     [0.89161655 1.44580827]]
    

## What changed structurally

| Tensor | Shape | Comes from |
|---|---|---|
| $X$ | $3 \times 2$ | given (memory sequence) |
| $Q$ | $2 \times 2$ | given directly (query sequence) |
| $K = XW_K$ | $3 \times 2$ | projection of $X$ |
| $V = XW_V$ | $3 \times 2$ | projection of $X$ |
| $A = \text{softmax}(QK^\top / \sqrt{d_k})$ | $2 \times 3$ | **rectangular**, not square |
| $O = AV$ | $2 \times 2$ | one output row per query |

The key structural change from a self-attention example (where $Q, K, V$ all derive from the same $X$): $A$ is no longer $n \times n$. It is $n_q \times n_k$ — here, 2 queries by 3 memory tokens. Each row of $O$ is still a $d_v$-dimensional weighted average of the rows of $V$, but the *number* of output rows now matches the number of supplied queries, fully decoupled from how many tokens live in $X$. This is exactly how a Transformer decoder's cross-attention layer reads a fixed encoder memory with however many decoder positions it currently has.
