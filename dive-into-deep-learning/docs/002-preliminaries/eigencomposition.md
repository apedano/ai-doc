# Eigendecomposition of Matrices

## Table of Contents

1. [Introduction](#introduction)
2. [Eigenvalues and Eigenvectors](#eigenvalues-and-eigenvectors)
3. [The Eigendecomposition Formula](#the-eigendecomposition-formula)
4. [Step-by-Step Calculation](#step-by-step-calculation)
5. [Geometric Interpretation](#geometric-interpretation)
6. [When Does Eigendecomposition Exist?](#when-does-eigendecomposition-exist)
7. [Special Cases](#special-cases)
8. [Applications](#applications)
9. [Eigendecomposition vs SVD](#eigendecomposition-vs-svd)
10. [Code Examples](#code-examples)

---

## Introduction

Eigendecomposition (also called *spectral decomposition*) is the factorization of a square matrix into a canonical form
using its **eigenvalues** and **eigenvectors**. It is one of the most fundamental operations in linear algebra,
underpinning techniques across machine learning, physics, statistics, and engineering.

The core idea: rather than treating a matrix as an opaque transformation, eigendecomposition reveals the **natural axes
** along which the transformation acts as pure scaling.

---

## Eigenvalues and Eigenvectors

### Definition

For a square matrix $\mathbf{A} \in \mathbb{R}^{n \times n}$, a scalar $\lambda$ and a non-zero
vector $\mathbf{v} \neq \mathbf{0}$ satisfy:

$$\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$$

- $\mathbf{v}$ is an **eigenvector** of $\mathbf{A}$ — a direction unchanged by the transformation
- $\lambda$ is the corresponding **eigenvalue** — the scaling factor along that direction

### Finding Eigenvalues — Characteristic Equation

Rearranging the definition:

$$\mathbf{A}\mathbf{v} - \lambda\mathbf{v} = \mathbf{0} \implies (\mathbf{A} - \lambda\mathbf{I})\mathbf{v} = \mathbf{0}$$

For a non-trivial solution ($\mathbf{v} \neq \mathbf{0}$), the matrix $(\mathbf{A} - \lambda\mathbf{I})$ must be
singular:

$$\det(\mathbf{A} - \lambda\mathbf{I}) = 0$$

This is the **characteristic equation**. Solving it yields eigenvalues $\lambda_1, \lambda_2, \ldots, \lambda_n$.

### Finding Eigenvectors

For each eigenvalue $\lambda_i$, substitute back and solve the homogeneous system:

$$(\mathbf{A} - \lambda_i\mathbf{I})\,\mathbf{v} = \mathbf{0}$$

The solution space — the null space of $(\mathbf{A} - \lambda_i\mathbf{I})$ — gives the corresponding eigenvector(s).

### Worked Example ($2 \times 2$)

$$\mathbf{A} = \begin{pmatrix} 3 & 1 \\ 0 & 2 \end{pmatrix}$$

**Step 1 — Characteristic equation:**

$$\det(\mathbf{A} - \lambda\mathbf{I}) = \det\begin{pmatrix} 3-\lambda & 1 \\ 0 & 2-\lambda \end{pmatrix} = (3-\lambda)(2-\lambda) = 0$$

$$\boxed{\lambda_1 = 3, \quad \lambda_2 = 2}$$

**Step 2 — Eigenvector for $\lambda_1 = 3$:**

$$(\mathbf{A} - 3\mathbf{I})\,\mathbf{v} = \begin{pmatrix} 0 & 1 \\ 0 & -1 \end{pmatrix}\begin{pmatrix} v_1 \\ v_2 \end{pmatrix} = \mathbf{0} \implies v_2 = 0,\; v_1 \text{ free}$$

$$\boxed{\mathbf{v}_1 = \begin{pmatrix} 1 \\ 0 \end{pmatrix}}$$

**Step 3 — Eigenvector for $\lambda_2 = 2$:**

$$(\mathbf{A} - 2\mathbf{I})\,\mathbf{v} = \begin{pmatrix} 1 & 1 \\ 0 & 0 \end{pmatrix}\begin{pmatrix} v_1 \\ v_2 \end{pmatrix} = \mathbf{0} \implies v_1 = -v_2$$

$$\boxed{\mathbf{v}_2 = \begin{pmatrix} -1 \\ 1 \end{pmatrix}}$$

---

## The Eigendecomposition Formula

If $\mathbf{A}$ has $n$ linearly independent eigenvectors, it can be decomposed as:

$$\boxed{\mathbf{A} = \mathbf{P}\,\mathbf{\Lambda}\,\mathbf{P}^{-1}}$$

where:

| Symbol                 | Description                                                   |
|------------------------|---------------------------------------------------------------|
| $\mathbf{P}$           | Matrix whose **columns are the eigenvectors** of $\mathbf{A}$ |
| $\mathbf{\Lambda}$ | **Diagonal matrix** of eigenvalues                            |
| $\mathbf{P}^{-1}$      | Inverse of the eigenvector matrix                             |

Explicitly:

$$\mathbf{P} = \begin{pmatrix} \mathbf{v}_1 & \mathbf{v}_2 & \cdots & \mathbf{v}_n \end{pmatrix}, \qquad \mathbf{\Lambda} = \begin{pmatrix} \lambda_1 & 0 & \cdots & 0 \\ 0 & \lambda_2 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & \lambda_n \end{pmatrix}$$

### For Symmetric Matrices

When $\mathbf{A} = \mathbf{A}^\top$ (symmetric), eigendecomposition takes the special orthogonal form:

$$\mathbf{A} = \mathbf{Q}\,\mathbf{\Lambda}\,\mathbf{Q}^\top$$

where $\mathbf{Q}$ is **orthogonal** ($\mathbf{Q}^{-1} = \mathbf{Q}^\top$), with orthonormal eigenvectors as columns.
This is always possible for real symmetric matrices by the **Spectral Theorem**.

---

## Step-by-Step Calculation

Given a matrix $\mathbf{A} \in \mathbb{R}^{n \times n}$:

**Step 1 — Compute eigenvalues.** Solve the characteristic equation:

$$\det(\mathbf{A} - \lambda\mathbf{I}) = 0 \quad \longrightarrow \quad \lambda_1, \lambda_2, \ldots, \lambda_n$$

**Step 2 — Compute eigenvectors.** For each $\lambda_i$, solve:

$$(\mathbf{A} - \lambda_i\mathbf{I})\,\mathbf{v}_i = \mathbf{0} \quad \longrightarrow \quad \mathbf{v}_i$$

**Step 3 — Construct $\mathbf{P}$ and $\mathbf{\Lambda}$:**

$$\mathbf{P} = \bigl[\,\mathbf{v}_1 \;\big|\; \mathbf{v}_2 \;\big|\; \cdots \;\big|\; \mathbf{v}_n\,\bigr], \qquad \mathbf{\Lambda} = \operatorname{diag}(\lambda_1,\, \lambda_2,\, \ldots,\, \lambda_n)$$

**Step 4 — Compute $\mathbf{P}^{-1}$** using standard matrix inversion (or $\mathbf{Q}^\top$ if $\mathbf{A}$ is
symmetric).

**Step 5 — Verify:**

$$\mathbf{P}\,\mathbf{\Lambda}\,\mathbf{P}^{-1} \stackrel{?}{=} \mathbf{A}$$

---

## Geometric Interpretation

A matrix $\mathbf{A}$ represents a linear transformation. Eigendecomposition breaks $\mathbf{A}\mathbf{x}$ into three
steps:

$$\mathbf{A}\mathbf{x} = \mathbf{P}\,\mathbf{\Lambda}\,\mathbf{P}^{-1}\mathbf{x}$$

1. $\mathbf{P}^{-1}\mathbf{x}$ — Express $\mathbf{x}$ in the eigenvector coordinate system
2. $\mathbf{\Lambda}(\cdots)$ — Scale each coordinate by the corresponding eigenvalue $\lambda_i$
3. $\mathbf{P}(\cdots)$ — Transform back to the original coordinate system

> **Key insight:** In the eigenvector basis, the transformation is *purely diagonal* — it only stretches or compresses
> along each natural axis. Eigendecomposition finds the "natural axes" of the transformation.

### What Eigenvalues Tell You

| Condition                | Effect on $\mathbf{v}$                      |
|--------------------------|---------------------------------------------|
| $\lambda > 1$            | Stretches                                   |
| $0 < \lambda < 1$        | Compresses                                  |
| $\lambda = 1$            | No change                                   |
| $\lambda = 0$            | Collapses to $\mathbf{0}$ (singular matrix) |
| $\lambda < 0$            | Flips direction and scales                  |
| $\lambda \in \mathbb{C}$ | Rotation involved                           |

---

## When Does Eigendecomposition Exist?

### Always Exists (over $\mathbb{C}$)

By the **Fundamental Theorem of Algebra**, any $n \times n$ matrix has exactly $n$ eigenvalues (counting algebraic
multiplicity) over $\mathbb{C}$.

### Diagonalizable (Real Eigendecomposition)

$\mathbf{A}$ is **diagonalizable** if and only if it has $n$ linearly independent eigenvectors, i.e.:

$$\sum_{i} \dim\,\ker(\mathbf{A} - \lambda_i \mathbf{I}) = n$$

**Sufficient conditions:**

- $\mathbf{A}$ has $n$ **distinct** eigenvalues
- $\mathbf{A}$ is **symmetric** ($\mathbf{A} = \mathbf{A}^\top$)
- $\mathbf{A}$ is **normal** ($\mathbf{A}\mathbf{A}^\top = \mathbf{A}^\top\mathbf{A}$)

### Defective Matrices (Not Diagonalizable)

Occurs when the **geometric multiplicity** is strictly less than the **algebraic multiplicity** for some eigenvalue.
Example:

$$\mathbf{A} = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}$$

Here $\lambda = 1$ has algebraic multiplicity $2$, but $\dim\ker(\mathbf{A} - \mathbf{I}) = 1$ — only one independent
eigenvector. Thus $\mathbf{A}$ is **not diagonalizable**.

---

## Special Cases

### Symmetric Matrices ($\mathbf{A} = \mathbf{A}^\top$)

- All eigenvalues $\lambda_i \in \mathbb{R}$
- Eigenvectors from distinct eigenvalues are **orthogonal**: $\mathbf{v}_i^\top \mathbf{v}_j = 0$ for $i \neq j$
- Always diagonalizable: $\mathbf{A} = \mathbf{Q}\mathbf{\Lambda}\mathbf{Q}^\top$

### Positive Definite Matrices

- All eigenvalues satisfy $\lambda_i > 0$
- Equivalent condition: $\mathbf{x}^\top\mathbf{A}\mathbf{x} > 0$ for all $\mathbf{x} \neq \mathbf{0}$

### Orthogonal Matrices ($\mathbf{Q}^\top\mathbf{Q} = \mathbf{I}$)

- All eigenvalues satisfy $|\lambda_i| = 1$ (i.e., $\lambda_i = \pm 1$ or complex on the unit circle)
- Represent pure rotations and/or reflections

### Diagonal Matrices

- Eigenvalues **are** the diagonal entries: $\lambda_i = a_{ii}$
- Eigenvectors are the standard basis vectors $\mathbf{e}_1, \mathbf{e}_2, \ldots, \mathbf{e}_n$

### Idempotent Matrices ($\mathbf{A}^2 = \mathbf{A}$)

- Eigenvalues satisfy $\lambda_i^2 = \lambda_i$, so $\lambda_i \in \{0, 1\}$
- Represent orthogonal projection operators

---

## Applications

### 1. Principal Component Analysis (PCA)

Eigendecompose the covariance matrix $\mathbf{\Sigma} = \mathbf{Q}\mathbf{\Lambda}\mathbf{Q}^\top$. The columns
of $\mathbf{Q}$ are the **principal components** — directions of maximum variance — and the eigenvalues $\lambda_i$
quantify the variance explained by each component.

### 2. Matrix Powers

Computing $\mathbf{A}^k$ naively requires $O(n^3 k)$ operations. With eigendecomposition:

$$\mathbf{A}^k = \mathbf{P}\,\mathbf{\Lambda}^k\,\mathbf{P}^{-1}, \qquad \mathbf{\Lambda}^k = \operatorname{diag}\!\left(\lambda_1^k,\, \lambda_2^k,\, \ldots,\, \lambda_n^k\right)$$

This reduces the cost to a single decomposition plus $O(n)$ scalar exponentiations.

### 3. Systems of Differential Equations

For the system $\dfrac{d\mathbf{x}}{dt} = \mathbf{A}\mathbf{x}$, the closed-form solution is:

$$\mathbf{x}(t) = \mathbf{P}\,e^{\mathbf{\Lambda} t}\,\mathbf{P}^{-1}\,\mathbf{x}(0), \qquad e^{\mathbf{\Lambda} t} = \operatorname{diag}\!\left(e^{\lambda_1 t},\, e^{\lambda_2 t},\, \ldots,\, e^{\lambda_n t}\right)$$

### 4. Google's PageRank

The PageRank vector $\mathbf{\pi}$ is the **dominant eigenvector** of the stochastic link matrix $\mathbf{M}$:

$$\mathbf{M}\mathbf{\pi} = \lambda_{\max}\mathbf{\pi}, \qquad \lambda_{\max} = 1$$

### 5. Stability Analysis

A linear dynamical system $\mathbf{x}_{k+1} = \mathbf{A}\mathbf{x}_k$ is **asymptotically stable** if and only if:

$$|\lambda_i| < 1 \quad \forall\, i \qquad \text{(discrete time)}$$

For the continuous system $\dot{\mathbf{x}} = \mathbf{A}\mathbf{x}$:

$$\operatorname{Re}(\lambda_i) < 0 \quad \forall\, i \qquad \text{(continuous time)}$$

### 6. Spectral Clustering

Graph partitioning uses eigendecomposition of the **graph Laplacian** $\mathbf{L} = \mathbf{D} - \mathbf{W}$,
where $\mathbf{D}$ is the degree matrix and $\mathbf{W}$ is the adjacency matrix. Clusters correspond to the
eigenvectors associated with the smallest non-zero eigenvalues.

### 7. Quantum Mechanics

Every observable $\hat{O}$ is a Hermitian operator. Its eigenvalues $\lambda_i$ are the **measurable values**, and the
eigenvectors $|\psi_i\rangle$ are the corresponding quantum states:

$$\hat{O}\,|\psi_i\rangle = \lambda_i\,|\psi_i\rangle$$

---

## Eigendecomposition vs SVD

Singular Value Decomposition (SVD) generalizes eigendecomposition to **any** $m \times n$ matrix:

$$\mathbf{A} = \mathbf{U}\,\mathbf{\Sigma}\,\mathbf{V}^\top$$

| Property         | Eigendecomposition                                           | SVD                                                         |
|------------------|--------------------------------------------------------------|-------------------------------------------------------------|
| Matrix shape     | Square only                                                  | Any ($m \times n$)                                          |
| Always exists    | No (requires $n$ indep. eigenvectors)                        | Yes                                                         |
| Decomposition    | $\mathbf{A} = \mathbf{P}\mathbf{\Lambda}\mathbf{P}^{-1}$ | $\mathbf{A} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^\top$ |
| Diagonal entries | Eigenvalues $\lambda_i$ (can be $< 0$)                       | Singular values $\sigma_i \geq 0$                           |
| Left/right bases | Same basis $\mathbf{P}$                                      | Different: $\mathbf{U} \neq \mathbf{V}$                     |
| Best for         | Symmetric/square matrices                                    | General matrices, PCA, LSA                                  |

> For symmetric positive semi-definite matrices, the two coincide: $\sigma_i = \lambda_i$
> and $\mathbf{U} = \mathbf{V} = \mathbf{Q}$.

---

## Code Examples

### Python (NumPy) — General Matrix

```python
import numpy as np

A = np.array([[4, 2],
              [1, 3]])

# Compute eigenvalues and eigenvectors
eigenvalues, P = np.linalg.eig(A)

print("Eigenvalues:", eigenvalues)
print("Eigenvector matrix P:\n", P)

# Reconstruct Λ
Lambda = np.diag(eigenvalues)

# Verify: A = P @ Λ @ P⁻¹
P_inv = np.linalg.inv(P)
A_reconstructed = P @ Lambda @ P_inv

print("\nReconstructed A:\n", np.round(A_reconstructed, 10))
```

### Python — Symmetric Matrix (use `eigh` for stability)

```python
import numpy as np

A = np.array([[4, 2],
              [2, 3]])

# eigh guarantees real eigenvalues and orthonormal eigenvectors
eigenvalues, Q = np.linalg.eigh(A)

print("Eigenvalues:", eigenvalues)
print("Orthonormal eigenvectors Q:\n", Q)

# Verify orthogonality: Q @ Qᵀ ≈ I
print("\nQ @ Qᵀ:\n", np.round(Q @ Q.T, 10))

# Reconstruct: A = Q @ diag(λ) @ Qᵀ
A_reconstructed = Q @ np.diag(eigenvalues) @ Q.T
print("\nReconstructed A:\n", np.round(A_reconstructed, 10))
```

### Python — Matrix Power via Eigendecomposition

```python
import numpy as np


def matrix_power_eigen(A, k):
    """Compute A^k using eigendecomposition: A^k = P Λ^k P⁻¹"""
    eigenvalues, P = np.linalg.eig(A)
    P_inv = np.linalg.inv(P)
    Lambda_k = np.diag(eigenvalues ** k)
    return P @ Lambda_k @ P_inv


A = np.array([[3, 1], [0, 2]])
print("A^10 =\n", np.round(matrix_power_eigen(A, 10)))
print("Verify:\n", np.round(np.linalg.matrix_power(A, 10)))
```

---

## Summary

The eigendecomposition $\mathbf{A} = \mathbf{P}\mathbf{\Lambda}\mathbf{P}^{-1}$ in five steps:

| Step | Action                                                                           | Result                                            |
|------|----------------------------------------------------------------------------------|---------------------------------------------------|
| 1    | Solve $\det(\mathbf{A} - \lambda\mathbf{I}) = 0$                                 | Eigenvalues $\lambda_1, \ldots, \lambda_n$        |
| 2    | Solve $(\mathbf{A} - \lambda_i\mathbf{I})\mathbf{v}_i = \mathbf{0}$              | Eigenvectors $\mathbf{v}_1, \ldots, \mathbf{v}_n$ |
| 3    | Build $\mathbf{P} = [\mathbf{v}_1 \mid \cdots \mid \mathbf{v}_n]$                | Eigenvector matrix                                |
| 4    | Build $\mathbf{\Lambda} = \operatorname{diag}(\lambda_1, \ldots, \lambda_n)$ | Eigenvalue matrix                                 |
| 5    | Verify $\mathbf{P}\mathbf{\Lambda}\mathbf{P}^{-1} = \mathbf{A}$              | Confirm decomposition                             |

**Key conditions:**

- ✅ **Exists** when $\mathbf{A}$ has $n$ linearly independent eigenvectors
- ✅ **Simplified** for symmetric $\mathbf{A}$: use $\mathbf{A} = \mathbf{Q}\mathbf{\Lambda}\mathbf{Q}^\top$ with
  orthogonal $\mathbf{Q}$
- ❌ **Fails** for defective matrices where geometric multiplicity $<$ algebraic multiplicity