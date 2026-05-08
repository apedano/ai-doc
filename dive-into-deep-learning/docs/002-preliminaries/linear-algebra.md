# Linear algebra

## Scalars

> We denote scalars by ordinary lower-cased letters (e.g., $x$, $y$, and $z$) and the space of all (continuous) 
*real-valued* scalars by $\mathbb{R}$. 

For expedience, we will skip past rigorous definitions of *spaces*: just remember that the expression $x \in \mathbb{R}$
is a formal way to say that $x$ is a real-valued scalar.The symbol $\in$ (pronounced "in") denotes membership in a set.
For example, $x, y \in \{0, 1\}$ indicates that $x$ and $y$ are variables that can only take values $0$ or $1$.

> **Scalars are implemented as tensors that contain only one element.**)

Below, we assign two scalars and perform the familiar addition, multiplication, division, and exponentiation operations

```python
x = torch.tensor(3.0)
y = torch.tensor(2.0)

x + y, x * y, x / y, x**y
```

```text
(tensor(5.), tensor(6.), tensor(1.5000), tensor(9.))
```

## Vectors

We denote vectors by bold lowercase letters, 
(e.g., $\mathbf{x}$, $\mathbf{y}$, and $\mathbf{z}$).

> Vectors are implemented as $1^{\textrm{st}}$-order tensors.
In general, such tensors can have arbitrary lengths,
subject to memory limitations. Caution: in Python, as in most programming languages, vector indices start at $0$, also known as *zero-based indexing*, whereas in linear algebra subscripts begin at $1$ (one-based indexing).

```python
x = torch.arange(3)
x
```

```text
tensor([0, 1, 2])
```

We can refer to an element of a vector by using a subscript.
For example, $x_2$ denotes the second element of $\mathbf{x}$. 
Since $x_2$ is a scalar, we do not bold it.
By default, we visualize vectors 
by stacking their elements vertically.

$$\mathbf{x} =\begin{bmatrix}x_{1}  \\ \vdots  \\x_{n}\end{bmatrix},$$

```python
x[2]
```
```text
tensor(2)
```

## Matrices

Just as scalars are $0^{\textrm{th}}$-order tensors
and vectors are $1^{\textrm{st}}$-order tensors,
matrices are $2^{\textrm{nd}}$-order tensors.
We denote matrices by bold capital letters
(e.g., $\mathbf{X}$, $\mathbf{Y}$, and $\mathbf{Z}$),
and represent them in code by tensors with two axes.
The expression $\mathbf{A} \in \mathbb{R}^{m \times n}$
indicates that a matrix $\mathbf{A}$ 
contains $m \times n$ real-valued scalars,
arranged as $m$ rows and $n$ columns.
When $m = n$, we say that a matrix is *square*.
Visually, we can illustrate any matrix as a table.
To refer to an individual element,
we subscript both the row and column indices, e.g.,
$a_{ij}$ is the value that belongs to $\mathbf{A}$'s
$i^{\textrm{th}}$ row and $j^{\textrm{th}}$ column:

$$\mathbf{A}=\begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \\ \end{bmatrix}.$$
:eqlabel:`eq_matrix_def`


In code, we represent a matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$
by a $2^{\textrm{nd}}$-order tensor with shape ($m$, $n$).
[**We can convert any appropriately sized $m \times n$ tensor 
into an $m \times n$ matrix**] 
by passing the desired shape to `reshape`:

```python
A = torch.arange(6).reshape(3, 2)
A
```

```text
tensor([[0, 1],
        [2, 3],
        [4, 5]])

```

### Traspose

$$
\mathbf{A} \in \mathbb{R}^{m \times n} \rightarrow \mathbf{B} = \mathbf{A}^\top, b_{i,j}=a_{j,i}  
$$



$$
\mathbf{A}^\top =
\begin{bmatrix}
    a_{11} & a_{21} & \dots  & a_{m1} \\
    a_{12} & a_{22} & \dots  & a_{m2} \\
    \vdots & \vdots & \ddots  & \vdots \\
    a_{1n} & a_{2n} & \dots  & a_{mn}
\end{bmatrix}.
$$

So that

$\mathbf{A} \in \mathbb{R}^{m \times n} \rightarrow \mathbf{B} \in \mathbb{R}^{n \times m}$ 

```python
A.T
```

```text
tensor([[0, 2, 4],
        [1, 3, 5]])
```

## Tensors

> Tensors (**give us a generic way of describing extensions to $n^{\textrm{th}}$-order arrays.**)
We call software objects of the *tensor class* "tensors" precisely because they too can have arbitrary numbers of axes.

We denote general tensors by capital letters with a special font face (e.g., $\mathsf{X}$, $\mathsf{Y}$, and $\mathsf{Z}$)
and their indexing mechanism (e.g., $x_{ijk}$ and $[\mathsf{X}]_{1, 2i-1, 3}$) 
follows naturally from that of matrices.

Tensors will become more important when we start working with images.
Each image arrives as a $3^{\textrm{rd}}$-order tensor with axes corresponding to the height, width, and *channel*.
At each spatial location, the intensities of each color (red, green, and blue) are stacked along the channel. 
Furthermore, a collection of images is represented in code by a $4^{\textrm{th}}$-order tensor,
where distinct images are indexed along the first axis. Higher-order tensors are constructed, as were vectors and matrices,
by growing the number of shape components.

```python
image_sizes = 3*20*20
image_tensor = torch.arange(image_sizes).reshape(3, 20, 20)
image_red = image_tensor[0]
image_red.shape
```

```text
torch.Size([20, 20])
```

### Elementwise operations

> Elementwise operations produce outputs that have the same shape as their operands.

#### Sum

```python
A = torch.arange(6, dtype=torch.float32).reshape(2, 3)
B = A.clone()  # Assign a copy of A to B by allocating new memory
A, A + B
```

```text
(tensor([[0., 1., 2.],
         [3., 4., 5.]]),
 tensor([[ 0.,  2.,  4.],
         [ 6.,  8., 10.]]))
```

#### Hadamard product $\odot$

We can spell out the entries 
of the Hadamard product of two matrices 
$\mathbf{A}, \mathbf{B} \in \mathbb{R}^{m \times n}$:


$$
\mathbf{A} \odot \mathbf{B} =
\begin{bmatrix}
    a_{11}  b_{11} & a_{12}  b_{12} & \dots  & a_{1n}  b_{1n} \\
    a_{21}  b_{21} & a_{22}  b_{22} & \dots  & a_{2n}  b_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{m1}  b_{m1} & a_{m2}  b_{m2} & \dots  & a_{mn}  b_{mn}
\end{bmatrix}.
$$

```python
A * B
```

```text
tensor([[ 0.,  1.,  4.],
        [ 9., 16., 25.]])
```

### Scalar operations

> <span style="color:red">Adding or multiplying a scalar and a tensor</span> produces a result with the same shape as the original tensor. Here, each element of the

```python
a = 2
X = torch.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```

```text
(tensor([[[ 2,  3,  4,  5],
          [ 6,  7,  8,  9],
          [10, 11, 12, 13]],
 
         [[14, 15, 16, 17],
          [18, 19, 20, 21],
          [22, 23, 24, 25]]]),
 torch.Size([2, 3, 4]))
```

### Reduction

### Vector reduction
> [**the sum of a tensor's elements.**]
 $\mathbf{x}$ of length $n$,
we write $\sum_{i=1}^n x_i$.
 
```python
x = torch.arange(3, dtype=torch.float32)
x, x.sum()
```

```text
(tensor([0., 1., 2.]), tensor(3.))
```

#### Matrix and generic reduction
> [**sums over the elements of tensors of arbitrary shape**],
the sum of the elements of an $m \times n$ matrix $\mathbf{A}$ 
could be written $\sum_{i=1}^{m} \sum_{j=1}^{n} a_{ij}$.

```python
A.shape, A.sum()
```
```
(torch.Size([2, 3]), tensor(15.))
```

#### Reduction over an axis

> [**specify the axes 
along which the tensor should be reduced.**]
To sum over all elements along the rows (axis 0),
we specify `axis=0` in `sum`.
Since the input matrix reduces along axis 0
to generate the output vector,
this axis is missing from the shape of the output. 

So that on a $m \times n$ matrix the sum with `axis=0` means that $m$ is fixed (sum by colum)


$$s_{i} = \sum_{j=1}^{n} a_{ij}$$ 


```python
(A.shape,
 A.sum(axis=0), #sum by column (along the axis 0 which is the row)
 A.sum(axis=0).shape, #
 A.sum(axis=1), #sum by row
 A.sum(axis=1).shape)
``` 

```text
(torch.Size([2, 3]),
 tensor([3., 5., 7.]),
 torch.Size([3]),
 tensor([ 3., 12.]),
 torch.Size([2]))
```

> Sum by all axis is the normal reduction 

```python
A.sum(axis=[0, 1]) == A.sum()  # Same as A.sum()
```

#### Mean reduction

```python
A.mean(), A.sum() / A.numel()
```

```text
(tensor(2.5000), tensor(2.5000))
```

### Mean reduction per axis

```python
A.mean(axis=0), A.sum(axis=0) / A.shape[0]
```

```text
(tensor([1.5000, 2.5000, 3.5000]), tensor([1.5000, 2.5000, 3.5000]))
```

### Non reduction sum

>Sometimes it can be useful to [<span style="color:red">**keep the number of axes unchanged**</span>] when invoking the function for calculating the sum or mean. 

This matters when we want to use the broadcast mechanism.

```python
sum_A = A.sum(axis=1, keepdims=True)
sum_A, sum_A.shape
```

```text
(tensor([[ 3.],
         [12.]]),
 torch.Size([2, 1]))
```

For instance, since `sum_A` keeps its two axes after summing each row,
we can (**divide `A` by `sum_A` with broadcasting**) 
to create a matrix where each row sums up to $1$.

```python
A / sum_A
```

```text
tensor([[0.0000, 0.3333, 0.6667],
        [0.2500, 0.3333, 0.4167]])
```

#### Cumulative sum

If we want to calculate [**the cumulative sum of elements of A along some axis], say axis=0 (row by row), we can call the cumsum function. 

By design, this function does not reduce the input tensor along any axis.

```python
A.cumsum(axis=0)
```

```text
tensor([[0., 1., 2.],
        [3., 5., 7.]])

```

## Dot product

> Given two vectors $\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$,
their <span style="color:red">*dot product*</span> $\mathbf{x}^\top \mathbf{y}$ (also known as <span style="color:red">*inner product*</span>, $\langle \mathbf{x}, \mathbf{y}  \rangle$) 
is a sum over the products of the elements at the same position: 
$\mathbf{x}^\top \mathbf{y} = \sum_{i=1}^{d} x_i y_i$.

```python
x = torch.arange(3, dtype = torch.float32)
y = torch.ones(3, dtype = torch.float32)
x, y, torch.dot(x, y)
```

```text
(tensor([0., 1., 2.]), tensor([1., 1., 1.]), tensor(3.))
```

Equivalently, (**we can calculate the dot product of two vectors 
by performing an elementwise multiplication followed by a sum:**)

```python
torch.sum(x * y) # tensor(3.)
```
Dot products are useful in a wide range of contexts.

For example, given some set of values, denoted by a vector $\mathbf{x}  \in \mathbb{R}^n$,
and a set of weights, denoted by $\mathbf{w} \in \mathbb{R}^n$, the weighted sum of the values in $\mathbf{x}$
according to the weights $\mathbf{w}$ could be expressed as the dot product $\mathbf{x}^\top \mathbf{w}$.

When the weights are nonnegative and sum to $1$, i.e., $\left(\sum_{i=1}^{n} {w_i} = 1\right)$,
the dot product $\mathbf{w}^\top \mathbf{x} = \sum_{i=1}^{d} w_i x_i$ expresses a <span style="color:red">*weighted average*</span>.

After normalizing two vectors to have unit length, the dot products express the cosine of the angle between them.
Later in this section, we will formally introduce this notion of *length*.


### Matrix--Vector product

Now that we know how to calculate dot products, we can begin to understand the *product*
between an $m \times n$ matrix $\mathbf{A}$ 
and an $n$-dimensional vector $\mathbf{x}$.
To start off, we visualize our matrix
in terms of its row vectors

$$\mathbf{A}=
\begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix},$$

where each $\mathbf{a}^\top_{i} \in \mathbb{R}^n$
is a row vector representing the $i^\textrm{th}$ row 
of the matrix $\mathbf{A}$.

> <span style="color:red">[**The matrix--vector product $\mathbf{A}\mathbf{x}$
is simply a column vector of length $m$,
whose $i^\textrm{th}$ element is the dot product 
$\mathbf{a}^\top_i \mathbf{x}$:**]</span>

$$
\mathbf{A}\mathbf{x}
= \begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix}\mathbf{x}
= \begin{bmatrix}
 \mathbf{a}^\top_{1} \mathbf{x}  \\
 \mathbf{a}^\top_{2} \mathbf{x} \\
\vdots\\
 \mathbf{a}^\top_{m} \mathbf{x}\\
\end{bmatrix} = \begin{bmatrix}
 \sum_{i=1}^{n} a_{1i} x_i  \\
 \sum_{i=1}^{n} a_{2i} x_i \\
\vdots\\
 \sum_{i=1}^{n} a_{mi} x_i\\
\end{bmatrix}
$$

>We can think of multiplication with a matrix $\mathbf{A}\in \mathbb{R}^{m \times n}$
as a transformation that projects vectors from $\mathbb{R}^{n}$ to $\mathbb{R}^{m}$.


These transformations are remarkably useful.

For example, we can represent rotations as multiplications by certain square matrices.

Matrix--vector products also describe the key calculation involved in computing the outputs of each layer in a neural network given the outputs from the previous layer.

### Implementation

To express a matrix--vector product in code, we use the `mv` function. 

> Note that the column dimension of `A`  (its length along axis 1) must be the same as the dimension of `x` (its length). 

Python has a convenience operator `@` that can execute both matrix--vector and matrix--matrix products(depending on its arguments). 

Thus we can write `A@x`.

```python
A = torch.arange(6, dtype=torch.float32).reshape(2, 3)
x = torch.arange(3, dtype=torch.float32)

A, x, A@x, torch.mv(A, x), torch.mv(A, x).shape
```

$$
\mathbf{A}\mathbf{x} = 
\begin{bmatrix}0 & 1 & 2 \\ 3 & 4 & 5\end{bmatrix}\begin{bmatrix}0 & 1 & 2\end{bmatrix} =
\begin{bmatrix}
    \begin{bmatrix} 0 & 1 & 2  \end{bmatrix}\begin{bmatrix} 0 & 1 & 2  \end{bmatrix} \\
    \begin{bmatrix} 3 & 4 & 5  \end{bmatrix}\begin{bmatrix} 0 & 1 & 2  \end{bmatrix} 
\end{bmatrix} = \begin{bmatrix}4 & 15\end{bmatrix}.
$$

```text
(tensor([[0., 1., 2.],
         [3., 4., 5.]]),
 tensor([0., 1., 2.]),
 tensor([ 5., 14.]),
 tensor([ 5., 14.]), 
 torch.Size([2]))
```

## Matrix--Matrix Multiplication

Once you have gotten the hang of dot products and matrix--vector products,
then *matrix--matrix multiplication* should be straightforward.

Say that we have two matrices 
$\mathbf{A} \in \mathbb{R}^{n \times k}$ 
and $\mathbf{B} \in \mathbb{R}^{k \times m}$:

$$\mathbf{A}=\begin{bmatrix}
 a_{11} & a_{12} & \cdots & a_{1k} \\
 a_{21} & a_{22} & \cdots & a_{2k} \\
\vdots & \vdots & \ddots & \vdots \\
 a_{n1} & a_{n2} & \cdots & a_{nk} \\
\end{bmatrix},\quad
\mathbf{B}=\begin{bmatrix}
 b_{11} & b_{12} & \cdots & b_{1m} \\
 b_{21} & b_{22} & \cdots & b_{2m} \\
\vdots & \vdots & \ddots & \vdots \\
 b_{k1} & b_{k2} & \cdots & b_{km} \\
\end{bmatrix}.$$


Let $\mathbf{a}^\top_{i} \in \mathbb{R}^k$ denote 
the row vector representing the $i^\textrm{th}$ row 
of the matrix $\mathbf{A}$
and let $\mathbf{b}_{j} \in \mathbb{R}^k$ denote 
the column vector from the $j^\textrm{th}$ column 
of the matrix $\mathbf{B}$:

$$\mathbf{A}=
\begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_n \\
\end{bmatrix},
\quad \mathbf{B}=\begin{bmatrix}
 \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
\end{bmatrix}.
$$


To form the matrix product $\mathbf{C} \in \mathbb{R}^{n \times m}$,
we simply compute each element $c_{ij}$
as the dot product between 
the $i^{\textrm{th}}$ row of $\mathbf{A}$
and the $j^{\textrm{th}}$ column of $\mathbf{B}$,
i.e., $\mathbf{a}^\top_i \mathbf{b}_j$:

$$\mathbf{C} = \mathbf{AB} = \begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_n \\
\end{bmatrix}
\begin{bmatrix}
 \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
\end{bmatrix}
= \begin{bmatrix}
\mathbf{a}^\top_{1} \mathbf{b}_1 & \mathbf{a}^\top_{1}\mathbf{b}_2& \cdots & \mathbf{a}^\top_{1} \mathbf{b}_m \\
 \mathbf{a}^\top_{2}\mathbf{b}_1 & \mathbf{a}^\top_{2} \mathbf{b}_2 & \cdots & \mathbf{a}^\top_{2} \mathbf{b}_m \\
 \vdots & \vdots & \ddots &\vdots\\
\mathbf{a}^\top_{n} \mathbf{b}_1 & \mathbf{a}^\top_{n}\mathbf{b}_2& \cdots& \mathbf{a}^\top_{n} \mathbf{b}_m
\end{bmatrix}.
$$

> [<span style="color:red">**We can think of the matrix--matrix multiplication $\mathbf{AB}$
as performing $m$ matrix--vector products 
or $m \times n$ dot products 
and stitching the results together 
to form an $n \times m$ matrix.</span>**]

In the following snippet, we perform matrix multiplication on `A` and `B`.
Here, `A` is a matrix with two rows and three columns,and `B` is a matrix with three rows and four columns.

After multiplication, we obtain a matrix with two rows and four columns.

```python
A = torch.arange(6, dtype=torch.float32).reshape(2, 3)
B = torch.arange(12, dtype=torch.float32).reshape(3, 4)
C = torch.mm(A,B)

A,B,C,C.shape 
```

```text

(tensor([[0., 1., 2.],
         [3., 4., 5.]]),
 tensor([[ 0.,  1.,  2.,  3.],
         [ 4.,  5.,  6.,  7.],
         [ 8.,  9., 10., 11.]]),
 tensor([[20., 23., 26., 29.],
         [56., 68., 80., 92.]]),
 torch.Size([2, 4]))
```

## Norms 
:label:`subsec_lin-algebra-norms`

Some of the most useful operators in linear algebra are *norms*.
> Informally, the norm of a vector tells us how *big* it is. 

> For instance, the $\ell_2$ norm measures the (Euclidean) length of a vector.

Here, we are employing a notion of *size* that concerns the magnitude of a vector's components
(not its dimensionality). 

A norm is a function $\| \cdot \|$ that maps a vector
to a scalar and satisfies the following three properties:

1. Given any vector $\mathbf{x}$, if we scale (all elements of) the vector 
   by a scalar $\alpha \in \mathbb{R}$, its norm scales accordingly:
   $$\|\alpha \mathbf{x}\| = |\alpha| \|\mathbf{x}\|.$$
2. For any vectors $\mathbf{x}$ and $\mathbf{y}$:
   norms satisfy the triangle inequality:
   $$\|\mathbf{x} + \mathbf{y}\| \leq \|\mathbf{x}\| + \|\mathbf{y}\|.$$
3. The norm of a vector is nonnegative and it only vanishes if the vector is zero:
   $$\|\mathbf{x}\| > 0 \textrm{ for all } \mathbf{x} \neq 0.$$

Many functions are valid norms and different norms 
encode different notions of size. 


>The <span style="color:red">**Euclidean norm (called **the $\ell_2$ *norm***)**</span> that we all learned in elementary school geometry
when calculating the hypotenuse of a right triangle
is the square root of the sum of squares of a vector's elements.

Formally, this is  and expressed as

**$$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2}.$$**

The method `norm` calculates the $\ell_2$ norm.

```python
u = torch.tensor([3.0, -4.0])
torch.norm(u) #tensor(5.)
```

> [<span style="color:red">The  ℓ1  norm</span>] is also common and the associated measure is called the <span style="color:red">Manhattan distance</span>. 
> By definition, the  ℓ1  norm sums the absolute values of a vector's elements:

**$$\|\mathbf{x}\|_1 = \sum_{i=1}^n \left|x_i \right|.$$**

Compared to the $\ell_2$ norm, it is less sensitive to outliers.

```python
torch.abs(u).sum() #tensor(7.)
```

Both the $\ell_2$ and $\ell_1$ norms are special cases
of the more general $\ell_p$ *norms*:

$$\|\mathbf{x}\|_p = \left(\sum_{i=1}^n \left|x_i \right|^p \right)^{1/p}.$$

In the case of matrices, matters are more complicated. 

After all, matrices can be viewed both as collections of individual entries 
*and* as objects that operate on vectors and transform them into other vectors. 

For instance, we can ask by how much longer 
the matrix--vector product $\mathbf{X} \mathbf{v}$ 
could be relative to $\mathbf{v}$. 

This line of thought leads to what is called the <span style="color:red">*spectral* norm<span style="color:red">. 

> For now, we introduce [**the <span style="color:red">*Frobenius norm*</span>, 
which is much easier to compute**] and defined as
the square root of the sum of the squares 
of a matrix's elements:

[**$$\|\mathbf{X}\|_\textrm{F} = \sqrt{\sum_{i=1}^m \sum_{j=1}^n x_{ij}^2}.$$**]

The Frobenius norm behaves as if it were 
an $\ell_2$ norm of a matrix-shaped vector.
Invoking the following function will calculate 
the Frobenius norm of a matrix.

```python
torch.norm(torch.ones((4, 9))) #tensor(6.)
```