# Linear regression


><span style="color:red">**Regression**</span> problems pop up whenever we want to predict a numerical value.

Common examples include:
* predicting prices (of homes, stocks, etc.),
* predicting the length of stay (for patients in the hospital),
* forecasting demand (for retail sales), ....

Not every prediction problem is one of classical regression.
Later on, we will introduce classification problems,
where the goal is to predict membership among a set of categories.

As a running example, suppose that we wish to **estimate the prices of houses (in dollars)
based on their area (in square feet) and age (in years)**.

## Terminology

> In the terminology of machine learning, the dataset is called a <span style="color:red">**training dataset**</span> or <span style="color:red">**training set**</span>,
and each row (containing the data corresponding to one sale) is called an <span style="color:red">**example**</span> (or <span style="color:red">**data point**</span>, <span style="color:red">**instance**</span> or <span style="color:red">**sample**</span>).

>The value we are trying to predict (price) is called a <span style="color:red">**label**</span> (or <span style="color:red">**target**</span>).
The variables (age and area) upon which the predictions are based are called <span style="color:red">**features**</span> (or <span style="color:red">**covariates**</span>).

## Basics

<span style="color:red">**Linear regression**</span> is both the simplest
and most popular among the standard tools for tackling regression problems.


> First, we assume that the relationship between features $\mathbf{x}$ and target $y$
is approximately linear,
i.e., that the conditional mean $E[Y \mid X=\mathbf{x}]$
can be expressed as a weighted sum
of the features $\mathbf{x}$.
 

This setup allows that the target value may still deviate from its expected value
on account of observation noise.

> Next, we can impose the assumption that any such **noise is well behaved, following a Gaussian distribution**.

Typically, we will use $n$ to denote the number of examples in our dataset.

We use superscripts to enumerate samples and targets, and subscripts to index coordinates.
More concretely,
* $\mathbf{x}^{(i)}$ denotes the $i^{\textrm{th}}$ sample
* and $x_j^{(i)}$ denotes its $j^{\textrm{th}}$ coordinate (feature) of that sample.

### Model

> At the heart of every solution is a <span style="color:red">**model**</span> that describes how features can be transformed
into an estimate of the target.

The <span style="color:red">**assumption of linearity**</span> means that the expected value of the target (price) can be expressed
as a weighted sum of the features (area and age):

 $$\textrm{price} = w_{\textrm{area}} \cdot \textrm{area} + w_{\textrm{age}} \cdot \textrm{age} + b.$$


Here $w_{\textrm{area}}$ and $w_{\textrm{age}}$ are called *weights*, and $b$ is called a <span style="color:red">**bias**</span>
(or <span style="color:red">**offset**</span> or <span style="color:red">**intercept**</span>).


The weights determine the influence of each feature on our prediction.

The bias determines the value of the estimate when all features are zero.

Even though we will never see any newly-built homes with precisely zero area,
we still need the bias because it allows us to express all linear functions of our features
(rather than restricting us to lines that pass through the origin).

> Strictly speaking, the <span style="color:red">**linear regression model**</span> is an *affine transformation* of input features, which is characterized by a *linear transformation* of features via a weighted sum, combined with a *translation* via the added bias.

Given a dataset, our goal is to choose the weights $\mathbf{w}$ and the bias $b$
that, on average, make our model's predictions fit the true prices observed in the data as closely as possible.

When our inputs consist of $d$ features, we can assign each an index (between $1$ and $d$)
and express our prediction $\hat{y}$ (in general the "hat" symbol denotes an estimate) as

> $$\hat{y} = w_1  x_1 + \cdots + w_d  x_d + b.$$

Collecting all features into a vector $\mathbf{x} \in \mathbb{R}^d$
and all weights into a vector $\mathbf{w} \in \mathbb{R}^d$,
we can express our model compactly via the dot product
between $\mathbf{w}$ and $\mathbf{x}$:

> $$\hat{y} = \mathbf{w}^\top \mathbf{x} + b.$$


The vector $\mathbf{x}$ corresponds to the features of a single example.
We will often find it convenient
to refer to features of our entire dataset of $n$ examples
via the <span style="color:red">**design matrix**</span> $\mathbf{X} \in \mathbb{R}^{n \times d}$.

* one row per sample
* one column per feature

For a collection of features $\mathbf{X}$,
the predictions $\hat{\mathbf{y}} \in \mathbb{R}^n$
can be expressed via the matrix--vector product,  resulting in the vector $\hat{\mathbf{y}}$:

> $${\hat{\mathbf{y}}} = \mathbf{X} \mathbf{w} + b,$$


where broadcasting (:numref:`subsec_broadcasting`) is applied during the summation.


Given features of a training dataset $\mathbf{X}$
and corresponding (known) labels $\mathbf{y}$,
the goal of linear regression is to find
the weight vector $\mathbf{w}$ and the bias term $b$
such that, given features of a new data example
sampled from the same distribution as $\mathbf{X}$,
the new example's label will (in expectation)
be predicted with the smallest error (even if the real sistem is perfectly linear). 

For example, whatever instruments we use to observe
the features $\mathbf{X}$ and labels $\mathbf{y}$, there might be a small amount of measurement error.

Thus, even when we are confident
that the underlying relationship is linear,
we will incorporate a noise term to account for such errors.

Before we can go about searching for the best *parameters*
(or *model parameters*) $\mathbf{w}$ and $b$,
we will need two more things:
* (i) a measure of the quality of some given model;
* (ii) a procedure for updating the model to improve its quality.

### Loss Function

Naturally, fitting our model to the data requires that we agree on some measure of <span style="color:red">**fitness**</span>
(or, equivalently, of *unfitness*).

> <span style="color:red">**Loss functions**</span> quantify the distance
between the *real* and *predicted* values of the target.

The loss will usually be a nonnegative number where smaller values are better
and perfect predictions incur a loss of 0.

For regression problems, the most common loss function is the squared error.

> $$l^{(i)}(\mathbf{w}, b) = \frac{1}{2} \left(\hat{y}^{(i)} - y^{(i)}\right)^2.$$

* $l^{(i)}$: <span style="color:red">**squared error**</span>  of the sample $i$. It is a function of the model parameters.
* $\hat{y}^{(i)}$: prediction for sample $i$
* $y^{(i)}$: real value for the sample $i$


The constant $\frac{1}{2}$ makes no real difference
but proves to be notationally convenient,
since it cancels out when we take the derivative of the loss.


In *Fig. 3.1.1*, we visualize the fit of a linear regression model
in a problem with one-dimensional inputs.

![Fitting a linear regression model to one-dimensional data.](img/fit-linreg.png)
*Fig. 3.1.1*


Note that large differences between estimates $\hat{y}^{(i)}$ and targets $y^{(i)}$
lead to even larger contributions to the loss, due to its quadratic form
(this quadraticity can be a double-edge sword; while it encourages the model to avoid large errors
it can also lead to excessive sensitivity to anomalous data).

To measure the quality of a model on the entire dataset of $n$ examples,
we simply average (or equivalently, sum) the losses on the training set:

$$L(\mathbf{w}, b) =\frac{1}{n}\sum_{i=1}^n l^{(i)}(\mathbf{w}, b) =\frac{1}{n} \sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2.$$

> When training the model, we seek parameters ($\mathbf{w}^*, b^*$)
that minimize the total loss across all training examples:

>$$\mathbf{w}^*, b^* = \operatorname*{argmin}_{\mathbf{w}, b}\  L(\mathbf{w}, b).$$

### Analytic Solution

We can find the optimal parameters (as assessed on the training data)
analytically by applying a simple formula as follows.

First, we can subsume the bias $b$ into the parameter $\mathbf{w}$
by appending a column to the design matrix consisting of all 1s.

$$X'=\begin{bmatrix}
1 & x_{11} & \cdots & x_{1d} \\
1 & x_{21} & \cdots & x_{2d} \\
\vdots & \vdots & \ddots & \vdots \\
1 & x_{n1} & \cdots & x_{nd} \\
\end{bmatrix}$$

and the weights

$$
w'=
\begin{bmatrix}
b \\
w
\end{bmatrix}
$$

so this is the model $${\hat{\mathbf{y}}} = \mathbf{X'} \mathbf{w'}$$



Then our prediction problem is to minimize the total predition error:

$$L(w)=\|\mathbf{X}\mathbf{w} - \mathbf{y} \|^2 $$.




Taking the derivative of the loss with respect to $\mathbf{w}$
and setting it equal to zero yields:

$$\begin{aligned}
    ∇_{\mathbf{w}} \|\mathbf{X}\mathbf{w} - \mathbf{y} \|^2 =
    2 \mathbf{X}^\top (\mathbf{X} \mathbf{w} - \mathbf{y}) = 0 \end{aligned}$$

hence

$$
\mathbf{X}^\top \mathbf{y} = \mathbf{X}^\top \mathbf{X} \mathbf{w}
$$

Solving for $\mathbf{w}$ provides us with the optimal solution
for the optimization problem.

Note that this solution 

$$\mathbf{w}^* = (\mathbf X^\top \mathbf X)^{-1}\mathbf X^\top \mathbf{y}$$

exists if the $\mathbf X^\top \mathbf X$ matrix is invertible

#### Design matrix full rank requirement

* The $\mathbf X^\top \mathbf X$ matrix is <span style="color:red">**squared**</span> and <span style="color:red">**symmetric**</span> by construction
    * the norm $\|\mathbf{X}\mathbf{w}\|^2=\mathbf{X}^T\mathbf{w}^T\mathbf{X}\mathbf{w} \ge 0$ $\forall \mathbf{w}$
    * therefore $X^TX⪰0$ (<span style="color:red">**semidefinite positive**</span>)


* if $rank(X)=d$ (full rank) then
  * no feature is a linear combination of others
  * columns are linearly independent 
    * if there is a linear dependency i.e. $x_3=x_2w_2+3x_1w_1$
      * multiple  $w$ will result in the value of the loss function 
      * the loss function will have the same flat value in the sample space
      * the gradient will be zero in that area invalidating optimization 
  * the null space $\mathcal{N}(X)={0}$
    * so $\mathbf{X}\mathbf{w} = 0 ⟺ \mathbf{w}=0$
    * this makes the semidefinite positive $\mathbf X^\top \mathbf X$ a <span style="color:red">**definite positive**</span>

For a theorem 

> A symmetric matrix is invertible iff it is positive definite.
 
Therefore 
> the $\mathbf X^\top \mathbf X$ matrix is <span style="color:red">**invertible**</span>

#### Existence of optimal solution

> If the columns of $\mathbf{X}$ are linearly independent (full rank), then the loss function has exactly one stationary point, and that point is the unique global minimum.

The quadratic form of the loss function creates a curved surface.

For positive definite matrices:

* every direction curves upward
* the origin is a strict minimum
* shapes are ellipsoids

Think of a multidimensional bowl.

### Minibatch Stochastic Gradient Descent

Fortunately, even in cases where we cannot solve the models analytically,
we can still often train models effectively in practice.

Moreover, for many tasks, those hard-to-optimize models
turn out to be so much better that figuring out how to train them
ends up being well worth the trouble.

> The key technique for optimizing nearly every deep learning model
consists of iteratively reducing the error
by updating the parameters in the direction
that incrementally lowers the loss function.
This algorithm is called <span style="color:red">**gradient descent**</span>.

#### Problem with the gradient descent

The most naive application of gradient descent
consists of taking the derivative of the loss function,
which is an average of the losses computed
on every single example in the dataset.
In practice, this can be extremely slow:
we must pass over the entire dataset before making a single update,
even if the update steps might be very powerful :cite:`Liu.Nocedal.1989`.
Even worse, if there is a lot of redundancy in the training data,
the benefit of a full update is limited.

#### SGD
The other extreme is to consider only a single example at a time and to take
update steps based on one observation at a time.
The resulting algorithm, <span style="color:red">**stochastic gradient descent (SGD)**</span>
can be an effective strategy :cite:`Bottou.2010`, even for large datasets.


Unfortunately, SGD has drawbacks, both computational and statistical.

* Computational problems of SGD
  * processors are a lot faster multiplying and adding numbers than they are at moving data from main memory to processor cache
  * It is up to an order of magnitude more efficient to perform a matrix--vector multiplication than a corresponding number of vector--vector operations.
  * process a single sample at the time (vector) is slower than a full batch (matrix)

####  Minibatch Stochastic Gradient Descent

The solution is to pick an intermediate strategy:

> rather than taking a full batch or only a single sample at a time,
we take a *minibatch* of observations.

The specific choice of the size of the said minibatch depends on many factors,
such as the amount of memory, the number of accelerators,
the choice of layers, and the total dataset size.

> Despite all that, a number between 32 and 256,
preferably a multiple of a large power of $2$, is a good start.
This leads us to *minibatch stochastic gradient descent*.

At iteration $t$

* random select samples of the fixed size $s$ $\mathcal{B}_t$
* $|\mathcal{B}| = s$ fixed size
* Calculate the gradient of the average loss of the batch based on the current model parameters
* Multiply the gradient by a predetermined small positive value $\eta$,
called the *learning rate*, and subtract the resulting term from the current parameter values

We can express the update as follows:

$$(\mathbf{w},b)_{t+1} \leftarrow (\mathbf{w},b)_{t} - \frac{\eta}{|\mathcal{B_t}|} \sum_{i \in \mathcal{B}_t} \partial_{(\mathbf{w},b)} l^{(i)}(\mathbf{w},b).$$



#### Explanation

For a single sample the error is $L = (\hat{y} - y)^2$

for $|\mathcal{B_t}|=s$ minibatch samples, we normalize the sum of errror for each

$$
L=\frac{1}{s}\sum_{i}{(\hat{y_i} - y_i)^2}
$$

which, in matric notation $\sum_{i}{(\hat{y_i} - y_i)^2}$ is $||\hat{Y} - Y||^2$ the <span style="color:red">**the squared L2 norm**</span>

$$
L=\frac{1}{s}||\hat{Y} - Y||^2
$$

> the loss is normalized by $\frac{1}{s}$ so it does not scale with the number of samples in the minibatch

The update of $w$ and $b$ becomes (given the <span style="color:red">**error**</span> $\epsilon=\hat{Y} - Y$)

$$
w'=w-\eta\frac{\delta{L}}{\delta{w}}=w-\frac{2\eta}{s}(\hat{Y} - Y)\frac{\delta{\hat{Y}}}{\delta{w}}=w-\frac{2\eta\epsilon}{s}\frac{\delta{(wX+b)}}{\delta{w}}=w-\frac{2\eta\epsilon}{s}X
$$

$$
b'=b-\eta\frac{\delta{L}}{\delta{b}}=b-\frac{2\eta}{s}(\hat{Y} - Y)\frac{\delta{\hat{Y}}}{\delta{b}}=b-\frac{2\eta\epsilon}{s}\frac{\delta{(wX+b)}}{\delta{b}}=b-\frac{2\eta\epsilon}{s}
$$

#### Procedure

First we can simulate true features an targets in code

```python
import numpy as np

# The seed is given an integer value to ensure that the results of pseudo-random generation are reproducible. By re-using a seed value, the same sequence should be reproducible from run to run as long as multiple threads are not running. Reproducibility is a very important concept that ensures that anyone who re-runs the code gets the exact same outputs.
# 42 is the Answer to the great question of “Life, the universe, and everything”!
np.random.seed(42)
n, m = 200, 2 

#generates the design matrix
X = np.random.uniform(-5, 5, (n, m))        # shape (n, m)

#values for true weights and bias
true_w = np.array([[3.0], [-2.0]])           # shape (m, 1)
true_b = 1.5

noise = np.random.normal(0, 1.0, (n, 1)) #normal distribution mean=0 stand_dev=1

y = X @ true_w + true_b + noise             # shape (n, 1)
```
This means the true target is

$$y=3x_1-2x_2+1.5 + noise$$

That is what we will try to find


In summary, minibatch SGD proceeds as follows:

1. initialize the values of the model parameters, typically at random;

```python
w = np.random.randn(m, 1)    # shape (2, 1)
b = 0.0
eta = 0.01 #learning rate
epochs = 71 #itarations
s = 2               # pure Minibatch SGD — s=2 sample at a
```
2. iteratively sample random minibatches from the data,
3. updating the parameters in the direction of the negative gradient.
   1. For quadratic losses and affine transformations,
   this has a closed-form expansion:
```python
for epoch in range(epochs):
    indices = np.random.permutation(n)        # ← shuffle once per epoch

    for start in range(0, n, s):
        idx = indices[start:start + s] #we take a minibatch size of sample indices
        xi = X[idx]        # shape (s, m)
        yi = y[idx]        # shape (s, 1)

        y_pred = xi @ w + b
        error = y_pred - yi
        
        dw = 2 * xi.T @ error    # shape (2, 1)
        db = 2 * np.mean(error) #scalar

        w -= 2*eta/s * dw
        b -= 2*eta/s * db
    #track the loss at every epoch
    y_all_pred = X @ w + b           # shape (n, 1)
    mse = np.mean((y_all_pred - y) ** 2) #mean squared error -> mean((ŷ - y)²)
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | MSE: {mse:.4f} | w: {w.ravel()} | b: {b:.4f}")
print(f"\nLearned:  w = {w.ravel()}, b = {b:.4f}")
print(f"True:     w = {true_w.ravel()}, b = {true_b}")
```

$$\begin{aligned} \mathbf{w} & \leftarrow \mathbf{w} - \frac{\eta}{s} \sum_{i \in \mathcal{B}_t} \partial_{\mathbf{w}} l^{(i)}(\mathbf{w}, b) && = \mathbf{w} - \frac{\eta}{s} \sum_{i \in \mathcal{B}_t} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)\\ b &\leftarrow b -  \frac{\eta}{s} \sum_{i \in \mathcal{B}_t} \partial_b l^{(i)}(\mathbf{w}, b) &&  = b - \frac{\eta}{s} \sum_{i \in \mathcal{B}_t} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right). \end{aligned}$$
:eqlabel:`eq_linreg_batch_update`

Since we pick a minibatch $\mathcal{B}$
we need to normalize by its size $|\mathcal{B}|=s$.

> Frequently, minibatch size $s$ and learning rate $\eta$ are user-defined
so they are not updated in the training loop. 
For this reason they are called <span style="color:red">**hyperparameters**</span>.


Output:

```text
Epoch   0 | MSE: 1.4098 | w: [ 3.17733381 -1.89433841] | b: 1.4795
Epoch  10 | MSE: 1.0244 | w: [ 3.0443967  -1.95495537] | b: 1.4454
Epoch  20 | MSE: 1.0250 | w: [ 2.92328966 -2.0389086 ] | b: 1.4178
Epoch  30 | MSE: 1.1455 | w: [ 3.12536326 -1.97868658] | b: 1.4643
Epoch  40 | MSE: 0.9824 | w: [ 2.96708134 -1.97749799] | b: 1.4663
Epoch  50 | MSE: 0.9899 | w: [ 3.03715343 -2.00535642] | b: 1.4708
Epoch  60 | MSE: 1.9324 | w: [ 2.66398143 -2.00124091] | b: 1.4441
Epoch  70 | MSE: 1.2686 | w: [ 3.17137694 -2.00112623] | b: 1.4509

Learned:  w = [ 3.17137694 -2.00112623], b = 1.4509
True:     w = [ 3. -2.], b = 1.5
```


> In the end, the quality of the solution is
typically assessed on a separate <span style="color:red">**validation dataset**</span> (or *validation set*).

>After training for some predetermined number of iterations
(or until some other stopping criterion is met), we record the estimated model parameters,
denoted $\hat{\mathbf{w}}, \hat{b}$.

Note that even if our function is truly linear and noiseless,
**these parameters will not be the exact minimizers of the loss**, nor even deterministic.
Although the algorithm converges slowly towards the minimizers
it typically will not find them exactly in a finite number of steps.
Moreover, the minibatches $\mathcal{B}$
used for updating the parameters are chosen at random.
This breaks determinism.

Linear regression happens to be a learning problem
with a global minimum (whenever $\mathbf{X}$ is full rank, or equivalently,
whenever $\mathbf{X}^\top \mathbf{X}$ is invertible).
However, the loss surfaces for deep networks contain many saddle points and minima.

>Fortunately, we typically do not care about finding
an exact set of parameters but merely any set of parameters
that leads to accurate predictions (and thus low loss).

>The more formidable task is to find parameters
that lead to accurate predictions on previously unseen data (outside the training set),
a challenge called <span style="color:red">**generalization**</span>.

## Vectorization for Speed

When training our models, we typically want to process
whole minibatches of examples simultaneously.
Doing this efficiently requires that (**we**) (~~should~~)
(**vectorize the calculations and leverage
fast linear algebra libraries
rather than writing costly for-loops in Python.**)

To see why this matters so much, let's (**consider two methods for adding vectors.**)
To start, we instantiate two 10,000-dimensional vectors containing all 1s.
In the first method, we loop over the vectors with a Python for-loop.
In the second, we rely on a single call to `+`. 

```python
import time
import torch
n = 10000
a = torch.ones(n)
b = torch.ones(n)

c = torch.zeros(n)
t = time.time()
#For loop method
for i in range(n):
    c[i] = a[i] + b[i]
print(f'{time.time() - t:.5f} sec') #0.05414 sec
#Vector sum
t = time.time()
d = a + b
print(f'{time.time() - t:.5f} sec') #0.00022 sec
```

> We see that the vector sum is 2 orders faster than the for loop sum.

## The Normal Distribution and Squared Loss

So far we have given a fairly functional motivation of the squared loss objective:

> the optimal parameters return the conditional expectation $E[Y\mid X]$
whenever the underlying pattern is truly linear, and the loss assigns large penalties for outliers.

We can also provide a more formal motivation for the squared loss objective
by making probabilistic assumptions about the distribution of noise.


$$p(x) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{1}{2 \sigma^2} (x - \mu)^2\right).$$

See [Gaussian distribution section](../002-preliminaries/probability_statistics.md#normal-gaussian-distribution)

>One way to motivate linear regression with squared loss
is to assume that observations arise from noisy measurements,
where the noise $\epsilon$ follows the normal distribution 
$\mathcal{N}(0, \sigma^2)$:

>$$y = \mathbf{w}^\top \mathbf{x} + b + \epsilon \textrm{ where } \epsilon \sim \mathcal{N}(0, \sigma^2).$$

Thus, we can now write out the <span style="color:red">**likelihood**</span> 
of seeing a particular $y$ for a given $\mathbf{x}$ via

$$P(y \mid \mathbf{x}) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{1}{2 \sigma^2} (y - \mathbf{w}^\top \mathbf{x} - b)^2\right).$$

As such, the likelihood factorizes.
According to <span style="color:red">**the principle of maximum likelihood**</span>, the best values of parameters $\mathbf{w}$ and $b$ are those
that maximize the *likelihood* of the entire dataset:

$$P(\mathbf y \mid \mathbf X) = \prod_{i=1}^{n} p(y^{(i)} \mid \mathbf{x}^{(i)}).$$

The equality follows since all pairs $(\mathbf{x}^{(i)}, y^{(i)})$
were drawn independently of each other.
Estimators chosen according to the principle of maximum likelihood
are called <span style="color:red">**maximum likelihood estimators**</span>.

While, maximizing the product of many exponential functions,
might look difficult,
we can simplify things significantly, without changing the objective,
by maximizing the logarithm of the likelihood instead.
For historical reasons, optimizations are more often expressed
as minimization rather than maximization.

>So, without changing anything,
we can <span style="color:red">***minimize* the *negative log-likelihood***</span>,
which we can express as follows:

>$$-\log P(\mathbf y \mid \mathbf X) = \sum_{i=1}^n \frac{1}{2} \log(2 \pi \sigma^2) + \frac{1}{2 \sigma^2} \left(y^{(i)} - \mathbf{w}^\top \mathbf{x}^{(i)} - b\right)^2.$$

If we assume that $\sigma$ is fixed,
we can ignore the first term,
because it does not depend on $\mathbf{w}$ or $b$.

The second term is identical
to the squared error loss introduced earlier,
except for the multiplicative constant $\frac{1}{\sigma^2}$.
Fortunately, the solution does not depend on $\sigma$ either.

>It follows that **minimizing the mean squared error
is equivalent to the maximum likelihood estimation
of a linear model under the assumption of additive Gaussian noise**.

> That means that even if we don't add the gaussian noise to our loss function 
> we don't loose likelihood of our estimation $\hat{y}$

## Linear Regression as a Neural Network

While linear models are not sufficiently rich
to express the many complicated networks
that we will introduce in this book,
(artificial) neural networks are rich enough
to subsume linear models as networks
in which every feature is represented by an input neuron,
all of which are connected directly to the output.

:numref:`fig_single_neuron` depicts
linear regression as a neural network.
The diagram highlights the connectivity pattern,
such as how each input is connected to the output,
but not the specific values taken by the weights or biases.

![Linear regression is a single-layer neural network.](./img/singleneuron.png)
:label:`fig_single_neuron`

The inputs are $x_1, \ldots, x_d$. 

We refer to $d$ as the *number of inputs*
or the *feature dimensionality* in the input layer.
The output of the network is $o_1$.
Because we are just trying to predict
a single numerical value,
we have only one output neuron.
Note that the input values are all *given*.
There is just a single *computed* neuron.

>In summary, we can think of linear regression
as a single-layer fully connected neural network.
We will encounter networks
with far more layers
in later chapters.