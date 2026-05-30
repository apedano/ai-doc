# Softmax regression

https://d2l.ai/chapter_linear-classification/softmax-regression.html

In this section, we focus on classification problems where we put aside how much? questions and instead focus on _which
category_? questions.

* Does this email belong in the spam folder or the inbox?
* Is this customer more likely to sign up or not to sign up for a subscription service?
* Does this image depict a donkey, a dog, a cat, or a rooster?
* Which movie is Aston most likely to watch next?
* Which section of the book are you going to read next?

Colloquially, machine learning practitioners overload the word classification to describe two subtly different problems:

> (i) those where we are interested only in <span style="color:red">**hard assignments of examples to categories (
classes)**</span>;


> (ii) those where we wish to make <span style="color:red">**soft assignments**</span>, i.e., to assess the probability
> that each category applies. The distinction tends to get blurred, in part, because often, even when we only care about
> hard assignments, we still use models that make soft assignments.

Even more, there are cases where more than one label might be true.

For instance, **a news article might simultaneously cover the topics of entertainment, business, and space flight, but
not the topics of medicine or sports**.

Thus, categorizing it into one of the above categories on their own would not be very useful.

This problem is commonly known as <span style="color:red">**multi-label classification**</span>.

## Classification

:label:`subsec_classification-problem`

Let's start with a simple **image classification problem**.

Here, each input consists of a $2\times2$ grayscale image.

We can represent **each pixel value** with a single scalar,
giving us four features $x_1, x_2, x_3, x_4$.

Further, let's assume that each image belongs to one
among the categories "cat", "chicken", and "dog".

Next, we have to choose how to represent the labels.

We have two obvious choices.

Perhaps the most natural impulse would be to choose $y \in \{1, 2, 3\}$, where the integers represent
$\{\textrm{dog}, \textrm{cat}, \textrm{chicken}\}$ respectively.

This is a great way of *storing* such information on a computer.
If the categories had some **natural ordering among them**,
say if we were trying to predict
$\{\textrm{baby}, \textrm{toddler}, \textrm{adolescent}, \textrm{young adult}, \textrm{adult}, \textrm{geriatric}\}$,
then it might even make sense to cast this as
an [ordinal regression](https://en.wikipedia.org/wiki/Ordinal_regression) problem
and keep the labels in this format.

In general, **classification problems do not come
with natural orderings among the classes**.

Fortunately, statisticians long ago invented a simple way
to represent categorical data: the *one-hot encoding*.

> A <span style="color:red">**one-hot encoding**</span> is a vector
> with as many components as we have categories.
> The component corresponding to a particular instance's category is set to 1
> and all other components are set to 0.


In our case, a label $y$ would be a three-dimensional vector,
with $(1, 0, 0)$ corresponding to "cat", $(0, 1, 0)$ to "chicken",
and $(0, 0, 1)$ to "dog":

$$y \in \{(1, 0, 0), (0, 1, 0), (0, 0, 1)\}.$$

### Linear Model

In order to estimate the conditional probabilities
associated with all the possible classes,
**we need a model with multiple outputs**, one per class.

To address classification with linear models,
we will need **as many affine functions as we have outputs**.

Strictly speaking, we only need one fewer,
since the final category has to be the difference
between $1$ and the sum of the other categories,
but for reasons of symmetry
we use a slightly redundant parametrization.

Each output corresponds to its own affine function.

In our case, since we have 4 features and 3 possible output categories,
we need 12 scalars to represent the weights ($w$ with subscripts),
and 3 scalars to represent the biases ($b$ with subscripts).

This yields:

$$
\begin{aligned}
o_1 &= x_1 w_{11} + x_2 w_{12} + x_3 w_{13} + x_4 w_{14} + b_1,\\
o_2 &= x_1 w_{21} + x_2 w_{22} + x_3 w_{23} + x_4 w_{24} + b_2,\\
o_3 &= x_1 w_{31} + x_2 w_{32} + x_3 w_{33} + x_4 w_{34} + b_3.
\end{aligned}
$$

The corresponding neural network diagram is shown in :numref:`fig_softmaxreg`.

Just as in linear regression, we use a single-layer neural network.

And since the calculation of each output, $o_1, o_2$, and $o_3$,
depends on every input, $x_1$, $x_2$, $x_3$, and $x_4$,
the output layer can also be described as a *fully connected layer*.

![Softmax regression is a single-layer neural network.](./img/softmaxreg.svg)
:label:`fig_softmaxreg`

For a more concise notation we use vectors and matrices:

$$\mathbf{o} = \mathbf{W} \mathbf{x} + \mathbf{b}$$

Note that we have gathered all of our weights into a $3 \times 4$ matrix and all biases
$\mathbf{b} \in \mathbb{R}^3$ in a vector.

To summerize

| Categories (one-hot encoding)   | Features (#pixels)              | Weights                           | Bias                            |
|---------------------------------|---------------------------------|-----------------------------------|---------------------------------|
| $\mathbf{o} \in \mathbb{R}^{m}$ | $\mathbf{X} \in \mathbb{R}^{n}$ | $\mathbf{W} \in \mathbb{R}^{m,n}$ | $\mathbf{o} \in \mathbb{R}^{m}$ |

### The Softmax
:label:`subsec_softmax_operation`

Assuming a suitable loss function,we could try, directly, to minimize the difference
between $\mathbf{o}$ and the labels $\mathbf{y}$.

While it turns out that treating classification as a vector-valued regression problem works surprisingly well,
it is nonetheless unsatisfactory in the following ways:

* There is no guarantee that the outputs $o_i$ sum up to $1$ in the way we expect probabilities to behave.
* There is no guarantee that the outputs $o_i$ are even nonnegative, even if their outputs sum up to $1$, or that they do not exceed $1$.

Both aspects render the estimation problem difficult to solve
and the solution very brittle to outliers.
For instance, if we assume that there
is a positive linear dependency
between the number of bedrooms and the likelihood
that someone will buy a house,
the probability might exceed $1$
when it comes to buying a mansion!
As such, we need a mechanism to "squish" the outputs.

## The $softmax$ function

> A way to accomplish this goal
(and to ensure nonnegativity) is to use
an exponential function $P(y = i) \propto \exp o_i$.
This does indeed satisfy the requirement
that the conditional class probability
increases with increasing $o_i$, it is monotonic,
and all probabilities are nonnegative.
We can then transform these values so that they add up to $1$
by dividing each by their sum.
This process is called *normalization*.
Putting these two pieces together
gives us the *softmax* function:

$$\hat{\mathbf{y}} = \mathrm{softmax}(\mathbf{o}) \quad \textrm{where}\quad \hat{y}_i = \frac{\exp(o_i)}{\sum_j \exp(o_j)}.$$
:eqlabel:`eq_softmax_y_and_o`

Note that the largest coordinate of $\mathbf{o}$
corresponds to the most likely class according to $\hat{\mathbf{y}}$.
Moreover, because the softmax operation
preserves the ordering among its arguments,
we do not need to compute the softmax
to determine which class has been assigned the highest probability. Thus,

$$
\operatorname*{argmax}_j \hat y_j = \operatorname*{argmax}_j o_j.
$$

