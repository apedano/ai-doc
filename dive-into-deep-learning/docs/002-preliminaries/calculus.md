# Calculus

For a long time, how to calculate 
the area of a circle remained a mystery.
Then, in Ancient Greece, the mathematician Archimedes
came up with the clever idea 
to inscribe a series of polygons 
with increasing numbers of vertices
on the inside of a circle
(Fig. 2.4.1). 
For a polygon with $n$ vertices,
we obtain $n$ triangles.
The height of each triangle approaches the radius $r$ 
as we partition the circle more finely. 
At the same time, its base approaches $2 \pi r/n$, 
since the ratio between arc and secant approaches 1 
for a large number of vertices. 
Thus, the area of the polygon approaches
$n \cdot r \cdot \frac{1}{2} (2 \pi r/n) = \pi r^2$.

![Finding the area of a circle as a limit procedure.](img/circle_area_approx.png)
*Fig. 2.4.1*

This limiting procedure is at the root of both 
<span style="color:red">*differential calculus*</span> and <span style="color:red">*integral calculus*</span>. 

> The former can tell us how to increase
or decrease a function's value by
manipulating its arguments. 
This comes in handy for the *optimization problems*
that we face in deep learning,
where we repeatedly update our parameters 
in order to decrease the loss function.


Optimization addresses how to fit our models to training data,
and calculus is its key prerequisite.

However, do not forget that our ultimate goal is to perform well on *previously unseen* data.
That problem is called *generalization* and will be a key focus of other chapters.

## Derivatives and Differentiation

> Put simply, a *derivative* is <span style="color:red"> the rate of change
in a function with respect to changes in its arguments</span>.

>Derivatives can tell us how rapidly a loss function
would increase or decrease were we 
to *increase* or *decrease* each parameter
by an infinitesimally small amount.

Formally, for functions $f: \mathbb{R} \rightarrow \mathbb{R}$,
that map from scalars to scalars,
[**the *derivative* of $f$ at a point $x$ is defined as**]

<span style="color:cyan">**$$f'(x) = \lim_{h \rightarrow 0} \frac{f(x+h) - f(x)}{h}.$$**</span>
:eqlabel:`eq_derivative`

This limit <span style="color:red">tells us what the ratio between a perturbation $h$
and the change in the function value $f(x + h) - f(x)$ converges to 
as we shrink its size to zero<span style="color:red">.

When $f'(x)$ exists, $f$ is said to be <span style="color:red">*differentiable*</span> at $x$

We say that $f$ is <span style="color:red">differentiable on a set $[a,b]$ </span> when

$$ \exists f'(x)  \forall x \in [a,b] $$


Not all functions are differentiable, including many that we wish to optimize,
such as accuracy and the area under the receiving operating characteristic (AUC).
However, because computing the derivative of the loss is a crucial step in nearly all 
algorithms for training deep neural networks, we often optimize a differentiable *surrogate* instead.


> We can interpret the derivative $f'(x)$ as the *instantaneous* rate of change of $f(x)$ with respect to $x$.

### Derivative function example

Let's develop some intuition with an example.
(**Define $u = f(x) = 3x^2-4x$.**)

```python
def f(x):
    return 3 * x ** 2 - 4 * x
```

and the derivative 

```python
def f_1(x, h):
    return (f(x+h)-f(x))/h
```

[**Setting $x=1$, we see that $\frac{f(x+h) - f(x)}{h}$**] (**approaches $2$
as $h$ approaches $0$.**)
While this experiment lacks 
the rigor of a mathematical proof,
we can quickly see that indeed $f'(1) = 2$.

```python
for h_v in 10.0**np.arange(-1, -6, -1):
    print(f'h={h_v:.5f}, numerical limit={f_1(1,h_v):.5f}')
```

Results in 
```text
h=0.10000, numerical limit=2.30000
h=0.01000, numerical limit=2.03000
h=0.00100, numerical limit=2.00300
h=0.00010, numerical limit=2.00030
h=0.00001, numerical limit=2.00003
```

### Notations

There are several equivalent notational conventions for derivatives.
Given $y = f(x)$, the following expressions are equivalent:

$$f'(x) = y' = \frac{dy}{dx} = \frac{df}{dx} = \frac{d}{dx} f(x) = Df(x) = D_x f(x),$$

### Common function derivatives

$$\begin{aligned} \frac{d}{dx} C & = 0 && \textrm{for any constant $C$} \\ \frac{d}{dx} x^n & = n x^{n-1} && \textrm{for } n \neq 0 \\ \frac{d}{dx} e^x & = e^x \\ \frac{d}{dx} \ln x & = x^{-1}. \end{aligned}$$

### Derivatives rules

> Functions composed from differentiable functions 
are often themselves differentiable.


The following rules come in handy for working with compositions of any differentiable functions 
$f$ and $g$, and constant $C$.

$$\begin{aligned} \frac{d}{dx} [C f(x)] & = C \frac{d}{dx} f(x) && \textrm{Constant multiple rule} \\ \frac{d}{dx} [f(x) + g(x)] & = \frac{d}{dx} f(x) + \frac{d}{dx} g(x) && \textrm{Sum rule} \\ \frac{d}{dx} [f(x) g(x)] & = f(x) \frac{d}{dx} g(x) + g(x) \frac{d}{dx} f(x) && \textrm{Product rule} \\ \frac{d}{dx} \frac{f(x)}{g(x)} & = \frac{g(x) \frac{d}{dx} f(x) - f(x) \frac{d}{dx} g(x)}{g^2(x)} && \textrm{Quotient rule} \end{aligned}$$

Using this, we can apply the rules 
to find the derivative of $3 x^2 - 4x$ via

$$\frac{d}{dx} [3 x^2 - 4x] = 3 \frac{d}{dx} x^2 - 4 \frac{d}{dx} x = 6x - 4.$$

Plugging in $x = 1$ shows that, indeed, the derivative equals $2$ at this location. 
Note that derivatives tell us the *slope* of a function at a particular location.

## Visualization

[**We can visualize the slopes of functions using the `matplotlib` library**].
We need to define a few functions. 
As its name indicates, `use_svg_display` tells `matplotlib` to output graphics in SVG format for crisper images. 

The comment `#@save` is a special modifier that allows us to save any function, 
class, or other code block to the `d2l` package so that we can invoke it later 
without repeating the code, e.g., via `d2l.use_svg_display()`.
