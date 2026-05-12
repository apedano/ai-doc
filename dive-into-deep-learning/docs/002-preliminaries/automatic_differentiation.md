# Automatic differentiation

## Introduction: the need for automatic differentiation
Gradients are used in machine learning to update parameters during training:

$$
 \theta \leftarrow \theta - \eta \nabla_{\theta} L
$$

where:

* $\theta $ = parameters
* $L$ = loss function
* $\eta$ = learning rate

PyTorch computes these gradients automatically using `autograd`.

### Forward and backward pass 

The <span style="color:red">**backward pass**</span> is the step where PyTorch Autograd documentation
 computes gradients using the chain rule of calculus.

In machine learning there are usually two phases:

* **Forward pass**
    * Compute outputs and the loss.
    * `x  -> operations -> y` as `Input → prediction → loss` 
    * (Ex: `image -> neural network -> cat probability -> loss`)
* **Backward pass**
    * Compute how much each parameter contributed to the loss.
    * For every weight in the network $∂parameter/∂loss$
    * PyTorch uses these gradients to update weights during training.

### Pytorch code

```python
# Can also create x = torch.arange(4.0, requires_grad=True)
x = torch.arange(3.0)
x.requires_grad_(True)
x.grad  # The gradient is None by default
```

Create a function $y = 2 \sum_{i=0}^{3}x_i^2$

```python
y = 2 * torch.dot(x, x) 
```

PyTorch records all operations:

* square
* multiplication
* addition

This builds a computation graph.

Now we can call the **backward pass** to compute the gradients:

```python
y.backward()
```

So that `x.grad` stores the computed gradient:

∂x / ∂y


after `backward()` is called.

```python
x.grad # tensor([ 0.,  4.,  8., 12.])
```

considering that, in this case:

$$\nabla y=4x$$

```text
x = [0,1,2,3]
4x = [0,4,8,12]
```

## Example for neural network training

```python
w = torch.tensor(2.0, requires_grad=True)

loss = (w - 5)**2

loss.backward()

print(w.grad)
```

Loss function:

$$L=(w−5)^2$$

Derivative:

$$\frac {\delta L} {\delta w} = 2(w-5)$$

At $w=2$:

```python
tensor(-6.)
```

Meaning:

* increasing $w$ decreases the loss
* gradient descent will move $w$ upward toward 5