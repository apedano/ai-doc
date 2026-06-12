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
import torch

# Simulating one weight in a network
W = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

# Forward pass: predictions for 3 samples
x = torch.tensor([1.0, 2.0, 3.0])
predictions = W * x + b       # [3, 5, 7]

# Loss: mean squared error vs target [3, 5, 8]
target = torch.tensor([3.0, 5.0, 8.0])
loss = ((predictions - target) ** 2).mean()  # scalar

# Backward pass: compute gradients
loss.backward()

print(W.grad)   # d(loss)/dW 
print(b.grad)   # d(loss)/db
# Use these gradients to update W and b (gradient descent)
```

Loss function:

$$L=avg{(\hat y-y)^2)}$$



Meaning:

* increasing $w$ decreases the loss
* gradient descent will move $w$ upward toward 5



## Detach

```python
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)

d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))
```

Here the $x$ and $y$ tensors are passed to the `plot` function with `detach()` which 
<span style="color:red">**pass only tensor data, not keeping track of the gradient operations**</span>.

In the example above it is needed because often `plot` functions create `numpy` versions of the input 

```python
x.numpy() 
```
if the tensor is not detached, it will result in 

```text
RuntimeError:
Can't call numpy() on Tensor that requires grad.
Use tensor.detach().numpy() instead.
```

For that reason might be convenient to `detach` internally

```python
tensor.detach().numpy()
```

The tensors are detached only for visualization; the original x and y remain connected to the computation graph for gradient calculations.

## `ones_like`

`torch.ones_like(tensor)` Creates a tensor of all 1s with the same shape as the input. 
Most commonly used as the initial upstream gradient when calling `.backward()` on a non-scalar tensor.

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

y = x ** 2          # y = [1, 4, 9]

# Can't call y.backward() directly — y is not a scalar!
# Pass ones_like(y) to say: "treat each output as equally weighted"
# This is mathematically equivalent to summing the outputs first: y.sum().backward().
y.backward(torch.ones_like(y))

print(x.grad)       # tensor([2., 4., 6.])  ← dy/dx = 2x at each point
```

