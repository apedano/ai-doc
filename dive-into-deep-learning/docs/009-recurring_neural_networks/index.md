# Recurrent neural networks RNNs

https://www.geeksforgeeks.org/machine-learning/introduction-to-recurrent-neural-network/
https://www.geeksforgeeks.org/machine-learning/ml-back-propagation-through-time/
https://chatgpt.com/c/6a3fd898-4f8c-83ed-88d1-984508f0cdde
https://cs231n.github.io/rnn/
https://srdas.github.io/DLBook/RNNs.html

A great many learning tasks require <span style="color:red">**dealing with sequential data**</span>.

| Problems with sequence output                        | Problems with sequence input                                              |
|------------------------------------------------------|---------------------------------------------------------------------------|
| Image captioning, speech synthesis, music generation | time series prediction, video analysis, and musical information retrieval |

Sometimes problems have

| Problems with sequence for both I/O                                                                 |
|-----------------------------------------------------------------------------------------------------|
| translating text from one natural language to another, engaging in dialogue, or controlling a robot |

![rnns_types.png](img/rnns_types.png)


> <span style="color:red">**Recurrent neural networks (RNNs)**</span> are deep learning models that capture the dynamics
> of sequences via recurrent
> connections, which can be thought of as cycles in the network of nodes.

> Recurrent Neural Networks (RNNs) were specifically designed for sequential data, where the order of the inputs
> matters.

Example: "A dog bites a man" or "A man bites a dog"

This might seem counterintuitive at first. After all, it is the feedforward nature of neural networks that makes the
order of computation unambiguous.

| Forward neural networks                                                                                                   | RNNs                                                                              |
|---------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| connections are applied synchronously to propagate each layer’s activations to the subsequent layer at the same time step | recurrent connections are dynamic, passing information across adjacent time steps |

![fnn_vs_rnn.png](img/fnn_vs_rnn.png)

### Hidden state

With this approach a RNN can keep memory of the sequence of inputs deriving a context out of it.
For example a specific word can change its meaning based on the place or the context where it is used.

Taking the language translation as example, the encoder RNN reads one word of a sentence at a time .

| Time | Input  | Hidden state    | $h_t$ |
|------|--------|-----------------|-------|
| 1    | I      | I               | $h_1$ |
| 2    | am     | I am            | $h_2$ |
| 3    | eating | I am eating     | $h_3$ |
| 4    | an     | I am eating an  | $h_4$ |
| 5    | apple  | Entire sentence | $h_5$ |

After reading the last word, the hidden state summarizes the sentence. A decoder RNN can then generate the French
translation:

`Je mange une pomme`

Each generated word is conditioned on both:

* the encoded meaning of the input sentence, and
* the words already generated.

## RNN architecture

Seen as a black box, at every time step, we submit an input to the RNN, which enters the network internal state and 
combines with the internal state of the previous time step; the outputs is connected to this combination.

![rnn_simple.png](img/rnn_simple.png)

The unrolled schema shows this in a sequence of a time series imput $x_1, x_2, \dots, x_t$ (i.e. frames in a video).

![unroll_rnn.png](img/unroll_rnn.png)

> We can represent the hidden state as a function of the input vector at the given time and of the hidden state (as vector) at the $t-1$ time step

$$h_t=f(h_{t-1}, x_t)$$

We are not committed to the size of the input sequence (the number of frames in the video)

## Single hidden state RNNs - Vanilla RNNs

### Forward step 

To represent the hidden state update we need to wight matrices $W_{xh}$ and $W_{hh}$ both projecting, respectively, the
projection of the input and the previous time step state to the current state  (squished by the activation function $\sigma$, usually $sigmoid$ or $tanh$):

$$a_t=W_{hh}h_{t-1}+ W_{xh}x{t} + b_h \:\: (\ast)$$ 

$$h_t=\tanh\left(a_t\right) \:\: (\star)$$

![vanilla_rnn_mformula_1.png](img/vanilla_rnn_mformula_1.png)

The preditction then becomes a function of the state accordingly, based on a wight matrix $W_{hy}$ projecting the hidden 
state onto the prediction

$$z_t=W_{hy}h_t+b_y \:\: (\circ)$$

$$ \hat y_t=softmax(z_t) \:\: (\bigcirc)$$

where $y,\hat y \in \mathbb{R}^K$

and $\hat y_i =\frac{e^{z_i}}{\sum_{j=1}^{K}e^j}$

![vanilla_rnn_mformula_2.png](img/vanilla_rnn_mformula_2.png)

### Loss (cross-entropy) calculation

For each time step $t$ the cross entropy loss is calculated as 


$$L_t=-\sum_{i=1}^{K}y_i\log(\hat y_i)$$

since $y$ is a one-hot vector, it becomes the logarithm of the estimation of the true class

Example:

* $y=[0,0,1]$,
* RNN Logits (unscaled outputs): $z=[1.2, 0.8, 2.5]$
* $\hat y = softmax(e)=[0.17, 0.12, 0.71]$
* $L_t= -\log(0.71)=0.342$

> So the total loss becomes the <span style="color:red">**scalar**</span> given by the sum of all those $L_t$

$$L=\sum_{t=1}^{T}L_t$$

### Backpropagation BTT (Backpropagation Through Time)

We update the weights $W_{hh}$ by getting the derivative of the loss at the very last time step $L_t$ with respect to $W_{hh}$

> The matrix update is unique for all time steps $W_h$ does not depend on the time $t$. 

> See in the backpropagation, how the output at the very last timestep affects the weights backward to the previous steps

> The derivative of the loss will be then the <span style="color:red">**sum of all the derivative at every time step**</span>

![backpropagation.png](img/backpropagation.png)

$$\frac{\partial L}{\partial W_{h}}=\sum_{t=1}^T\frac{\partial L_t}{\partial W_{h}} \:\:(\beta)$$

The single element in the sum (at the generic time $t$) can be rewritten with the **chain rule**: 

$$\frac{\partial L_t}{\partial W_{h}}=\frac{\partial L_t}{\partial h_{t}}\frac{\partial h_t}{\partial W_{h}}$$

If we introduce the $t-1$ step in the same way, using the chain rule, we obtain

$$\frac{\partial L_t}{\partial W_{h}}=\frac{\partial L_t}{\partial h_{t}}\frac{\partial h_{t}}{\partial h_{t-1}} \frac{\partial h_{t-1}}{\partial W_{h}}$$

if we continue to the step 1 we obtain

$$\frac{\partial L_t}{\partial W_{h}}=\frac{\partial L_t}{\partial h_{t}}\left(\prod_{k=2}^{t} \frac{\partial h_{k}}{\partial h_{k-1}} \right)\frac{\partial h_{1}}{\partial W_{h}} \:\:(\gamma)$$


If we extract how the derivative of the hidden state at time step $k$ depends on the
hidden state at $k-1$ from $(\alpha)$we have

$$\frac{\partial h_k}{\partial h_{k-1}} =  tanh^{'}(W_{hh}h_{k-1} + W_{xh}x_k)W_{hh} $$

and we apply it to $(\gamma)$ we have

$$\frac{\partial L_t}{\partial W_{h}}=\frac{\partial L_t}{\partial h_{t}}\left(\prod_{k=2}^{t} tanh^{'}(W_{hh}h_{k-1} + W_{xh}x_k)W_{hh} \right)\frac{\partial h_{1}}{\partial W_{h}} \:\:(\epsilon)$$

So we can write the <span style="color:red">**final form of the loss gradient**</span>


> $$\frac{\partial L}{\partial W_{h}}=\sum_{t=1}^T\sum_{k=1}^t\frac{\partial L_t}{\partial h_{t}}\left(\prod_{j=k+1}^{t}tanh^{'}(W_{hh}h_{j-1})\right)\frac{\partial h_k}{\partial W_{h}} $$ 

The derivative is made of three factors

1) $\frac{\partial L_t}{\partial h_{t}}$ → just standard backprop from the output layer.
2) $\prod_{j=k+1}^{t}tanh^{'}(W_{hh}h_{j-1})$ → hidden state to hidden state contribution back in time
3) $\frac{\partial h_k}{\partial W_{h}}$ → local contribution at step $k$


#### Issue with the sequence length

The longer the time sequence is, the more contributions it requires. For this reason:

* Run forward and backward through chunks of the sequence instead of whole sequence

* Carry hidden states forward in time forever, but only backpropagate for some smaller
number of time steps ($T$ in the derivative formula)

![backpropagation_2.png](img/backpropagation_2.png)

#### Limitations of BTT in vanilla RNNs

Depending on the behavior of the hidden state to hidden state contributions we have the two cases

* <span style="color:red">**Vanishing gradient**</span>: 
  * $\frac{\partial h_k}{\partial h_{k-1}}<1$
  * since the $tanh^{'}$ function can have very small values (the function will go from -1 to 1), the more elements we add to the products the smaller the final derivative will become. The reduction is **exponential**
  * The network will become less sensible to changes in the past (loosing memory), considering that we need memory of the sequence for the training.
  * The backpropagation will create very little variations of the wight matrix, making the network very "stubborn" to changes 

* <span style="color:red">**Exploding gradient**</span>: 
  * $\frac{\partial h_k}{\partial h_{k-1}}>1$
  * unstable training and updates due to big variations of the weight matrix related to small variations of the input/hidden state

In practice, we can **treat the exploding gradient problem through gradient clipping**, which is clipping large gradient values to a maximum threshold. 
However, since vanishing gradient problem still exists in cases where largest singular value of $W_{hh}$ matrix is less than one, LSTM was designed to avoid this problem.


#### Long-Short Term Memory (LSTM)

In practice, we actually will rarely ever use Vanilla RNN formula. 
Instead, we will use what we call a <span style="color:red">**Long-Short Term Memory (LSTM) RNN**</span>.

https://cs231n.github.io/rnn/#lstm-formulation


## Example
https://cs231n.github.io/rnn/#rnn-example-as-character-level-language-model


