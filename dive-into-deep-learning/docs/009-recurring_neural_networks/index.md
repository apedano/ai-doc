# Recurrent neural networks RNNs

https://www.geeksforgeeks.org/machine-learning/introduction-to-recurrent-neural-network/
https://www.geeksforgeeks.org/machine-learning/ml-back-propagation-through-time/
https://chatgpt.com/c/6a3fd898-4f8c-83ed-88d1-984508f0cdde
https://cs231n.github.io/rnn/

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

