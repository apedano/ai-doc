# Recurrent neural networks RNNs

https://www.geeksforgeeks.org/machine-learning/introduction-to-recurrent-neural-network/
https://www.geeksforgeeks.org/machine-learning/ml-back-propagation-through-time/
https://chatgpt.com/c/6a3fd898-4f8c-83ed-88d1-984508f0cdde

A great many learning tasks require <span style="color:red">**dealing with sequential data**</span>.

| Problems with sequence output                        | Problems with sequence input                                              |
|------------------------------------------------------|---------------------------------------------------------------------------|
| Image captioning, speech synthesis, music generation | time series prediction, video analysis, and musical information retrieval |

Sometimes problems have

| Problems with sequence for both I/O                                                                 |
|-----------------------------------------------------------------------------------------------------|
| translating text from one natural language to another, engaging in dialogue, or controlling a robot |


> <span style="color:red">**Recurrent neural networks (RNNs)**</span> are deep learning models that capture the dynamics of sequences via recurrent
connections, which can be thought of as cycles in the network of nodes. 

> Recurrent Neural Networks (RNNs) were specifically designed for sequential data, where the order of the inputs matters.

This might seem counterintuitive at first. After all, it is the feedforward nature of neural networks that makes the order of computation unambiguous. 
However, recurrent edges are defined in a precise way that ensures that no such ambiguity can arise. 

Recurrent neural networks are unrolled across time steps (or sequence steps), with the same underlying parameters applied at each step. 
While the standard connections are applied synchronously to propagate each layer’s activations to the subsequent layer at the same time
step, the recurrent connections are dynamic, passing information across adjacent time steps. As the unfolded view in
Fig. 9.1 reveals, RNNs can be thought of as feedforward neural networks where each layer’s parameters (both conventional
and recurrent) are shared across time steps.