# Transformers and attention mechanisms

## Documentation

STANFORD: https://www.youtube.com/watch?v=RQowiOF_FvQ
https://arxiv.org/html/2604.00965v1

https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/
https://jalammar.github.io/illustrated-transformer/

## RNNs Vs Transformers

We have seen that RNNs are able to solve many kind of problem 
* one-to-one: 
* one-to-many: from a work to an image that represents that word
* many-to-one: image classification problem
* many-to-many: from a video to a description of it

For instance, for image classification we have also CNN.

>After a very important article, <span style="color: red">**[Attention is all you need](https://arxiv.org/abs/1706.03762)**</span>, 
the concept of attention and transformers are considered a valid solution for all the mentioned 
problems taking over the traditional CNN and RNN operators

> Nowaday transformers (often in combination with sel-attention) are used <span style="color: red">**everywhere**</span>

## Sequence to sequence with RNNs: DECODER-ENCODER

Based on the example of language translation from English to Italian

* **Input**: sequence $x_1,x_2,\dots,x_T$
* **Output**: sequence $y_1,y_2,\dots,y_T$. The number of words in the input language can be different from the number of words in the translation

The encoder is a RNN where the inner state sequence is a function of the input and the 
previous hidden state

> <span style="color: red">**ENCODER**</span>: $h_t=f_w\left(x_{t},h_{t-1}\right)$

![rnn.png](img/rnn.png)

The decoder produces an internal state from the input. We move towards the output with 
a DECODER. 

All the processing of the input sequence in the decoder is represented in a the 
<span style="color: red">**CONTEXT VECTOR**</span> $C$ 

Often $C=h_T$ considering that the hidden state at the final time step incorporates information form all the previous hidden states in the past. 

The same way for the decoder, the ENCODER is a RNN with its <span style="color: red">**decoder hidden state**</span> $s_t$

> <span style="color: red">**ENCODER**</span>: $s_t=g_u\left(y_{t-1},c,s_{t-1}\right)$

![encoder_decoder.png](img/encoder_decoder.png)

We see that the 4 input words generate 3 distinct output value.

### The problem of the fixed lenght of $C$

The only connection between the input sequence and the output sequence it <span style="color: red">**the fixed length of the context vector $C$**</span>
which might be not enough for very long sequences (a paragraph, a book) or overidimensioned for small sequences (small sentences).

> The **solution** is to look back at the entire input sequence at every time step to generate the output

## Attention in RNN based sequence-to-sequence

Compared to the previous setup, we still have an encoder RNN and the second decoder RNN with the 
hidden state $s_t$

> This time the decoder hidden state at time stamp $t$ will be the scalar result of the linear activation  
> of a combination of the encoder hidden state and the previous decoder hidden state called <span style="color: red">**alignment scores**</span>

$$e_{t,i}=f_{att}\left(s_{t-1},h_i\right) \:(scalar)$$

where

* $t$ is the decoder time step
* $i$ is the encoder time step

  

Among all possible ways to have $f_{att}$ combine the vectors a linear transformation is ofter used 

The alignent scores are arbitrary scalar, so we need to bound their values using the $softmax$ function in order
to obtain a probability distribution from them called <span style="color: red">**attention weight**</span>.

$$a_{t,i}=softmax(e_{t,i})$$

$0 \lt a_{t,i} \lt 1$ and $\sum_{i}a_{t,i}=1$ for each time step

> the attention score is best understood as **how much attention the decoder should pay to the i-th input word 
> when generating the t-th output word**.


We can now compute a new <span style="color: red">**context vector**</span> as the weighted some of all attention weights over the corresponding encoder 
hidden state

$$c_{t}=\sum_ia_{t,i}h_i$$



<span style="color: red">**T:12:22**</span> 










