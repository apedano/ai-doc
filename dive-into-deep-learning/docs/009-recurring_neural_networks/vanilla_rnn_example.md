# Vanilla RNN example

https://chatgpt.com/c/6a3fd898-4f8c-83ed-88d1-984508f0cdde?mweb_fallback=1

> The goal is to predict whether a movie review is positive or negative after reading the words one by one.

| Input             | Output                           |
|-------------------|----------------------------------|
| `movie was great` | Positive (`1`) or negative (`0`) |

The input is a numeric value embedded to every single word

| Word    | Input x |
|---------|---------|
| `movie` | 0.2     |
| `was`   | 0.1     |
| `great` | 0.8     |

## RNN architecture

> One hidden layer

| RNN feature     | formula                       |
|-----------------|-------------------------------|
| hidden state    | $tanh(W_xx+W_{h}h_{t-1}+b_h)$ |
| expected output | $\hat y=\sigma(W_yh_3+b_y)$   |

| Parameter | value |
|-----------|-------|
| $W_x$     | 0.5   |
| $W_h$     | 0.8   |
| $W_y$     | 1.2   |
| $b_h$     | 0     |
| $b_y$     | 0     |
| $h_0$     | 0     |

## Forward propagation


| Input             | hidden state value                                    |
|-------------------|-------------------------------------------------------|
| `movie` $x_1=0.2$ | $h_1=tanh(0.5 \times 0.2 + 0.8 \times 0)=0.0997$      |
| `was` $x_2=0.1$   | $h_2=tanh(0.5 \times 0.1 + 0.8 \times 0.0997)=0.1291$ |
| `great` $x_3=0.8$ | $h_3=tanh(0.5 \times 0.8 + 0.8 \times 0.1291)=0.4647$ |


## Prediction

| Neuron output                   | $z=1.2 \times 0.4647=0.5577$ |
|---------------------------------|------------------------------|
| prediction (sigmoid activation) | $\hat y=\sigma (z)=0.6359$   |

| Prediction        | Real label |
|-------------------|------------|
| $\hat y = 63.6 $% | $y=1$      |

## Loss calculation

Using the binary cross entropy loss function

$$L = -y log(\hat y)=-log(0.6359)=0.452$$

## Backward propagation

For binary cross-entropy with sigmoid

$$\frac{\partial L}{\partial z}=\hat - y =0.6359-1=-0.3641$$

