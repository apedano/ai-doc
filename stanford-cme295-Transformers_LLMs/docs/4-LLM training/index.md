# LLM training

The traditional way of training language model was to train on a specific task

Training language models is highly expensive, so the idea is to use a pre-trained model and fine tune it for the 
specific task

```mermaid
flowchart LR
    classDef success fill:#d4edda, stroke: #28a745, color: #155724;
    classDef danger fill:#f8d7da, stroke: #dc3545, color: #721c24;
    classDef warning fill:#fff3cd, stroke: #ffc107, color: #856404;
    classDef blue fill:#6497b1, stroke:	#03396c, color: #011f4b;
    
    Node1[Spam detection]:::success
    Node2[Sentiment extraction]:::warning
    Node3[Translation model]:::danger
    Node4[Pre-trained LLM]:::blue -.-> Node1
    Node4[Pre-trained LLM]:::blue -.-> Node2
    Node4[Pre-trained LLM]:::blue -.-> Node3
    
```

This is what BERT does as explained here: [bert_deep_dive.md](../2-Tranformers%20based%20models/bert_deep_dive.md)

>So the pretrained model is trained with (MLM or NSP) to learn the "basics" of the target language. Then the same model is trained with the specific task trianing set

> The pre-training is <u>always</u> to predict the next token.

## Training data for pre-trained models

* natural language 
  * Common crawl (https://commoncrawl.org/) - Over 300 billion pages spanning 19 years.
  * Wikipedia
* Code
  * Github
  * StackOverflow

Examples are:

* GPT-3 trained on a training set of 300Bi of tokens
* Llama 3 15 Trilions

## Notation FLOP, FLOPS

![flop.png](img/flop.png)

![flops.png](img/flops.png)

## Performance 

How the model performance evolves as function of the model size and the training size

![performance.png](img/performance.png)

The following picture shows that for a certain amount of input tokens on the x-axis, a bigger model (the colors towards yellow)
produce less loss, therefore better performance

![performance_per_size.png](img/performance_per_size.png)

### Chinchilla law

Based on a fix amount of compute (FLOPS) what are the ideal model size and training set size.

![chichilla_law_1.png](img/chichilla_law_1.png)

The coloured lines are per compute size.

![chichilla_law_1.png](img/chichilla_law_1.png)

As a rule of thumbs, the ideal sizes are <u>training set size 20 times the model size</u>.

### Knowledge cutoff

> The knowledge of the model is updated up to the time of the update time of the training data provided.

This means that if the question is about information after that date the model can retrieve that info from 
external sources

```mermaid
flowchart LR
    classDef success fill:#d4edda, stroke: #28a745, color: #155724;
    classDef danger fill:#f8d7da, stroke: #dc3545, color: #721c24;
    classDef warning fill:#fff3cd, stroke: #ffc107, color: #856404;
    classDef blue fill:#6497b1, stroke:	#03396c, color: #011f4b;
    
    Node1[Cutoff knowledge]:::success
    Node2[Extrarnal source - Web search]:::warning
    Node4[LLM]:::blue -.-> Node1
    Node4[LLM]:::blue -.-> Node2
    
```

It is hard to add new knowledge to an already pre-trained model, because the weights are already determined and are 
not changed at inference time.

The risk is to introduce <span style="color:#FF0000">**plagiarism**</span>, the tendency of copying parts of answers from external sources and not 
as part of elaboration.

## Training in decoder only LLMs - Adam optimizer

The problem of training decoder-only LLMs models involves updated different type of weights for each transformer layer

* the <u>masked self attention layer</u>: $Q=XW_Q$, $K=XW_K$,$V=XW_V$ so that we have $Attention(Q,K,V)=softmax(\frac{QK^\mathsf{T}+M}{\sqrt{d_k}})V$
* the <u>output</u> $O=Attention(Q,K,V)W_O$
* the FFN(x) with the typical two projections $n \rightarrow m \rightarrow n$: $\sigma(xW_1+b_1)W_2+b_2$
* token embeddings (RoPE)
* Layer normalizations
* MoE (if present)

> The training compares the next token probability $P(mat∣The cat is sitting on the)$ with the target to calculate the loss

### The traditional SGD is not used

The Stochastic Gradient Descent 

$\Theta_{t+1}=\Theta_{t}-\eta g_t$ ($\eta$ learning rate) ($g_t=\nabla_\Theta L_t$)

An LLM may have billions of parameters, and different parameters can have very different gradient behavior.

Adam addresses this by maintaining information about the gradients over time.

### The Adam optimizer

For each parameter $\Theta$, Adam memorizes two values

* <span style="color:#FF0000">**First moment**</span>: a exponentially moving average of the gradient

$$m_t=\beta_tm_{t-1}(1-\beta_t)m_{t}$$

* <span style="color:#FF0000">**Second moment**</span>: a exponentially moving average of the gradient

$$m_t=\beta_tm_{t-1}(1-\beta_t)m_{t}$$

https://chatgpt.com/c/6aa56667-8c10-83eb-a618-eab9d7a7bd88





