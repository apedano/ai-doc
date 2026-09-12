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

