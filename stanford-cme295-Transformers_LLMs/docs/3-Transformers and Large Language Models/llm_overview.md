# LLMs

https://www.youtube.com/watch?v=Q5baLehv5So

Slides: https://cme295.stanford.edu/slides/fall25-cme295-lecture3.pdf


## Definition

 > <span style="color: red">**LLM - Large Language Model**</span>
 
> <span style="color: red">**Language Model**</span>: a statistical or machine learning model that assigns probabilities to
sequences of tokens.

> <span style="color: red">**Large**</span>: involves 100s of Bilions of parameter and large computation time for both inference and training

Most of LLMs are <span style="color: red">**decoder-only transformer based**</span>

![decoder-only.png](img/decoder-only.png)

Despite <u>**BERT which is a encoder-only transformer based model**</u>.

```mermaid
flowchart LR
    classDef input fill:none,stroke:none;
    classDef danger fill:#008080,stroke:#008000,color:#FFA500;
    

    Node1[x]:::input --> Node2[BIG MODEL]:::danger
    Node2 --> Node3[y^]:::input
```

### Examples of LLMs

GPT, Gamma (Google), DeepSeek, LLama etc...

## MoE - Mixture of Experts

The <span style="color: red">**intuition**</span> is of involve only subparts of the model to generate a specific output to reduce the computation overhead 
need by the complete model

```mermaid
flowchart LR
    classDef input fill:none,stroke:none;
    classDef danger fill:#008080,stroke:#008000,color:#FFA500;
    

    Node1[x]:::input --> Node2[BIG MODEL subpart]:::danger
    Node2 --> Node3[y^]:::input
```

> The idea is to divide the model into subparts called <span style="color: red">**experts**</span> $E_i \rightarrow i \in [1,\dots,n]$

> Given the input $x$, the <span style="color: red">**gate or route $G_i(x)$**</span> selects how the corresponding expert contributes in the output generation, so that

$$\hat y=\sum_{i=1}^nG_i(x)E_i(x)$$

### Dense MoE

> All available experts are involved in the output generation with a contribution weighted by the applicable gate

$$G_i(x) \in \mathbf{R} [0,1]$$

![dense_moe.png](img/dense_moe.png)

### Sparse MoE

> Only a <span style="color: red">**top-k**</span> selection of gates is used for the output generation

$K\times G_i(x) \in \mathbf{I}[0,1]$

![sparse_moe.png](img/sparse_moe.png)

## Integration of MoE in decoder based LLMs

In a **decoder-only base LLM** such as **GPT**-style models, a **Mixture of Experts (MoE)** replaces the normal dense feed-forward network (**FFN**) in some or all Transformer layers with a collection of specialized FFNs called <span style="color: red">**experts**</span>.

> <u>**IDEA**</u>: For each token, the model dynamically chooses a small number of experts instead of running the token through one shared FFN.

```mermaid
flowchart LR
    classDef noDecoration fill:none,stroke:none;
    classDef attention fill:#008080,stroke:#008000,color:#FFA500;
    classDef an fill:#FFFACD,stroke:#FFD700,color:#FF8C00;
    classDef ffn fill:#4682B4,stroke:#1E90FF,color:#AFEEEE;
    

    token[tokens]:::noDecoration --> MMHA
    token[tokens]:::noDecoration --> AN1
    
    subgraph Trans1 [Decoder Transformer Layer]
        direction LR
        MMHA[Masked 
        Multi-Head
        attention]:::attention--> AN1[Add and Norm]:::an
        
        AN1-->FFN[FFN]:::ffn
        FFN-->AN2[Add and Norm]:::an
        AN1-->AN2
            
    end
    AN2-->output[output]:::noDecoration
    
```


https://chatgpt.com/c/6a89c687-98f0-83eb-b522-452367a67409


