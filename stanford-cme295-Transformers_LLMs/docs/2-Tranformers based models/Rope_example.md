# RoPE Rotary Position Embedding example

Consider a single pair of 2D Query and Key vectors $q = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$ and $k = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$ with base frequency $\theta = \frac{\pi}{6}$ radians ($30^\circ$).


The attention score of the rotated vectors, based on the relative position $\Delta=m-n$ and on $\theta$ is

$q'_mk^{'{\mathsf{T}}}_n=q_mR(\Delta)k_n^\mathsf{T}$


```python
import numpy as np
```


```python
q=np.array([1, 0])
k=np.array([1, 0])
theta=np.pi/6 #30 degrees

def rotation_matrix(delta:float):
    angle_rad = delta*theta
    return np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])

def get_rope_attention_score(query_vector :np.array, key_vector :np.array, m: int , n:int ):
    delta=n-m
    r_m_n=rotation_matrix(delta)
    #print(f"r_m_n: {r_m_n}")
    return np.matmul(q,np.matmul(rotation_matrix(delta),k))

att_score_1_2=get_rope_attention_score(q, k, 1,2)
print(f"att_score_1_2: {att_score_1_2}")
att_score_10_11=get_rope_attention_score(q, k, 10,11)
print(f"att_score_10_11: {att_score_10_11}")


```

    att_score_1_2: 0.8660254037844387
    att_score_10_11: 0.8660254037844387
    

The result will be the same for all positions $m,n$ next to each other where $\Delta=1$

If we increase the position distance with $m=10$ and $n=13$ ($\Delta=3$)


```python
att_score_10_13=get_rope_attention_score(q, k, 10, 13)
print(f"att_score_10_13: {round(att_score_10_13,4)}")
```

    att_score_10_13: 0.0
    

We see that the attention score becomes 0
