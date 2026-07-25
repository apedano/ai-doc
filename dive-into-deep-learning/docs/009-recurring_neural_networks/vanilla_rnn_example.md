#Implementation of vannilla RRN with training 


# Initial data


```python
import numpy as np
import scipy as sp
```

# Functions

## Forward step


```python
def forward_step(time_step :int, h_previous : np.array, W_hh :np.array, W_hx:np.array, b_h:np.array, b_y:np.array, x:np.array, y_true_class: int) -> np.array:
    a_current=np.matmul(W_hh,h_previous)+np.matmul(W_hx,x)+b_h
    print(f"a_{time_step}:{a_current}")
    h_current=np.tanh(a_current)
    print(f"h_{time_step}:{h_current}")
    z_current=np.matmul(W_hy,h_current)+b_y
    print(f"z_{time_step}:{z_current}")
    y_est_current=sp.special.softmax(z_current)
    print(f"y_est_{time_step}:{y_est_current}")
    y_est_true_class=y_est_current[y_true_class, 0]
    print(f"y_true_class: {y_est_true_class}")
    l_current=-np.log(y_est_current[y_true_class,0])
    print(f"l_{time_step}:{l_current}")
    return h_current,l_current,y_est_current

```

### One hot vector


```python
def one_hot_column_vector(index, length):
    # Create a 2D array of zeros with shape (length, 1)
    vector = np.zeros((length, 1))

    # Set the '1' at the specific row index in the first column
    vector[index, 0] = 1.0

    return vector
```

## Error signal $\delta_t$


```python
def error_signal(h_t: np.array, weights_hh: np.array, delta_future:np.array ) -> np.array:

    delta_t_temp=np.matmul(np.transpose(weights_hh),delta_future)
    h_t_squared=h_t**2
    #hadamand product multiply (1-h^2)
    delta_t=np.multiply(delta_t_temp, (np.ones(h_t.shape)-h_t_squared))
    return delta_t
```

# Initial data


```python
W_hx = np.array([[0.5, -0.3],
               [0.1, 0.8]])
print(f"Shape of W_hx:{W_hx.shape}")

W_hh = np.array([[0.2, 0.4],
               [-0.5, 0.3]])
print(f"Shape of W_hh:{W_hh.shape}")

W_hy = np.array([[0.3, -0.2],[0.1, 0.4], [-0.3, 0.2], [0.2, 0.1]])
print(f"Shape of W_hy:{W_hy.shape}")

b_h = np.array([[0.1], [-0.1]])
print(f"Shape of b_h:{b_h.shape}")

b_y = np.array([[0], [0], [0], [0]])
print(f"Shape of b_y:{b_y.shape}")

h_0=np.array([[0], [0]])
print(f"Shape of h_0:{h_0.shape}")

#learning rate
eta=0.1

```

    Shape of W_hx:(2, 2)
    Shape of W_hh:(2, 2)
    Shape of W_hy:(4, 2)
    Shape of b_h:(2, 1)
    Shape of b_y:(4, 1)
    Shape of h_0:(2, 1)
    

## True classes



```python
y_1=2
y_2=0
y_3=3
```

# Input data


```python
x_1=np.array([[1], [0.5]])
print(f"Shape of x_1:{x_1.shape}")

x_2=np.array([[0.5], [-1]])
print(f"Shape of x_2:{x_2.shape}")


x_3=np.array([[-1], [1]])
print(f"Shape of x_3:{x_3.shape}")





```

    Shape of x_1:(2, 1)
    Shape of x_2:(2, 1)
    Shape of x_3:(2, 1)
    

# Forward pass

## t=1


```python
h_1,l_1,y_est_1 =forward_step(1,h_0,W_hh,W_hx,b_h,b_y,x_1,y_1)
```

    a_1:[[0.45]
     [0.4 ]]
    h_1:[[0.42189901]
     [0.37994896]]
    z_1:[[ 0.05057991]
     [ 0.19416949]
     [-0.05057991]
     [ 0.1223747 ]]
    y_est_1:[[0.24197645]
     [0.27934008]
     [0.21869555]
     [0.25998792]]
    y_true_class: 0.2186955488365349
    l_1:1.520074704334452
    

## t=2


```python
h_2,l_2,y_est_2 =forward_step(2,h_1,W_hh,W_hx,b_h,b_y,x_2,y_2)
```

    a_2:[[ 0.88635939]
     [-0.94696481]]
    h_2:[[ 0.7095909 ]
     [-0.73840587]]
    z_2:[[ 0.36055844]
     [-0.22440326]
     [-0.36055844]
     [ 0.06807759]]
    y_est_2:[[0.35845562]
     [0.19970537]
     [0.17428432]
     [0.26755469]]
    y_true_class: 0.35845561810987603
    l_2:1.0259504254628646
    

## t=3


```python
h_3,l_3,y_est_3 =forward_step(3,h_2,W_hh,W_hx,b_h,b_y,x_3,y_3)
```

    a_3:[[-0.85344417]
     [ 0.02368279]]
    h_3:[[-0.6928645 ]
     [ 0.02367836]]
    z_3:[[-0.21259502]
     [-0.05981511]
     [ 0.21259502]
     [-0.13620506]]
    y_est_3:[[0.20945334]
     [0.24402749]
     [0.32043871]
     [0.22608046]]
    y_true_class: 0.22608045720641756
    l_3:1.4868643377265847
    

## Backward pass

### Output Error signal
$$\delta^{z}_{t}=\frac{\partial L_t}{\partial z_t}=\hat y_t - y_t$$



```python
dz_3=y_est_3-one_hot_column_vector(y_3, 4)
print(f"Derivative of dz_3:{dz_3}")
dz_2=y_est_2-one_hot_column_vector(y_2, 4)
print(f"Derivative of dz_2:{dz_2}")
dz_1=y_est_1-one_hot_column_vector(y_1, 4)
print(f"Derivative of dz_1:{dz_1}")
```

    Derivative of dz_3:[[ 0.20945334]
     [ 0.24402749]
     [ 0.32043871]
     [-0.77391954]]
    Derivative of dz_2:[[-0.64154438]
     [ 0.19970537]
     [ 0.17428432]
     [ 0.26755469]]
    Derivative of dz_1:[[ 0.24197645]
     [ 0.27934008]
     [-0.78130445]
     [ 0.25998792]]
    

### Hidden state and pre-activation error signals

$$\delta^{h}_{t}=(\delta^{h}_{t})_{local}+(\delta^{h}_{t})_{rec}=W_{hy}^\mathsf{T}\delta^{z}_{t}+W_{hh}^\mathsf{T}\delta^{a}_{t+1}$$
$$\delta^{a}_{t}=(1-h^2_t)\odot\delta^{h}_{t}$$

### t=3
$\delta^{h}_{T+1}:=0 \rightarrow \delta^{h}_{T}=W_{hy}^\mathsf{T}\delta^{z}_{T}$

$\delta^{a}_{T}=(1-h^2_T)\odot\delta^{h}_{T}$



```python
dh_3=np.matmul(np.transpose(W_hy),dz_3)
print(f"dh_3:{dh_3}")
da_3=(1-h_3**2) * dh_3
print(f"da_3:{da_3}")
```

    dh_3:[[-0.16367677]
     [ 0.04241612]]
    da_3:[[-0.0851019 ]
     [ 0.04239234]]
    

### t=2


```python
dh_2=np.matmul(np.transpose(W_hy),dz_2)+np.matmul(np.transpose(W_hh),da_3)
print(f"dh_2:{dh_2}")
da_2=(1-h_2**2) * dh_2
print(f"da_2:{da_2}")


```

    dh_2:[[-0.20948368]
     [ 0.2484803 ]]
    da_2:[[-0.10400462]
     [ 0.1129981 ]]
    

### t=1


```python
dh_1=np.matmul(np.transpose(W_hy),dz_1)+np.matmul(np.transpose(W_hh),da_2)
print(f"dh_1:{dh_1}")
da_1=(1-h_1**2) * dh_1
print(f"da_1:{da_1}")

```

    dh_1:[[ 0.30961589]
     [-0.07462377]]
    da_1:[[ 0.25450464]
     [-0.06385099]]
    

## Parameter gradients


Derivative $\frac{\partial L}{\partial W_{hy}}=\sum_{t=1}^{T}\delta^z_{t}h_t^T$


```python
dL_dWhy=np.matmul(dz_3,np.transpose(h_3))+np.matmul(dz_2,np.transpose(h_2))+np.matmul(dz_1,np.transpose(h_1))
print(f"dL_dWhh:{dL_dWhy}")

```

    dL_dWhh:[[-0.49826722  0.57061835]
     [ 0.09048443 -0.03555048]
     [-0.42798161 -0.41796091]
     [ 0.8357644  -0.11710696]]
    

Derivative $\frac{\partial L}{\partial W_{hh}}=\sum_{t=1}^{T}\delta^a_{t}h_{t-1}^T$


```python
dL_dWhh=np.matmul(da_3,np.transpose(h_2))+np.matmul(da_2,np.transpose(h_1))+np.matmul(da_1,np.transpose(h_0))
print(f"dL_dWhh:{dL_dWhh}")
```

    dL_dWhh:[[-0.10426698  0.0233233 ]
     [ 0.077755    0.01163076]]
    

Derivative $\frac{\partial L}{\partial W_{hx}}=\sum_{t=1}^{T}\delta^a_{t}x_{t}^T$



```python
dL_dWhx=np.matmul(da_3,np.transpose(x_3))+np.matmul(da_2,np.transpose(x_2))+np.matmul(da_1,np.transpose(x_1))
print(f"dL_dWhx:{dL_dWhx}")
```

    dL_dWhx:[[ 0.28760423  0.14615504]
     [-0.04974428 -0.10253126]]
    

Derivative $\frac{\partial L}{\partial b_{h}}=\sum_{t=1}^{T}\delta^a_{t}$



```python
dL_dbh=da_3+da_2+da_1
print(f"dL_dbh:{dL_dbh}")
```

    dL_dbh:[[0.06539813]
     [0.09153944]]
    

Derivative $\frac{\partial L}{\partial b_{y}}=\sum_{t=1}^{T}\delta^z_{t}$


```python
dl_dby=dz_3+dz_2+dz_1
print(f"dl_dby:{dl_dby}")
```

    dl_dby:[[-0.1901146 ]
     [ 0.72307295]
     [-0.28658142]
     [-0.24637693]]
    

## Update step

$$W_{hy}^{new}=W_{hy}^{old}-\eta \delta_{z_t}h_t^T, \:\: b_{y}^{new}=b_{y}^{old}-\eta\delta_{z_t}$$

$$W_{hx}^{new}=\begin{bmatrix}0.471&-0.315\0.105&0.810\end{bmatrix}\quad
W_{hh}^{new}=\begin{bmatrix}0.210&0.398\-0.508&0.299\end{bmatrix}\quad
W_{hy}^{new}=\begin{bmatrix}0.350&-0.257\0.091&0.404\-0.257&0.242\0.116&0.112\end{bmatrix}$$


```python
W_hy=W_hy-eta*dL_dWhy
print(f"Updated W_hy:{W_hy}")
W_hh=W_hh-eta*dL_dWhh
print(f"Updated W_hy:{W_hy}")
W_hx=W_hx-eta*dL_dWhx
print(f"Updated W_hx:{W_hx}")
b_h=b_h-eta*dL_dbh
print(f"Updated b_h:{b_h}")
b_y=b_y-eta*dl_dby
print(f"Updated b_y:{b_y}")

```

    Updated W_hy:[[ 0.34982672 -0.25706183]
     [ 0.09095156  0.40355505]
     [-0.25720184  0.24179609]
     [ 0.11642356  0.1117107 ]]
    Updated W_hy:[[ 0.34982672 -0.25706183]
     [ 0.09095156  0.40355505]
     [-0.25720184  0.24179609]
     [ 0.11642356  0.1117107 ]]
    Updated W_hx:[[ 0.47123958 -0.3146155 ]
     [ 0.10497443  0.81025313]]
    Updated b_h:[[ 0.09346019]
     [-0.10915394]]
    Updated b_y:[[ 0.01901146]
     [-0.07230729]
     [ 0.02865814]
     [ 0.02463769]]
    
