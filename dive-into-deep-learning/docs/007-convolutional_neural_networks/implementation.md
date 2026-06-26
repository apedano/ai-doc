## CNNs Pytorch implementation

[Link to page](https://www.datacamp.com/tutorial/pytorch-cnn-tutorial?utm_cid=23340058065&utm_aid=192632748929&utm_campaign=230119_1-ps-dscia~dsa-tofu~python_2-b2c_3-emea_4-prc_5-na_6-na_7-le_8-pdsh-go_9-nb-e_10-na_11-na&utm_loc=9064696-&utm_mtd=-c&utm_kw=&utm_source=google&utm_medium=paid_search&utm_content=ps-dscia~emea-en~dsa~tofu~tutorial~python&gad_source=1&gad_campaignid=23340058065&gbraid=0AAAAADQ9WsFGfYzgNZDhmU7EAgv8Xha3w&gclid=CjwKCAjwuuPRBhAnEiwA2Ji8egcFveM7S4Ldz-cimIASKTCXJVBZVc2Tt-1xtVdb6fc4vQHO-NgxwxoCcAYQAvD_BwE)

### Import libraries

```python
# Load in relevant libraries, and alias where appropriate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# !pip install torchvision
import torchvision

import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# !pip install torchmetrics
import torchmetrics
```

### Define constants
```python
# Define relevant variables for the ML task
batch_size = 60
num_classes = 10
learning_rate = 0.001
num_epochs = 20

# Device will determine whether to run the training on GPU or CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### Load train and test datasets

PyTorch also comes with a rich ecosystem of tools and extensions, including `torchvision`, a module for computer vision. 

Torchvision includes **several image datasets that can be used for training and testing neural networks**.

```python
batch_size = 60

#Downloads and converts the MINST dataset to a Pythorch tensor (similar to a numpy array but with GPU acceleration capabilities)
train_dataset = datasets.MNIST(root="dataset/", download=True, train=True, transform=transforms.ToTensor())

# The DataLoader handles batching and shuffling, where we can also apply transformations too (normalizations, scaling etc..)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root="dataset/", download=True, train=False, transform=transforms.ToTensor())

test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
```

Every image is a number to be recognized

```python
def imshow(img):
   npimg = img.numpy()
   plt.imshow(np.transpose(npimg, (1, 2, 0)))
   plt.show()

# get some random training images (one batch of 60)
dataiter = iter(train_loader)
images, labels = next(dataiter)
labels
# show images
imshow(torchvision.utils.make_grid(images))
```

### Define the CNN architecture

We use the `torch.nn` (neural network) module

```python
#%%

class CNN(nn.Module):
   def __init__(self, in_channels, num_classes):

       """
       Building blocks of convolutional neural network for a
       CNN with two convolutional layers, followed by a fully connected layer.

       Parameters:
           * in_channels: Number of channels in the input image (for grayscale images, 1)
           * num_classes: Number of classes to predict. In our problem, 10 (i.e digits from  0 to 9).
       """
       super(CNN, self).__init__()

       # 1st convolutional layer
       self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, padding=1)
       # Max pooling layer
       self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
       # 2nd convolutional layer
       self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
       # Fully connected layer
       self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

   def forward(self, x):
       """
       Define the forward pass of the neural network.

       Parameters:
           x: Input tensor.

       Returns:
           torch.Tensor
               The output tensor after passing through the network.
       """
       x = F.relu(self.conv1(x))  # Apply first convolution and ReLU activation
       x = self.pool(x)           # Apply max pooling
       x = F.relu(self.conv2(x))  # Apply second convolution and ReLU activation
       x = self.pool(x)           # Apply max pooling
       x = x.reshape(x.shape[0], -1)  # Flatten the tensor
       x = self.fc1(x)            # Apply fully connected layer
       return x
       x = x.reshape(x.shape[0], -1)  # Flatten the tensor
       x = self.fc1(x)            # Apply fully connected layer
       return x
```

Create the model instance we will use

```python
device = "cuda" if torch.cuda.is_available() else "cpu"

model = CNN(in_channels=1, num_classes=num_classes).to(device)
print(model)
```

The `print(model)` will print all network's architecture

```text
CNN(
  (conv1): Conv2d(1, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (fc1): Linear(in_features=784, out_features=10, bias=True)
)
```

### Training phase

Using the cross-entropy loss function:  [Link to doc](../../004-linear_neural_networks_for_classification/softmax_regression/#the-cross-entropy)

available in PyTorch as `nn.CrossEntropyLoss`

We will also use Adam optimizer, one of the most popular optimization algorithms.

```python
# Define the loss function
cross_en_loss = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
```
The training loop (10 epochs)

```python
num_epochs=10

#Epoch loop: Each epoch represents a full pass through the entire training dataset
for epoch in range(num_epochs):
 # Iterate over training batches
   print(f"Epoch [{epoch + 1}/{num_epochs}]")
   #Batch Loop:  over batches of data from dataloader_train. 
   # tqdm is used to display a progress bar for the loop.
   for batch_index, (data, targets) in enumerate(tqdm(train_loader)):
       #move the data and targets to the specified device (e.g., CPU or GPU) 
       # for computation
       data = data.to(device)
       targets = targets.to(device)
       
       #Forward Pass: computes the model's predictions for the current batch.
       scores = model(data)
       
       #Loss Calculation: calculates the loss between the model's predictions and the true targets
       loss = cross_en_loss(scores, targets)
       
       #Backward pass
       optimizer.zero_grad() # clears old gradients from the last step. 
       loss.backward() #computes the gradient of the loss with respect to model parameters.
       optimizer.step() #updates the model parameters based on the computed gradients.
```

Output (at the end of the training):

```text
Epoch [1/10]
100%|██████████| 1000/1000 [00:05<00:00, 168.65it/s]
...
...
Epoch [10/10]
100%|██████████| 1000/1000 [00:05<00:00, 170.43it/s]
```

### Evaluating the model (inference mode)

Once the model is trained, we can evaluate its performance on the **test dataset**. 

We will use <span style="color:red">**accuracy**</span>: it measures the proportion of correctly classified cases from the total number of objects in the dataset.

Next, we use the `.eval` method of the model to put the model in evaluation mode, because some layers in PyTorch models behave differently at training versus testing stages. We also add a Python context with torch.no_grad, indicating we will not be performing gradient calculation.

Then, we iterate over test examples with no gradient calculation. For each test batch, we get model outputs, take the most likely class, and pass it to the accuracy function along with the labels. Finally, we compute the metrics and print the results. We got a 0.98 accuracy score, which means that our model correctly classified 98% of the digits. Not bad!

```python
from torchmetrics import Accuracy

#we set up the accuracy metric from `torchmetrics`. 
acc = Accuracy(task="multiclass", num_classes=10)

#puts the model in inference mode
#(network is actively analyzing new, unseen data to make predictions, rather than learning from data)
model.eval()
#no need to keep track of the gradient for the backward step
with torch.no_grad():
    for images, labels in test_loader:
        #forward pass, produces logits of shape [batch_size, 10]
        outputs = model(images)
        #argmax across the class dimension, giving the predicted class index per sample;
        # _ discards the actual max values
        _, preds = torch.max(outputs, 1)
        #each metric object accumulates the batch's true/false positive/negative counts internally,
        # no computation is anything yet
        acc(preds, labels)

test_accuracy  = acc.compute()


print(f"Test accuracy:  {test_accuracy:.4f}") 
```

The output is: 

```text
Test accuracy:  0.9862
```

This means ~%98.6 of correct classification... not bad!!!
