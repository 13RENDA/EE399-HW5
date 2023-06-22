# EE399-HW4

### Author: Shiyu Chen

## Abstract

In this project, we will train a three-layer Feed Forward Neuro Network(FFNN) with two datasets, one is the data set from homework 1 and the other is MNIST dataset. We use different training data and testing data to fit the FFNN model and compare the results to discover the influence of training data on the performance of FFNN in data forecasting. We also will perform FFNN on classifying data in PCA-processed MNIST dataset and compare the performance of FFNN with the other three classifiers LSTM, SVM, and decision tree. This project will investigate the efficiency of FFNN in processing data forecasting and classification.

## Introduction

This project contains two tasks. In the first task, we will reconsider the data from homework 1, which contains 30 two-dimensional data points. We first fit those data to a three-layer FFNN. This involves defining the architecture, activation functions, and number of nodes in each layer, and then training the neural network using the data. Then we separate data into testing and training sets. The first scenario is using the first 20 data as training data and using the last 10 data as the testing set. The second scenario is combining the first 10 and last 10 data points as training set, and using the middle 10 data points as the testing data. We will compute the least square error for each model over the training points and also evaluate the model's performance by computing the least square error on the test data.

Moving to the second part, we will train the FFNN on The MNIST dataset, which consists of handwriting digit images. We first compute the first 20 PCA modes of digital images and build an FFNN to classify the digits. Then we will compare the result of the neural network against the other three classifiers, namely LSTM, SVM (support vector machines), and decision tree classifiers, and finally evaluate the performance of each classifier and compare their accuracy in classifying the digits.

By performing the analysis above, we can have a comprehensive understanding of the effectiveness of FFNN in data forecasting and classifying.

## Theoretical Background

Feed Forward Neuro Network (FFNN) is a type of artificial neural network in which information flows in one direction, from the input layer to the output layer. The network consists of multiple layers of interconnected nodes, or neurons. Each neuron receives input from the previous layer and applies an activation function to produce an output. The input layer receives the raw input data, and subsequent hidden layers process the data through a series of transformations. Finally, the output layer provides the network's final output or prediction.

Long Short-Term Memory is a type of recurrent neural network (RNN) architecture. Unlike feed-forward neural networks, which process input data in a single direction, LSTM networks have feedback connections that allow information to flow in loops. This looped structure enables LSTM networks to retain and utilize information from previous time steps, making them particularly effective for processing sequential or time-series data. The LSTM cell consists of the input gate, forget gate, and output gate that controls the information flow as well as decide whether to retain or discard information. The LSTM cell can selectively remember or forget information over long periods, which helps in capturing important patterns and relationships in sequential data.

## Algorithm Implementation

**Part I**

```
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
import torch.optim as optim

```
The code starts with importing necessary libraries.

```
# Define the neural network architecture
class Net(nn.Module):
   def __init__(self):
       super(Net, self).__init__()
       self.fc1 = nn.Linear(1,10)
       self.fc2 = nn.Linear(10,10)
       self.fc3 = nn.Linear(10,1)


   def forward(self, x):
       # x = x.view(-1, 1) # flatten the input image
       x = torch.relu(self.fc1(x))
       x = torch.relu(self.fc2(x))
       x = self.fc3(x)
       return x
```
We then build the neuro network architecture using the PyTorch library. The neural network defined in this code has three fully connected layers (also known as linear layers) and uses the ReLU activation function. 

In the constructor (__init__ method), the three fully connected layers are defined:

self.fc1 = nn.Linear(1, 10) creates the first layer with an input size of 1 and an output size of 10. This means it expects an input of size 1 and produces a tensor of size 10 as its output.

self.fc2 = nn.Linear(10, 10) creates the second layer with an input size of 10 and an output size of 10.

self.fc3 = nn.Linear(10, 1) creates the third layer with an input size of 10 and an output size of 1.

The forward method defines the forward pass of the neural network:

x = torch.relu(self.fc1(x)) performs the matrix multiplication of the input tensor x with the weights of the first layer (self.fc1) and applies the ReLU activation function to the result.

x = torch.relu(self.fc2(x)) performs the same operations for the second layer.

x = self.fc3(x) performs the matrix multiplication and linear transformation of the third layer, without applying any activation function.

```
# Split the data into training and test sets
X_train, X_test = X[:20], X[20:]
Y_train, Y_test = Y[:20], Y[20:]

# Convert the data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).reshape(-1, 1)
Y_train = torch.tensor(Y_train, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32).reshape(-1, 1)
Y_test = torch.tensor(Y_test, dtype=torch.float32).reshape(-1, 1)
# Initialize the neural network
net = Net()

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

num_epochs = 10
# Train the neural network
for epoch in range(num_epochs):
   for i in range(X_train.shape[0]):
       optimizer.zero_grad()
       outputs = net(X_train[i])
       loss = criterion(outputs, Y_train[i])
       loss.backward()
       optimizer.step()
      
       if (i+1) % 10 == 0:
           print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, X_train.shape[0], loss.item()))

# Compute the least square error for each data point in the test set
Y_test_pred = net(X_test).detach().numpy().flatten()
test_error = np.mean((Y_test_pred - Y_test.numpy())**2)
print('Test Error: {:.4f}'.format(test_error))
```
After splitting the data into training and testing sets, we transfer them into tensor for fitting FFNN. 
The neural network is trained in a nested loop. The outer loop iterates over the epochs, while the inner loop iterates over the training samples. Within each iteration, the optimizer is zeroed (optimizer.zero_grad()), the output of the neural network for the current input (outputs) is computed, and the loss between the predicted output and the actual output (Y_train[i]) is calculated using the defined loss function. The gradients are then calculated (loss.backward()) and the optimizer updates the model's parameters (optimizer.step()). After training, the least square error is computed for each data point in the test set.

**Part II**

```
# Define the neural network architecture
class Net(nn.Module):
   def __init__(self):
       super(Net, self).__init__()
       self.fc1 = nn.Linear(20, 128)
       self.fc2 = nn.Linear(128, 64)
       self.fc3 = nn.Linear(64, 10)

```
We change the number of nodes in each layer of our neuro network for performing on the MNIST dataset.

```
# Compute the first 20 PCA modes of the digit images
pca = PCA(n_components=20)
X_train_pca = pca.fit_transform(X_train)

```
We then compute the first 20 PCA modes and transform training data using the PCA modes. The following step of training the network is the same as above.

```
# Test the network
with torch.no_grad():
   correct = 0
   total = 0
   for images, labels in test_loader:
       outputs = net(images)
       _, predicted = torch.max(outputs.data, 1)
       total += labels.size(0)
       correct += (predicted == labels).sum().item()
```
This code evaluates the performance of a neural network on the test data. It iterates over the test loader and uses the neural network to make predictions on the test images. The predicted labels are obtained by taking the maximum values along the predicted dimension. The number of correctly predicted samples and the total number of samples are counted. This information will be used to calculate the accuracy of the network on the test data.

```
# Scaling the data from -1 to 1
scaling = MinMaxScaler(feature_range=(-1,1)).fit(x_train_pca)
x_train_pca = scaling.transform(x_train_pca)
x_test_pca = scaling.transform(x_test_pca)




# train SVM classifier
svc = SVC(kernel='linear')
svc.fit(x_train_pca, y_train)
y_svc_pred = svc.predict(x_test_pca)
print(f'SVM accuracy: {accuracy_score(y_test, y_svc_pred)}')


# train decision tree classifier
dtc = DecisionTreeClassifier()
dtc.fit(x_train_pca, y_train)
y_dtc_pred = dtc.predict(x_test_pca)
print(f'Decision tree accuracy: {accuracy_score(y_test, y_dtc_pred)}')


```
We then apply SVM and Decision Tree classifier on the data set and compute the accuracy of them.
```
# Define the LSTM model
class LSTMClassifier(nn.Module):
   def __init__(self, input_size, hidden_size, num_layers, num_classes):
       super(LSTMClassifier, self).__init__()
       self.hidden_size = hidden_size
       self.num_layers = num_layers
       self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
       self.fc = nn.Linear(hidden_size, num_classes)
      
   def forward(self, x):
       h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
       c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
       out, _ = self.lstm(x, (h0, c0))
       out = self.fc(out[:, -1, :])
       return out
```
To perform LSTM, we first define the architecture of LSTM model in PyTorch. The forward method defines the forward pass of the model. Inside the forward method, initial hidden and cell states (h0 and c0) are created as tensors of zeros with the appropriate dimensions. The last time step of the output sequence is extracted using out[:, -1, :]. This selects the last element along the time dimension for each sample in the batch. The selected time step output is passed through the fully connected layer self.fc, and the result is returned as the output of the model.

## Result

### Part I

**The least-square error of different models on training data**

<img width="682" alt="image" src="https://github.com/13RENDA/EE399-HW4/assets/122130043/bafcfe28-9614-47dd-8857-76ae78571703">

Scenario I: First 20 data points are used for training and the last 10 data points are used for testing

Scenario II: First 10 and last 10 data points are used for training and the middle 10 data points are used for testing

According to the result, linear regression model performs best among all models in forecasting this data set. FFNN has large error in both scenario and the reason might be overfitting. The dataset is simple and straightforward, so performing FFNN on the dataset may cause overfitting as performing 19th degree polynomial regression. 

**Accuracy of different models in classifying MNIST data**

<img width="684" alt="image" src="https://github.com/13RENDA/EE399-HW4/assets/122130043/7c37bb4c-c5d7-41ab-8afb-4458e77fbf8f">

Based on the provided accuracy values, the FFNN and LSTM models performed exceptionally well, with the LSTM model showing the highest accuracy among all the models. The SVM model also achieved a respectable accuracy, while the Decision Tree model had the lowest accuracy. This result shows the exceptional performance of neural network in dealing with large-scale dataset.

## Conclusion

In this project, we have explored the application of three_layered FFNN in data forecasting and classification tasks. By training the model with different data structures, FFNN exhibits different performances in data forecasting, though the accuracy is not satisfying. However, neuro network has impressive performance in classifying large scale datasets like MNIST dataset. The reason might be the neuro network will overfit the small dataset and therefore cause large errors in forecasting.

Overall, this project provides valuable insights into the effectiveness of FFNN in data forecasting and classification, highlighting its potential as a reliable and versatile tool in various machine learning applications. 










