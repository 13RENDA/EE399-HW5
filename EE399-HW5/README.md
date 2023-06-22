# EE399-HW5

### Author: Shiyu Chen

## Abstract

In this project, we will implement machine learning techniques on Lorenz equations. We will train neuro networks to advance the solution of Lorenz equations for rho = 10, 28, and 40, and then observe the performance of NN in predicting future states for rho = 17 and rho = 15. In the second part, we will compare the performances of LSTM, RNN, and Echo State Networks for forecasting the dynamics of Lorenz equations.  By working on this project, insights can be gained into the most effective NN architecture for predicting future states in the Lorenz system.

## Introduction

Lorenz equation represents atmospheric convention which is vital in weather forecasting and other related applications.  To study the influence of rho value on the dynamics of the Lorenz equations, we train a Neural Network (NN) to advance the solution from time t to t + ∆t for different values of ρ = 10, 28, and 40. This training process entails generating a dataset of input-output pairs and designing different NN architectures, including a feed-forward network, LSTM, RNN, or Echo State Network. Once trained, the NN can be used for future state prediction by changing ρ to new values 17 and 35. The trained NN takes the initial state as input and produces the predicted state at the future time step. To compare different architectures, feed-forward, LSTM, RNN, and ESN models can be implemented and evaluated based on their accuracy, computational efficiency, and overall performance in forecasting the dynamics of the Lorenz equations. By following this procedure, insights can be gained into the most effective NN architecture for predicting future states in the Lorenz system.


## Theoretical Background 

**Lorenz equation**

The Lorenz equations are a set of three nonlinear differential equations that describe a simplified model of atmospheric convection. The equations are given by:

dx/dt = σ * (y - x)
dy/dt = x * (ρ - z) - y
dz/dt = x * y - β * z

In these equations, x, y, and z represent the variables that describe the state of the system over time. The parameters σ, ρ, and β are constants that control the behavior of the system. The variable t represents time, and dx/dt, dy/dt, and dz/dt represent the rates of change of x, y, and z with respect to time.

The Lorenz equations are known for their chaotic behavior, where small changes in initial conditions can lead to drastically different outcomes. They have been extensively studied and have become a popular example in the field of nonlinear dynamics and chaos theory due to their rich dynamics and sensitivity to initial conditions.

**Feed Forward Neuro Network (FFNN)** is a type of artificial neural network in which information flows in one direction, from the input layer to the output layer. The network consists of multiple layers of interconnected nodes, or neurons. Each neuron receives input from the previous layer and applies an activation function to produce an output. The input layer receives the raw input data, and subsequent hidden layers process the data through a series of transformations. Finally, the output layer provides the network's final output or prediction.

**Recurrent Neural Networks (RNN)** is specifically suited for tasks where the current input depends not only on the current state but also on the previous inputs or states in the sequence. In an RNN, each neuron or processing unit maintains an internal state that is updated based on the current input and the previous state. This updated state is then used to generate the output and is also fed back into the network for the next time step. This feedback loop enables the RNN to remember information from previous steps and use it to influence future predictions.

**Long Short-Term Memory (LSTM)** is a type of RNN. LSTM addresses the vanishing gradient problem, which can occur in standard RNNs and hinder their ability to capture long-term dependencies. The LSTM cell consists of the input gate, forget gate, and output gate that controls the information flow as well as decide whether to retain or discard information. The LSTM cell can selectively remember or forget information over long periods, which helps in capturing important patterns and relationships in sequential data.

**Echo State Network(ESN)** is another type of RNN. The key characteristic of an ESN is its reservoir, which is a fixed random network of recurrently connected neurons. The reservoir is initialized randomly and remains unchanged during training. It produces complex temporal dynamics due to its random weights and nonlinear activation functions. During training, only the connections between the reservoir and the output layer are modified. The input data is fed into the reservoir, and the reservoir's internal states evolve over time. The final reservoir states are used as features for prediction or classification tasks. This characteristic of reservoir simplifies the training process and reduces computational complexity compared to traditional RNNs. ESN is suitable for time series prediction tasks because of its ability to capture temporal dependencies and process sequential data. However, one challenge with ESN is that it requires careful tuning of hyperparameters.


## Algorithm implementation

```
dt = 0.01
T = 8
t = np.arange(0,T+dt,dt)
beta = 8/3
sigma = 10
rho_train = [10, 28, 40]
rho_test = [17, 35]
nn_input_train = np.zeros((100*(len(t)-1)*len(rho_train), 3))
nn_output_train = np.zeros_like(nn_input_train)
nn_input_test = np.zeros((100*(len(t)-1)*len(rho_test), 3))
nn_output_test = np.zeros_like(nn_input_test)
```
The code starts with declaring the value of parameters and building data sets for training and testing.

```
def lorenz_deriv(x_y_z, t0, sigma=sigma, beta=beta, rho=rho_train[0]):
   x, y, z = x_y_z
   return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]
```
Then we define the Lorenz equation as a function for future use.

```
# Generate training data
np.random.seed(321)
x0_train = -15 + 30 * np.random.random((100, 3))

k=0
for rho in rho_train:
 x_t = np.asarray([integrate.odeint(lorenz_deriv, x0_j, t, args=(rho,))
                     for x0_j in x0_train])
                     
 for j in range(100):
     nn_input_train[k*(len(t)-1):(k+1)*(len(t)-1), :] = x_t[j, :-1, :]
     nn_output_train[k*(len(t)-1):(k+1)*(len(t)-1), :] = x_t[j, 1:, :]
     k += 1

```
Then randomly generates initial states (x0_train) for the Lorenz equations. There are 100 sets of initial states for each of the three variables (x, y, and z). The outer loop iterates over the training rho values and stores the resulting trajectories of the Lorenz equations in the x_t array. The inner loop creates input-output pairs for training the neuro network. We use the same proceeds testing data and make plots to the trajectories for each rho value.

**Part 2**

```
# Define the neural network architecture
class Net(nn.Module):
   def __init__(self):
       super(Net, self).__init__()
       self.fc1 = nn.Linear(3,10)
       self.fc2 = nn.Linear(10,10)
       self.fc3 = nn.Linear(10,3)


   def forward(self, x):
       x = torch.relu(self.fc1(x))
       x = torch.relu(self.fc2(x))
       x = self.fc3(x)
       return x


# Initialize the neural network
net = Net()
# Convert the data to PyTorch tensors
X_train = torch.tensor(nn_input_train, dtype=torch.float32).reshape(-1, 3)
Y_train = torch.tensor(nn_output_train, dtype=torch.float32).reshape(-1, 3)
X_test = torch.tensor(nn_input_test, dtype=torch.float32).reshape(-1, 3)
Y_test = torch.tensor(nn_output_test, dtype=torch.float32).reshape(-1, 3)


# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)


# Train the model
for epoch in range(30):
   optimizer.zero_grad()
   outputs = net(X_train)
   loss = criterion(outputs, Y_train)
   loss.backward()
   optimizer.step()
   print(f"Epoch {epoch+1}, loss={loss.item():.4f}")


# Test the model
net.eval()  # Set the model to evaluation mode
with torch.no_grad():
   test_outputs = net(X_test)
   test_loss = criterion(test_outputs, Y_test)
   print(f"Test loss: {test_loss.item():.4f}")


``` 
In the second part, we use the FFNN that was built in HW4, and follow the same steps to train and test the FFNN with the input-output sets we created above.

```
# Define the LSTM model
class LSTMModel(nn.Module):
   def __init__(self):
       super(LSTMModel, self).__init__()
       self.lstm = nn.LSTM(input_size=3, hidden_size=10, num_layers=1, batch_first=True)
       self.fc = nn.Linear(in_features=10, out_features=3)
      
   def forward(self, x):
       x, _ = self.lstm(x)
       x = self.fc(x[:, -1, :])
       return x


# Define the RNN model
class RNNModel(nn.Module):
   def __init__(self):
       super(RNNModel, self).__init__()
       self.rnn = nn.RNN(input_size=3, hidden_size=10, num_layers=1, batch_first=True)
       self.fc = nn.Linear(in_features=10, out_features=3)
      
   def forward(self, x):
       x, _ = self.rnn(x)
       x = self.fc(x[:, -1, :])
       return x


# Define the Echo State Network (ESN) model
class ESNModel(nn.Module):
   def __init__(self):
       super(ESNModel, self).__init__()
       self.esn = nn.RNN(input_size=3, hidden_size=10, num_layers=1, batch_first=True)
       self.fc = nn.Linear(in_features=10, out_features=3)
      
   def forward(self, x):
       x, _ = self.esn(x)
       x = self.fc(x[:, -1, :])
       return x
``` 
We then apply the same procedures to RNN, LSTM and ESN models.

```
# Define arrays to store the average testing losses for each run
fnn_avg_test_losses = []
lstm_avg_test_losses = []
rnn_avg_test_losses = []
esn_avg_test_losses = []
```
This code uses arrays to store the average testing losses for each model, which would be used for evaluating their performances.

## Result

**Loss Performance in each round**

![image](https://github.com/13RENDA/EE399-HW5/assets/122130043/cf332274-b654-405f-815f-cba46b6e3bc6)

![image](https://github.com/13RENDA/EE399-HW5/assets/122130043/b5e02d31-d548-4eb1-aceb-c36f05f34a01)

![image](https://github.com/13RENDA/EE399-HW5/assets/122130043/5bf8dc45-158d-414a-844f-827809eb2e08)

![image](https://github.com/13RENDA/EE399-HW5/assets/122130043/46e20984-3463-4894-94a9-54d6f4a1443c)

![image](https://github.com/13RENDA/EE399-HW5/assets/122130043/c47c2056-baed-4759-9603-d37fa295dce7)

![image](https://github.com/13RENDA/EE399-HW5/assets/122130043/8fa6052c-555a-4474-98c6-07e8e672fadf)

![image](https://github.com/13RENDA/EE399-HW5/assets/122130043/2b954e42-6f5d-4b2f-aec0-0c0271d071e1)

![image](https://github.com/13RENDA/EE399-HW5/assets/122130043/ade3dd88-eb01-446f-95f5-bca1fc4d81f5)

![image](https://github.com/13RENDA/EE399-HW5/assets/122130043/97a615a9-7347-4e4b-bd00-a37c0248b3cc)

![image](https://github.com/13RENDA/EE399-HW5/assets/122130043/dc2c1c7d-b05b-4373-88a0-d4752844b3a3)

**Average Losses of different models**

<img width="514" alt="image" src="https://github.com/13RENDA/EE399-HW5/assets/122130043/74f05214-8844-4f44-9c0c-c21820aaaf43">

The FFNN model has the lowest average training loss but the highest average testing loss, meanwhile LSTM has the highest average training loss but lowest average testing loss. RNN is the second best in training tasks but also the second worst in testing tasks. ESN behaves conversely to the RNN.
This observation drove the result that the model which is strong in training is usually weak in testing. 

The higher training loss and lower testing loss of LSTM models can be attributed to their ability to capture long-term dependencies but also to their potential to overfit the training data. Conversely, the FFNN model, lacking recurrent connections and memory mechanisms, may have a lower training loss due to its simplicity but may struggle to generalize effectively to new data, leading to a higher testing loss.



## Conclusion

Overall, the RNN and ESN models have comparatively balanced performances, and FFNN and LSTM have exempt but extreme performances. By discussing the possible reason behind the result, we conclude that each model has its own strengths and weaknesses in analyzing the dynamic of Lorenz equation. When we are choosing the model for processing data, we need to make a tradeoff between its strength and drawbacks.


