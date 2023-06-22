# EE399-HW1

Shiyu Chen

### Abstract
This project is using python optimized function to fit the model f(x) = Acos(Bx)+Cx+D to the data with least-square error approach, then play with the result parameters to generate loss landscapes, and finally practice data training.

### Introduction
This project is to fit a mathematical model to data and estimate the optimal values using the least-squares error <img width="211" alt="image" src="https://user-images.githubusercontent.com/122130043/233039863-28581b8b-37f9-4145-84f6-3eddaa0d49cf.png">
 approach. The model is, f(x) = Acos(Bx)+Cx+D, where A, B, C, and D are parameters.
This project is to fit a mathematical model to data and estimate the optimal values using the least-squares error  approach. The model is, f(x) = Acos(Bx)+Cx+D, where A, B, C, and D are parameters.

Task (i) aims to determine the optimized value of parameters A, B, C, and D, and find the minimum error between the model predictions and actual data points by least-square error function. 

Task (ii) uses the optimized parameters from task(i), two of the parameters are fixed and the others are swept to generate 2D loss landscapes. Explore all combinations of parameters and visualize with a gird plot with .pcolor. 

In task (iii), the first 20 data work as training data to fit three models: a line, parabola, and 19th-degree polynomial. Compute the least squares error for each model over the training points. The remaining data work as testing data. The trained models are evaluated on the testing data to calculate the least-squares error for comparison. 

Task (iv) repeats task (iii) but use the first 10 and the last 10 as the training data and the remaining data as testing data.


### Theoretical Background

Model fitting is a key step in machine learning, where a mathematical model is adjusted to approximate observed data. In this project, we approach for model fitting by minimizing least-squares error. The function of the least-squares error is <img width="211" alt="image" src="https://user-images.githubusercontent.com/122130043/233040163-a55ee545-7c97-4cb6-855b-0fb14ccd620a.png">, where f(x) is the function of fitted model, y is the true data, and $\Sigma$ denotes the sum over all data points. The result indicates the discrepancy between the predicted values and true values from the model and the true data, in other words, the smaller error, the more accurate model.

Data training is another crucial step in developing machine learning model. It exposing the model to a data set to learn patterns, relationships, and representations form the data. In this project, we mainly explored supervised learning and overfitting.

In Supervised learning, the model is trained on labeled data, which means the outputs of the input data are known, and the model learns to predict new output values based on the input features and their associated labels during training.

Overfitting means the model learns to perform well on training data but fails to generalize to new unseen data.


### Algorithm Implementation and Development
```
#define the objective function
def velfit_l(c, x, y):    
    return (np.sqrt(np.sum(((c[0] * x) + c[1] - y)**2)/ 10))

# optimize fitting    
res= opt.minimize(velfit_l, v0, args=(x_20, y_20), method='Nelder-Mead')

```


I first defined the least-squares error as a function and return the result to be used in the optimization method provided by Python scipy package. 

Optimization Algorithms are used to update model’s parameters during training in order to minimized loss function. In this project we apply scipy.optimize package and ‘Nelder-Mead’ method to approach the result. 

The optimization process includes curve fitting, which requires hyperparameters for the function. Hyperparamters are parameters set by users prior to training. These parameters influence the training process and tuning them appropriately is significant of training machine learning models.

This optimization algorithm will calculate the least-squares error of each data point to the predicted curve, and adjust the fitted curve to get the smallest total least-squares error among all the data points.



### Computational Results

**question(i):**

The optimized coefficients are: 
A =2.1716818723637914, 
B =0.9093249029166655, 
C =0.7324784894461773, 
D =31.45291849616531
The minimum error is: 0.04521457161268927

**question(ii):**

In question(ii), I keep two parameters and swept the other two, and tried all combinations. The corresponding loss landscapes and number of minima are shown below

f1(x) = Acos(Bx)+Cx+D

1 minima

<img width="353" alt="image" src="https://user-images.githubusercontent.com/122130043/233041926-551bd9e7-9c0a-4b54-a9a6-8c12b97a9d2b.png">

f2(x) = Acos(Bx)+Dx+C

1 minima

<img width="363" alt="image" src="https://user-images.githubusercontent.com/122130043/233042006-22d68784-5ad5-43e8-b96c-a0c41e7b2302.png">

f3(x) = Acos(Cx)+Bx+D

1 minima

<img width="362" alt="image" src="https://user-images.githubusercontent.com/122130043/233042090-b8e783ab-01fa-474c-bf91-c43b651e7c4a.png">

f4(x) = Ccos(Bx)+Ax+D

1 minima

<img width="374" alt="image" src="https://user-images.githubusercontent.com/122130043/233042171-f80121f7-fa27-4e8b-9926-c194d61596b6.png">

f5(x) = Dcos(Bx)+Cx+A

2 minimas

<img width="367" alt="image" src="https://user-images.githubusercontent.com/122130043/233042205-c63fe5f5-eb7e-41d7-9fe6-7fa9c3a3bd3f.png">


**question(iii):**

In this question, we use first 20 raw data to train the model (linear, parabola, 19th-degree polynomial) and use the last 10 raw data to test the model. The results are shown below.

Linear

<img width="400" alt="image" src="https://user-images.githubusercontent.com/122130043/233042272-23243f40-de11-452f-9ac7-4c8762304c80.png">

The least square error for linear model is 3.527853270811783

Parabola:

<img width="388" alt="image" src="https://user-images.githubusercontent.com/122130043/233042363-45765da1-07bd-4aab-ac74-99feb5783862.png">

The least square error for parabola model is 8.713552037984044

19th-degree polynomial:

<img width="412" alt="image" src="https://user-images.githubusercontent.com/122130043/233042423-eafd336f-0d80-4ec7-b620-df7385c76d49.png">

The least square error for 19th-degree polynomial model is: 28617752784.428474

**question(iv):**

In this question we use the first and last 10 raw data to train the model (linear, parabola, 19th-degree polynomial) and the rest of the data to test the model. The results are shown below:

Linear:

<img width="395" alt="image" src="https://user-images.githubusercontent.com/122130043/233042489-a75f3667-d71a-4bda-aa01-d5f6e6c979e4.png">

The least square error for new linear model is: 2.948750927000568, which is 0.5791023438112153 smaller than the result from question (iii)

Parabola:

<img width="395" alt="image" src="https://user-images.githubusercontent.com/122130043/233042556-2d9bd53b-83ee-44c4-94d2-ebcd819c215a.png">

The least square error for new porabola model is: 2.935308118157717, which is 5.778243919826327 smaller than the result from question (iii)

19th-degree polynomial:

<img width="419" alt="image" src="https://user-images.githubusercontent.com/122130043/233042594-d546516b-9a84-472e-b0bd-bdf209f278eb.png">

The least square error for new polynomial model is: 81.93347036362427, which is 28617752702.495003 smaller than the result from question (iii)


### Summary and conclusion

By the loss landscapes from question (ii), we can see that switching the parameters will cause the fitted curve changes and generate different amount of minima.

According to the curve fitting plot and the least-squares errors in question(iii), the linear model predicts data best, then the parabola, and 19th-degree polynomial obviously overfitting.

However, in question (iv), the parabola fits best and then the linear model (though they looks similar) and the polynomial is still overfitted. Meanwhile, all the model in question (iv) has smaller least-square error than models in question(iii), especially in the overfitting case (polynomial), which means that training model with the head and tail data performs better than training model with head to middle data.


In summary, this project involves fitting a mathematical model to data and estimating the optimal values using the least-squares error approach. The optimized parameters are then used to generate loss landscapes, and data training is practiced to evaluate the accuracy of the model.

The project includes four tasks, which involve optimization, hyperparameter tuning, data training, and overfitting. The optimization process is carried out using the Scipy package's 'Nelder-Mead' method, which calculates the least-squares error of each data point to the predicted curve and adjusts the fitted curve to get the smallest total least-squares error among all the data points.

The results of the project include the optimized coefficients, minimum error, loss landscapes, and least-squares errors for linear, parabola, and 19th-degree polynomial models. The trained models are evaluated on the testing data to calculate the least-squares error for comparison. The report provides a detailed explanation of the theoretical background, algorithm implementation, and computational results.

In conclusion, this project provides a comprehensive demonstration of fitting a mathematical model to data using the least-squares error approach, hyperparameter tuning, and data training. The results obtained show the importance of these processes in developing machine learning models and their impact on model accuracy.

