# EE399-HW3

## Analyse MNIST Dataset With Singular Value Decomposition(SVD) And Supervised Learning Techniques: LDA, SVM and Decision Tree

### Author: Shiyu Chen

## Abstract
In this project, we perform analysis to MNIST data set. We start with doing SVD to the data set and figure out how many modes are necessary to reconstruct the good image by looking at its singular value spectrum. We obtain the U, Σ, and V^T by SVD analysis, and project the data set onto three selected modes. Then we project the dataset into PCA space and build classifiers to identify the data. We compare the performance of Linear classifier (LDA), Support Vector Machines (SVM), and decision tree classifiers.

## Introduction
The MNIST dataset consists of a collection of 70,000 handwritten digits (0-9) as training examples and an additional 10,000 images for testing. Each image is a grayscale 28x28 pixel image, making it a relatively small and manageable dataset. In this project, we use it for evaluating the performance of image classification algorithms and comparing the performance of different classifiers.

We first reshape each image to a column vector, each column of the resulting data matrix represents a different image. Then perform SVD on the data matrix to decompose it into three matrices: U, Σ, and V^T. We then plot the singular value spectrum to discover how many modes are required for good image reconstruction. This can be determined by observing the point where adding more modes does not significantly improve reconstruction. We also project the digit images onto three selected V-modes and create a 3D plot where each projected point is colored based on its digit label.

After completing SVD analysis, we project the data into the PCA space and use LDA to determine the most difficult and easiest pair of digits to separate. Do the same thing with SVM and decision tree classifier, and compare performances between these classifiers on the hardest and easiest pair of digits to separate. 

## Theoretical Background

Singular Value Decomposition (SVD) is a matrix factorization technique that decomposes a matrix into three separate matrices U, Σ, and V^T. Given a matrix A, the Singular Value Decomposition is expressed as:

A = UΣV^T

Where:

U is an orthogonal matrix, and its columns are the left singular vectors of A.
	
Σ is a diagonal matrix containing the singular values of A.

V^T is the transpose of an orthogonal matrix V, and its rows are the right singular vectors of A.

In the context of SVD, given a matrix A, the singular values are the square roots of the eigenvalues of the positive semi-definite matrix A^T A or A A^T . They are denoted by σ₁, σ₂, ..., σₖ, and arranged in descending order, meaning that σ₁ is the largest singular value, σ₂ is the second largest, and so on. The singular value spectrum is a plot that represents the magnitudes of the singular values, usually in decreasing order. By analyzing the singular value spectrum, we can see how much each mode contributes to reconstructing the image.

Principal Component Analysis is a dimensionality reduction technique that transforms high-dimension data into a lower-dimensional space while preserving the most important information or patterns presented in the data. In this project, we use PCA to reduce the dimension of data to 3 for better applying classifiers to the data.

Linear Discriminant Analysis(LDA) aims to find a linear combination of features that maximizes the separation between different classes while minimizing the variation within each class. The main goal of LDA is to project the original high-dimensional data onto a lower-dimensional space while maximizing class separability.

Support Vector Machines(SVM) helps us to find an optimal hyperplane in a N-dimensional space that distinctly classifies the data points.Support vectors are the data points that are closest to the hyperplane. Using these support vectors, we maximize the magrin of classifier. SVM is helpful in classifying data in high dimensional space.

Decision tree classifier is a flowchart-like structure that makes decisions based on feature values to ultimately reach a predicted outcome. It chooses the most relevant features to split the data based on their information gain, Gini impurity, or entropy. By deciding split point, data will be divided into subsets and creating child nodes for each subset. Recursive this process until reach met the stopping condition. Therefore, when classify new samples, traverse the decision tree from the root node to a leaf node based on the sample's feature values. The leaf node reached represents the predicted class label for the input sample.


## Algorithm Implementation and Development

```
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# Load the MNIST data
mnist = fetch_openml('mnist_784')


# Scale the data to [0, 1]
X = mnist.data / 255.0


# Convert feature values from string to integer
X = mnist.data.astype('int')
Y = mnist.target.astype('int')


# Reshape images into column vectors
X = X.T
```
The code started by importing libraries and import MNIST data set. Then we scale and reshape data into column vectors.

```
from scipy.linalg import svd
import matplotlib.pyplot as plt


# Perform SVD on the data
U, s, Vt = svd(X, full_matrices = False)
V = Vt.T


# Plot the singular value spectrum
plt.plot(s)
plt.xlabel('Mode')
plt.ylabel('Singular Value')
plt.title('MNIST Singular Value Spectrum')
plt.show()


# Determine the number of modes necessary for good image reconstruction
energy = np.cumsum(s ** 2) / np.sum(s ** 2)
r = np.argmax(energy > 0.95) + 1
print(f"Rank of the digit space: {r}")
```
We then perform SVD to the data and plot the singular value spectrum.

We use energy to calculate the energy, or contribution, of each mode, and compute the number of necessary modes for good image reconstruction.

```
# Create the 3D scatter plot
fig = plt.figure(figsize = (8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(V[:, 5], V[:, 6], V[:, 7], c=mnist.target.astype(int), s=20)
```
This code projects data onto mode 6, 7, and 8 and creates a 3D plot to display the result.

```
# Linear classifier for two digits
x_lc = mnist.data[(Y == 4) | (Y == 9)]
y_lc = Y[(Y == 4) | (Y == 9)]

```
To use LDA, we first need to pick out two digits from the dataset.

```
# PCA
pca = PCA(n_components=3)
x_lc_pca = pca.fit_transform(x_lc)
x_lc_pca = x_lc_pca.astype('float64')
y_lc = y_lc.astype('float64')

```
Then apply PCA to chosen data, which reduces the dimensionality to 3.

```
# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(x_lc_pca, y_lc, test_size=0.2)

# train LDA classifier
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
y_pred = lda.predict(X_test)
print(f'LDA accuracy for 4 vs 9: {accuracy_score(y_test, y_pred)}')

```
Split processed data into training and testing sets. Using training set to train the LDA model and using testing set to test the model. Finally calculate the accuracy score for LDA classifier.

```
# train SVM classifier
svc = SVC(kernel='linear')
svc.fit(X_train01, y_train01)
y_svc_pred01 = svc.predict(X_test01)
print(f'SVM accuracy for 0 vs 1: {accuracy_score(y_test01, y_svc_pred01)}')

# train decision tree classifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train01, y_train01)
y_dtc_pred01 = dtc.predict(X_test01)
print(f'Decision tree accuracy for 0 vs 1: {accuracy_score(y_test01, y_dtc_pred01)}')
```
Same procedure for SVM and decision tree classifiers. 


## Results

### Singular Value Spectrum

<img width="415" alt="image" src="https://github.com/13RENDA/EE399-HW3/assets/122130043/43db032d-09e3-4b88-832c-5634f415dc9d">

Rank of the digit space: 102

About 102 modes are necessary for good image reconstruction, which is consistent with the plot.

### Interpretation of SVD matrices

U matrix shape: (784, 784)

U represents the basis vectors or eigendigits that capture the primary patterns and features within the dataset.

Sigma matrix shape: (784,)

Σ contains the singular values that quantify the importance of each mode in the dataset.

V matrix shape: (70000, 784)

V^T matrix represents the coefficients or weights that describe the contribution of each mode to the images.

### 3D plot for projecting data onto mode 6, 7, and 8

<img width="371" alt="image" src="https://github.com/13RENDA/EE399-HW3/assets/122130043/6459e88d-440e-42f4-8e66-22ceeda0c0e7">

Following are the accuracy of LDA, SVM, and decision tree classifier in separating digital numbers in the MNIST dataset.  Need to notice that the result may vary every time. 

### Table 1

<img width="685" alt="image" src="https://github.com/13RENDA/EE399-HW3/assets/122130043/256eb3c5-d7c7-40ac-834b-74fa789cc60f">

According to table 1, Decision Tree has the highest accuracy in identifying the easiest separable set (0,1), and SVM performs best in separating 4 and 9.

### Table 2

<img width="683" alt="image" src="https://github.com/13RENDA/EE399-HW3/assets/122130043/f3b0ab7e-92c1-4b8b-8ec9-4e745911883c">

To rank the performance of the three models, SVM is the best, LDA comes next, and Decision Tree performs worst. SVM has the best performance in classifying digital data in MNIST data set, which has the highest average accuracy, and highest accuracy in identifying the easiest and hardest separable set. 

## Conclusion

In this project, we perform a comprehensive analysis of the MNIST data set, including SVD, PCA, and digit classification, and evaluate the performance of different classifiers. 

**SVD**

By computing U, Σ, and V^T from SVD analysis, we can build singular value spectrum and figure out that 102 modes are required for reconstructing good images. By interpreting matrices gained from SVD, we can project the data into PCA space and have a more comprehensive understanding of the underlying data structure, therefore making a more informed decision in choosing models to process data.In general,SVD and PCA offers a robust approach for analyzing data and reducing its dimensionality in various machine learning applications.

**Analysis performance of different classifiers**

Based our assessment of LDA, SVM, and decision tree classifiers on separating various digit pairs from MNIST dataset, we found that the SVM has superior performance compared to the other two. It is noteworthy that different classifiers have different results in determining the easiest and hardest separable digit sets, but SVM demonstrated exceptional performance on both pairs and impressive average accuracy of 0.984005362. In all, our result indicates that SVM is the most accurate classifier among the three tested models.


