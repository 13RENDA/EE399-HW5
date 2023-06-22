# EE399-HW2
## Author
Shiyu Chen
## Abstract
In this project, we work with the dataset that consists of 2414 greyscale faces, each image has been downsampled to 32*32 pixels and stored as colums within a 1024*2414 matrix. We extract the first 100 images from the matrix and compute the correlation matrix, eigen vectors, SVD and percentage of various to the matrix. By analyzing the correlation matrix, we can find the most and least correlated images in dataset. In the last task, we figure out the eigenfaces of this dataset. Overall, this project provides an introduction to the face recognition in machine learning.

## Introduction and Overview
This project is consist of seven steps and aiming to accomplish three tasks:
Use correlation metrix to find the most correlated and uncorrelated images in dataset. (Step (a) (b) (c))
Discover the relationship between eigenvectors and principle component directions of SVD. (Step (d) (e) (f))
Find the eigenfaces of the dataset. (Step (g))

For task 1, we extract the first 100 images from the raw dataset to matrix X, and apply dot product to X^T and X to get a 100*100 correlation matrix. By plotting this matrix, we can figure out the correlation between different images and therefore the most correlated and uncorrelated pairs. In step (c) we shrink the correlation matrix to 10*10 scale, which only compare 10 designated images, to see the correlation more clearly. 

In task 2, we create the matrix Y = XX^T and apply SVD to matrix X. By comparing the first six eigenvectors with largest eigenvalues in matrix Y and first six principle component directions from SVD, and compute the norm of difference of their absolute values, we figure out that these two set of values are essentially the same.

In task 3, we compute the percentage of variance captured by each of the first 6 SVD modes by ij100% 
And then plot these 6 modes we can get the eigenfaces of the dataset, which represent the most representative features of face images in datatset.

## Theoretical Background

Correlation matrix: 
Correlation matrix represents the correlation coefficients between a set of variables and it can be compute by dot product of two matrices. The correlation coefficient measures the strength and direction of the linear relationship between two variables. Larger the coefficient means higher correlation between two vectors, and 0 means no correlation. It is often visualized using a heat map, where the strength of the correlation is indicated by different colors or shades.

SVD:
Singular Value Decomposition (SVD) is a matrix factorization method in linear algebra that can decompose any rectangular matrix into three simpler matrices: a left singular matrix, a diagonal singular value matrix, and a right singular matrix. 
Given an m x n matrix A, its singular value decomposition can be represented as:

A = UΣV^T

Where U is an m x m orthogonal matrix whose columns are the left singular vectors of A, Σ is an m x n diagonal matrix whose non-zero entries are the singular values of A, and V is an n x n orthogonal matrix whose columns are the right singular vectors of A.

Singular values in Σ are sorted in descending order and can be used to determine the rank of the matrices. 

Eigenfaces: 
Eigenfaces refer to a set of principal components that are derived from a set of face images. These principal components capture the most important variations in facial features across the set of images and are used to represent each face as a linear combination of these components.

Percentage of variance: 
Percentage of variance refers to the amount of variability in a dataset that is explained by a set of variables or factors. It is often used in multivariate analysis, such as principal component analysis (PCA), to understand the relationship between variables and to identify the most important variables that contribute to the variability in the data.

## Algorithm Implementation and Development

```
# Load yalefaces file
results=loadmat('yalefaces.mat')
X=results['X']

# Take the first 100 images 
X_100j = X[:,:100]

# Compute correlation matrix of the first 100 images
C = np.dot(X_100j.T, X_100j)
```
We use `loadmat` method to load the data from `yalefaces.mat` to X. And then extrace the first 100 columns, which represents the first 100 faces, to matrix X_100j. Then we get the correlation matrix C by making dot product of X_100j^T and X_100j, andd use pcolor to plot the matrix. The result is shown in Figure1.
In step(c) we do the same thing but use different columns in matrix X.

```
#make a copy of the correlation matrix
C_copy = np.copy(C)

#eliminate the result of the image comparing with itself 
np.fill_diagonal(C, 0)
np.fill_diagonal(C_copy, 1000)

most_idx = np.where(C == np.max(C))
least_idx = np.where(C_copy == np.min(C_copy))
```
The most correlated and the most uncorrelated images can be get from the correlation matrix. Larger the value means higher correlation, and vice versa. The indices of columns and rows of correlation matrix is the index of the images, therefore the diagonal value are computed by the image with itself and will have the highest correlation. To eliminate this, we fill the diagonal value of C to 0 (means these values have no correlation), and when we use np.where to get the indices of the largest value in C. 
To get the most uncorrelated images, we compute the minimum value in the copy of correlation matrix because we changed diagonal values to 0 in C. By analyzing the correlation matrix, image 87 and 89 are the most similar faces, and image 55 and 65 are the most different faces. Pictures are shown in result section.

```
# Create matrix Y
Y = np.dot(X, X.T)

# Compute the eigenvectors and eigenvalues of Y
eigenvalues, eigenvectors = np.linalg.eig(Y)

# Find the indices that sort the eigenvalues in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]

# Get the first six eigenvectors with the largest magnitude eigenvalue
v = eigenvectors[:, sorted_indices[:6]]
```
We use `np.linalg.eig()` to get the eigenvectors and corresponding eigenvalues. Then sorted eigenvalues in descending order and save the indices, which are then used for getting the corresponding eigenvectors. The first six eigenvectors within the sorted_indices are the eigenvectors with the largest magnitude eigenvalue.

```
# Compute the SVD of X
U, S, Vt = np.linalg.svd(X)

# Get the first six principal component directions
u = U[:, :6]
```
We use np.linalg.svd() to compute SVD to the matrix X. The first six principle component directions are the first six vectors in U matrix.


## Computational Results

(a)

<img width="487" alt="image" src="https://user-images.githubusercontent.com/122130043/233779103-c72f9e7c-35b9-464e-8391-973181290df2.png">

Figure1. Correlation Matrix of the first 100 images

Red pixels represent the low correlation between images and blue pixels represent the high correlation. From this plot, the index of the most correlated images should be around 90.

(b)

<img width="318" alt="image" src="https://user-images.githubusercontent.com/122130043/233779129-2f6e42b4-e320-4138-a6fa-143320a4abfd.png">

The most correlated two faces are image 89 and image 87, which is consistent with the correlation matrix plot.

<img width="325" alt="image" src="https://user-images.githubusercontent.com/122130043/233779141-2c22ecf7-9787-414f-86a2-bb8818816aaa.png">

The most uncorrelated two faces are images 55 and 65.

(c)

<img width="355" alt="image" src="https://user-images.githubusercontent.com/122130043/233779150-6d0a6b0d-aec4-487c-a89f-9d69e08b873d.png">

From Figure 3, ignore the diagonal values, the most correlated images should be image 7 and image 9, but the most uncorrelated pictures are hard to recognize.

(d)

```
v: [[ 0.02384327  0.04535378  0.05653196  0.04441826 -0.03378603  0.02207542]
    [ 0.02576146  0.04567536  0.04709124  0.05057969 -0.01791442  0.03378819]
    [ 0.02728448  0.04474528  0.0362807   0.05522219 -0.00462854  0.04487476]
     ...
    [ 0.02082937 -0.03737158  0.06455006 -0.01006919  0.06172201  0.03025485]
    [ 0.0193902  -0.03557383  0.06196898 -0.00355905  0.05796353  0.02850199]
    [ 0.0166019  -0.02965746  0.05241684  0.00040934  0.05757412  0.00941028]]
```

(e)

```
u: [[-0.02384327 -0.04535378 -0.05653196  0.04441826 -0.03378603  0.02207542]
     [-0.02576146 -0.04567536 -0.04709124  0.05057969 -0.01791442  0.03378819]
     [-0.02728448 -0.04474528 -0.0362807   0.05522219 -0.00462854  0.04487476]
     ...
     [-0.02082937  0.03737158 -0.06455006 -0.01006919  0.06172201  0.03025485]
     [-0.0193902   0.03557383 -0.06196898 -0.00355905  0.05796353  0.02850199]
     [-0.0166019   0.02965746 -0.05241684  0.00040934  0.05757412  0.00941028]]


```
(f)
The normal difference is: 1.261524498888461e-15.

(g)

<img width="717" alt="image" src="https://user-images.githubusercontent.com/122130043/233779229-31217ca6-7eef-4a36-9846-7f171da48ab0.png">

The percentage of variance of the first 6 SVD modes are: 
[72.92756746909564, 15.281762655694362, 2.5667449429852702, 1.8775248514714737, 0.6393058444446512, 0.5924314415034914]


## Summary and Conclusions

In this project, the correlation matrix provides a method to find the correlation between vectors and therefore figure out the most correlated and uncorrelated vectors. The higher value in correlation matrix represents the higher correlation between data.
We also compute the first six largest eigenvectors by Y = XX^T and the first six principal component directions by SVD matrix X. These two results are basically the same, and the norm of difference between the absolute values of the largest magnitude eigenvector from (d) and the first principal component direction from (e) is close to zero, means these two matrices barely have differences and should capture the same underlying structure in the data.
We finally compute the percentage of variance captured by each of the first six SVD modes. These modes, also called eigenfaces, represent the first six most significant features of faces in the dataset. The percentage of the variance of these six modes in total reaches 93%, which suggests that these six modes capture considerable variance of the dataset.

