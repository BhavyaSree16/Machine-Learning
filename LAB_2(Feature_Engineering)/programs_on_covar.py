"""Import Libraries"""
import numpy as np

def compute_mean_and_variance(vector):
    mean = np.mean(vector)
    variance = np.var(vector)
    return mean, variance

N = 10
vector = np.random.rand(N)

mean, variance = compute_mean_and_variance(vector)
print("Vector (1xN):", vector)
print("Mean of the vector:", mean)
print("Variance of the vector:", variance)

"""Create two vectors each of dimension 1XM each representing N-
dimensional feature vector of a sample. Write a program to

compute the Covariance between them.
"""

import numpy as np
def covarience(vector1,vector2):
    cov=np.cov(vector1,vector2)
    return cov
N = 5
vector1 = np.random.rand(N)
vector2 = np.random.rand(N)
cov=covarience(vector1,vector2)
print("Vector 1:", vector1)
print("Vector 2:", vector2)
print(cov)

"""Create two vectors each of dimension 1XN. Write a program to
compute the Correlation between them.
"""

import numpy as np
def correlation(vector1,vector2):
    corr=np.corrcoef(vector1,vector2)
    return corr
N = 5
vector1= np.random.rand(N)
vector2= np.random.rand(N)
corr=correlation(vector1,vector2)
print("Vector 1:", vector1)
print("Vector 2:", vector2)
print(corr)

"""Create a Matrix of MXN dimension representing the M-dimensional
feature vector for N number of samples i. e (i,j)th entry of the matrix
represents the ith feature of jth sample. Write a program to compute
the covariance matrix and correlation matrix. Comment on
takeaways from these matrixes.
"""

import numpy as np
M = 5
N = 3
matrix = np.random.rand(M,N)
cov_matrix = np.cov(matrix)
corr_matrix = np.corrcoef(matrix)
print("Matrix (MXN):", matrix)
print("Covariance Matrix:")
print(cov_matrix)
print("Correlation Matrix:")
print(corr_matrix)