#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import math
import matplotlib.pyplot as plt
import designmatrix as dm

(countries, features, values) = a1.load_unicef_data()

## Select first column as the values
targets = values[:,1]
## Select rest of the columns as the features
x = values[:,7:]
#x = a1.normalize_data(x)

## Cut off value for the train set
N_TRAIN = 100;

## Training Features (100x33 matrix)
## Testing Features (95x33 matrix)
x_train = x[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]

## Training values (Nx1 matrix) (100x1 matrix)
## Testing values (Nx1 matrix) (95x1 mtrix)
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]

## Polynomial Model Construction Limit
pDegree = 6

## To store Root-Mean-Square error, created vector of dimension M, one for each polynomial error.
trainError = {}
testError = {}

## Python range function doesn't include last specified number. Running loop from degree 1 to 6.
for i in range(1,pDegree+1):
	trainMatrix = dm.calculateDesignMatrix(x_train,i,categ="Poly")
	testMatrix = dm.calculateDesignMatrix(x_test,i,categ="Poly")

	## weights = pinv(trainMatrix'*trainMatrix)*trainMatrix'*targetValue;
	weights = np.linalg.pinv(trainMatrix)*t_train

	trainError[i] = math.sqrt(np.sum(np.square(np.dot(trainMatrix,weights) - t_train))/np.shape(t_train)[0])
	testError[i] = math.sqrt(np.sum(np.square(np.dot(testMatrix,weights) - t_test))/np.shape(t_test)[0])

# Produce a plot of results.
plt.plot(trainError.keys(), trainError.values())
plt.plot(testError.keys(), testError.values())
plt.ylabel('RMS Error')
plt.legend(['Train error','Test error'],loc="best")
plt.title('Fit with polynomials, no regularization')
plt.xlabel('Polynomial degree')
plt.show()

import normalized