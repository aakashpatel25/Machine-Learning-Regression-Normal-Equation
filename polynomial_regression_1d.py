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
x = values[:,7:15]
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
pDegree = 3

featureDimension = np.shape(x)[1]

## To store Root-Mean-Square error, created vector of dimension M, one for each polynomial error.
trainError = np.zeros((featureDimension,1))
testError = np.zeros((featureDimension,1))

## Python range function doesn't include last specified number. Running loop from degree 1 to 6.
for i in range(0,featureDimension):
	trainMatrix = dm.calculateDesignMatrix(x_train[:,i:i+1],pDegree,categ="Poly")
	testMatrix = dm.calculateDesignMatrix(x_test[:,i:i+1],pDegree,categ="Poly")

	## weights = pinv(trainMatrix'*trainMatrix)*trainMatrix'*targetValue;
	weights = np.linalg.pinv(trainMatrix)*t_train

	trainError[i] = (math.sqrt(np.sum(np.square(np.dot(trainMatrix,weights) - t_train))/np.shape(t_train)[0]))
	testError[i] = (math.sqrt(np.sum(np.square(np.dot(testMatrix,weights) - t_test))/np.shape(t_test)[0]))

ind = np.arange(featureDimension)
width = 0.35

fig, ax = plt.subplots(figsize=(10,8.3))
rects1 = ax.bar(ind, trainError, width, color='r')
rects2 = ax.bar(ind + width, testError, width, color='b')
ax.set_ylabel('RMS Error')
ax.set_xlabel('Feature Number')
ax.set_title('RMS Error Plot Feature Wise')
ax.set_xticks(ind + width)
ax.set_xticklabels(('Total Population', 'Annual Births', 'Annual Deaths', 'GNI', 'Life Expectancy','Literacy Rate','School Enrolment','Low Birthweight'))
ax.legend((rects1[0], rects2[0]), ('Train Error', 'Test Error'))
plt.xticks(rotation=30)
plt.show()

import visualize_1d