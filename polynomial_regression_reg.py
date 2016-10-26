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
x = a1.normalize_data(x)

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

## Function to calculate design matrix.
def calculateDesignMatrix(matrix,degree):
    n = np.shape(matrix)[0]
    baseMatrix = np.ones((n,1))
    for i in range(1,degree+1):
        baseMatrix = np.hstack((baseMatrix,np.power(matrix,i))) 
    return baseMatrix


pDegre = 2
lambdas =[0,0.01,0.1,1,10,100,1000,10000]

validationErrorSet = list()

for i in range(0,100,10):
	trainSet = np.vstack((x_train[0:i,:],x_train[(i+10):,:]))
	targetTrainSet = np.vstack((t_train[0:i],t_train[i+10:]))
	validationSet = x_train[i:(i+10),:]
	targetValidationSet = t_train[i:(i+10)]
	assert (validationSet not in trainSet), "Validation Set is part of Train Set"
	assert (targetValidationSet not in targetTrainSet), "Validation Target Set is part of Train Target Set"
	valError= []
	for lmbd in lambdas:
		trainMatrix = dm.calculateDesignMatrix(trainSet,pDegre,categ="Poly")
		validationMatrix = dm.calculateDesignMatrix(validationSet,pDegre,categ="Poly")
		weights = np.linalg.inv((np.eye(np.shape(trainMatrix)[1])*lmbd) + trainMatrix.T.dot(trainMatrix)).dot(trainMatrix.T)*targetTrainSet
		validationError = math.sqrt(np.sum(np.square(np.dot(validationMatrix,weights) - targetValidationSet))/np.shape(targetValidationSet)[0])
		valError.append(validationError)
	validationErrorSet.append(valError)

validationErrorSet =  np.array(validationErrorSet).T
validationError = []

for i in range(0,np.shape(validationErrorSet)[0]):
	validationError.append(np.mean(validationErrorSet[i,:]))

minVal = min(validationError)
val = lambdas[validationError.index(minVal)]

lmb = "Best Lambda: "+str(val)
error = "Error at Best Lambda: %.4f"%minVal

# Produce a plot of results.
plt.semilogx(lambdas, validationError,label='Validation error')
plt.semilogx(val,minVal,marker='o',color='r',label="Best Labmda")
plt.ylabel('RMS Error')
plt.text(5, 116, lmb, fontsize=15)
plt.text(5, 109, error, fontsize=15)
plt.legend()
plt.title('Ridge Regression')
plt.xlabel('Lambda')
plt.show()