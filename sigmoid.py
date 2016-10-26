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
x = values[:,10:11]
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

mu=[100,10000]
sd=2000.0

trainMatrix = dm.calculateDesignMatrix(x_train,mus=mu,sd=sd,categ="Sigmoid")
testMatrix = dm.calculateDesignMatrix(x_test,mus=mu,sd=sd,categ="Sigmoid")
	
weights = np.linalg.pinv(trainMatrix)*t_train

trainError = (math.sqrt(np.sum(np.square(np.dot(trainMatrix,weights) - t_train))/np.shape(t_train)[0]))
testError = (math.sqrt(np.sum(np.square(np.dot(testMatrix,weights) - t_test))/np.shape(t_test)[0]))
trE = "Train Error: %.4f"%trainError
tstE = "Test Error: %.4f"%testError

x_ev = np.linspace(np.asscalar(min(min(x_train),min(x_test))), np.asscalar(max(max(x_train),max(x_test))), num=500)
y_ev = dm.calculateDesignMatrix(np.reshape(x_ev,(500,1)),mus=mu,sd=sd,categ="Sigmoid")*weights

plt.plot(x_train,t_train,'bo',color='r',label="Train Data")
plt.plot(x_test,t_test,'bo',color='b',marker="*",label="Test Data")
plt.plot(x_ev,y_ev,'g.-',label="Learned Polynomial")
plt.text(100000, 100, trE, fontsize=15)
plt.text(100000, 113, tstE, fontsize=15)
plt.legend()
plt.xlabel("GNI Per Capita (US$)")
plt.ylabel("Child Mortality Rate Under 5")
plt.title('Regression Estimate for GNI using Sigmoid Function')
plt.show()
 