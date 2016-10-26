#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt
import designmatrix as dm


(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,10:13]
#x = a1.normalize_data(x)

N_TRAIN = 100
pDegree = 3

plt.figure(figsize=(17, 8)) 

poly_label = []
x_label = ["GNI Per Capita (US$)","Life Expectancy at Birth","Literacy Rate"]
y_label = ["Child Mortality Rate Under 5","Child Mortality Rate Under 5","Child Mortality Rate Under 5"]
title = ['Regression Estimate for GNI Per Capita','Regression Estimate for Life Expectancy at Birth','Regression Estimate for Literacy']

for i in range(0,3):
	x_train = x[0:N_TRAIN,i]
	t_train = targets[0:N_TRAIN]
	x_test = x[N_TRAIN:,i]
	t_test = targets[N_TRAIN:]

	x_ev = np.linspace(np.asscalar(min(min(x_train),min(x_test))), np.asscalar(max(max(x_train),max(x_test))), num=500)

	trainMatrix = dm.calculateDesignMatrix(x_train,pDegree,categ="Poly")
	weights = np.linalg.pinv(trainMatrix)*t_train
	y_ev = dm.calculateDesignMatrix(np.reshape(x_ev,(500,1)),pDegree,categ="Poly")*weights

	if i==2:
		ax = plt.subplot2grid((2,2), (1,0),colspan=2)
	else:
		ax = plt.subplot2grid((2,2), (0, i))
	ax.plot(x_ev,y_ev,'g',label="Learned Polynomial")
	ax.plot(x_train,t_train,'bo',color='r',label="Train Data")
	ax.plot(x_test,t_test,'bo',color='b',marker="*",label="Test Data")
	ax.legend(loc="best")
	ax.set_xlabel(x_label[i])
	ax.set_ylabel(y_label[i])
	ax.set_title(title[i])

plt.show()