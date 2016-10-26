import numpy as np
import math

def calculateSigmoid(matrix,mu,sd):
	return np.divide(1,np.add(1,np.power(math.exp(1),(np.divide((np.ones(np.shape(matrix))*mu - matrix),sd)))))

def calculateDesignMatrix(matrix,degree=0,mus=[0],sd=2000.0,categ="Poly"):
	baseMatrix = np.ones((np.shape(matrix)[0],1))
	if categ=="Poly":
		for i in range(1,degree+1):
			baseMatrix = np.hstack((baseMatrix,np.power(matrix,i)))
	elif categ=="Sigmoid":
		for mu in mus:
			baseMatrix = np.hstack((baseMatrix,calculateSigmoid(matrix,mu,sd)))
	else:
		pass
	return baseMatrix