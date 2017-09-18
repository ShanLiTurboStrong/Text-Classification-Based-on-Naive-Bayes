import math
import numpy as np

# The logProd function takes a vector of numbers in logspace 
# (i.e., x[i] = log p[i]) and returns the product of those numbers in logspace
# (i.e., logProd(x) = log(product_i p[i]))
def logProd(x):
	## Inputs ## 
	# x - 1D numpy ndarray
	## Outputs ##
	# log_product - float

	log_product = 0
	for each in x[:]:
		log_product=log_product+each
	return log_product

# The NB_XGivenY function takes a training set XTrain and yTrain and
# Beta parameters beta_0 and beta_1, then returns a matrix containing
# MAP estimates of theta_yw for all words w and class labels y
def NB_XGivenY(XTrain, yTrain, beta_0, beta_1):
	## Inputs ## 
	# XTrain - (n by V) numpy ndarray
	# yTrain - 1D numpy ndarray of length n
	# alpha - float
	# beta - float
	
	## Outputs ##
	# D - (2 by V) numpy ndarray, element in (i,j) means P(Xj=1|Y=i)
	D = np.zeros((2, XTrain.shape[1]))
	num_y0=0.0
	num_y1=0.0
	num_xy0=0.0
	num_xy1=0.0
	numerater0=beta_0-1.0
	numerater1=beta_1-1.0
	denominator=beta_1+beta_0-2.0;
	for i in range(yTrain.shape[0]):
		if yTrain[i]==0:
			num_y0+=1
		else:
			num_y1+=1
	for i in range(XTrain.shape[1]):
		for j in range(XTrain.shape[0]):
			if XTrain[j][i]==1:
				if yTrain[j]==0:	
					num_xy0+=1
				else:
					num_xy1+=1
		D[0][i]=(num_xy0+numerater0)/(num_y0+denominator)
		D[1][i]=(num_xy1+numerater1)/(num_y1+denominator)
		num_xy0=0.0
		num_xy1=0.0
	return D
	
# The NB_YPrior function takes a set of training labels yTrain and
# returns the prior probability for class label 0
def NB_YPrior(yTrain):
	## Inputs ## 
	# yTrain - 1D numpy ndarray of length n

	## Outputs ##
	# p - float
	p = 0.0
	count=0.0
	for i in range(yTrain.shape[0]):
		if yTrain[i]==0 :
			count+=1
	p=count/yTrain.shape[0]
	return p

# The NB_Classify function takes a matrix of MAP estimates for theta_yw,
# the prior probability for class 0, and uses these estimates to classify
# a test set.
def NB_Classify(D, p, XTest):
	## Inputs ## 
	# D - (2 by V) numpy ndarray
	# p - float
	# XTest - (m by V) numpy ndarray
	
	## Outputs ##
	# yHat - 1D numpy ndarray of length m


	yHat = np.ones(XTest.shape[0])
	predict_y0_matrix=np.array(XTest)
	predict_y1_matrix=np.array(XTest)
	for i in range(XTest.shape[0]):
		for j in range(XTest.shape[1]):
			if predict_y0_matrix[i][j]==1:
				predict_y0_matrix[i][j]=math.log(D[0][j])
				predict_y1_matrix[i][j]=math.log(D[1][j])
			else:
				predict_y0_matrix[i][j]=math.log(1-D[0][j])
				predict_y1_matrix[i][j]=math.log(1-D[1][j])
	predict_y0=np.zeros(predict_y0_matrix.shape[0])
	predict_y1=np.zeros(predict_y1_matrix.shape[0])
	for i in range(XTest.shape[0]):
		predict_y0[i]=logProd(predict_y0_matrix[i][:])+math.log(p)
		predict_y1[i]=logProd(predict_y1_matrix[i][:])+math.log(1-p)
	for i in range(yHat.shape[0]):
		if predict_y0[i]>=predict_y1[i]:
			yHat[i]=0
		else:
			yHat[i]=1
	return yHat

# The classificationError function takes two 1D arrays of class labels
# and returns the proportion of entries that disagree
def classificationError(yHat, yTruth):
	## Inputs ## 
	# yHat - 1D numpy ndarray of length m
	# yTruth - 1D numpy ndarray of length m
	
	## Outputs ##
	# error - float

	error = 0
	for i in range(yHat.shape[0]):
		if yHat[i]!=yTruth[i]:
			error+=1
	return error/(0.0+yHat.shape[0])
