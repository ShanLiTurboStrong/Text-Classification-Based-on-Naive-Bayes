import os
import csv
import numpy as np
import NB
import math

# Point to data directory here
# By default, we are pointing to '../data/'
data_dir = os.path.join('..','data')

# Read vocabulary into a list
# You will not need the vocabulary for any of the homework questions.
# It is provided for your reference.
with open(os.path.join(data_dir, 'vocabulary.csv'), 'rb') as f:
    reader = csv.reader(f)
    vocabulary = list(x[0] for x in reader)

# Load numeric data files into numpy arrays
XTrain = np.genfromtxt(os.path.join(data_dir, 'XTrain.csv'), delimiter=',')
yTrain = np.genfromtxt(os.path.join(data_dir, 'yTrain.csv'), delimiter=',')
XTrainSmall = np.genfromtxt(os.path.join(data_dir, 'XTrainSmall.csv'), delimiter=',')
yTrainSmall = np.genfromtxt(os.path.join(data_dir, 'yTrainSmall.csv'), delimiter=',')
XTest = np.genfromtxt(os.path.join(data_dir, 'XTest.csv'), delimiter=',')
yTest = np.genfromtxt(os.path.join(data_dir, 'yTest.csv'), delimiter=',')

# TODO: Test logProd function, defined in NB.py
# x=[math.log(1) ,math.log(2),math.log(3),math.log(4),math.log(5),math.log(6),math.log(7)]
# print(NB.logProd(x))
# TODO: Test NB_XGivenY function, defined in NB.py
xT=XTrain
yT=yTrain
matrix_XGivenY=NB.NB_XGivenY(xT, yT, 5, 7)
# count=0
# for x in range(matrix_XGivenY.shape[1]):
# 	tmp=matrix_XGivenY[0][x]
# 	if tmp==0:
# 		count+=1
# 	tmp=matrix_XGivenY[1][x]
# 	if tmp==0:
# 		count+=1
# print(count)
# TODO: Test NB_YPrior function, defined in NB.py
y_prior=NB.NB_YPrior(yT)
# TODO: Test NB_Classify function, defined in NB.py
result=NB.NB_Classify(matrix_XGivenY, y_prior, XTest)
# TODO: Test classificationError function, defined in NB.py
error=NB.classificationError(result, yTest)
print(error)
# TODO: Run experiments outlined in HW2 PDF