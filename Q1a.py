import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import math
import sys
import os

data_dir = sys.argv[1]
out_dir = sys.argv[2]
dataX = os.path.join(sys.argv[1], 'linearX.csv')
dataY = os.path.join(sys.argv[1], 'linearY.csv')
out = os.path.join(sys.argv[2], 'Q1.txt')
outfile = open(out, "w")

# 1. Linear Regression
print("#################### 1. Linear Regression ####################", file=outfile)

trainX = np.loadtxt(dataX)
trainY = np.loadtxt(dataY)

def normalize(X):
	mu = np.mean(X)
	sigma = np.std(X)
	return (X - mu)/sigma

X = normalize(trainX)
X = np.column_stack((np.ones(X.shape), X))
Y = trainY.reshape(-1,1)
m = len(Y)

# (a) Batch Gradienct Descent
def hw(theta, x):
	return np.dot(x, theta)

def Jw(y, h):
	return (0.5/m)*np.sum((y - h)**2)

def dJw(x, y, h):
	return (1.0/m)*np.dot(x.T, (y - h))

def batchGradientDescent(x, y, eta):
	print("\nLearning Rate = {}".format(eta), file=outfile)
	theta = np.zeros((x.shape[1], 1))
	h = hw(theta, x)
	prevCost = Jw(y, h)
	thetaL = [theta]
	costL = [prevCost]
	converged = False
	itr = 0

	while not converged:
		itr += 1
		theta = theta + eta*dJw(x, y, h)
		h = hw(theta, x)
		cost = Jw(y, h)
		thetaL.append(theta)
		costL.append(cost)
		error = abs(cost - prevCost)
		prevCost = cost
		if error < 1e-20 or itr > 20000:
			converged = True
		if itr % 10 == 0:
			print('iteration {}: error = {} cost = {}'.format(itr, error, cost), end = '', file=outfile)
			print(' theta = {0},{1}'.format(theta[0], theta[1]), file=outfile)

	print("Stopping Criteria: Error < 1e-20", file=outfile)
	print("Max Iterations = ", itr, file=outfile)
	print("Final Cost =", cost, file=outfile)
	print("Final Parameters = {0},{1}\n".format(theta[0], theta[1]), file=outfile)
	return thetaL, costL, itr

thetaL1, costL1, maxit1 = batchGradientDescent(X, Y, 0.09)