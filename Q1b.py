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

# 1. Linear Regression
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
	return thetaL, costL, itr

thetaL1, costL1, maxit1 = batchGradientDescent(X, Y, 0.09)

# (b) Plotting Data & Hypothesis on 2-D plot
def plot2D(theta, eta):
	x = np.array(trainX)
	y = np.dot(X, theta)
	plt.plot(x, y, '-r')
	plt.plot(trainX, trainY, '+')
	plt.title('Regression Line - ' + r'$\eta = {0}$'.format(eta))
	plt.xlabel('acidity')
	plt.ylabel('density')
	regression_line = os.path.join(out_dir, 'Q1bRegressionLine.png') 
	plt.savefig(regression_line)
	plt.close()

plot2D(thetaL1[-1], 0.09)