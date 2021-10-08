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

# (d) Contour to show error function
def computeCost(theta0, theta1):
	theta = np.empty((0,1), float)
	theta = np.append(theta, np.array([[theta0]]), axis=0)
	theta = np.append(theta, np.array([[theta1]]), axis=0)
	h = hw(theta, X)
	cost = Jw(Y, h)
	return cost

def plotContour(thetaL, costL, eta):
	fig = plt.figure(figsize=(7,7))
	xv = np.array([a[0] for a in thetaL])
	yv = np.array([a[1] for a in thetaL])
	xs = np.linspace(min(xv.min(), 0), max(xv.max(), 2), 20)
	ys = np.linspace(min(yv.min(), -1), max(yv.max(), 1), 20)
	x, y = np.meshgrid(xs, ys)
	zs = np.array([computeCost(theta0, theta1) for theta0, theta1 in zip(np.ravel(x), np.ravel(y))])
	z = zs.reshape(x.shape)

	plt.contour(x,y,z,cmap = cm.viridis)
	
	plt.title("Gradient Descent(Contours) - " + r'$\eta = {0}$'.format(eta))
	plt.xlabel(r'$\theta_0$')
	plt.ylabel(r'$\theta_1$')

	for itr in range(len(thetaL)):
		plt.scatter(thetaL[itr][0], thetaL[itr][1], color='orange', marker='.')
		#plt.pause(0.001)

	contour = os.path.join(out_dir, "Q1dContour.png")
	plt.savefig(contour)
	plt.close()

plotContour(thetaL1, costL1, 0.09)