import numpy as np
from test_linreg_univariate import plotData1D
from test_linreg_univariate import plotRegLine1D
from linreg import LinearRegression
filePath = "C:\\Users\\DELL\\AI\\hw1_skeleton\\data\\univariateData.dat"
file = open(filePath,'r')
allData = np.loadtxt(file,delimiter=',')
X = np.matrix(allData[:,:-1])
y = np.matrix((allData[:,-1])).T
n,d = X.shape
X = np.c_[np.ones((n,1)), X]
lr_model = LinearRegression(alpha = 0.01, n_iter = 1500)
lr_model.fit(X,y)
plotRegLine1D(lr_model,X,y)