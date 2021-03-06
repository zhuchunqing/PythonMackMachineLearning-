import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import datasets

iris=datasets.load_iris();
X=iris.data[:,2:]
Y=iris.target
plt.scatter(X[Y==0,0],X[Y==0,1])
plt.scatter(X[Y==1,0],X[Y==1,1])
plt.scatter(X[Y==2,0],X[Y==2,1])
plt.show()