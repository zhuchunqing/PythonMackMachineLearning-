import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from collections import Counter
from sklearn import datasets
import Test as test
import KNNpackaging as knn
import metrics as metrics
digits=datasets.load_digits()
# print(digits.keys())
# print(digits.DESCR)
X=digits.data
Y=digits.target
# print(X.shape)
# print(Y.shape)
# print(Y)
# print(X[:10])
some_digit=X[666]
some_digit_image=some_digit.reshape(8,8)
plt.imshow(some_digit_image,cmap=plt.cm.binary)
X_train,Y_train,X_test,Y_test=test.train_test_split(X,Y,test_redio=0.2)
my_knn_clf=knn.KNNClassifier(k=3)
my_knn_clf.fit(X_train,Y_train)
y_predic=my_knn_clf.predict(X_test)
prdictaccuracy=metrics.accuracy_score(Y_test,y_predic)
print(prdictaccuracy)

