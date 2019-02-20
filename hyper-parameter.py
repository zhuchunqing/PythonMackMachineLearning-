import KNNpackaging as knn 
from sklearn import datasets
import numpy as np
import Test as test
best_score=0.0
best_k=-1
best_method=""
digits=datasets.load_digits()
X=digits.data
Y=digits.target
X_train,Y_train,X_test,Y_test=test.train_test_split(X,Y,test_redio=0.2)
# for method in ["uniform","distance"]:
for a in range(1,11):
        knn_clf=knn.KNNClassifier(k=a)
        knn_clf.fit(X_train,Y_train)
        score=knn_clf.score(X_test,Y_test)
        if score>best_score:
            best_k=a
            best_score=score
print(best_k)
print(best_score)