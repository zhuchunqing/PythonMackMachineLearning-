import numpy as np 
from math import sqrt
from collections import Counter
import metrics as metrics 
class KNNClassifier:
    def __init__(self,k):
        "初始化KNN分类器"
        assert k>=1,"k must be valid"
        self.k=k
        self._X_train=None
        self._Y_train=None
    def fit(self,X_train,Y_train):
        "根据训练数据集X_train和Y_train训练KNN分类器"
        assert X_train.shape[0]==Y_train.shape[0],"the size of X_train must be equal to the size of Y_train"
        assert self.k<=X_train.shape[0],"the size of x_train must be at least k."
        self._X_train=X_train
        self._Y_train=Y_train
        return self
    def predict(self,X_predict):
        "给定带预测数据集x_predict ，返回表示x_predic的结果向量"
        assert self._X_train is not None and self._Y_train is not None,\
        "must fit before predict"
        assert X_predict.shape[1]==self._X_train.shape[1],\
        "the feature number of x_predict must be equal to X_train"
        Y_predict=[self._predict(x) for x in X_predict]
        return np.array(Y_predict)
    def _predict(self,x):
        "给定单个待预测数据集x,返回x的预测结果值"
        assert x.shape[0]==self._X_train.shape[1],\
        "the feature number of x must be equal to X_train"
        distances=[]
        for x_train in self._X_train:
            d=sqrt(np.sum((x_train-x)**2))
            distances.append(d)
        nearest=np.argsort(distances)
        topK_y=[self._Y_train[i] for i in nearest[:self.k]]
        votes=Counter(topK_y)
        return votes.most_common(1)[0][0]
    def score(self,X_test,y_test):
        ttttt=self.predict(X_test)
        return metrics.accuracy_score(y_test,ttttt)
     
    def __repr__(self):
        return "KNN(k=%d)"%self.k





    
