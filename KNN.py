import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from collections import Counter
import KNNpackaging as knn 
import Test as test
from sklearn import datasets
iriss=datasets.load_iris()
X = iriss.data
Y = iriss.target
X_train,Y_train,X_test,Y_test=test.train_test_split(X,Y,0.2,None)

# from sklearn.neighbors import KNeighborsClassifier
# KNN_classfier=KNeighborsClassifier(n_neighbors=3)

# raw_data_X=[[3.3,2.3],[3.1,1.78],[1.34,3.36],[3.58,4.678],[2.28,2.866],
#             [7.42,4.69],[5.74,3.53],[7.17,2.51],[7.79,3.42],[7.94,0.79]]
# raw_data_Y=[0,0,0,0,0,1,1,1,1,1]
# X_train=np.array(raw_data_X)
# Y_train=np.array(raw_data_Y)
knnclf=knn.KNNClassifier(k=3)
knnclf.fit(X_train,Y_train)
# X=np.array([8.09,3.37])
# knnclf=knn.KNNClassifier(k=6)
# knnclf.fit(X_train,Y_train)
# X_predict=X.reshape(1,-1)
y_predict=knnclf.predict(X_test)
sum(y_predict==Y_test)
print(sum(y_predict==Y_test)/len(Y_test))
# X_predict=X.reshape(1,-1)
# print(KNN_classfier.predict(X_predict))
# # print(X_train[Y_train==0,0])
# # print(X_train[Y_train==0,1])
# plt.scatter(X_train[Y_train==0,0],X_train[Y_train==0,1],color='g')
# plt.scatter(X_train[Y_train==1,0],X_train[Y_train==1,1],color='r')
# plt.scatter(X[0],X[1],color='b')
# # plt.show()
# distances=[]
# for x_train in X_train:
#     d=sqrt(np.sum((x_train-X)**2))
#     distances.append(d)
# # print(np.argsort(distances))
# nearest=np.argsort(distances)
# # print(nearest)
# K=6
# # 最近六个点的y坐标
# topK_y=[Y_train[i] for i in nearest[:K]]
# topK_x=[X_train[i] for i in nearest[:K]]
# # print(topK_x)
# # Counter({1: 5, 0: 1}) 健值是1有5个元素 健值是0有1一个元素
# votes=Counter(topK_y)
# # votesx=Counter(topK_x)
# predict_y=votes.most_common(1)[0][0]
# # predict_x=votesx.most_common(1)[0][0]
# # print(Counter(topK_y))
# print(predict_y)
# # Counter({0:1,1:5})
