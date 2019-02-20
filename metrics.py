import numpy as np 
def accuracy_score(y_true,y_predic):
    "计算y_true和y_predic的准确度"
    assert y_true.shape[0]==y_predic.shape[0],\
    "the size of y_truemust be equal to the size of y_predict"
    return sum(y_true==y_predic)/len(y_true)