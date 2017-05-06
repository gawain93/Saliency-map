# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 21:49:58 2017

@author: ga63koh
"""

import os
import h5py
import numpy as np
from keras.models import load_model
from keras.utils import np_utils
from supersaliency import evaluation
import matplotlib.pyplot as plt


class ROC(object):
    
    def __init__(self,test_path,model_path,tau,rows,cols,test_num):
        self.test_path=test_path
        self.model=load_model(model_path)
        self.tau=tau
        self.rows=rows
        self.cols=cols
        self.test_num=test_num
     
    @staticmethod    
    def load_test_data_default(data_path):
        read_file=h5py.File(data_path,'r')
        cubes=read_file['cubes'][:]
        label=read_file['labels'][:]
        test_data=[None]*9
        
        for j in range(9):
            test_data[j]=cubes[:,j,...]
            test_data[j]=test_data[j].reshape((test_data[j].shape[0],14,60,40,1))
            
        test_labels=np_utils.to_categorical(label,2)
        
        return test_data,test_labels

          
    def test_model_default(self):
        test_list=sorted(os.listdir(self.test_path))
        predicted=[None]
        actual=[None]
        print predicted,actual 
        
        for batch in test_list:
            print str(batch)
            test_data,test_labels=self.load_test_data(self.test_path+'/'+batch)
            test_labels=test_labels[:,1]
            test_labels=test_labels.tolist()
            predict_on_batch=self.model.predict(test_data)[:,1]            # extract the positive possibilities
            predict_on_batch=predict_on_batch.tolist()
            
            predicted=np.concatenate((predicted, predict_on_batch), axis=0)
            actual=np.concatenate((actual, test_labels), axis=0)
        
        predicted=predicted[1:]
        actual=actual[1:]
        return predicted,actual
        

def test():
    test_path='/mnt/scratch/mikhail/salieny_cnn_data/__3DCNN_preprocessed__/Gazecom_batches/test'
    model_path='/home/ga63koh/self study/keras/FP Project/my_model110.h5'
    roc=ROC(test_path,model_path)
    predicted,actual=roc.test_model()
    auc=evaluation.calculate_area_under_curve(predicted, actual, cls=1, random_state=42)
    tp, fp=evaluation.get_roc_curve(predicted, actual, cls=1)
    plt.plot(fp, tp)
    plt.xlabel('False positive rate')
    plt.ylabel('Ture positive rate')
    plt.title('Roc Curve')
    
    return auc,predicted,actual
        

if __name__ == "__main__":
     auc,predicted,actual=test()  































def get_roc_curve(predicted, actual, cls=1):
    sorted_by_prob = np.argsort(-predicted)
    tp = np.cumsum(np.single(actual[sorted_by_prob] == cls))
    fp = np.cumsum(np.single(actual[sorted_by_prob] != cls))
    tp /= max(np.sum(actual == cls), 1.0)
    fp /= max(np.sum(actual != cls), 1.0)
    tp = np.hstack((0.0, tp, 1.0))
    fp = np.hstack((0.0, fp, 1.0))
    return tp, fp

def calculate_area_under_curve(predicted, actual, cls, random_state=42):
    # assure that @predicted and @actual are rows, not columns
    if predicted.shape[0] > 1:
        predicted = predicted.reshape((-1, ))
    if actual.shape[0] > 1:
        actual = actual.reshape((-1, ))

    # create a pseudorandom number generator (PRNG) separate from that of the external code (not to interfere)
    prng = np.random.RandomState(random_state)
    # shuffle the points to get a uniform shuffling (to get ~50% AUCs for chance-models)
    permutation_indices = prng.permutation(len(predicted))
    predicted = predicted[permutation_indices]
    actual = actual[permutation_indices]

    tp, fp = get_roc_curve(predicted, actual, cls)
    h = np.diff(fp)  # steps on X axis
    auc = np.sum(h * (tp[1:] + tp[:-1])) / 2  # trapezoid rule
    return auc