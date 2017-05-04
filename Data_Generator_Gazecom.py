# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 12:40:13 2017

@author: ga63koh
"""

import numpy as np
from random import randrange
import os
import h5py
from keras.utils import np_utils

class Data_Generator_Gazecom(object):
        
    def __init__(self,data_path,batch_size):
        self.data_path=data_path
        self.batch_size=batch_size
        
        
    def append_cubes(self,cube,label):
        while cube.shape[0]<self.batch_size/2:
            cube=np.append(cube,cube,axis=0)
            label=np.concatenate([label,label])
            
        cube=np.append(cube,cube[:self.batch_size-cube.shape[0],...],axis=0)
        label=np.concatenate([label,label[:self.batch_size-len(label)]])
        
        return cube,label
        

    def generate_vali_data(self):              # all the validationdata are generated together, for every batch, the validation data are the same
        listing_vali=os.listdir(self.data_path+'/'+'validation')
        os.chdir(self.data_path+'/'+'validation')
        valid_data=[None]*9
        while 1:
            i=0
            for file in listing_vali:
                read_file=h5py.File(file,'r')
                cubes=read_file['cubes'][:]
                label=read_file['labels'][:]
                if i==0:
                    for j in range(9):
                        valid_data[j]=cubes[:,j,...]
                        valid_data[j]=valid_data[j].reshape((cubes.shape[0],cubes.shape[2],cubes.shape[3],cubes.shape[4],1))
                    valid_labels=np_utils.to_categorical(label,2)
                    i+=1
                else:
                    for j in range(9):
                        tem=cubes[:,j,...]
                        tem=tem.reshape((tem.shape[0],14,60,40,1))
                        valid_data[j]=np.append(valid_data[j],tem,axis=0)
                        
                    label=np_utils.to_categorical(label,2)
                    valid_labels=np.append(valid_labels,label,axis=0)
                    i=i+1
             
            return valid_data,valid_labels
                        
      
    def generate_batch_data(self):
        
        listing=os.listdir(self.data_path+'/'+'train')       
        os.chdir(self.data_path+'/'+'train')
        while 1:
            No_file=range(len(listing))
            for i in range(len(listing)):
              random_index = randrange(0,len(No_file))
              index=No_file[random_index]
              train_data=[None]*9          
             
              selected_file=listing[index]
              
              read_file = h5py.File(selected_file, 'r')
              cubes = read_file['cubes'][:]
              label = read_file['labels'][:]
              
              if len(label)==0 or cubes.shape[0]==0:
                 continue
             
              #if len(label)<self.batch_size:
                 #cubes,label=self.append_cubes(cubes,label,self.batch_size)
                  
              assert cubes.shape[0] == len(label)
              
              for j in range(9):
                  train_data[j]=cubes[:,j,...]
                  train_data[j]=train_data[j].reshape((cubes.shape[0],cubes.shape[2],cubes.shape[3],cubes.shape[4],1)) 
                  #print train_data[j].shape
              
              train_labels=np_utils.to_categorical(label,2)
              del No_file[random_index]
              yield (train_data,train_labels)
                 

    def generate_test_data(self):
        listing_test=sorted(os.listdir(self.data_path+'/'+'test'))
        os.chdir(self.data_path+'/'+'test')
        test_data=[None]*9
        while 1:
         for file in listing_test:
            read_file=h5py.File(file,'r')
            cubes=read_file['cubes'][:]
            label=read_file['labels'][:]
            for j in range(9):
                test_data[j]=cubes[:,j,...]
                test_data[j]=test_data[j].reshape((cubes.shape[0],cubes.shape[2],cubes.shape[3],cubes.shape[4],1))
            test_labels=np_utils.to_categorical(label,2)
        
            yield (test_data,test_labels)
            
        
    def generate_predict_data(self):
        listing_test=sorted(os.listdir(self.data_path+'/'+'test'))
        os.chdir(self.data_path+'/'+'test')
        test_data=[None]*9
        while 1:
         for file in listing_test:
            read_file=h5py.File(file,'r')
            cubes=read_file['cubes'][:]
            label=read_file['labels'][:]
            test_labels=np_utils.to_categorical(label,2)
            print test_labels.shape
            for j in range(9):
                test_data[j]=cubes[:,j,...]
                test_data[j]=test_data[j].reshape((cubes.shape[0],cubes.shape[2],cubes.shape[3],cubes.shape[4],1))
            yield test_data
           
             
    def num_batches(self):
        num_train=len(os.listdir(self.data_path+'/'+'train'))
        num_validation=len(os.listdir(self.data_path+'/'+'validation'))
        num_test=len(os.listdir(self.data_path+'/'+'test'))
        
        return num_train,num_validation,num_test
        
              
def test():
    path='/mnt/scratch/mikhail/salieny_cnn_data/__3DCNN_preprocessed__/Gazecom_batches'
    DG=Data_Generator_Gazecom(path,batch_size=50)
    valid_data,valid_labels=DG.generate_vali_data()
    return valid_data,valid_labels
    
if __name__ == "__main__":
    valid_data,valid_labels=test() 