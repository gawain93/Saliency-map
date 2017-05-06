# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 11:36:43 2017

@author: ga63koh
"""

'''
The file used to train the network and evaluate the results
inputs:
      data_sets: The name of the datasets
      data_path: The path to the preprocessed data
      channels: number of channels of the network
      tau: number of time dimension of the convolution kernnel
      rows:number of rows of the convolution kernel
      cols:number of columns of the convolution kernel
      batch_size: number of video volumes in each training batch
      num_epoch: number of epoches for training
      model_path: if the training is started from the begining, if 1 , start from begin
      num_test: number of points used for testing
'''

from MergeModel import Merge_Model 
import matplotlib.pyplot as plt
import os
from keras.models import load_model
from Data_Generator_Gazecom_default import Data_Generator_Gazecom_default
from Data_Generator_Gazecom import Data_Generator_Gazecom
from DataLayer_Hollywood import DataLayer_Hollywood
from supersaliency import evaluation

def Train(data_sets,data_path,channels,tau,rows,cols,batch_size,num_epoch,model_path,num_test):
    
    default_tau=14
    default_rows=60
    default_cols=40
        
    if model_path is None:
       M=Merge_Model()
       model=M.constru_model(channels,tau,rows,cols)
    else:
       model=load_model(model_path)   
       
    if data_sets=='Gazecom':
       if tau!=default_tau or rows!=default_rows or cols!=default_cols:
          Data_generator=Data_Generator_Gazecom(data_path,rows,cols,tau,batch_size,videos_per_batch=5)
          train_generator=Data_generator.batch_generator()
          num_points=Data_generator.num_points()
          valid_data,valid_labels=Data_generator.valid_data()
          history=model.fit_generator(train_generator,steps_per_epoch =num_points//batch_size+1 ,epochs=num_epoch,validation_data=(valid_data,valid_labels))    
          test_generator=Data_generator.test_generator()
          
       else:
           Data_generator=Data_Generator_Gazecom_default(data_path,batch_size)
           train_generator=Data_generator.generate_batch_data()
           valid_data,valid_labels=Data_generator.generate_vali_data()
           num_train,num_validation,num_test=Data_generator.num_batches()              
           d,l=train_generator.next()
           print [d[i].shape for i in range(9)] , num_train
           history=model.fit_generator(train_generator,steps_per_epoch =num_train ,epochs=num_epoch,validation_data=(valid_data,valid_labels)) 
           
    if data_sets=='Hollywood':
       Data_generator=DataLayer_Hollywood(data_path,rows,cols,tau,batch_size,videos_per_batch=5)
       train_generator=Data_generator.batch_generator()
       num_points=Data_generator.num_points()
       valid_data,valid_labels=Data_generator.valid_data()
       history=model.fit_generator(train_generator,steps_per_epoch =num_points//batch_size+1 ,epochs=num_epoch,validation_data=(valid_data,valid_labels))        
       test_generator=Data_generator.test_generator()
       
    os.chdir(data_path+'/'+'results')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    model.save('my_model.h5')
    model.save_weights('weights.h5')
    
    labels=[None]*num_test
    predicts=[None]*num_test
    for i in range(num_test):    
        test_data, label=test_generator.next()
        predict=model.predict(test_data)[:,1]  
        labels[i]=label
        predicts[i]=predict
            
    return test_generator,predicts,labels
       
test_generator,predicts,labels=Train('Gazecom',data_path,9,14,60,40,50,1,model_path=None,num_test=50)    # data_path for Gazecom:  '/mnt/scratch/mikhail/salieny_cnn_data/__3DCNN_preprocessed__/Gazecom_batches'
 
                                                                                                        # data_path for hollywood:'/mnt/scratch/mikhail/salieny_cnn_data/__3DCNN_preprocessed__/hollywood'
auc=evaluation.calculate_area_under_curve(predicts, labels, cls=1, random_state=42)
tp, fp=evaluation.get_roc_curve(predicts, labels, cls=1)
plt.plot(fp, tp)
plt.xlabel('False positive rate')
plt.ylabel('Ture positive rate')
plt.title('Roc Curve')
    