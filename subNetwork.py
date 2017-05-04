# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 16:10:54 2017

@author: ga63koh

"""

from keras.layers import Conv3D,MaxPooling3D,Flatten
import keras.layers

class subNetwork(object):
    
    # The class to cinstruct one sub channel of thre input data.
    # The InputImg could be the time series images Grayscale, 
    # Gradient in X or Y directions or OpticalFloe in X and Y directions
    # output: model       
                                                                                
    @staticmethod
    def ConstructLayer(input_video):
         
        #input_video=Input(shape=(14,60,40,3))    # H1 tau,rows,cols,channels      channels,tau,rows,cols,
        
        # add the first convolution layer, as described in the paper,
        # 2 types of filter is used , the size of the filter is (7,7,3)
        # The first dimensition is channel !!!!!!!!!!!!!!!!!!!!!!!!!!
        '''
        case 1
        
        out1=Conv3D(1,(3,7,7), padding='valid', activation='relu')(input_video)   # C2_1, activation mode can be changed in optimization
        out2=Conv3D(1,(3,7,7), padding='valid', activation='relu')(input_video)   # C2_2
        
        # Spatial subsample of (2,2)
        out1=MaxPooling3D((1,2,2), padding='valid')(out1)    # S3_1
        out2=MaxPooling3D((1,2,2), padding='valid')(out2)    # S3_2
        # Since the the convolutional layer after S3 are are consist of one filter,
        # So it is not necessary to construct different convolutional layer for 
        # different output from the previous layer
        
        print 'The shape of layer S3 is: ',out1.shape
        # kernel size is (7,6,3)
        out1_1=Conv3D(1,(3, 7, 6),strides=(1, 1, 1),padding='valid')(out1)   # C4_1
        out1_2=Conv3D(1,(3, 7, 6),strides=(1, 1, 1),padding='valid')(out1)
        out1_3=Conv3D(1,(3, 7, 6),strides=(1, 1, 1),padding='valid')(out1)
        
        out2_1=Conv3D(1,(3, 7, 6),strides=(1, 1, 1),padding='valid')(out2)   # C4_2
        out2_2=Conv3D(1,(3, 7, 6),strides=(1, 1, 1),padding='valid')(out2)
        out2_3=Conv3D(1,(3, 7, 6),strides=(1, 1, 1),padding='valid')(out2)
                  
        # add (3,3) subsample
        out1_1=MaxPooling3D((1, 3, 3),padding='valid')(out1_1)   # S5_1
        out1_2=MaxPooling3D((1, 3, 3),padding='valid')(out1_2)
        out1_3=MaxPooling3D((1, 3, 3),padding='valid')(out1_3)
        
        out2_1=MaxPooling3D((1, 3, 3),padding='valid')(out2_1)   # S5_2
        out2_2=MaxPooling3D((1, 3, 3),padding='valid')(out2_2)
        out2_3=MaxPooling3D((1, 3, 3),padding='valid')(out2_3)
        
        print 'The shape of layer S5 is: ',out1.shape
        #Flatten the both
        out1_1=Flatten()(out1_1)
        out1_2=Flatten()(out1_2)
        out1_3=Flatten()(out1_3)
        
        out2_1=Flatten()(out2_1)
        out2_2=Flatten()(out2_2)
        out2_3=Flatten()(out2_3)
        
        # concatenate the two parts
        concatenated = keras.layers.concatenate([out1_1,out1_2,out1_3,out2_1,out2_2,out2_3])
        
        return concatenated
       
        '''
        '''
        case2
        '''      
      
        out=Conv3D(2,(3,7,7), padding='valid', activation='relu')(input_video)
        out=MaxPooling3D((1,2,2), padding='valid')(out)
        out=Conv3D(3,(3, 7, 6),strides=(1, 1, 1),padding='valid')(out)
        out=MaxPooling3D((1, 3, 3),padding='valid')(out)
        out=Flatten()(out)
                
        
        return out
        