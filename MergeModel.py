# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 15:22:29 2017

@author: ga63koh
"""
'''
input:
     channels: number of channels of the network
     tau: number of time dimension of the convolution kernnel
     rows:number of rows of the convolution kernel
     cols:number of columns of the convolution kernel
'''


from keras.layers import Input,Dense
from subNetwork import subNetwork 
from keras.models import Model
import keras.layers
from keras import optimizers

class Merge_Model(object):
        
  
    def constru_model(self,channels,tau,rows,cols):
        input_channels=[None]*channels
        base_model=[None]*channels
        
        for i in range(channels):
            input_channels[i]=Input(shape=(tau,rows,cols,1))
            base_model[i]=subNetwork.ConstructLayer(input_channels[i])

        out=keras.layers.concatenate(base_model)

        #out = Dense(1000, activation='softmax')(out)
        out = Dense(100, activation='softmax')(out)
        out = Dense(10, activation='softmax')(out)
        out = Dense(2, activation='softmax')(out)

        model=Model(input_channels,out)
        sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy']) 
        #change optimizer as  sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        
        return model

def test():
    M=Merge_Model()
    model=M.constru_model(channels=9,tau=14,rows=60,cols=40)
    print 'The number os input channels is: ' ,len(model.input)
    return model
          

if __name__ == "__main__":
    net = test()