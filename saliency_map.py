# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 17:28:59 2017

@author: ga63koh
"""

from keras.models import load_model
from hardwired_class import hardwired
from supersaliency import util
from  Generate_blobs import  Generate_blobs
import h5py
import numpy as np
import os
import cv2

class saliency_map(object):
    
    def __init__(self,model_path,test_path,dst_path,out_size,downsize_factor,tau,crows,ccols):     # test_path: the path to the test data test data is the raw video
        self.model_path=model_path
        self.test_path=test_path
        self.dst_path=dst_path                       # the path used to store the output of the hardwired
        self.out_size=out_size
        self.downsize_factor=downsize_factor
        self.model=load_model(model_path)
        self.tau=tau
        self.crows=crows
        self.ccols=ccols
        os.chdir(self.dst_path)
        
        h= hardwired(self.test_path,self.dst_path,self.out_size)
        self.num_frames,self.rows,self.cols=h.process_video()                                    # convert the raw video into five channels
           
    def extract_volumes(self,video_blobs,t,r,c):
        volume=video_blobs[t-self.tau/2:t+self.tau/2,r-self.crows/2:r+self.crows/2,c-self.ccols/2:c+self.ccols/2,:]
        print volume.shape
        volume=volume.reshape([1,self.tau,self.crows,self.ccols,1])
        return volume
        
    def saliency_map(self):
        G= Generate_blobs(self.dst_path,self.rows,self.cols,self.downsize_factor,self.crows,self.ccols,self.tau,self.out_size)
        Gray,GrdX1,GrdX2,GrdX3,GrdY1,GrdY2,GrdY3,OptX,OptY=G.generate_blobs()
        saliency_map=np.zeros(Gray.shape[:3])
        
        for t in range(self.tau,self.num_frames+self.tau):
            for r in range(self.crows,self.crows+self.out_size[0],self.downsize_factor[0]):
                for c in range(self.ccols,self.ccols+self.out_size[1],self.downsize_factor[1]):
                    gray=self.extract_volumes(Gray,t,r,c)
                    grdx1=self.extract_volumes(GrdX1,t,r,c)
                    grdx2=self.extract_volumes(GrdX2,t,r,c)
                    grdx3=self.extract_volumes(GrdX3,t,r,c) 
                    grdy1=self.extract_volumes(GrdY1,t,r,c)
                    grdy2=self.extract_volumes(GrdY2,t,r,c)
                    grdy3=self.extract_volumes(GrdY3,t,r,c) 
                    optx=self.extract_volumes(OptX,t,r,c)
                    opty=self.extract_volumes(OptY,t,r,c) 
                    
                    volume=[gray,grdx1,grdx2,grdx3,grdy1,grdy2,grdy3,optx,opty]
                    
                    p=self.model.predict(volume)
                    print t,r,c
                    saliency_map[t,r,c]=p[0,1]     #[0]
                    
        saliency_map=saliency_map[self.tau:self.tau+self.num_frames,self.crows:self.crows+self.out_size[0]:self.downsize_factor[0],self.ccols:self.ccols+self.out_size[1]:self.downsize_factor[1]]  
        saliency_map_interp=[cv2.resize(saliency_map[i,...],(self.out_size[1],self.out_size[0])) for i in range(self.num_frames)]
        saliency_map_interp=np.asarray(saliency_map_interp)
        return saliency_map_interp
                    
  
def test():
    model_path='/home/ga63koh/self study/keras/FP Project/my_model110.h5'
    test_path='/mnt/scratch/mikhail/salieny_cnn_data/__3DCNN_preprocessed__/saliency map_256/video'
    dst_path='/mnt/scratch/mikhail/salieny_cnn_data/__3DCNN_preprocessed__/saliency map_256/channels'
    map_path='/mnt/scratch/mikhail/salieny_cnn_data/__3DCNN_preprocessed__/saliency map_256/saliency map'
    
    out_size=[72*2,128*2]
    downsize_factor=[2,2]
    tau=14
    crows=60
    ccols=40
    s_map=saliency_map(model_path,test_path,dst_path,out_size,downsize_factor,tau,crows,ccols)
    smap=s_map.saliency_map()
    
    os.chdir(map_path)
    '''
    for f in range(smap.shape[0]):
        sf=smap[f,...]*255
        if f <10:
           cv2.imwrite('00'+str(f)+'.png',sf)
        elif f<100:
           cv2.imwrite('0'+str(f)+'.png',sf) 
        else:
           cv2.imwrite(str(f)+'.png',sf)
    '''     
    dump_file = h5py.File('s'+'.h5', 'w')
    dump_file.create_dataset('cubes', data=smap)
    dump_file.close() 
    
    util.VideoHandler.write(smap,'test.avi',{'fps':30},out_size=util.Size(720,1280),equalize_hist=True)
    return smap
   
   
if __name__ == "__main__":
    smap=test()         
    


       
