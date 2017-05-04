# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 19:10:24 2017

@author: ga63koh
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 10:31:50 2017

@author: ga63koh
"""



import cv2
import os
import numpy as np
import scipy.misc
import h5py
import glob
from supersaliency import util

path1="/mnt/syno8/data/Pursuit/Gazecom/videos"  # path of the raw file
path2="/mnt/scratch/mikhail/salieny_cnn_data/__3DCNN_preprocessed__/gazcom_h5"  # path of the video after process

os.chdir(path1)
rows_resize=144
cols_resize=256



##########################################################

def Normalize(all_framesx,all_framesy,VideoName):
    maxX=np.amax(all_framesx)
    minX=np.amin(all_framesx)
    maxY=np.amax(all_framesy)
    minY=np.amin(all_framesy)
    all_framesx-=minX
    all_framesy-=minY
    all_framesx/= (maxX-minX)
    #all_framesx*=255
    all_framesy /= (maxY-minY)
    #all_framesy*=255


video_num=0

for videos in sorted(glob.glob('*.avi')):
    video=cv2.VideoCapture(path1+'/'+videos)  # read one video
    if video is None:
       print ("This is the last video")
       break
    width=int(video.get(3))
    height=int(video.get(4))       # get the size of the video
    fps=video.get(5)               # get the stream rate of the radio
    frames=video.get(7)
    all_framesx=np.ndarray([int(frames),rows_resize,cols_resize])
    all_framesy=np.ndarray([int(frames),rows_resize,cols_resize])     # opical flow
    Gray=np.ndarray([int(frames),rows_resize,cols_resize])
    Gradx1=np.ndarray([int(frames),rows_resize,cols_resize])
    Gradx2=np.ndarray([int(frames),rows_resize,cols_resize])
    Gradx3=np.ndarray([int(frames),rows_resize,cols_resize])
    Grady1=np.ndarray([int(frames),rows_resize,cols_resize])
    Grady2=np.ndarray([int(frames),rows_resize,cols_resize])
    Grady3=np.ndarray([int(frames),rows_resize,cols_resize])
    
    VideoName,_=videos.split('.')  # extract the name of each video
    print width,height,fps         # for testing

    iteration=1
   
    # interation inside of each video
    while True:
          frame=video.read()[1]       # extract every frame in the video
          if frame is None:        # if no frames left, quit
             print ("This is the last frame")
             break
          frame=scipy.misc.imresize(frame,(rows_resize,cols_resize))
          w,h,c=frame.shape[:3]    # shape of the frame
          print iteration, (h,w,c) # for testing


          ## start to process the frames
          gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)                          #convert to gray image
          grdx=cv2.Sobel(frame,cv2.CV_8U,1,0,ksize=3,scale=1,delta=0)    
                                                                               #use sobel filter to calculate the gradient
          grdy=cv2.Sobel(frame,cv2.CV_8U,0,1,ksize=3,scale=1,delta=0)          #boarderType=cv2.BORDER_DEFAULT
          
          
          if iteration is 1:
             all_framesx[0,...]=np.zeros(gray.shape).astype('float64')
             all_framesy[0,...]=np.zeros(gray.shape).astype('float64')
             Gray[0,...]=gray/255.
             Gradx1[0,...]=grdx[...,0]/255.
             Gradx2[0,...]=grdx[...,1]/255.
             Gradx3[0,...]=grdx[...,2]/255.
             Grady1[0,...]=grdy[...,0]/255.
             Grady2[0,...]=grdx[...,1]/255.
             Grady3[0,...]=grdx[...,2]/255.
             LastFrame=gray
             

          if iteration is not 1:
             flow = cv2.calcOpticalFlowFarneback(LastFrame,gray,0.5, 3, 15, 3, 5, 1.2, 0)       
                                                                
             all_framesx[iteration-2,...]=flow[...,0]
             all_framesy[iteration-2,...]=flow[...,1]
             Gray[iteration-1,...]=gray/255.
             Gradx1[iteration-1,...]=grdx[...,0]/255.
             Gradx2[iteration-1,...]=grdx[...,1]/255.
             Gradx3[iteration-1,...]=grdx[...,2]/255.
             Grady1[iteration-1,...]=grdy[...,0]/255.
             Grady2[iteration-1,...]=grdx[...,1]/255.
             Grady3[iteration-1,...]=grdx[...,2]/255.
             
             
          
          iteration=iteration+1
          LastFrame=gray      
          
          # save the frames after processing
          
     # The end of the interation inside each video
    Normalize(all_framesx,all_framesy,VideoName)
    dump_file = h5py.File(path2+'/'+VideoName+'.h5', 'w')
    dump_file.create_dataset('channels', data=np.asarray([Gray,Gradx1,Gradx2,Gradx3,Grady1,Grady2,Grady3,all_framesx,all_framesy]))
    dump_file.create_dataset('VideoName', data=VideoName)
    del Gray
    del Gradx1
    del Gradx2
    del Gradx3
    del Grady1
    del Grady2
    del Grady3
    del all_framesx
    del all_framesy
    gc = util.datasets.load_gazecom(out_size=util.Size(144,256))
    
    file_name= gc['ground_truth'][video_num]['filename']
    file_name=file_name.split('/')[-1]   # the last '/' in the file name is the video name
    file_name=file_name.split('_')[0]
    assert videos.split('.')[0]==file_name
    
    augmented_points, augmented_labels = util.GroundTruthHandler.augment_with_negative_samples(gc['ground_truth'][video_num]) 
    dump_file.create_dataset('augmented_points', data=augmented_points['data'])
    dump_file.create_dataset('augmented_labels',data=augmented_labels)
    dump_file.create_dataset('num_points',data=len(augmented_labels))
    dump_file.close()
    video_num+=1



             
                                                  
           


