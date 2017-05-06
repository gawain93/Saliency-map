# -*- coding: utf-8 -*-
"""
Created on Sat May  6 15:58:18 2017

@author: ga63koh
"""

# -*- coding: utf-8 -*-
"""
The class used to generate batches of training data into the Neuro Network
Input: 
     data_path: path of the data of the preprocessed videos
     crows:  height of the video volumes
     ccols:  width of the video volumes
     tau:  time length of the video volumes
     batch_size: number of volumes in each batch
     
Output:
     batch:one batch of volumes
     labels: corresponding labels of the batch

Created on Thu Apr 20 15:23:30 2017

@author: ga63koh
"""

import os
import numpy as np
import h5py
from keras.utils import np_utils

class Data_Generator_Gazecom(object):
    
    def __init__(self,data_path,crows,ccols,tau,batch_size,videos_per_batch):  #data_path /mnt/scratch/mikhail/salieny_cnn_data/__3DCNN_preprocessed__/hollywood_trainh5
        self.data_path=data_path
        self.train_path=data_path+'/gazecom_train_h5'
        self.test_path=data_path+'/gazecom_test_h5'
        self.crows=crows
        self.ccols=ccols
        self.tau=tau
        self.batch_size=batch_size     
        self.videos_per_batch=videos_per_batch
        self.points_per_video=self.batch_size//self.videos_per_batch
        
        
    def generate_cards(self):    
        videos_list=sorted(os.listdir(self.train_path))
        cards=[None]*len(videos_list)
        i=0
        num_points=0
        point_lists=[None]
    
        for video in videos_list:
            print i
            read_file=h5py.File(self.train_path+'/'+video,'r')
            augmented_points=read_file['augmented_points'][:]
            if augmented_points.shape[0] is 0:                   # when there contains no interest points
               print str(video),'skip',i
               continue
            augmented_labels=read_file['augmented_labels'][:]
            augmented_labels=augmented_labels.reshape([len(augmented_labels),1])
            shuffler=np.concatenate((augmented_points,augmented_labels),axis=1)
            np.random.shuffle(shuffler)
            augmented_points=shuffler[:,:3]
            augmented_labels=shuffler[:,3]
            del shuffler
            video_name=video.split(".")[0]
            cards[i]=(video_name,augmented_points,augmented_labels) #
            print video_name
            i+=1
            
        cards=cards[:i]
        while len(cards) is not 0:        
            for i in range(len(cards)):
                num_points+=len(cards[i][2])
            
            num_points/=1.0           
            possibility=[len(cards[i][2])/num_points for i in range(len(cards))]
            print num_points
            print len(cards)
            print np.sum(possibility)
        
            video_index=np.random.choice(len(cards), 1, possibility)[0]
        
            if len(cards[video_index][2])<self.points_per_video:
                for i in range(len(cards[video_index][2])):
                    point=(cards[video_index][0],cards[video_index][1][0,:],cards[video_index][2][0])
                    point_lists.append(point)
                    cards[video_index]=(cards[video_index][0],cards[video_index][1][1:,:],cards[video_index][2][1:])
                del cards[video_index]
                print 'last points'
            else:
                for i in range(self.points_per_video):
                    point=(cards[video_index][0],cards[video_index][1][0,:],cards[video_index][2][0])
                    point_lists.append(point)
                    cards[video_index]=(cards[video_index][0],cards[video_index][1][1:,:],cards[video_index][2][1:])
                print 'to continue'
            num_points=0
        
        del point_lists[0]
        return point_lists
        
        
    def extract_video(self,video_blobs,point_coord):         
        video_blobs=np.asarray(video_blobs)
        channels,num_frames,rows,cols=video_blobs.shape
       
        t,x,y=point_coord
        
        sliced_images=np.ndarray([channels,self.tau,rows+2*self.crows,cols+2*self.ccols])
             
        if  t+self.tau/2>num_frames:
            sliced_images[:,0:int(self.tau/2),int(self.crows):int(rows+self.crows),int(self.ccols):int(self.ccols+cols)]=video_blobs[:,int(t-self.tau/2):int(t),...]
            sliced_images[:,int(self.tau/2):int(num_frames+self.tau/2-t),int(self.crows):int(rows+self.crows),int(self.ccols):int(self.ccols+cols)]   \
                                                                                                         =video_blobs[:,int(t):int(num_frames),...]
            sliced_images[:,int(num_frames+self.tau/2-t):int(self.tau),int(self.crows):int(rows+self.crows),int(self.ccols):int(self.ccols+cols)]  \
                                                 =video_blobs[:,int(num_frames):int(2*num_frames-self.tau/2-t-1):-1,...]
                       
            del video_blobs
            
        elif t-self.tau/2<=0:
            sliced_images[:,int(self.tau/2):int(self.tau),int(self.crows):int(rows+self.crows),int(self.ccols):int(self.ccols+cols)]=video_blobs[:,int(t):int(t+self.tau/2),...]
            sliced_images[:,int(self.tau/2-t):int(self.tau/2),int(self.crows):int(rows+self.crows),int(self.ccols):int(self.ccols+cols)]=video_blobs[:,0:int(t),...]
            sliced_images[:,0:int(self.tau/2-t),int(self.crows):int(rows+self.crows),int(self.ccols):int(self.ccols+cols)]=np.flip(video_blobs[:,0:int(self.tau/2-t),...],1) 
                        
            del video_blobs
        
        else:
            sliced_images[:,:,int(self.crows):int(rows+self.crows),int(self.ccols):int(self.ccols+cols)]=video_blobs[:,int(t-self.tau/2):int(t+self.tau/2),...]
                       
            del video_blobs
            
        return sliced_images,rows,cols
    
    
    def extract_volumes(self,sliced_images,point_coord,rows,cols):
        
        sliced_images[:,:,0:int(self.crows),:]=sliced_images[:,:,int(2*self.crows-1):int(self.crows-1):-1,:]
        sliced_images[:,:,int(rows+self.crows):int(rows+2*self.crows),:]=sliced_images[:,:,int(self.crows+rows-1):int(rows-1):-1,:]
        
        sliced_images[:,:,:,0:int(self.ccols)]=sliced_images[:,:,:,int(2*self.ccols-1):int(self.ccols-1):-1]
        sliced_images[:,:,:,int(cols+self.ccols):int(cols+2*self.ccols)]=sliced_images[:,:,:,int(self.ccols+cols-1):int(cols-1):-1]
        
        _,x,y=point_coord
        volume=sliced_images[:,:,int(y+self.crows/2):int(y+3*self.crows/2),int(x+self.ccols/2):int(x+3*self.ccols/2)]
        
        return volume
    
    
    def slice_cubes(self,video_blobs,point_coord):
         
        sliced_images,rows,cols=self.extract_video(video_blobs,point_coord)
        volume=self.extract_volumes(sliced_images,point_coord,rows,cols)
    
        return volume
    
    def num_points(self):
        points_lists=self.generate_cards()
        return len(points_lists)
        
        
    def valid_data(self):
        points_lists=self.generate_cards()[:100]
        print len(points_lists)
       
        previous_video=points_lists[0][0]   # video name is in first dim
        print previous_video
    
        with h5py.File(self.train_path+'/'+previous_video+'.h5','r') as video_file:
              video_blobs=video_file['channels'][:]
              print video_blobs.shape
        
        while 1:                
              valid_data=np.ndarray([len(points_lists),9,self.tau,self.crows,self.ccols,1])
              labels=np.ndarray([len(points_lists),2])
            
              for i in range(len(points_lists)):
                  video_name=points_lists[i][0]
                
                  if video_name is not previous_video:
                      with h5py.File(self.train_path+'/'+video_name+'.h5','r') as video_file:
                          video_blobs=video_file['channels'][:]
                          print 'changed'
    
                  previous_video=video_name
                
                  point_coord=points_lists[i][1]
                  volume=self.slice_cubes(video_blobs,point_coord)
                  volume=volume.reshape([9,self.tau,self.crows,self.ccols,1])
                
                  label=points_lists[i][2]          # the last item is the label
                  label=np_utils.to_categorical(label,2)
                
                  valid_data[i,...]=volume
                  labels[i,:]=label
                
              valid_data=valid_data.transpose([1,0,2,3,4,5])
              valid=[valid_data[j,...] for j in range(9)]
                      
              return valid,labels
    
    
    def test_generator(self):
         videos_list=sorted(os.listdir(self.test_path))       
        
         while 1:
             for video in videos_list:
                 video_file=h5py.File(self.test_path+'/'+video,'r')
                 points_in_video=video_file['augmented_points'][:]
                 video_blobs=video_file['channels'][:]
                 augmented_labels=video_file['augmented_labels'][:]
                 for p in range(points_in_video.shape[0]):
                    test_data=np.ndarray([1,9,self.tau,self.crows,self.ccols,1])
                    volume=self.slice_cubes(video_blobs,points_in_video[p,:])
                    volume=volume.reshape([9,self.tau,self.crows,self.ccols,1])
                    label=augmented_labels[p]
                    test_data[0,...]=volume
                    test_data=test_data.transpose([1,0,2,3,4,5])
                    yield test_data, label
        
        
    def batch_generator(self):
    
        points_lists=self.generate_cards()[100:-1]
        num_points=len(points_lists)
        print len(points_lists)
        num_batch=num_points//self.batch_size+1
       
        previous_video=points_lists[0][0]   # video name is in first dim
        print previous_video
    
        with h5py.File(self.train_path+'/'+previous_video+'.h5','r') as video_file:
              video_blobs=video_file['channels'][:]
              print video_blobs.shape
        
        while 1:
            No_batch=0
            while No_batch<num_batch:
                if (No_batch+1)*self.batch_size<=num_points:
                   current_list=points_lists[No_batch*self.batch_size:(No_batch+1)*self.batch_size]
                else:
                   current_list=points_lists[No_batch*self.batch_size:]
                
                batch_data=np.ndarray([len(current_list),9,self.tau,self.crows,self.ccols,1])
                labels=np.ndarray([len(current_list),2])
            
                for i in range(len(current_list)):
                    video_name=current_list[i][0]
                
                    if video_name is not previous_video:
                       with h5py.File(self.train_path+'/'+video_name+'.h5','r') as video_file:
                            video_blobs=video_file['channels'][:]
                            print 'changed'
    
                    previous_video=video_name
                
                    point_coord=current_list[i][1]
                    volume=self.slice_cubes(video_blobs,point_coord)
                    volume=volume.reshape([9,self.tau,self.crows,self.ccols,1])
                
                    label=current_list[i][2]          # the last item is the label
                    label=np_utils.to_categorical(label,2)
                
                    batch_data[i,...]=volume
                    labels[i,:]=label
                
                batch_data=batch_data.transpose([1,0,2,3,4,5])
                batch=[batch_data[j,...] for j in range(9)]
            
                No_batch+=1
            
                yield batch,labels
        
            
def test():
    dl=Data_Generator_Gazecom('/mnt/scratch/mikhail/salieny_cnn_data/__3DCNN_preprocessed__/Gazecom_batches',60,40,14,50,5)
    generator=dl.batch_generator()
    return generator,dl
    
if __name__ == "__main__":
    generator ,dl= test()
    b,l=generator.next()
           
        
    