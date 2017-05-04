# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 14:33:52 2017

@author: ga63koh
"""

'''
 the class use to cut the raw input video and generate the labels
input:
      DataAdd: input data address
      channel_list: the name of the channels
      Dist: output address
      tau
      l
'''

from supersaliency import util
import numpy as np
import h5py
import os
import cv2

class DataLayer_Gazecom(object):
    
    def __init__(self,DataAdd,channel_list,Dist,tau,crows,ccols):
        self.DataAdd=DataAdd
        self.listing=np.sort(os.listdir(self.DataAdd))
        self.num_videos=len(self.listing)/len(channel_list)
        self.out_size=util.Size(72*2, 128*2)
        self.gc = util.datasets.load_gazecom(handlabelling_expert='ioannis',out_size=self.out_size)   # get the labels of the data ['video_parameters', 'ground_truth', 'video_names']
        self.channel_list=channel_list
        self.Dist=Dist
        self.tau=tau
        self.crows=crows
        self.ccols=ccols
    
    def padarray(self,input_folder):           # input_folder is an absolute path !!!!!!!!!!!!!!!!
        current_folder='/home/ga63koh/processed'+'/'+str(input_folder)
        print current_folder
        num_frames=os.listdir(current_folder)
        num_frames=np.sort(num_frames)
        images=[cv2.resize(cv2.imread(os.path.join(current_folder,image_name)), \
                (self.out_size.width,self.out_size.height)) for image_name in num_frames]   # read and resize the images
        images=np.array(images)
        
        num_frames,rows,cols,rgb=images.shape
        print num_frames,rows,cols,rgb
        padded_images=np.ndarray([num_frames+2*self.tau,rows+2*self.crows,cols+2*self.ccols,rgb])
        padded_images[self.tau:self.tau+num_frames,self.crows:self.crows+rows,self.ccols:self.ccols+cols,:]=images
        padded_images[self.tau:self.tau+num_frames,0:self.crows,:,:]=padded_images[self.tau:self.tau+num_frames,2*self.crows-1:self.crows-1:-1,:,:]
        print 'left'
        padded_images[self.tau:self.tau+num_frames,rows+self.crows:rows+2*self.crows,:,:] \
        =padded_images[self.tau:self.tau+num_frames,self.crows+rows-1:rows-1:-1,:,:]
        print 'right'
        padded_images[self.tau:self.tau+num_frames,:,0:self.ccols,:]=padded_images[self.tau:self.tau+num_frames,:,2*self.ccols-1:self.ccols-1:-1,:]
        print 'up'
        padded_images[self.tau:self.tau+num_frames,:,cols+self.ccols:cols+2*self.ccols,:] \
        =padded_images[self.tau:self.tau+num_frames,:,self.ccols+cols-1:cols-1:-1,:]
        print 'down'
        padded_images[0:self.tau,...]=padded_images[2*self.tau-1:self.tau-1:-1,...]
        padded_images[num_frames+self.tau:num_frames+2*self.tau,...]=padded_images[self.tau+num_frames-1:num_frames-1:-1,...]
        print 'time'
        
        print padded_images.shape
        print 'function pa finished'
        return padded_images
        
    @staticmethod    
    def generate_labels(num):
        if num<10:
            return '000'+str(num)
        elif num<100:
            return '00'+str(num)
        elif num<1000:
            return '0'+str(num)
        else:
            return str(num)
            
            
    def slice_video(self):
        number=0
        for i in range(18):#range(self.num_videos):+range(16,18)
            augmented_points, augmented_labels=util.GroundTruthHandler.augment_with_negative_samples(self.gc['ground_truth'][i])
            print len(augmented_labels)
            # above to process on one given video of the output of the preprocessing
            interest_points=augmented_points['data']+[self.tau,self.crows,self.ccols]     # convert the coordinate to the padded mode
            print np.max(augmented_points['data'][:,0])
            print np.max(augmented_points['data'][:,2])
            print np.max(augmented_points['data'][:,1])
            print np.max(interest_points[:,0])
            print np.max(interest_points[:,2])
            print np.max(interest_points[:,1])
            assert np.max(interest_points[:,0])<660
            assert np.max(interest_points[:,2])<2*self.ccols+2*self.out_size[0]
            assert np.max(interest_points[:,1])<2*self.crows+2*self.out_size[1]
            num_labels=interest_points.shape[0]
            output_cubes=np.ndarray([50,9,self.tau,self.crows,self.ccols])  #[num_labels,channels, t,x,y]
            
            video_folders=self.listing[i*5:(i+1)*5]
            print list([video_folders])
            
            '''
            for channels in list(video_folders):
                if 'GrdX' in channels:
                    padded_channelgx=self.padarray(channels)
                    print padded_channelgx.shape
                    #padded_channelgx1=padded_channelgx[...,0]
                    #padded_channelgx2=padded_channelgx[...,1]
                    #padded_channelgx3=padded_channelgx[...,2]
                    print 'GrdX finished'
                if 'GrdY' in channels:
                    padded_channelgy=self.padarray(channels)
                    print padded_channelgy.shape
                    #padded_channelgy1=padded_channelgy[...,0]
                    #padded_channelgy2=padded_channelgy[...,1]
                    #padded_channelgy3=padded_channelgy[...,2]
                    print 'GrdY finished'
                if 'Gray' in channels:
                    padded_channelgry=self.padarray(channels)
                    print padded_channelgry.shape
                    print 'Gray finished'
                if 'OptX':
                    padded_channelopx=self.padarray(channels)
                    print padded_channelopx.shape
                    print 'OptX finished'
                if 'OptY':
                    padded_channelopy=self.padarray(channels)
                    print padded_channelopy.shape
                    print 'OptY finished'
            '''
            padded_channelgry=self.padarray(video_folders[0])
            print padded_channelgry[...,0].shape
            print 'Gray finished'
            padded_channelgx=self.padarray(video_folders[1])
            print padded_channelgx[...,0].shape
            print 'GrdX finished'
            padded_channelgy=self.padarray(video_folders[2])
            print padded_channelgy[...,0].shape
            print 'GrdY finished'
            padded_channelopx=self.padarray(video_folders[3])
            print padded_channelopx[...,0].shape
            print 'OptX finished'
            padded_channelopy=self.padarray(video_folders[4])
            print padded_channelopy[...,0].shape
            print 'OptY finished'
            
                
            #padded_channels=np.array([padded_channelgry[...,0],padded_channelgx,padded_channelgy,padded_channelopx,padded_channelopy])
            padded_channels=np.array([padded_channelgry[...,0],padded_channelgx[...,0],padded_channelgx[...,1],padded_channelgx[...,2],
                                      padded_channelgy[...,0],padded_channelgy[...,1],padded_channelgy[...,2],padded_channelopx[...,0],
                                      padded_channelopy[...,0]])
            print padded_channels.shape      # should be (t,x,y,9)  
            print 'pad array finished'
                
            del padded_channelgry
            del padded_channelopx
            del padded_channelopy
            del padded_channelgy
            del padded_channelgx
               
            count=0
            batch=0
            ######################## to schuffer the data#################
            augmented_labels=augmented_labels.reshape([augmented_labels.shape[0],1])
            shuffler=np.concatenate((interest_points,augmented_labels),axis=1)
            np.random.shuffle(shuffler)
            interest_points=shuffler[:,:3]
            augmented_labels=shuffler[:,3]
            del shuffler
            ###############################################################
            for interest_point in interest_points:
                print interest_point
                print interest_point[0]-self.tau/2,interest_point[2]-self.crows/2,interest_point[1]-self.ccols/2
                print interest_point[0]+self.tau/2,interest_point[2]+self.crows/2,interest_point[1]+self.ccols/2
                
                output_cubes[count,:,:,:,:]=padded_channels[:,int(interest_point[0]-self.tau/2):int(interest_point[0]+self.tau/2),
                                                              int(interest_point[2]-self.crows/2):int(interest_point[2]+self.crows/2),
                                                              int(interest_point[1]-self.ccols/2):int(interest_point[1]+self.ccols/2)]  #x,y is reversed!!!!!
                print count                
                count+=1
                if count >= 50:
                    os.chdir(self.Dist)
                    '''
                    data={}
                    data['cubes']=output_cubes
                    data['labels']=augmented_labels[int(batch*50):int((batch+1)*50)]
                    data['name']=self.listing[i*5]+str(count)
                    pickle.dump(data,open(self.generate_labels(number)+self.listing[i*5]+str(batch)+'.p','wb'))
                    '''
                    dump_file = h5py.File(self.generate_labels(number)+self.listing[i*5]+str(batch)+'.h5', 'w')
                    dump_file.create_dataset('cubes', data=output_cubes)
                    dump_file.create_dataset('labels', data=augmented_labels[int(batch*50):int((batch+1)*50)])
                    dump_file.close()                    
                    
                    batch+=1
                    count=0
                    number+=1
                    output_cubes=np.zeros([50,9,self.tau,self.crows,self.ccols])
                    print ('video'+self.listing[i*5]+str(batch)+'sliced')
            
            os.chdir(self.Dist)
            '''
            data={}
            data['cubes']=output_cubes[0:int(num_labels-batch*50),...]
            data['labels']=augmented_labels[int(batch*50):int(num_labels)]
            data['name']=self.listing[i*5]+str(count)
            pickle.dump(data,open(self.generate_labels(number)+self.listing[i*5]+str(batch)+'.p','wb'))
            '''
            dump_file = h5py.File(self.generate_labels(number)+self.listing[i*5]+str(batch)+'.h5', 'w')
            dump_file.create_dataset('cubes', data=output_cubes[0:int(num_labels-batch*50),...])
            dump_file.create_dataset('labels', data=augmented_labels[int(batch*50):int(num_labels)])
            dump_file.close()
            number+=1
            count=0
            batch=0
            print 'all done for one video'
                    
'''           
def test():
    DataAdd='/home/ga63koh/processed'
    channel_list=['Gray','GrdX','GrdY','OptX','OptY']
    Dist='/mnt/scratch/mikhail/salieny_cnn_data/__3DCNN_preprocessed__/train_new'
    tau=14
    row=60
    col=40
    datalayer=DataLayer(DataAdd,channel_list,Dist,tau,row,col)
    datalayer.slice_video()


    
if __name__ == "__main__":
    test()       
            
'''               
                
                    
                
                    

                    
                    
                    
                    
                    
                
                
                
                
            
        
                                             

