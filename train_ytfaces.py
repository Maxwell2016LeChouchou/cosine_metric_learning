import functools
import os
import numpy as np 
import scipy.io as sio 
import train_app
from datasets import youtube_faces
import nets.deep_sort.network_definition as net 

class youtube_faces(object):
    
    def __init__(self, image_list, bbox_files, num_validation_y=0.1, seed=1234):
        self._image_list = image_list
        self._bbox_files = bbox_files
        self._num_validation_y = num_validation_y
        self._seed = seed
    
    def read_train(self):
        image_list, bbox_files = youtube_faces.read_train_split_to_str(self._image_list, self._bbox_files)

        

