import functools
import os
import numpy as np 
import scipy.io as sio 
import train_app
from datasets import youtube_faces
import nets.deep_sort.network_definition as net 

class youtube_faces(object):
    
    def __init__(self, dataset_dir, num_validation_y=0.1, seed=1234):
        self._dataset_dir = dataset_dir
        self._num_validation_y = num_validation_y
        self._seed = seed
    
    def read_train(self):
        image_list, yt_person_name, yt_dir_file = youtube_faces.read_train_split_to_str(self._dataset_dir)
        train



        

