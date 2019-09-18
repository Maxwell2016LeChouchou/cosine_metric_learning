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
        image_list, yt_person_name, yt_dir_file = youtube_faces.read_train_split_to_str(
            self._dataset_dir)
        train_indices, _ = util.create_validation_split(
            np.asarray(ids, np.int64), self._num_validation_y, self._seed)
        
        image_list = [image_list[i] for i in train_indices]
        yt_person_name = [yt_person_name[i] for i in train_indices]
        yt_dir_file = [yt_dir_file[i] for i in train_indices]

        return image_list, yt_person_name, yt_dir_file
    
    def read_validation(self):
        image_list, yt_person_name, yt_dir_file = youtube_faces.read_train_split_to_str(
            self._dataset_dir)
        _, valid_indices = util.create_validation_split(
            np.asarray(ids, np.int64), self._num_validation_y, self._seed)

        image_list = [image_list[i] for i in valid_indices]
        yt_person_name = [yt_person_name[i] for i in valid_indices]
        yt_dir_file = [yt_dir_file[i] for i in valid_indices]

        return image_list, yt_person_name, yt_dir_file

    def read_test(self):
        return youtube_faces.read_test_split_to_str(self._dataset_dir)

def main():


        

        

