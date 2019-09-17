# vim: expandtab:ts=4:sw=4
import os
import numpy as np
import cv2
import scipy.io as sio


# The maximum person ID in the dataset.
# MAX_LABEL = 1501

# IMAGE_SHAPE = 128, 64, 3


def _parse_filename(filename):
    """Parse meta-information from given filename.

    Parameters
    ----------
    filename : str
        A Market 1501 image filename.

    Returns
    -------
    (int, int, str, str) | NoneType
        Returns a tuple with the following entries:

        * Unique ID of the individual in the image
        * Index of the camera which has observed the individual
        * Filename without extension
        * File extension

        Returns None if the given filename is not a valid filename.

    """
    # filename_base, ext = os.path.splitext(filename)
    # if '.' in filename_base:
    #     # Some images have double filename extensions.
    #     filename_base, ext = os.path.splitext(filename_base)
    # if ext != ".jpg":
    #     return None
    # person_id, cam_seq, frame_idx, detection_idx = filename_base.split('_')
    # return int(person_id), int(cam_seq[1]), filename_base, ext

    filename_base, ext = os.path.splitext(filename)

    person_name, dir_file, frame_idx = filename_base.split('/')
    return str(person_name), str(dir_file), filename_base, ext
    

def read_train_split_to_str(dataset_dir):
    # """Read training data to list of filenames.

    # Parameters
    # ----------
    # dataset_dir : str
    #     Path to the Market 1501 dataset directory.

    # Returns
    # -------
    # (List[str], List[int], List[int])
    #     Returns a tuple with the following values:

    #     * List of image filenames (full path to image files).
    #     * List of unique IDs for the individuals in the images.
    #     * List of camera indices.

    # """

    image_list = []
    yt_person_name = []
    yt_dir_file = []

    image_path = '/home/max/Downloads/cosine_metric_learning/datasets/train_bbox_image/'
    for home, dirs, files in os.walk(image_path):
        for filename in files:
            meta_data = _parse_filename(filename)
            if meta_data is None:
                continue
            image_list.append(os.path.join(home,filename))
            yt_person_name.append(meta_data[0])
            yt_dir_file.append(meta_data[1])
    print ("successfully get train image list")  

    return image_list, yt_person_name, yt_dir_file

def read_train_split_to_image(dataset_dir):
 

    image_list, yt_person_name, yt_dir_file = read_train_split_to_str(dataset_dir)
    
    images = np.zeros((len(image_list), 128, 64, 3), np.uint8)
    for i, filename in enumerate(image_list):
        images[i] = cv2.imread(image_list, cv2.IMREAD_COLOR)
    
    return images, yt_person_name, yt_dir_file


def read_test_split_to_str(dataset_dir):

    image_list = []
    yt_person_name = []
    yt_dir_file = []

    image_path = '/home/max/Downloads/cosine_metric_learning/datasets/test_bbox_image/'
    for home, dirs, files in os.walk(image_path):
        for filename in files:
            meta_data = _parse_filename(filename)
            if meta_data is None:
                continue
            image_list.append(os.path.join(home,filename))
            yt_person_name.append(meta_data[0])
            yt_dir_file.append(meta_data[1])
    print ("successfully get test image list")  

    return image_list, yt_person_name, yt_dir_file


def read_test_split_to_image(dataset_dir):
   

    image_list, yt_person_name, yt_dir_file = read_test_split_to_str(dataset_dir)
    
    images = np.zeros((len(image_list), 128, 64, 3), np.uint8)
    for i, filename in enumerate(image_list):
        images[i] = cv2.imread(image_list, cv2.IMREAD_COLOR)
    
    return images, yt_person_name, yt_dir_file