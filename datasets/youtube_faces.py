# vim: expandtab:ts=4:sw=4
import os
import numpy as np
import cv2
import scipy.io as sio
import csv

# The maximum person ID in the dataset.
# MAX_LABEL = 1501
MAX_LABEL = 159 


IMAGE_SHAPE = 64, 64, 3

def row_csv2dict(csv_file):
    dict_club={}
    with open(csv_file)as f:
        reader=csv.reader(f,delimiter=',')
        for row in reader:
            dict_club[row[0]]=row[1]
            
    return dict_club

def read_train_directory_to_str(directory):
    
    """Read bbox_train directory.

    Parameters
    ----------
    directory : str
        Path to bbox_train directory.

    Returns
    -------
    (List[str], List[int], List[int], List[int])
        Returns a tuple with the following entries:

        * List of image filenames.
        * List of corresponding unique IDs for the individuals in the images.
        * List of camera indices.
        * List of tracklet indices.

    """
    
    image_filenames = []
    ids = []
    camera_indices = []
    tracklet_indices = []
    youtube_dic = row_csv2dict('/home/max/Desktop/yt_test_data/train_image_pairs.txt')
    train_dir = '/home/max/Desktop/yt_test_data/bbox_train/'

    for file_dir in os.listdir(directory):
        txt = os.path.join(directory, file_dir)
        for line in open(txt, "r"):
            data = line.split(",")
            filename = data[0]
            image_path = os.path.join(train_dir,filename)
            file_info = filename.split("/")
            person = file_info[0]
            person_ids = youtube_dic[person]
            camera = file_info[1]
            tracklet = camera

            image_filenames.append(image_path)
            ids.append(person_ids)
            camera_indices.append(camera)
            tracklet_indices.append(tracklet)

    return image_filenames, ids, camera_indices, tracklet_indices


def read_test_directory_to_str(directory):
    """Read bbox_test directory.

    Parameters
    ----------
    directory : str
        Path to bbox_test directory.

    Returns
    -------
    (List[str], List[int], List[int], List[int])
        Returns a tuple with the following entries:

        * List of image filenames.
        * List of corresponding unique IDs for the individuals in the images.
        * List of camera indices.
        * List of tracklet indices.

    """

    image_filenames= []
    ids = []
    camera_indices = []
    tracklet_indices = []
    youtube_dic = row_csv2dict('/home/max/Desktop/yt_test_data/test_image_pairs.txt')
    train_dir = '/home/max/Desktop/yt_test_data/bbox_test/'

    for file_dir in os.listdir(directory):
        txt = os.path.join(directory, file_dir)
        for line in open(txt, "r"):
            data = line.split(",")
            filename = data[0]
            image_path = os.path.join(train_dir,filename)
            file_info = filename.split("/")
            person = file_info[0]
            person_ids = youtube_dic[person]
            camera = file_info[1]
            tracklet = camera

            image_filenames.append(image_path)
            ids.append(person_ids)
            camera_indices.append(camera)
            tracklet_indices.append(tracklet)

    return image_filenames, ids, camera_indices, tracklet_indices


def read_train_directory_to_image(directory, image_shape=(64, 64)): #Completed 
    """Read images in bbox_train/bbox_test directory.

    Parameters
    ----------
    directory : str
        Path to bbox_train/bbox_test directory.
    image_shape : Tuple[int, int]
        A tuple (height, width) of the desired image size.

    Returns
    -------
    (np.ndarray, np.ndarray, np.ndarray, np.ndarray)
        Returns a tuple with the following entries:

        * Tensor of images in BGR color space.
        * One dimensional array of unique IDs for the individuals in the images.
        * One dimensional array of camera indices.
        * One dimensional array of tracklet indices.

    """
    reshape_fn = (
        (lambda x: x) if image_shape == IMAGE_SHAPE[:2]
        else (lambda x: cv2.resize(x, image_shape[::-1])))
    
    filenames, ids, camera_indices, tracklet_indices = (
        read_train_directory_to_str(directory))

    images = np.zeros((len(filenames), ) + image_shape + (3, ), np.uint8)
    for i, filename in enumerate(filenames):
        if i % 1000 == 0:
            print("Reading %s, %d / %d" % (directory, i, len(filenames)))
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        images[i] = reshape_fn(image)    
    ids = np.asarray(ids, dtype=np.int64)
    camera_indices = np.asarray(camera_indices, dtype=np.int64)
    tracklet_indices = np.asarray(tracklet_indices, dtype=np.int64)
    return images, ids, camera_indices, tracklet_indices


def read_test_directory_to_image(directory, image_shape=(64, 64)): #Completed 
    """Read images in bbox_train/bbox_test directory.

    Parameters
    ----------
    directory : str
        Path to bbox_train/bbox_test directory.
    image_shape : Tuple[int, int]
        A tuple (height, width) of the desired image size.

    Returns
    -------
    (np.ndarray, np.ndarray, np.ndarray, np.ndarray)
        Returns a tuple with the following entries:

        * Tensor of images in BGR color space.
        * One dimensional array of unique IDs for the individuals in the images.
        * One dimensional array of camera indices.
        * One dimensional array of tracklet indices.

    """
    reshape_fn = (
        (lambda x: x) if image_shape == IMAGE_SHAPE[:2]
        else (lambda x: cv2.resize(x, image_shape[::-1])))
    
    filenames, ids, camera_indices, tracklet_indices = (
        read_test_directory_to_str(directory))

    images = np.zeros((len(filenames), ) + image_shape + (3, ), np.uint8)
    for i, filename in enumerate(filenames):
        if i % 1000 == 0:
            print("Reading %s, %d / %d" % (directory, i, len(filenames)))
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        images[i] = reshape_fn(image)    
    ids = np.asarray(ids, dtype=np.int64)
    camera_indices = np.asarray(camera_indices, dtype=np.int64)
    tracklet_indices = np.asarray(tracklet_indices, dtype=np.int64)
    return images, ids, camera_indices, tracklet_indices



def read_train_split_to_str(dataset_dir): # Completed 
    """Read training data to list of filenames.

    Parameters
    ----------
    dataset_dir : str
        Path to the MARS dataset directory; ``bbox_train`` should be a
        subdirectory of this folder.

    Returns
    -------
    (List[str], List[int], List[int], List[int])
        Returns a tuple with the following entries:

        * List of image filenames.
        * List of corresponding unique IDs for the individuals in the images.
        * List of camera indices.
        * List of tracklet indices.

    """
    train_dir = os.path.join(dataset_dir, "csv_train")
    return read_train_directory_to_str(train_dir)


def read_train_split_to_image(dataset_dir, image_shape=(64, 64)): # Completed 
    """Read training images to memory. This consumes a lot of memory.

    Parameters
    ----------
    dataset_dir : str
        Path to the MARS dataset directory; ``bbox_train`` should be a
        subdirectory of this folder.
    image_shape : Tuple[int, int]
        A tuple (height, width) of the desired image size.

    Returns
    -------
    (np.ndarray, np.ndarray, np.ndarray, np.ndarray)
        Returns a tuple with the following entries:

        * Tensor of images in BGR color space.
        * One dimensional array of unique IDs for the individuals in the images.
        * One dimensional array of camera indices.
        * One dimensional array of tracklet indices.

    """
    train_dir = os.path.join(dataset_dir, "csv_train")
    return read_train_directory_to_image(train_dir, image_shape)


def read_test_split_to_str(dataset_dir):   #Completed
    """Read training data to list of filenames.

    Parameters
    ----------
    dataset_dir : str
        Path to the MARS dataset directory; ``bbox_test`` should be a
        subdirectory of this folder.

    Returns
    -------
    (List[str], List[int], List[int], List[int])
        Returns a tuple with the following entries:

        * List of image filenames.
        * List of corresponding unique IDs for the individuals in the images.
        * List of camera indices.
        * List of tracklet indices.

    """
    test_dir = os.path.join(dataset_dir, "csv_test")
    return read_test_directory_to_str(test_dir)


def read_test_split_to_image(dataset_dir, image_shape=(64, 64)):  # Completed 
    """Read test images to memory. This consumes a lot of memory.

    Parameters
    ----------
    dataset_dir : str
        Path to the MARS dataset directory; ``bbox_test`` should be a
        subdirectory of this folder.
    image_shape : Tuple[int, int]
        A tuple (height, width) of the desired image size.

    Returns
    -------
    (np.ndarray, np.ndarray, np.ndarray, np.ndarray)
        Returns a tuple with the following entries:

        * Tensor of images in BGR color space.
        * One dimensional array of unique IDs for the individuals in the images.
        * One dimensional array of camera indices.
        * One dimensional array of tracklet indices.

    """
    test_dir = os.path.join(dataset_dir, "csv_test")
    return read_test_directory_to_image(test_dir, image_shape)

