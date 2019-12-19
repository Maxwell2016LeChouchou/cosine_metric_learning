import os
import numpy as np 
import csv


# def extract_file_dir(input_dir, output_dir):
#     array_dic = []
#     for line in open(input_dir, "r"):
#         data = line.split(",")
#         filename = data[0]
#         array_dic.append(np.array([filename]))
#     a = np.array(array_dic)
#     if a.shape[0] > 0:
#         np.savetxt(output_dir,a,fmt="%s")
#     print(output_dir)

# def main():
#     input_dir = '/home/maxwell/Desktop/yt_test_data/test_csv/'
#     output_dir = '/home/maxwell/Desktop/yt_test_data/csv_list/'

#     if not os.path.exists(output_dir):
#         os.mkdir(output_dir)
#     for filename in os.listdir(input_dir):
#         if os.path.isfile(os.path.join(input_dir, filename)):
#             extract_file_dir(input_dir+filename, output_dir+filename)

# if __name__ == "__main__":
#     main()


# def read_train_image(directory):
#     array=[]
#     for home, dirs, files in os.walk(directory):
#         for filename in files:
#             array.append(filename)
#     print(array)
#     print(len(array))

# read_train_image('/home/maxwell/Downloads/image/test_images/')

# def read_train_test_directory_to_str(directory):
#     """Read bbox_train/bbox_test directory.

#     Parameters
#     ----------
#     directory : str
#         Path to bbox_train/bbox_test directory.

#     Returns
#     -------
#     (List[str], List[int], List[int], List[int])
#         Returns a tuple with the following entries:

#         * List of image filenames.
#         * List of corresponding unique IDs for the individuals in the images.
#         * List of camera indices.
#         * List of tracklet indices.

#     """

#     def to_label(x):
#         return int(x) if x.isdigit() else -1

#     dirnames = os.listdir(directory)
#     image_filenames, ids, camera_indices, tracklet_indices = [], [], [], []
#     for dirname in dirnames:
#         filenames = os.listdir(os.path.join(directory, dirname))
#         filenames = [
#             f for f in filenames if os.path.splitext(f)[1] == ".jpg"]
#         image_filenames += [
#             os.path.join(directory, dirname, f) for f in filenames]
#         ids += [to_label(dirname) for _ in filenames]
#         camera_indices += [int(f[5]) for f in filenames]
#         tracklet_indices += [int(f[7:11]) for f in filenames]
#         print(filenames)
#     return image_filenames, ids, camera_indices, tracklet_indices

# read_train_test_directory_to_str('/home/maxwell/Desktop/yt_test_data/test_image_2/')


# def main():
#     input_dir = '/home/maxwell/Desktop/yt_test_data/test_csv/'
#     output_dir = '/home/maxwell/Desktop/yt_test_data/csv_list/'

#     if not os.path.exists(output_dir):
#         os.mkdir(output_dir)
#     for filename in os.listdir(input_dir):
#         if os.path.isfile(os.path.join(input_dir, filename)):
#             extract_file_dir(input_dir+filename, output_dir+filename)

# if __name__ == "__main__":
#     main()

# def test_file(directory):
#     image_dir = '/home/maxwell/Desktop/yt_test_data/test_image/'
#     for filename in os.listdir(directory):
#         txt = os.path.join(directory, filename)
#         for line in open(txt, "r"):
#             data = line.split(",")
#             filename = data[0]
#             image_path = os.path.join(image_dir, filename)
#             #image_filenames.append(image_path)

#             #filename_base, ext = os.path.splitext(line)
#             file_info = filename.split("/")
#             person_name = file_info[0]
#             camera_indices = file_info[1]
#             tracklet_indices = camera_indices
#             print(camera_indices)

# test_file('/home/maxwell/Desktop/yt_test_data/test_csv/')

