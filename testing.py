import os
import numpy as np
import cv2
import csv

def _parse_filename(csv_dir):
    csv_dir = '/home/max/Downloads/cosine_metric_learning/datasets/test_csv'
    for filename in os.listdir(csv_dir):
        for line in open(os.path.join(csv_dir,filename), "r"):
            data = line.split(",")
            filename = data[0]
            filename_base, ext = os.path.splitext(filename)
            person_name, dir_file, frame_idx = filename_base.split("/")
            print(person_name)
            print(dir_file)
            print(frame_idx)

    return str(person_name), str(dir_file), str(frame_idx)



def train_dataset(image_path)
    
    image_list = []
    yt_person_name = []
    yt_dir_file = []

    for home, dirs, files in os.walk(image_path):
        for filename in files:
            meta_data = _parse_filename(filename)
            if meta_data is None:
                continue
            image_list.append(os.path.join(home,filename))
            yt_person_name.append(meta_data[0])
            yt_dir_file.append(meta_data[1])
    print(image_list)
    print(yt_person_name)
    print(yt_dir_file) 


def main():
    input_path = '/home/max/Downloads/cosine_metric_learning/datasets/test_dataset/'
    csv_dir = '/home/max/Downloads/cosine_metric_learning/datasets/test_csv'
    train_dataset(input_path)


if __name__ == "__main__":
    main()