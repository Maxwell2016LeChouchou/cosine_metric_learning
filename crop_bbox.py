import os 
import numpy as np 
from PIL import Image 
import cv2

#image_dir = '/home/max/Desktop/image_file/'
#bbox_dir = '/home/max/Downloads/cosine_metric_learning-master/datasets/Youtube_faces/train_csv_bbox/'

def split(df,group):
    data = namedtuple('data',['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

def get_imagelist(image_path):
    image_list = []

    for home, dirs, files in os.walk(image_path):
        for filename in files:
            image_list.append(os.path.join(home,filename))
    
    print("successfully get image list")
    return image_list

def read_bbox(bbox_dir, output_dir):    
    files = []
    for f in sorted(os.listdir(bbox_dir)):
        domain = os.path.abspath(bbox_dir)
        f = os.path.join(domain,f)
        files += [f]
        for line in open(f, "r"):
            data = line.split(",")
            total_len = len(data)
            filename = data[0]
            bbox_info = [float(i) for i in data[2:total_len]]
            xmin = bbox_info[0]
            xmax = bbox_info[1]
            ymin = bbox_info[2]
            ymax = bbox_info[3]
            #cropped_image = get_imagelist(image_dir).crop(xmin, ymin, xmax, ymax)
            root = '/home/max/Downloads/cosine_metric_learning-master/datasets/Youtube_faces/Val_dataset/'
            im_filename = os.path.join(root, filename)
            im = cv2.imread(im_filename)
            
            cropped_image = im[int(ymin):int(ymax), int(xmin):int(xmax), :]
            save_to = os.path.join(output_dir, filename)            
            
            os.makedirs(os.path.split(save_to)[0], exist_ok=True)
            cv2.imwrite(save_to, cropped_image)
            print('Saving ', save_to)
            #cv2.imshow('s', im)
            #cv2.waitKey(0)
            #cv2.imshow('dd', cropped_image)
            #cv2.waitKey(0)

            #face_image = cropped_image.save(os.path.join(output_dir, group.filename))
        
    print("successfully get csv list")
    return files

def main(_):
    pass

read_bbox('/home/max/Downloads/cosine_metric_learning-master/datasets/Youtube_faces/val_csv_bbox/', '/home/max/Downloads/cosine_metric_learning-master/datasets/Youtube_faces/val_cropped_image/')

        

