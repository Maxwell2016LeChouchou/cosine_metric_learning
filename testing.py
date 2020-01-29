
import os
import csv

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
    youtube_dic = row_csv2dict('/home/max/Desktop/yt_test_data/train_image_pairs.txt')
    train_dir = '/home/max/Desktop/yt_test_data/test_image/'

    for file_dir in sorted(os.listdir(directory)):        
        txt = os.path.join(directory, file_dir)
        
        dic_temp = {}
        for line in open(txt, "r"):            
            data = line.split(",")
            filename = data[0]
            image_path = os.path.join(train_dir,filename)
            file_info = filename.split("/")
            person = file_info[0]
            person_ids = youtube_dic[person]
            camera = file_info[1]
            if camera not in dic_temp:
                dic_temp[camera] = [[image_path, person_ids, camera]]
            else:
                if len(dic_temp[camera]) < 15:                                        
                    dic_temp[camera].append([image_path, person_ids, camera])
        # print(dic_temp)
        for k, v in dic_temp.items():                        
            for image_path, person_ids, camera in v:
                image_filenames.append(image_path)
                ids.append(person_ids)
                camera_indices.append(camera)
    # for img_file in image_filenames:
    #     print(img_file)

    return image_filenames, ids, camera_indices
    

def main(_):
    pass

read_train_directory_to_str('/home/max/Desktop/yt_test_data/test/')

