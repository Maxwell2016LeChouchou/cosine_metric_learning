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
    arg_parser = train_app.create_default_argument_parser("youtube_faces")
    arg_parser.add_argument(
        "--dataset_dir", help="path to youtube_faces dataset directory.",
        default="cosine_metric_learning/datasets")
    args = arg_parser.parse_args()
    dataset = youtube_faces(args.dataset_dir, num_validation_y=0.1, seed=1234)

    if args.mode == "train":
        train_x, train_y, _ = dataset.read_train()
        print("Train set size: %d images, %d persons" % (
            len(train_x), len(np.unique(train_y))
        ))

        network_factory = net.create_network_factory(
            is_training=True, num_classes=youtube_faces.MAX_LABEL + 1,
            add_logits=arg.loss_mode == "cosine-softmax")
        
        train_kwargs = train_app.to_train_kwargs(args)
        train_app.train_loop(
            net.preprocess, network_factory, train_x, train_y,
            num_images_per_person=10, image_shape=None, **train_kwargs)
    
    elif args.mode == "eval":
        valid_x, valid_y, yt_dir_file = dataset.read_validation()
        print("Validation set size: %d images, %d persons" %(
            len(train_x), len(np.unique(valid_y))
        ))

        network_factory = net.create_network_factory(
            is_training=False, num_classes=youtube_faces.MAX_LABEL + 1,
            add_logits=arg.loss_mode == "cosine-softmax")
        eval_kwargs = train_app.to_eval_kwargs(args)
        train_app.eval_loop(
            net.preprocess, network_factory, train_x, train_y, 
            num_images_per_person = 10, image_shape=None, **eval_kwargs)

    # elif args.mode == "export":
    #     # Export one specific model.
    #     gallery_filenames, _, query_filenames, _, _ = dataset.read_test()

    #     network_factory = net.create_network_factory(
    #         is_training=False, num_classes=market1501.MAX_LABEL + 1,
    #         add_logits=False, reuse=None)
    #     gallery_features = train_app.encode(
    #         net.preprocess, network_factory, args.restore_path,
    #         gallery_filenames, image_shape=market1501.IMAGE_SHAPE)
    #     sio.savemat(
    #         os.path.join(args.sdk_dir, "feat_test.mat"),
    #         {"features": gallery_features})

    #     network_factory = net.create_network_factory(
    #         is_training=False, num_classes=market1501.MAX_LABEL + 1,
    #         add_logits=False, reuse=True)
    #     query_features = train_app.encode(
    #         net.preprocess, network_factory, args.restore_path,
    #         query_filenames, image_shape=market1501.IMAGE_SHAPE)
    #     sio.savemat(
    #         os.path.join(args.sdk_dir, "feat_query.mat"),
    #         {"features": query_features})

    elif args.mode == "finalize":
        network_factory = net.create_network_factory(
            is_training=False, num_classes=youtubefaces.MAX_LABEL + 1,
        )


    



        

        

