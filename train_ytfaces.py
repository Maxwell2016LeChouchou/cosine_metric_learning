import functools
import os
import numpy as np 
import scipy.io as sio 
import train_app
from datasets import youtube_faces
from datasets import util
import nets.deep_sort.network_definition as net 


class Youtube_faces(object):
    
    def __init__(self, dataset_dir, num_validation_y=0.1, seed=1234):
        self._dataset_dir = dataset_dir
        self._num_validation_y = num_validation_y
        self._seed = seed
    
    def read_train(self):
        filenames, ids, camera_indices = youtube_faces.read_train_split_to_image(
            self._dataset_dir)
        train_indices, _ = util.create_validation_split(
            np.asarray(ids, np.int64), self._num_validation_y, self._seed)
        
        # filenames = [filenames[i] for i in train_indices]
        # ids = [ids[i] for i in train_indices]
        # camera_indices = [camera_indices[i] for i in train_indices]

        filenames = np.array(filenames)[train_indices]
        ids = np.array(ids)[train_indices]
        camera_indices = np.array(camera_indices)[train_indices]
        return filenames, ids, camera_indices
    
    def read_validation(self):
        filenames, ids, camera_indices, = youtube_faces.read_train_split_to_image(
            self._dataset_dir)
        _, valid_indices = util.create_validation_split(
            np.asarray(ids, np.int64), self._num_validation_y, self._seed)

        # filenames = [filenames[i] for i in valid_indices]
        # ids = [ids[i] for i in valid_indices]
        # camera_indices = [camera_indices[i] for i in valid_indices]

        filenames = np.array(filenames)[valid_indices]
        ids = np.array(ids)[valid_indices]
        camera_indices = np.array(camera_indices)[valid_indices]

        return filenames, ids, camera_indices

    # def read_test_filenames(self):
    #     filename = os.path.join(self._dataset_dir, "info", "test_name.txt")
    #     with open(filename, "r") as file_handle:
    #         content = file_handle.read()
    #         lines = content.splitlines()

    #     image_dir = os.path.join(self._dataset_dir, "bbox_test")
    #     return [os.path.join(image_dir, f[:4], f) for f in lines]

def main():
    arg_parser = train_app.create_default_argument_parser("youtube_faces")
    arg_parser.add_argument(
        "--dataset_dir", help="path to youtube_faces dataset directory.",
        default="/home/maxwell/Desktop/yt_test_data/")
    args = arg_parser.parse_args()
    dataset = Youtube_faces(args.dataset_dir, num_validation_y=0.1, seed=1234)

    if args.mode == "train":
        train_x, train_y, _ = dataset.read_train()
        print("Train set size: %d images, %d persons" % (
            len(train_x), len(np.unique(train_y))
        ))

        network_factory = net.create_network_factory(
            is_training=True, num_classes=youtube_faces.MAX_LABEL + 1,
            add_logits=args.loss_mode=="cosine-softmax")
        train_kwargs = train_app.to_train_kwargs(args)
        train_app.train_loop(
            net.preprocess, network_factory, train_x, train_y,
            num_images_per_id=4, image_shape=youtube_faces.IMAGE_SHAPE, **train_kwargs)
    
    elif args.mode == "eval":
        valid_x, valid_y, camera_indices = dataset.read_validation()
        print("Validation set size: %d images, %d persons" %(
            len(valid_x), len(np.unique(valid_y))
        ))

        network_factory = net.create_network_factory(
            is_training=False, num_classes=youtube_faces.MAX_LABEL + 1,
            add_logits=args.loss_mode=="cosine-softmax")
        eval_kwargs = train_app.to_eval_kwargs(args)
        train_app.eval_loop(
            net.preprocess, network_factory, valid_x, valid_y, camera_indices,
            image_shape=youtube_faces.IMAGE_SHAPE, **eval_kwargs)

    elif args.mode == "export":
        raise NotImplementedError()
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
            is_training=False, num_classes=youtube_faces.MAX_LABEL + 1, 
            add_logits=False, reuse=None)
        train_app.finalize(
            functools.partial(net.preprocess, input_is_bgr=True),
            network_factory, args.restore_path, image_shape=IMAGE_SHAPE,
            output_filename="./youtube_faces.ckpt")
    elif args.mode == "freeze":
        network_factory = net.create_network_factory(
            is_training=False, num_classes=youtube_faces.MAX_LABEL + 1,
            add_logits=False, reuse=None)
        train_app.freeze(
            functools.partial(net.preprocess, input_is_bgr=True),
            network_factory, args.restore_path, image_shape=IMAGE_SHAPE,
            output_filename="./youtube_faces.ckpt")
    else:
        raise ValueError("Invalid mode argument.")

if __name__ == "__main__":
    main()
        


    



        

        

