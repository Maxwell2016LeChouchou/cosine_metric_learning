import argparse
import numpy as numpy
import tensorflow as tf 
import tensorflow.contrib.slim as slim

from datasets import util
import queued_trainer
import metrics
import losses 

def create_default_argument_parser(dataset_name):
    """Create an argument parser with default arguments.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset. This value is used to set default directories.

    Returns
    -------
    argparse.ArgumentParser
        Returns an argument parser with default arguments.

    """   
    parse = argparse.ArgumentParser(
        description="Metric trainer  (%s)" % dataset_name)
    parser.add_argument(
        "--batch_size", help="Train batch size", default=128, type=int)
    parser.add_argument(
        "--learning_rate", help="Learning rate", default=1e-3, type=float)
    parser.add_argument(
        "--eval_log_dir",
        help="Evaluation log directory",
        default="/tmp/%s_evaldir" % dataset_name)
    parser.add_argument(
        "--number_of_steps", help="Number of train/eval steps."
        "If None given, run indefinitely", default=None, type=int)
    parser.add_argument(
        "--log_dir", help="Log and checkpoints directory.",
        default="/tmp/%s_logdir" % dataset_name)
    parser.add_argument(
        "--loss_mode", help="one of 'cosine-softmax', 'magnet', 'triplet'", 
        type=str, default="cosine-softmax")
    parser.add_argument(
        "--mode", help="one of 'train', eval', 'finalize', 'freeze'.",
        type=str, default="train")
    parser.add_argument(
        "--restore_path", help="If not None, resume training of a given"
        "check (mode 'train').", default=None)
    parser.add_argument(
        "--run_id", help="An optional run-id. If None given, "
        "a new one is created", type=str, default=None)
    return parser
    
    def to_train_kwargs(args):
        """Parse command-line training arguments.

        Parameters
        ----------
        args : argparse.Namespace
            Namespace of an argument parser that was created with
            create_default_argument_parser.

        Returns
        -------
        Dict[str, T]
            Returns a dictionary of named arguments to be passed on to
            train_loop.

        """
        kwargs_dict = {
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "log_dir": args.log_dir,
            "loss_mode": args.loss_mode,
            "number_of_steps": args.number_of_steps,
            "restore_path": args.restore_path,
            "run_id": args.run_id,
        }
        return kwargs_dict

    def to_eval_kwargs(args):
        """Parse command-line evaluation arguments.

        Parameters
        ----------
        args : argparse.Namespace
            Namespace of an argument parser that was created with
            create_default_argument_parser.

        Returns
        -------
        Dict[str, T]
            Returns a dictionary of named arguments to be passed on to
            eval_loop.

        """
        kwargs_dict = {
            "eval_log_dir": args.eval_log_dir,
            "log_dir": args.log_dir,
            "loss_mode": args.loss_mode,
            "run_id": args.run_id,
        }
        return kwargs_dict
    
    def train_loop(preprocess_fn, network_factory, train_x, train_y, 
                num_images_per_person, batch_size, log_dir, 
                image_shape=None, restore_path=None, exclude_from_restore=None,
                run_id=None, number_of_steps=None, 
                loss_mode="cosine-softmax", learning_rate=1e-3, 
                trainable_scopes=None, save_summaries_secs=60,
                save_interval_secs=300):

        """Start training.

        Parameters
        ----------
        preprocess_fn : Callable[tf.Tensor] -> tf.Tensor
            A callable that applies preprocessing to a given input image tensor of
            dtype tf.uint8 and returns a floating point representation (tf.float32).
        network_factory : Callable[tf.Tensor] -> (tf.Tensor, tf.Tensor)
            A callable that takes as argument a preprocessed input image of dtype
            tf.float32 and returns the feature representation as well as a logits
            tensors. The logits may be set to None if not required by the loss.
        train_x : List[str] | np.ndarray
            A list of image filenames or a tensor of images.
        train_y : List[int] | np.ndarray
            A list or one-dimensional array of labels for the images in `train_x`.
        num_images_per_id : int
            Sample `num_images_per_id` images for each label at each training
            iteration. The number of identities sampled at each iteration is
            computed as `batch_size / num_images_per_id`. The `batch_size` must be
            divisible by this number.
        batch_size : int
            The number of images at each training iteration.
        log_dir : str
            Used to construct the log and checkpoint directory. They are stored in
            `log_dir/run_id`.
        image_shape : Tuple[int, int, int] | NoneType
            Image shape (height, width, channels) or None. If None, `train_x` must
            be an array of images such that the shape can be queries from this
            variable.
        restore_path : Optional[str]
            If not None, resumes training from the given checkpoint file.
        exclude_from_restore : Optional[List[str]]
            An optional list of variable scopes to be used in conjunction with
            `restore_path`. If not None, variables in the given scopes are not
            restored from the checkpoint file.
        run_id : Optional[str]
            A string that identifies the training run; used to construct the
            log and checkpoint directory `log_dir/run_id`. If None, a random
            string is created.
        number_of_steps : Optional[int]
            The total number of training iterations. If None, training runs
            indefenitely.
        loss_mode : Optional[str]
            A string that identifies the loss function used for training; must be
            one of 'cosine-softmax', 'magnet', 'triplet'. This value defaults to
            'cosine-softmax'.
        learning_rate : Optional[float]
            Adam learning rate; defaults to 1e-3.
        trainable_scopes : Optional[List[str]]
            Optional list of variable scopes. If not None, only variables within the
            given scopes are trained. Otherwise all variables are trained.
        save_summaries_secs : Optional[int]
            Save training summaries every `save_summaries_secs` seconds to the
            log directory.
        save_interval_secs : Optional[int]
            Save checkpoints every `save_interval_secs` seconds to the log
            directory.
        """
        
        if image_shape is None:
            assert type(train_x) == np.ndarray
            image_shape = train_x.shape[1:]
        elif type(train_x) == np.ndarray:
            assert train_x.shape[1:] == image_shape
        read_from_file = type(train_x) != np.ndarray

        trainer, train_op = create_trainer(
            preprocess_fn, network_factory, read_from_file, image_shape, 
            batch_size, loss_mode, learning_rate=learning_rate, 
            trainable_scopes=trainable_scopes)
        feed_generator = queued_trainer.random_sample_identities_forever(
                        batch_size, num_images_per_person, train_x, train_y)
        
        variables_to_restore = slim.get_variables_to_restore(
            exclude=exclude_from_restore)
        trainer.run(
            feed_generator, train_op, log_dir, restore_path=restore_path, 
            variables_to_restore=variables_to_restore, run_id=run_id,
            )
        

    def create_trainer(preprocess_fn, network_factory, read_from_file, 
                        image_shape, batch_size, loss_mode, 
                        learning_rate=1e-3, trainable_scopes=None):
        """Create trainer.

        Parameters
        ----------
        preprocess_fn : Callable[tf.Tensor] -> tf.Tensor
            A callable that applies preprocessing to a given input image tensor of
            dtype tf.uint8 and returns a floating point representation (tf.float32).
        network_factory : Callable[tf.Tensor] -> (tf.Tensor, tf.Tensor)
            A callable that takes as argument a preprocessed input image of dtype
            tf.float32 and returns the feature representation as well as a logits
            tensors. The logits may be set to None if not required by the loss.
        read_from_file:
            Set to True if images are read from file. If False, the trainer expects
            input images as numpy arrays (i.e., data loading must be handled outside
            of the trainer).
        image_shape: Tuple[int, int, int]
            Image shape (height, width, channels).
        batch_size:
            Number of images per batch.
        loss_mode : str
            One of 'cosine-softmax', 'magnet', 'triplet'. If 'cosine-softmax', the
            logits tensor returned by the `network_factory` must not be None.
        learning_rate: float
            Adam learning rate; defauls to 1e-3.
        trainable_scopes: Optional[List[str]]
            Optional list of variable scopes. If not None, only variables within the
            given scopes are trained. Otherwise all variables are trained.

        Returns
        -------
        QueuedTrainer
            Returns a trainer object to be used for training and evaluating the
            given TensorFlow model.

        """        

        num_channels = image_shape[-1] if len(image_shape) == 3 else 1

        with tf.device("/cpu:0"):
            label_var = tf.placeholder(tf.int64, (None, ))

            if read_from_file:
                filename_var = tf.placeholder(tf.string, (None, ))
                image_var = tf.map_fn(
                    lambda x: tf.image.decode_jpeg(
                        tf.read_file(x), channels=num_channels),
                    filename_var, back_prop=False, dtype=tf.unit8)
                image_var = tf.image.resize_images(image_var, image_shape[:2])
                input_vars = [filename_var, label_var]
            else:
                image_var = tf.placeholder(tf.unit8, (None,) + image_shape)
                input_vars = [image_var, label_var]
            
            enqueue_vars = [
                tf.map_fn(
                    lambda x: preprocess_fn(x, is_training=True),
                    image_var, back_prop=False, dtype=tf.float32),
                label_vars]
            
        trainer = queued_trainer.QueuedTrainer(enqueue_vars, inputs_vars)
        image_var, label_var = Trainer.get_input_vars(batch_size)
        tf.summary.image("images", image_var)
    
    feature_var, logit_var = network_factory(image_var)
    _create_loss(feature_var, logit_var, label_var, mode=loss_mode)

    if trainable_scopes is None:
        variables_to_train = tf.trainable_variables()
    else:
        variables_to_train = []
        for scope in trainable_scopes:
            variables = tf.get_collection(
                tf.Graphkeys.TRAINABLE_VARIABLES, scope)
            variables_to_train.extend(variables)
    
    global_step = tf.train.get_or_create_global_step()

    loss_var = tf.losses.get_total_loss()
    train_op = slim.learning.create_train_op(
        loss_var, tf.train.AdamOptimizer(learning_rate=learning_rate),
        gloabl_step, summarize_gradients=False, 
        variables_to_train=variables_to_train)
    tf.summary.scalar("total_loss", loss_var)
    tf.summary.scalar("learning_rate", learning_rate)

    regularization_var = tf.reduce_sum(tf.losses.get_regularization_loss())
    tf.summary.scalar("weight_loss", regularization_var)
    return trainer, train_op

def eval_loop(preprocess_fn, network_factory, data_x, data_y, yt_dir_file,
                log_dir, eval_log_dir, image_shape=None, run_id=None,
                loss_mode="cosine-softmax", num_galleries=10, 
                random_seed=4321):
    """Evaluate a running training session using CMC metric averaged over
    `num_galleries` galleries where each gallery contains for every identity a
    randomly selected image-pair.

    A call to this function will block indefinitely, monitoring the
    `log_dir/run_id` for saved checkpoints. Then, creates summaries in
    `eval_log_dir/run_id` that can be monitored with tensorboard.

    Parameters
    ----------
    preprocess_fn : Callable[tf.Tensor] -> tf.Tensor
        A callable that applies preprocessing to a given input image tensor of
        dtype tf.uint8 and returns a floating point representation (tf.float32).
    network_factory : Callable[tf.Tensor] -> (tf.Tensor, tf.Tensor)
        A callable that takes as argument a preprocessed input image of dtype
        tf.float32 and returns the feature representation as well as a logits
        tensors. The logits may be set to None if not required by the loss.
    data_x : List[str] | np.ndarray
        A list of image filenames or a tensor of images.
    data_y : List[int] | np.ndarray
        A list or one-dimensional array of labels for the images in `data_x`.
    camera_indices: Optional[List[int] | np.ndarray]
    yt_dir_file: Optional[List[int]]
        A list or one-dimensinal array of youtube celebrity's faces in 
        different videos, frames in each video is one file in this argument
    log_dir: str
        Should be equivalent to the `log_dir` passed into `train_loop` of the
        training run to monitor.
    eval_log_dir:
        Used to construct the tensorboard log directory where metrics are
        summarized.
    image_shape : Tuple[int, int, int] | NoneType
        Image shape (height, width, channels) or None. If None, `train_x` must
        be an array of images such that the shape can be queries from this
        variable.
    run_id : str
        A string that identifies the training run; must be set to the same
        `run_id` passed into `train_loop`.
    loss_mode : Optional[str]
        A string that identifies the loss function used for training; must be
        one of 'cosine-softmax', 'magnet', 'triplet'. This value defaults to
        'cosine-softmax'.
    num_galleries: int
        The number of galleries to be constructed for evaluation of CMC
        metrics.
    random_seed: Optional[int]
        If not None, the NumPy random seed is fixed to this number; can be used
        to produce the same galleries over multiple runs.
    """

    if image_shape is None:
        # If image-shape is not set, train_x must be an image array. 
        # Here we query the image shape from the array of images.

        assert type(data_x) == np.ndarray
        image_shape = data_x.shape[1:]

    elif type(data_x) == np.ndarray:
        assert data_x.shape[1:] == image_shape
    read_from_file = type(data_x) != np.ndarray

    # # Create num_galleries random CMC galleries to average CMC top-k over.
    # probes, galleries = [], []
    # for i in range(num_galleries):
    #     probe_indices, gallery_indices = util.create_cmc_probe_and_gallery(
    #         data_y, camera_indices, seed=random_seed + i)
    #     probes.append(probe_indices)
    #     galleries.append(gallery_indices)
    # probes, galleries = np.asarray(probes), np.asarray(galleries)
            




    


