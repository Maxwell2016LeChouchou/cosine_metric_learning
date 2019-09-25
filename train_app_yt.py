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
    


