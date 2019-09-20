import tensorflow as tf 
import tensorflow.contrib.slim as slim


def pdist(a, b=None):
    """Compute element-wise squared distance between `a` and `b`.

    Parameters
    ----------
    a : tf.Tensor
        A matrix of shape NxL with N row-vectors of dimensionality L.
    b : tf.Tensor
        A matrix of shape MxL with M row-vectors of dimensionality L.

    Returns
    -------
    tf.Tensor
        A matrix of shape NxM where element (i, j) contains the squared
        distance between elements `a[i]` and `b[j]`.

    """
    sq_sum_a = tf.reduce_sum(tf.square(a), reduction_indices=[1])
    if b is None:
        return -2 * tf.matmul(a, tf.transpose(a)) + \
            tf.reshape(sq_sum_a, (-1, 1)) + tf.reshape(sq_sum_a, (1, -1))
            
    sq_sum_b = tf.reduce_sum(tf.square(b), reduction_indices=[1])
    return -2 * tf.matmul(a, tf.transpose(b)) + \
        tf.reshape(sq_sum_a, (-1, 1)) + tf.reshape(sq_sum_b, (1, -1))

def cosine_distance(a, b=None):
    """Compute element-wise cosine distance between `a` and `b`.

    Parameters
    ----------
    a : tf.Tensor
        A matrix of shape NxL with N row-vectors of dimensionality L.
    b : tf.Tensor
        A matrix of shape NxL with N row-vectors of dimensionality L.

    Returns
    -------
    tf.Tensor
        A matrix of shape NxM where element (i, j) contains the cosine distance
        between elements `a[i]` and `b[j]`.

    """
    a_normed = tf.nn.l2_normalize(a, dim=1)
    b_normed = a_normed if b is None else tf.nn.l2_normalize(b, dim=1)
    return (
        tf.constant(1.0, tf.float32) -
        tf.matmul(a_normed, tf.transpose(b_normed)))

