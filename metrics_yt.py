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

# Orginal code has other two functions defined below 
"""
def recognition_rate_at_k(probe_x, probe_y, gallery_x, gallery_y,
                        k, measure=pdist):
# This function defined for calculating the CMC curve, which compute 
    recognition rate at a given level 'k'

def streaming_mean_cmc_at_k(probe_x, probe_y, gallery_x, gallery_y, 
                                k, measure=pdist):
# Compute cumulated matching charateristics (CMC) at level 'k' over 
    a stream of data (ie, multiple batches)
"""

def streaming_mean_average_precision(probe_x, probe_y, gallery_x, gallery_y,
                                    good_mask, measure=pdist):
    """Compute mean average precision (mAP) over a stream of data
    probe_x: tf.Tensor
        A tensor of N probe images.
    probe_y: tf.Tensor
        A tensor of N probe labels.
    gallery_x: tf.Tensor
        A tensor of M gallery images.
    gallery_y: tf.Tensor
        A tensor of M gallery labels
    measure: Callable[tf.Tensor, tf.Tensor] -> tf.Tensor
        A callable that computes for two matrices of row-vectors a matrix of
        element-wise distances. See `pdist` for an example.

    Returns
    -------
    Tuple[tf.Tensor, tf.Tensor]
        The first element in the tuple is the current result. The second element
        is an operation that updates the computed metric based on new data.
    """
    if good_mask.dtype != tf.float32:
        good_mask = tf.cast(good_mask, tf.float32)
    
    # Compute similarity measure and mask out diagonal (similarity to self)
    predictions = good_mask * tf.exp(-measure(probe_x, gallery_x))

    # Compute matrix of predicted labels
    k = tf.shape(gallery_y)[0]
    _, prediction_indices = tf.nn.top_k(predictions, k=k)
    predicted_label_mat = tf.gather(gallery_y, prediction_indices)
    label_eq_mat = tf.cast(tf.equal(
        predicted_label_mat, tf.reshape(probe_y, (-1,1))), tf.float32)
    
    # Computer statistics
    num_relevant = tf.reduce_sum(
        good_mask * label_eq_mat, reduction_indices=[1], keep_dims=True)
    true_positives_at_k = tf.cumsum(label_eq_mat, axis=1)
    retrieved_at_k = tf.cumsum(tf.ones_like(label_eq_mat), axis=1)
    precision_at_k = true_positives_at_k / retrieved_at_k
    relevant_at_k = label_eq_mat
    average_precision = (
        tf.reduce_sum(precision_at_k * relevant_at_k, reduction_indices=[1]) /
        tf.cast(tf.squeeze(num_relevant), tf.float32))       
    return slim.metrics.streaming_mean(average_precision)
    
    