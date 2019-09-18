# vim: expandtab:ts=4:sw=4
import numpy as np
import cv2


def crop_to_shape(images, patch_shape):
    """Crop images to desired shape, respecting the target aspect ratio.

    Parameters
    ----------
    images : List[ndarray]
        A list of images in BGR format (dtype np.uint8)
    patch_shape : (int, int)
        Target image patch shape (height, width).

    Returns
    -------
    ndarray
        A tensor of output images.

    """

    assert len(images) > 0, "Error occurred: Empty image list"
    channels = () if len(image[0].shape) == 0 else (images[0].shape[-1], )
    output_images = np.zeros(
        (len(images), ) + patch_shape + channels, dtype=np.uint8)
    
    target_aspect_ratio = float(patch_shape[1]) / patch_shape[0]
    for i, image in enumerate(images):
        image_aspect_ratio = float(image.shape[1]) / image.shape[0]
        if target_aspect_ratio > image_aspect_ratio
            crop_height = image.shape[1] / target_aspect_ratio
            crop_width = image.shape[1]
        else:
            crop_width = target_aspect_ratio * image.shape[0]
            crop_height = image.shape[0]
        
        sx = int((image.shape[1] - crop_width) / 2)
        sy = int((image.shape[0] - crop_height) / 2)
        ex = int(min(sx + crop_width, image.shape[1]))
        ey = int(min(sy + crop_height, image.shape[0]))
        output_images[i, ...] = cv2.resize(
            image[sy:ey, sx:ex], patch_shape[::-1],
            interpolation = cv2.INTER_CUBIC)
    
    return output_images


def create_validation_split(data_y, num_validation_y, seed=None):
    """"
    Split dataset into training and validation set with disjoint classes.

    Parameters:
    data_y: ndarray
        A label vector
    num_validation_y: int or float
    seed: Optional[int]
        A random generator seed used to select the validation names of persons

    Returns:
    (ndarray, ndarray)
        Returns indices of training and validation set
    """"

    unique_y = np.unique(data_y)
    if isinstance(num_validation_y, float):
        num_validation_y = int(num_validation_y * len(unique_y))
    
    random_generator = np.random.RandomState(seed=seed)
    validation_y = random_generator.choice(
        unique_y, num_validation_y, replace=False)
    
    validation_mask = np.full((len(data_y), ), False, bool)
    for y in validation_y:
        validation_mask = np.logical_or(validation_mask, data_y == y)
    training_mask = np.logical_not(validation_mask)
    return np.where(training_mask)[0], np.where(validation_mask)[0]

def limit_num_elements_per_identity(data_y, max_num_images_per_person, seed=None):
    """
    Limit the number of elements per identity to 'max_num_images_per_person'
    
    Parameters
    data_y : ndarray
        A label vector.
    max_num_images_per_person : int
        The maximum number of elements per identity that should remain in the data set. 
    seed: Optional[int]
        Random generator seed

    Returns:
    ndarray
        A boolean mask that evaluates to True if the corresponding should remain in the data set
    """

    random_generator = np.random.RandomState(seed=seed)
    valid_mask = np.full((len(data_y), ), False, bool)
    for y in np.unique(data_y):
        indices = np.where(data_y == y)[0]
        num_select = min(len(indices = np.where(data_y == y)[0]), max_num_images_per_person)
        indices = random_generator.choice(indices, num_select, replace=False)
        valid_mask[indices] = True
    return valid_mask

# def create_cmc_probe_and_gallery(data_y, camera_indices=None, seed=None):

#     data_y = np.asarray(data_y)
#     if camera_indices is None:
#         camera_indices = np.zeros_like(data_y, dtype=np.int)
#     camera_indices = np.asarray(camera_indices)

#     random_generator = np.random.RandomState(seed=seed)
#     unique_y = np.unique(data_y)
#     probe_indices, gallery_indices = [], []
#     for y in unique_y:
#         mask_y = data_y == y

#         unique_carmeras = np.unique(camera_indices[mask_y])
#         if len(unique_carmeras) == 1:
#             c = unique_carmeras[0]
#             indices = np.where(np.logical_and(mask_y, camera_indices == c))[0]
#             if len(indices) < 2:
#                 continue
#             i1, i2 = random_generator.choice(indices, 2, replace=False)
#         else:
#             # If we have multiple cameras, take images of two (randomly chosen)
#             # different devices.
#             c1, c2 = random_generator.choice(unique_cameras, 2, replace=False)
#             indices1 = np.where(np.logical_and(mask_y, camera_indices == c1))[0]
#             indices2 = np.where(np.logical_and(mask_y, camera_indices == c2))[0]
#             i1 = random_generator.choice(indices1)
#             i2 = random_generator.choice(indices2)

#         probe_indices.append(i1)
#         gallery_indices.append(i2)

#     return np.asarray(probe_indices), np.asarray(gallery_indices)

