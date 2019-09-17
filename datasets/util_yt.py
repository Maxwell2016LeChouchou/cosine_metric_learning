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
        image_aspect_ratio = float(image.shape[1] / image.shape[0])