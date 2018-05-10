try:
    from cStringIO import StringIO as BytesIO
except: # Python 3
    from io import BytesIO
import numpy as np
import PIL.Image
import shutil
from math import sqrt

def save_image_array(img, filename, rescale=True, fmt="png"):
    """
    Expects a 4D image array of (n_images, height, width, channels)

    rescale will rescale 1 channel images to the maximum value available

    Modified from implementation by Kyle McDonald

    https://github.com/kylemcdonald/python-utils/blob/master/show_array.py
    """

    if len(img.shape) != 4:
       raise ValueError("Expects a 4D image array of (n_images, height, width, channels)")

    if img.shape[0] != 1:
        n = len(img)
        side = int(sqrt(n))
        shp = img.shape
        if (side * side) == n:
            pass
        else:
            raise ValueError("Need input length that can be reshaped to a square (4, 16, 25, 36, etc)")
        n,h,w,c = img.shape
        img = img.reshape(side, side, h, w, c).swapaxes(1, 2).reshape(side*h, side*w, c)
    else:
        img = img[0]

    if rescale:
        img_max = np.max(img)
        img_min = np.min(img)
        #scale to 0, 1
        img = (img - img_min) / float(img_max - img_min)
        # scale 0, 1 to 0, 255
        img *= 255.

    if img.shape[-1] == 1:
       img = img[:, :, 0]

    img = np.uint8(np.clip(img, 0, 255))
    image_data = BytesIO()
    PIL.Image.fromarray(img).save(image_data, fmt)
    with open(filename, 'wb') as f:
        image_data.seek(0)
        shutil.copyfileobj(image_data, f)
