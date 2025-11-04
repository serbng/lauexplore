from typing import Iterable
from pathlib import Path
import fabio
import numpy as np
import matplotlib.pyplot as plt
import multiprocess as mp

from .._utils import linear_chunks # for mosaic
from ..visuals._utils import draw_colorbar

def display_image(data, roi=None, **kwargs):
    """Plot an image.
    
    Parameters
    ----------
    image_path        (str): Full path to the image
    ROI        (tuple[int]): Subset of pixels inside the image to plot. The format is:
                             (x_position, y_position, x_boxsize, y_boxsize)
    
    Keyword arguments
    ----------
    kwargs: passed to matplotlib.pyplot.imshow
    """
    if isinstance(data, str):
        with fabio.open(data) as f:
            image_data = f.data
    elif isinstance(data, np.ndarray):
        image_data = data
    else:
        raise TypeError(
            "data type must be in {str, np.ndarray}." + f" Got {type(data)}"
        )
    
    if roi is not None:
        y1 = roi[0] - (roi[2] // 2)
        y2 = roi[0] + (roi[2] // 2)
        x1 = roi[1] - (roi[3] // 2)
        x2 = roi[1] + (roi[3] // 2)
                
        image_data = image_data[x1:x2, y1:y2]
    
    ax = plt.gca()
    image = ax.imshow(image_data, **kwargs)
    draw_colorbar(image)
    ax.set_xlabel('X pixel')
    ax.set_ylabel('Y pixel')
    ax.set_aspect('equal')

