import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import ScalarFormatter

def cbarticks_zero_centered(vmin, vmax): 
    """
    Generate a TwoSlopeNorm normalization (centered at zero) and corresponding ticks for the colorbar. 
    If vmin and vmax have the same sign, the colorbar will be onesided.
    
    Parameters
    ----------
    vmin [float]: Minimum value for the data.
    vmax [float]: Maximum value for the data.
        
    Returns
    -------
    norm   [TwoSlopeNorm]: Normalization object with center zero.
    cbar_ticks [np.array]: Array of ticks for the colorbar.
    """
    if vmax < vmin:
        raise ValueError("vmin and vmax must be in ascending order")
    
    factors = [0.25, 0.5, 0.75, 1]
    positive_ticks = np.array([vmax*f for f in factors])
    negative_ticks = np.array([vmin*f for f in factors[::-1]])
    
    # If vmin and vmax have the same sign, adjust the onesided colorbar
    if vmin >= 0:
        vmin = -1e-8
        negative_ticks = np.array([])
    if vmax <= 0:
        vmax = +1e-8
        positive_ticks = np.array([])
        
    norm = TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)
    cbar_ticks = np.concatenate((negative_ticks, 0, positive_ticks), axis=None)
    
    return norm, cbar_ticks

def cbarticks_scientific():
    """
    Return proper format to display the colorbar ticks with scientific notation

    Example:
    ...
    fig, ax = plt.subplots()
    image = ax.imshow(data)
    format = cbarticks_scientific()
    plt.colorbar(image, format=format)
    # Or
    # draw_colorbar(image, format=format)
    """
    format = ScalarFormatter(useMathText=True)
    format.set_powerlimits((0, 0))
    return format # to use in format **kwarg of plt.colorbar()

def draw_colorbar(image, width=5, pad=0.1, **kwargs):
    """
    Make colorbars with the same width for a given image
    
    Parameters
    -----------
    image  (matplotlib.cm.ScalarMappable): Ex.: AxesImage, ContourSet, QuadMesh... 
                                           Object returned by any matplotlib plotting function
                                           that supports a colorbar
    axis           (matplotlib.axes.Axes): where the plot is located
    
    Keyword arguments
    -----------
    scientific  (bool): Format ticks using scientific notation
    orientation (str)
    """
    # scientific = kwargs.pop('scientific', None)
    # if scientific:
    #     format=cbarticks_scientific()
    
    divider = make_axes_locatable(image.axes)
    cbar_ax = divider.append_axes('right', f'{width}%', pad = pad)
    orient  = kwargs.pop('orientation', 'vertical')
   
    plt.colorbar(image, cax=cbar_ax, orientation=orient, **kwargs)