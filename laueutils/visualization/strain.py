import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.optimize import curve_fit
#from ..classes import FitFileSeries
from ..utils import tabular_data
from ._utils import (draw_colorbar, 
                     cbarticks_zero_centered, 
                     cbarticks_scientific,
                     limits_meanNstd)

def __gaussian__(x, A, mu, sigma):
    return A * np.exp(- (x - mu)**2 / (2 * sigma**2))

def strain_map(x, y, strain_tensors, **kwargs):
    """
    Plot six strain component maps (exx eyy, ezz, eₓxy, exz, and eyz) using a list of 3×3 strain tensors.
    
    Parameters
    ----------
    x                  [np.ndarray]: x-coordinate values (2D grid NxM).
    y                  [np.ndarray]: y-coordinate values (2D grid NxM).
    strain_tensors list[np.ndarray]: list of 3×3 strain tensors corresponding to each grid point.
    
    Keyword Arguments
    -----------------
    multiplier            [float]: Factor to multiply each strain component.  Default 1.0.
    reshape_order {'C', 'F', 'A'}: Order for numpy.reshape to match the grid shape. Default 'C'.
        scale {'mean[sigma_value]sigma', 'uniform', 'default'}: default 'default'
            Color scaling mode:
              - 'mean[sigma_value]sigma': Set limits as mean ± (sigma_value × std). E.g., 'mean3sigma'.
              - 'uniform': Apply same limits to all components, computed as ±max(|component|) across all components.
    cbar_width            [float]: Width of the colorbar, in percentage with respect to the axes width. Default 5
    cbar_norm              [bool]: If True, center the colorbar at zero. Default False
    ncols                   [int]: Number of columns in the subplots
    nrows                   [int]: Number of rows in the subplots
    figsize       tuple[int, int]: Size in inches of the figure
        
    Additional kwargs are passed to matplotlib.pyplot.pcolormesh.
    
    Returns
    -------
    fig   [matplotlib.figure.Figure]: The figure object containing the subplots.
    axes                [np.ndarray]: Flattened array of subplot axes.
    """
    # Extract custom keyword arguments.
    multiplier     = kwargs.pop('multiplier', 1.0)
    reshape_order  = kwargs.pop('reshape_order', 'C')
    scale          = kwargs.pop('scale', 'default')
    cbar_width     = kwargs.pop('cbar_width', 5)
    cbar_norm      = kwargs.pop('cbar_norm', False)
    figsize        = kwargs.pop('figsize', (12,10))
    ncols          = kwargs.pop('ncols', 3)
    nrows          = kwargs.pop('nrows', 2)
    cbar_ticks     = None # if cbar_norm = True, this value will be overwritten
    
    # Ensure the number of strain tensors matches the grid size.
    npoints = len(strain_tensors)
    if npoints != x.size:
        raise ValueError("Length of strain_tensors does not match the total number of grid points in x/y.")
    
    # Define the strain component keys and their respective indices.
    components_index = {'xx': (0, 0), 'yy': (1, 1),
                        'zz': (2, 2), 'xy': (0, 1),
                        'xz': (0, 2), 'yz': (1, 2)}
    
    # Extract and reshape each strain component.
    components = {}
    for comp, index in components_index.items():
        components[comp] = np.array([tensor[index] * multiplier for tensor in strain_tensors])
        components[comp] = components[comp].reshape(x.shape, order=reshape_order) 
    

    if scale == 'uniform':
        global_max = max([np.nanmax(np.abs(component[comp])) for comp in components_index.keys()])
        vmin, vmax = (-global_max, global_max)

    # Create a figure with six subplots arranged in 2 rows x 3 columns.
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()
    
    # Loop over each strain component and generate a plot.
    for i, (component, value) in enumerate(components.items()):
        # Start with a copy of the remaining kwargs for pcolormesh.
        singleplot_kwargs = kwargs.copy()
        
        if scale == 'default':
            vmin = kwargs.get('vmin', np.nanmin(value))
            vmax = kwargs.get('vmax', np.nanmax(value))
            singleplot_kwargs.update({'vmin': vmin, 'vmax': vmax})
            
        if scale.startswith('mean') and scale.endswith('sigma'):
            try:
                std_mul = float(scale[len('mean'): -len('sigma')])
            except ValueError:
                std_mul = 3.0
            vmin, vmax = limits_meanNstd(value, std_mul)
            singleplot_kwargs.update({'vmin': vmin, 'vmax': vmax})
            
        # If cbar_norm is enabled, center the colorbar using a custom norm and put vmin and vmax to None
        if cbar_norm:
            # Determine normalization limits either from provided values or data.  
            norm, cbar_ticks = cbarticks_zero_centered(vmin, vmax)
            singleplot_kwargs['norm'] = norm
            singleplot_kwargs['vmin'] = None
            singleplot_kwargs['vmax'] = None
        
        # Create the pcolormesh plot.
        mesh = axes[i].pcolormesh(x, y, value, **singleplot_kwargs)
        axes[i].set_title(f"$\\varepsilon_{{{component}}}$")
        axes[i].set_aspect('equal')
                           
        draw_colorbar(mesh, width=cbar_width, ticks=cbar_ticks)
    
    fig.suptitle(f"Deviatoric strain components ($\\times 10^{{{-np.log10(multiplier):3.1f}}}$)")
    fig.tight_layout()
    return fig, axes



def strain4map(x, y, strain_tensors, **kwargs):

    """
    Plot four strain component maps from a list of 3×3 strain tensors:
      (εxx+εyy)/2, εzz, εxy, (εxz+εyz)/2.

    Parameters
    ----------
    x, y               [np.ndarray]: 2D coordinate grids (same shape), typically from np.meshgrid.
    strain_tensors list[np.ndarray]: list/array of 3×3 strain tensors corresponding to each grid point.

    Keyword Arguments
    -----------------
    multiplier            [float]: Factor to multiply each strain component. Default 1.0.
    reshape_order {'C','F','A'}: Order for numpy.reshape to match the grid shape. Default 'C'.
        scale {'mean[sigma_value]sigma', 'uniform', 'default'}: default 'default'
            Color scaling mode:
              - 'mean[sigma_value]sigma': Set limits as mean ± (sigma_value × std). E.g., 'mean3sigma'.
              - 'uniform': Apply same symmetric limits to all components (±global max).
    cbar_width            [float]: Width of the colorbar, in % w.r.t. axes width. Default 5.
    cbar_norm              [bool]: If True, center the colorbar at zero. Default False.
    ncols                   [int]: Number of columns in the subplots. Default 2.
    nrows                   [int]: Number of rows in the subplots. Default 2.
    figsize       tuple[int,int]: Figure size in inches. Default (5, 10).

    Additional kwargs are passed to matplotlib.pyplot.pcolormesh.

    Returns
    -------
    fig   [matplotlib.figure.Figure]: The figure object containing the subplots.
    axes                [np.ndarray]: Flattened array of subplot axes.
    """
    # ---- extract kwargs aligned with strain_map ----
    multiplier     = kwargs.pop('multiplier', 1.0)
    reshape_order  = kwargs.pop('reshape_order', 'C')
    scale          = kwargs.pop('scale', 'default')
    cbar_width     = kwargs.pop('cbar_width', 5)
    cbar_norm      = kwargs.pop('cbar_norm', False)
    figsize        = kwargs.pop('figsize', (5, 10))
    ncols          = kwargs.pop('ncols', 4)
    nrows          = kwargs.pop('nrows', 1)

    cbar_ticks     = None  # will be set by cbarticks_zero_centered if cbar_norm=True

    # ---- basic checks ----
    if x.shape != y.shape:
        raise ValueError("x and y must have the same 2D shape.")
    npoints = x.size

    strain = np.asarray(strain_tensors, dtype=float)
    if strain.ndim != 3 or strain.shape[1:] != (3, 3):
        raise ValueError("strain_tensors must have shape (N, 3, 3).")
    if strain.shape[0] != npoints:
        raise ValueError("Length of strain_tensors does not match the total number of grid points in x/y.")

    # ---- scale tensors ----
    strain = strain * multiplier

    # ---- components (flattened) ----
    exx = strain[:, 0, 0]
    eyy = strain[:, 1, 1]
    ezz = strain[:, 2, 2]
    exy = strain[:, 0, 1]
    exz = strain[:, 0, 2]
    eyz = strain[:, 1, 2]

    components = {
        r'$(\varepsilon_{xx}+\varepsilon_{yy})/2$': (exx + eyy) / 2.0,
        r'$\varepsilon_{zz}$':                      ezz,
        r'$\varepsilon_{xy}$':                      exy,
        r'$(\varepsilon_{xz}+\varepsilon_{yz})/2$': (exz + eyz) / 2.0,
    }

    # ---- reshape back to 2D ----
    components_2d = {k: v.reshape(x.shape, order=reshape_order) for k, v in components.items()}

    # ---- uniform scaling (global across all four maps) ----
    if scale == 'uniform':
        global_max = max(np.nanmax(np.abs(arr)) for arr in components_2d.values())
        vmin_u, vmax_u = -global_max, global_max

    # ---- figure ----
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True)
    axes = axes.flatten()

    for i, (label, data2d) in enumerate(components_2d.items()):
        singleplot_kwargs = kwargs.copy()

        if scale == 'default':
            vmin = kwargs.get('vmin', np.nanmin(data2d))
            vmax = kwargs.get('vmax', np.nanmax(data2d))
            singleplot_kwargs.update({'vmin': vmin, 'vmax': vmax})

        elif isinstance(scale, str) and scale.startswith('mean') and scale.endswith('sigma'):
            # parse "meanNsigma", e.g., "mean3sigma"
            try:
                std_mul = float(scale[len('mean'):-len('sigma')])
            except ValueError:
                std_mul = 3.0
            vmin, vmax = limits_meanNstd(data2d, std_mul)
            singleplot_kwargs.update({'vmin': vmin, 'vmax': vmax})

        elif scale == 'uniform':
            singleplot_kwargs.update({'vmin': vmin_u, 'vmax': vmax_u})

        # zero-centered normalization if requested (re-using your helper)
        if cbar_norm:
            norm, cbar_ticks = cbarticks_zero_centered(
                singleplot_kwargs.get('vmin', np.nanmin(data2d)),
                singleplot_kwargs.get('vmax', np.nanmax(data2d)),
            )
            singleplot_kwargs['norm'] = norm
            singleplot_kwargs['vmin'] = None
            singleplot_kwargs['vmax'] = None

        # ---- plot ----
        mesh = axes[i].pcolormesh(x, y, data2d, **singleplot_kwargs)
        axes[i].set_title(label)
        axes[i].set_aspect('equal')

        # your shared helper
        draw_colorbar(mesh, width=cbar_width, ticks=cbar_ticks)

        # optional axes labels (like in your 6-map version)
        if i % ncols == 0:
            axes[i].set_ylabel('Position [µm]')
        if i // ncols == nrows - 1:
            axes[i].set_xlabel('Position [µm]')

    # ---- title with multiplier power-of-10 like strain_map ----
    fig.suptitle(f"Deviatoric strain components ($\\times 10^{{{-np.log10(multiplier):.1f}}}$)")
    fig.tight_layout()
    return fig, axes

    
# def strain4map(
#     x, y, strain_tensors, *,
#     multiplier=1.0,
#     reshape_order='C',
#     scale='default',          # 'default' | 'uniform' | 'mean3sigma' (or meanNsigma)
#     cbar_norm=False,
#     cbar_width=5,
#     figsize=(5, 10),
#     ncols=2,
#     nrows=2,
#     **kwargs                  # passed to plt.pcolormesh
# ):
#     """
#     Plot four strain component maps from a list/array of 3×3 strain tensors:
#       (εxx+εyy)/2, εzz, εxy, (εxz+εyz)/2.

#     Parameters
#     ----------
#     x, y : np.ndarray
#         2D coordinate grids (same shape), typically from np.meshgrid.
#     strain_tensors : array-like, shape (N, 3, 3)
#         One 3×3 tensor per grid point (flattened order must match x.ravel()).
#     multiplier : float, default 1.0
#         Factor applied to all components (e.g., 1e4).
#     reshape_order : {'C','F','A'}, default 'C'
#         Order used for reshaping 1D components back to the 2D grid.
#     scale : str, default 'default'
#         - 'default' : independent vmin/vmax per subplot
#         - 'uniform' : same symmetric limits across all subplots
#         - 'meanNsigma' (e.g., 'mean3sigma'): vmin/vmax = mean ± N·std for each subplot
#     cbar_norm : bool, default False
#         If True, centers colormap at zero using TwoSlopeNorm.
#     cbar_width : float, default 5
#         Colorbar width as a percentage of axes width (depends on your draw_colorbar helper).
#     figsize : tuple, default (10, 8)
#         Figure size in inches.
#     ncols, nrows : int, default 2, 2
#         Subplot grid layout.

#     Returns
#     -------
#     fig : matplotlib.figure.Figure
#     axes : np.ndarray
#         Array of Axes with shape (nrows*ncols,).
#     """
#     # ---- basic checks ----
#     if x.shape != y.shape:
#         raise ValueError("x and y must have the same 2D shape.")
#     N = x.size

#     strain = np.asarray(strain_tensors, dtype=float)
#     if strain.ndim != 3 or strain.shape[1:] != (3, 3):
#         raise ValueError("strain_tensors must have shape (N, 3, 3).")
#     if strain.shape[0] != N:
#         raise ValueError("Number of tensors (N) must match x.size == y.size.")

#     # ---- scale and extract components ----
#     strain *= multiplier

#     exx = strain[:, 0, 0]
#     eyy = strain[:, 1, 1]
#     ezz = strain[:, 2, 2]
#     exy = strain[:, 0, 1]
#     exz = strain[:, 0, 2]
#     eyz = strain[:, 1, 2]

#     components = {
#         r'$(\varepsilon_{xx}+\varepsilon_{yy})/2$': (exx + eyy) / 2.0,
#         r'$\varepsilon_{zz}$':                      ezz,
#         r'$\varepsilon_{xy}$':                      exy,
#         r'$(\varepsilon_{xz}+\varepsilon_{yz})/2$': (exz + eyz) / 2.0,
#     }

#     # reshape back to 2D maps
#     comps2d = {k: v.reshape(x.shape, order=reshape_order) for k, v in components.items()}

#     # ---- global limits for 'uniform' ----
#     if scale == 'uniform':
#         global_max = max(np.nanmax(np.abs(a)) for a in comps2d.values())
#         vmin_u, vmax_u = -global_max, global_max

#     # ---- figure ----
#     fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=True, sharex=True)
#     axes = axes.flatten()

#     for i, (title, data2d) in enumerate(comps2d.items()):
#         single_kwargs = kwargs.copy()

#         if scale == 'default':
#             vmin = kwargs.get('vmin', np.nanmin(data2d))
#             vmax = kwargs.get('vmax', np.nanmax(data2d))
#             single_kwargs.update({'vmin': vmin, 'vmax': vmax})

#         elif isinstance(scale, str) and scale.startswith('mean') and scale.endswith('sigma'):
#             # parse "meanNsigma" (e.g., "mean3sigma")
#             try:
#                 std_mul = float(scale[len('mean'):-len('sigma')])
#             except ValueError:
#                 std_mul = 3.0
#             mean = np.nanmean(data2d)
#             std  = np.nanstd(data2d)
#             vmin, vmax = mean - std_mul*std, mean + std_mul*std
#             single_kwargs.update({'vmin': vmin, 'vmax': vmax})

#         elif scale == 'uniform':
#             single_kwargs.update({'vmin': vmin_u, 'vmax': vmax_u})

#         # zero-centered normalization if requested
#         if cbar_norm:
#             vmin_c = single_kwargs.get('vmin', np.nanmin(data2d))
#             vmax_c = single_kwargs.get('vmax', np.nanmax(data2d))
#             norm = TwoSlopeNorm(vmin=vmin_c, vcenter=0, vmax=vmax_c)
#             single_kwargs['norm'] = norm
#             single_kwargs['vmin'] = None
#             single_kwargs['vmax'] = None

#         mesh = axes[i].pcolormesh(x, y, data2d, **single_kwargs)
#         axes[i].set_title(title)
#         axes[i].set_aspect('equal')

#         # assumes your helper exists
#         draw_colorbar(mesh, width=cbar_width)

#         if i % ncols == 0:
#             axes[i].set_ylabel('Position [µm]')
#         if i // ncols == nrows - 1:
#             axes[i].set_xlabel('Position [µm]')

#     # nice multiplier exponent in title (works if multiplier is a power of 10)
#     try:
#         expo = -np.log10(multiplier)
#         title_scale = f"×10$^{{{expo:.1f}}}$"
#     except Exception:
#         title_scale = f"×({multiplier:g})"

#     fig.suptitle(f"Deviatoric strain {title_scale}")
#     fig.tight_layout()
#     return fig, axes
# def strain_histogram(ffs: FitFileSeries, 
#                      multiplier: float = 1e4,
#                      fit: bool = False, **kwargs):
#     """Plot the histogram of the six strain components εxx, εyy, εzz, εxy, εxz, εyz.
    
#     Parameters
#     ----------
#     ffs          FitFileSeries: Objected containing the information of a folder of parsed .fit files
#     multiplier           float: Value multiplied to each strain component. Defaults to 1e4
#     fit                   bool: Fit the histogram with a gaussian. The fit parameters will be printed
#                                 as the title of each subplot.
    
#     Keyword arguments
#     ----------
#     kwargs: passed to matplotlib.pyplot.hist   
#     """
    
#     normalized_strain = ffs.deviatoric_strain_crystal_frame * multiplier
#     voigt_indices     = [(0,0), (1,1), (2,2), (0,1), (0,2), (1,2)]
#     strain_components = [normalized_strain[:, index[0], index[1]] for index in voigt_indices]
    
#     xlabels = ['ε$_{xx}$', 'ε$_{yy}$', 'ε$_{zz}$', 'ε$_{xy}$', 'ε$_{xz}$', 'ε$_{yz}$']
    
#     fig, axes = plt.subplots(2, 3, figsize = (11, 7))
    
#     for axidx, data, xlabel in zip(np.ndindex(axes.shape), 
#                                    strain_components, 
#                                    xlabels):
        
#         counts, bins, patches = axes[axidx].hist(data, **kwargs)
        
#         if fit:       
#             # Compute the bins centers. Used when evaluating the fitting function (__gaussian__)
#             bin_centers = bins[:-1] + np.diff(bins)/2
#             # Compute the fit_params [A, mu, sigma], returned covariance is trashed   
#             fit_params, _ = curve_fit(__gaussian__, bin_centers, counts, p0 = (50, 0, 2))
            
#             # Plot result
#             xlims = axes[axidx].get_xlim()
#             xvals = np.linspace(xlims[0], xlims[1], 200)
            
#             axes[axidx].plot(xvals, __gaussian__(xvals, *fit_params), color = 'red', linewidth = 2)
#             axes[axidx].set_title(f'A = {fit_params[0]:.1f}, μ = {fit_params[1]:.2f}, σ = {fit_params[2]:.2f}')
    
#         axes[axidx].set_xlabel(xlabel)
#         if axidx[1] == 0:
#             axes[axidx].set_ylabel('Counts')
        
#     fig.tight_layout()


# def plot_strain_for_slice(ffs: FitFileSeries, sample_x: np.ndarray, slice_range: tuple):
#     """
#     Plots the mean strain values (εzz and (εxx + εyy)/2) per column 
#     for a given range of x positions.

#     Parameters
#     ----------
#     ffs : FitFileSeries
#         Object containing the information of a folder of parsed .fit files.
#     sample_x : np.ndarray
#         2D mesh of the x positions of the xech motor.
#     slice_range : tuple
#         Range of indices to slice the x positions and strain data.
    
#     Returns
#     -------
#     fig, ax : matplotlib figure and axis objects.
#     """
    
#     normalized_strain = ffs.deviatoric_strain_crystal_frame 
    
#     # Extracting strain components and reshaping
#     ε_xx = normalized_strain[:, 0, 0].reshape(sample_x.shape, order='F')
#     ε_yy = normalized_strain[:, 1, 1].reshape(sample_x.shape, order='F')
#     ε_zz = normalized_strain[:, 2, 2].reshape(sample_x.shape, order='F')

#     # Computing the required strain components
#     mean_in_plane_strain = (ε_xx + ε_yy) / 2  # Mean in-plane strain
#     strain_components = [mean_in_plane_strain, ε_zz]  # List of required strain components
    
#     # Compute mean strain components per column
#     strain_mean = [np.nanmean(component, axis=0) for component in strain_components]
    
#     # Extract x positions and strain values for the given slice range
#     x = sample_x[0, slice_range[0]:slice_range[1]]  # Taking first row since x positions are the same for each column
#     epsilon_xx_yy_vals = strain_mean[0][slice_range[0]:slice_range[1]]
#     epsilon_zz_vals = strain_mean[1][slice_range[0]:slice_range[1]]

#     # Plot experimental strain values
#     fig, ax = plt.subplots(figsize=(7, 7))
#     ax.scatter(x, epsilon_zz_vals * 100, label=r'Experimental $\epsilon_{zz}$', color='orange', linewidth=2)
#     ax.scatter(x, epsilon_xx_yy_vals * 100, label=r'Experimental $(\epsilon_{xx} + \epsilon_{yy}) / 2$', color='purple', linewidth=2)

#     # Print mean and standard deviation
#     print(f'Mean εzz: {np.mean(epsilon_zz_vals * 1e4):.2f}')
#     print(f'Std εzz: {np.std(epsilon_zz_vals * 100):.2f}')
#     print(f'Mean (εxx + εyy)/2: {np.mean(epsilon_xx_yy_vals * 1e4):.2f}')
#     print(f'Std (εxx + εyy)/2: {np.std(epsilon_xx_yy_vals * 100):.2f}')
    
#     # Axis labels and formatting
#     ax.set_xlabel('Position (µm)')
#     ax.set_ylabel('Strain Shell Region (%)')
#     ax.legend()
#     ax.grid()
    
#     plt.show()
    
#     return fig, ax
