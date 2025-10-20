from warnings import warn
import numpy as np
import plotly.graph_objects as go

from lauexplore.scan import Scan

def _as_grid(arr: np.ndarray, scan: Scan) -> np.ndarray:
    nx, ny = scan.nbxpoints, scan.nbypoints
    arr = np.asarray(arr)
    
    if arr.ndim == 2:
        if arr.shape != (ny, nx):
            raise ValueError(f"2D data must have shape (ny, nx) = ({ny}, {nx}). Got ({arr.shape}).")
        return arr
    
    if arr.ndim == 1:
        # Warn if the scan is complete and number of elements doesn't match
        scan_complete = getattr(scan, "is_complete", default=False)
        if arr.size < scan.length and scan_complete:
            warn(f"1D data length {arr.size} < scan length {scan.length}. Is it intentional?", RuntimeWarning)
            
        data = np.empty((ny, nx), dtype=arr.dtype)
        for index in range(scan.length):
            i, j = scan.index_to_ij(index)
            data[j, i] = arr[index]
        return data
    
    raise ValueError("Unsupported data dimensionality: expected 1D or 2D array.")
            
def plot_heatmap(
    z: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    width: int = 600,
    height: int = 600,
    zmin: float | None = None,
    zmax: float | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    colorscale: str = "Viridis",
    log10: bool = False,
    customdata: np.ndarray | None = None,
    hovertemplate: str | None = None,
    cbartitle: str | None = None
) -> go.Figure:
    
    z = np.log10(z) if log10 else z
    
    heatmap = go.Heatmap(
        z=z,
        x=x,
        y=y,
        colorscale=colorscale,
        zmin=zmin,
        zmax=zmax,
        customdata=customdata,
        hovertemplate=hovertemplate,
        colorbar=dict(title=cbartitle)
    )
    
    fig = go.Figure(heatmap)
    fig.update_layout(
        title = title,
        xaxis = dict(title=xlabel, constrain="domain"),
        yaxis = dict(title=ylabel, scaleanchor="x", scaleratio=1, constrain="domain"),
        width = width,
        height = height,
        hoverlabel=dict(font=dict(family="Andale Mono, monospace", size=12))
    )
    
    return fig