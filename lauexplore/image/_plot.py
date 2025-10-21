from pathlib import Path
import numpy as np
import plotly.graph_objects as go

from lauexplore import plots
from lauexplore.image import read

def plot(
        image_data: str | Path | np.ndarray,
        width: int = 800,
        height: int = 800,
        zmin: float | None = 1000,
        zmax: float | None = None,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        colorscale: str = "gray",
        log10: bool = False,
        cbartitle: str | None = None
    ) -> go.Figure:
    
    if isinstance(image_data, (str, Path)):
        title = title or str(image_data)
        image_data = read(image_data)
    
    ny, nx = image_data.shape
    
    customdata, hovertemplate = plots.base_hovermenu(nx, ny)
    x = np.arange(nx)
    y = np.arange(ny)
    
    image_plot = plots.base.heatmap(
        image_data, x, y,
        customdata=customdata,
        hovertemplate=hovertemplate,
        width = width,
        height = height,
        zmin = zmin,
        zmax = zmax,
        title = title,
        xlabel = xlabel or "X pixel",
        ylabel = ylabel or "Y pixel",
        colorscale = colorscale,
        log10 = log10,
        cbartitle = cbartitle or "counts",
    )

    # Flip up-down yaxis
    image_plot.update_yaxes(autorange="reversed")
    
    return image_plot
