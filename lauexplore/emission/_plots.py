from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from lauexplore.emission.fluorescence import Fluorescence

def fluoplot(fluo: "Fluorescence",
    *,
    width : int = 600,
    height : int = 600,
    zmin: float | None = None,
    zmax: float | None = None,
    title: str | None = None,
    colorscale: str = "Viridis",
    log10: bool = False,
) -> go.Figure:
    """Create a Plotly figure of a fluorescence scan.

    Works with both 2D mesh scans and linear scans. If a `Scan` is not
    provided, falls back to plotting the raw array (1D scatter or 2D heatmap).

    Hover shows: (i, j) indices, (x, y) positions, and the linear `index`.

    Parameters
    ----------
    fluo : Fluorescence
        Fluorescence data container with `.data`, `.material`, and optional `.scan`.
    title : str, optional
        Figure title override.
    colorscale : str, default "Viridis"
        Plotly colorscale name.
    zmin, zmax : float, optional
        Color limits. If None, Plotly auto-scales.
    log10 : bool, default False
        If True, plot log10 of the intensity.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    scan = fluo.scan
    nbxpoints, nbypoints = scan.shape
    # shape of the transposed
    shapeT = tuple(reversed(scan.shape))
    
    data = np.full(shapeT, 0, dtype=int)

    # I use data.size just in case the scan is not finished
    # If it's not I won't assign values to the rest of the
    # array, and the counts will be 0
    for index in range(data.size):
        i, j = scan.index_to_ij(index)
        data[j,i] = fluo.data[index]
    
    z = np.log10(data) if log10 else data

    # Build customdata: [i, j, index, x, y]
    hoverdata = np.empty(shapeT + (5,), dtype=float)
    for j in range(nbypoints):
        for i in range(nbxpoints):
            index  = scan.ij_to_index(i, j)
            xx, yy = scan.ij_to_xy(i, j)
            hoverdata[j, i] = (i, j, index, xx, yy)

    hovertemplate = (
        "(i, j) = (%{customdata[0]}, %{customdata[1]})<br>"
        "(x, y) = (%{customdata[3]}, %{customdata[4]})<br>"
        "image index = %{customdata[2]}<br>"
        "value = %{z}<extra></extra>"
    )

    # Special handling for linear scans: plot as a line with the same hover fields
    if scan.is_linear:
        # Identify the varying axis
        if nbxpoints > 1 and nbypoints == 1:
            # Horizontal line
            cust = hoverdata[0, :, :]
        elif nbypoints > 1 and nbxpoints == 1:
            cust = hoverdata[:, 0, :]

    # 2D heatmap for mesh scans
    heat = go.Heatmap(
        z=z,
        colorscale=colorscale,
        zmin=zmin,
        zmax=zmax,
        customdata=hoverdata,
        hovertemplate=hovertemplate,
        colorbar=dict(title="Intensity"),
    )

    fig = go.Figure(heat)
    fig.update_layout(
        title=title or f"{fluo.material} — fluorescence map ({nbxpoints}×{nbypoints})",
        yaxis = dict(scaleanchor="x", scaleratio=1, constrain="domain"), # aspect ratio of 1
        xaxis = dict(constrain="domain"), # Limit plot to heatmap region
        width = width,
        height = height,
        hoverlabel=dict(font=dict(family="Andale Mono, monospace", size=12))
    )

    return fig