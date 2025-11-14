import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def clamp(x, lo, hi): 
    return max(lo, min(hi, x))

def _infer_paddings(
        nrows: int,
        ncols: int, 
        width: int,
        height: int,
    ) -> tuple[float, float]:

    """
    Infer (hspacing, vspacing) in figure fractions.
    Heuristics:
      - Linear stacks (N×1 or 1×N) use very small gaps.
      - For grids, start at 0.08 and shrink gaps if per-panel pixels are small.
    """

    # Linear column: tall stack
    if ncols == 1 and nrows > 1:
        v = 0.03 if nrows <= 6 else 0.02     # tiny vertical gaps
        h = 0.02                              # horizontal gap irrelevant but small
        return (h, v)

    # Linear row: wide strip
    if nrows == 1 and ncols > 1:
        if ncols <= 6:
            h = 0.04
        elif ncols <= 10:
            h = 0.03
        else:
            h = 0.02
        v = 0.02
        return (h, v)

    # 2D grid
    h = 0.08
    v = 0.08

    # Pixels available per panel (rough, margins ignored)
    per_w = width  / max(1, ncols)
    per_h = height / max(1, nrows)

    # If panels are small, reduce gaps so the data area grows
    if per_w < 220: h = 0.06
    if per_w < 160: h = 0.04
    if per_h < 220: v = 0.05
    if per_h < 160: v = 0.03

    # Keep within sane bounds
    h = clamp(h, 0.01, 0.15)
    v = clamp(v, 0.01, 0.15)
    return (h, v)

def tiles(
    data_list: list[np.ndarray],
    x: np.ndarray, 
    y: np.ndarray,
    *,
    nrows: int = 2,
    ncols: int = 3,
    colorscale: str = "balance",
    width: int = 1100,
    height: int = 700,
    customdata: np.ndarray | None = None,
    hovertemplate: str | None = None,
    hspacing: float | None = None,
    vspacing: float | None = None,
    cbar_title: str | None = None,
    cbar_width: int = 20,
    cbar_padding: int = 0,
    subplot_titles: list[str] | None = None,
    mask: np.ndarray | None = None,
    
) -> go.Figure:
    """
    Minimal version with per-subplot colorbars:
    - 6 heatmaps in an nrows × ncols grid (default 2×3)
    - Each subplot has its own natural z-limits and its own colorbar
    - 1:1 aspect in every subplot
    - Optional shared x/y so zooming one pans/zooms all
    - Each colorbar is positioned next to its subplot using the subplot's axis domains
    """

    # Components (εxx, εyy, εzz, εxy, εxz, εyz)
    
    n_panels = len(data_list)
    if nrows * ncols < n_panels:
        raise ValueError(f"Grid too small: need ≥ {n_panels} panels; got {nrows*ncols}.")

    if subplot_titles is None:
        subplot_titles = [f"subplots {i}" for i in range(n_panels)]
    
    if hspacing is None or vspacing is None:
        inf_h, inf_v = _infer_paddings(nrows, ncols, width, height)
        if hspacing is None:
            hspacing = inf_h
        if vspacing is None:
            vspacing = inf_v    
                  
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=subplot_titles[: nrows * ncols],
        horizontal_spacing=hspacing,
        vertical_spacing=vspacing,
    )
    
    subplot_indices = np.ndindex((nrows, ncols)) # returns a generator
    xaxes = [ax for ax in fig.select_xaxes()] # fetch all x axes from generator
    yaxes = [ax for ax in fig.select_yaxes()] # fetch all y axes from generator
    
    for idx, (r, c) in enumerate(subplot_indices):
        if idx >= n_panels:
            break
        
        # subplots are 1-indexed
        r += 1
        c += 1
        
        xax = xaxes[idx]
        yax = yaxes[idx]
        cbar_x  =  xax.domain[1] + cbar_padding
        cbar_y  = (yax.domain[1] + yax.domain[0]) * 0.5
        cbar_len =(yax.domain[1] - yax.domain[0]) * 0.95
        
        trace = go.Heatmap(
            z=data_list[idx],
            x=x, y=y,
            colorscale=colorscale,
            customdata=customdata,
            hovertemplate=hovertemplate,
            colorbar=dict(
                title=cbar_title, 
                thickness=cbar_width,
                x = cbar_x,
                y = cbar_y,
                len = cbar_len,
                xanchor = "left"
            )
        )
        
        fig.add_trace(trace, row=r, col=c)
        # Fetch current x axis, otherwise setting scaleanchor="x" will scale each
        # subplot with respect to the first x axis of the subplot
        anchor_id = xax._plotly_name.replace("axis", "")
        fig.update_yaxes(
            scaleanchor=anchor_id, # scale with respect to subplot xaxis
            scaleratio=1, # scale by a factor of 1, i.e. aspect_ratio="equal"
            constrain="domain", # plot restricts to where are values to plot
            matches="y", # zoom on all subplots
            row=r, col=c
        )
        fig.update_xaxes(
            constrain="domain",
            matches="x",
            row=r, col=c
        )
    
    fig.update_layout(
        width=width,
        height=height
    )

    return fig