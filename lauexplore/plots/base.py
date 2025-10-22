from warnings import warn
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pandas import DataFrame
    
import numpy as np
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display

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
        scan_complete = getattr(scan, "is_complete", False)
        if arr.size < scan.length and scan_complete:
            warn(f"1D data length {arr.size} < scan length {scan.length}. Is it intentional?", RuntimeWarning)
            
        data = np.empty((ny, nx), dtype=arr.dtype)
        for index in range(scan.length):
            i, j = scan.index_to_ij(index)
            data[j, i] = arr[index]
        return data
    
    raise ValueError("Unsupported data dimensionality: expected 1D or 2D array.")
            
def heatmap(
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

def indexation(peaks: "DataFrame") -> go.Figure:
    # ================ Data to display in the hover menu ================
    # numpy.ndarray containing data for the tooltip
    # data will be Xexp, Yexp, h, k, l, intensity, energy, spot_index
    custom_data = np.hstack((
                peaks[['Xexp', 'Yexp', 'h','k','l','Intensity','Energy']].values,
                np.atleast_2d(peaks.index.values).T # column vector with spot_index values
            ))
    
    # ===================== How to display the menu =====================
    hover_template = (
            "(%{customdata[0]}, %{customdata[1]})<br>"
            "Miller indices: [%{customdata[2]}, %{customdata[3]}, %{customdata[4]}]<br>"
            "I = %{customdata[5]:8.2f} [cts]<br>"
            "E = %{customdata[6]:6.3f} [keV]<br>"
            "spot index = %{customdata[7]}"
        )
    
    # ========== Create the figure widget with the two plots ===========
    fig = go.FigureWidget(
        data=[
            go.Scatter(
                x=peaks['Xexp'], y=peaks['Yexp'],
                mode='markers',
                name='Experimental',
                hovertemplate=hover_template,
                customdata=custom_data
            ),
            go.Scatter(
                x=peaks.get('Xtheo', []), y=peaks.get('Ytheo', []),
                mode='markers',
                name='Theoretical',
                visible=False,
                marker=dict(symbol='star', size=10, color='rgba(0,0,0,0)', line=dict(color='red', width=2))
            )
        ],
        layout=go.Layout(
            title='Indexed peak positions',
            width=800,
            height=700,
            xaxis=dict(title='Xexp', range=[0,2018]),
            yaxis=dict(title='Yexp', range=[2018,0]),
            showlegend=True
        )
    )
    # Grab the references to the traces
    exp_scatter, theo_scatter = fig.data

    # =================== Create the navigation widgets ====================
    toggle_button = widgets.ToggleButton(
        value=False,
        description='Camera positions ↔ Scattering angles',
        tooltip='Switch between (X, Y) and (2θ, χ) plots',
        layout=widgets.Layout(width='250px')
    )
    show_theo = widgets.Checkbox(
        value=False,
        description='Show Theoretical spots',
        layout=widgets.Layout(width='400px')
    )

    # ======================== Callback definition =========================
    def on_change(*args):
        if not toggle_button.value:
            # Camera space plot
            exp_x, exp_y = 'Xexp','Yexp'
            theo_x, theo_y = 'Xtheo','Ytheo'
            xlabel, ylabel = 'x pixel','y pixel'
            xlim, ylim = [0, 2018], [2018, 0]
        else:
            # Angle space plot
            exp_x, exp_y = '2θexp','χexp'
            theo_x, theo_y = '2θtheo','χtheo'
            xlabel, ylabel = '2θ','χ'
            xlim, ylim = [40, 140], [-40, 40]
        
        with fig.batch_update():
            exp_scatter.x = peaks[exp_x]
            exp_scatter.y = peaks[exp_y]
            fig.layout.xaxis.title = xlabel
            fig.layout.yaxis.title = ylabel
            fig.layout.xaxis.range = xlim
            fig.layout.yaxis.range = ylim

            # handle theoretical overlay: only if user checked it AND both theo cols exist
            if show_theo.value and theo_x in peaks.columns and theo_y in peaks.columns:
                theo_scatter.x = peaks[theo_x]
                theo_scatter.y = peaks[theo_y]
                theo_scatter.visible = True
            else:
                theo_scatter.visible = False

    # ====================== Link callback to buttons ======================
    toggle_button.observe(on_change, names='value')
    show_theo.observe(on_change, names='value')

    # ============================== Display ===============================
    ui = widgets.HBox([toggle_button, show_theo])
    display(ui, fig)
