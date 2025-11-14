from typing import Iterable, Literal
import warnings
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.express.colors import get_colorscale

from lauexplore.scan import Scan
from lauexplore.dataset import Dataset
from lauexplore import plots
from lauexplore.plots.base import _as_grid
from lauexplore.plots.composite import tiles
from lauexplore.plots.hovermenus import scan_hovermenu
from lauexplore._utils import compute_zlimits, nonlinear_colorscale


@dataclass
class Strain:
    """
    Container for deviatoric strain tensors with Plotly visualization support.

    Notes
    -----
    - If `scale` is provided, user-defined `zmin`, `zmax` and `zmid` 
      **will be ignored**.
    - z-limits are computed *per strain component*.
    - If zmid is available, a nonlinear colorscale adjustment is applied,
      preserving the full resolution of the original colormap.
    """

    tensors: np.ndarray
    scan_obj: Scan | None = None
    dataset: Dataset | None = None
    
 
    @property
    def scan(self):
        if self.scan_obj is not None:
            return self.scan_obj
        elif self.dataset.scan is not None:
            return self.dataset.scan
        else:
            raise ValueError("No scan available!")

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
        ref_frame: Literal["crystal", "sample"] = "crystal",
    ) -> "Strain":
        """
        Create a Strain container from a Dataset instance.
        """
        if ref_frame not in {"crystal", "sample"}:
            raise ValueError(f"Reference frame must be 'crystal' or 'sample'.")

        tensors = (
            dataset.deviatoric_strain_crystal_frame
            if ref_frame == "crystal"
            else dataset.deviatoric_strain_sample_frame
        )
        return cls(np.asarray(tensors), scan_obj=None, dataset=dataset)

    # -----------------------------------------------------------

    def plot(
        self,
        *,
        nrows: int = 2,
        ncols: int = 3,
        colorscale: str = "balance",
        width: int = 1100,
        height: int = 700,
        hspacing: float | None = None,
        vspacing: float | None = None,
        cbar_title: str = "× 1e-4",
        cbar_width: int = 20,
        cbar_padding: int = 0,
        subplot_titles: list[str] | None = None,
        mask: np.ndarray | None = None,
        multiplier: float = 1e4,
        title: str | None = None,
        zmin: float | None = None,
        zmax: float | None = None,
        zmid: float | None = None,
        scale: str | None = None,
    ) -> go.Figure:

        """
        Plot the six deviatoric strain components as heatmaps.

        Parameters
        ----------
        scale : str or None
            If provided, overrides any user-defined zmin, zmax, zmid.
            (i.e. these values will be ignored.)
        zmin, zmax, zmid : float or None
            Manual z-range settings (ignored if `scale` is used).

        Returns
        -------
        go.Figure
        """

        # ------------------------------------------------------
        # Validate tensors
        # ------------------------------------------------------
        
        tensors = np.asarray(self.tensors)
        if tensors.ndim != 3 or tensors.shape[1:] != (3, 3):
            raise ValueError("`tensors` must have shape (N, 3, 3).")
            
        if scale is not None and any(v is not None for v in (zmin, zmax, zmid)):
            warnings.warn(
                "You provided zmin/zmax/zmid, but `scale` was also specified. "
                "The values zmin, zmax and zmid will be ignored",

                UserWarning,
            )

        # Strain components
        components = [
            tensors[:, 0, 0],
            tensors[:, 1, 1],
            tensors[:, 2, 2],
            tensors[:, 0, 1],
            tensors[:, 0, 2],
            tensors[:, 1, 2],
        ]

        # Mask
        if mask is not None:
            components = [np.where(mask, c, np.nan) for c in components]

        # Convert to spatial grids
        grids = [
            plots._as_grid(c * multiplier, self.scan)
            for c in components
        ]

        # Titles
        default_titles = [
            r"$\varepsilon_{xx}$",
            r"$\varepsilon_{yy}$",
            r"$\varepsilon_{zz}$",
            r"$\varepsilon_{xy}$",
            r"$\varepsilon_{xz}$",
            r"$\varepsilon_{yz}$",
        ]
        subplot_titles = subplot_titles or default_titles

        # Hover
        customdata, hovertemplate = scan_hovermenu(self.scan)

        # Axes in µm
        x = self.scan.xpoints * 1e3
        y = self.scan.ypoints * 1e3

        # ------------------------------------------------------
        # Compute z-limits per component
        # ------------------------------------------------------
        zlimits = []
        if scale is not None:
            for comp in components:
                zmin_i, zmax_i, zmid_i = compute_zlimits(
                    comp, scale=scale, multiplier=multiplier
                )
                zlimits.append((zmin_i, zmax_i, zmid_i))
        else:
            zlimits = [(zmin, zmax, zmid)] * len(components)

        # ------------------------------------------------------
        # Build main figure using tiles()
        # ------------------------------------------------------
        fig = tiles(
            grids,
            x=x,
            y=y,
            nrows=nrows,
            ncols=ncols,
            colorscale=colorscale,
            width=width,
            height=height,
            hspacing=hspacing,
            vspacing=vspacing,
            customdata=customdata,
            hovertemplate=hovertemplate,
            cbar_title=cbar_title,
            cbar_width=cbar_width,
            cbar_padding=cbar_padding,
            subplot_titles=subplot_titles,
            mask=mask,
        )

        # Load base colorscale from Plotly Express
        base_colorscale = get_colorscale(colorscale)

        for trace, (zmin_i, zmax_i, zmid_i) in zip(fig.data, zlimits):

            if zmin_i is not None:
                trace.update(zmin=zmin_i)
            if zmax_i is not None:
                trace.update(zmax=zmax_i)

            if zmid_i is not None:
                shifted = nonlinear_colorscale(
                    base_colorscale,
                    zmin=zmin_i,
                    zmax=zmax_i,
                    zmid=zmid_i,
                )
                trace.update(colorscale=shifted)

        # ------------------------------------------------------
        # Final layout
        # ------------------------------------------------------
        fig.update_layout(
            width=width,
            height=height,
            title=title
           )

        return fig


    def plot4(
        self,
        *,
        nrows: int = 2,
        ncols: int = 2,
        colorscale: str = "balance",
        width: int = 1100,
        height: int = 700,
        hspacing: float | None = None,
        vspacing: float | None = None,
        cbar_title: str = "× 1e-4",
        cbar_width: int = 20,
        cbar_padding: int = 0,
        subplot_titles: list[str] | None = None,
        mask: np.ndarray | None = None,
        multiplier: float = 1e4,
        title: str | None = None,
        zmin: float | None = None,
        zmax: float | None = None,
        zmid: float | None = None,
        scale: str | None = None,
    ) -> go.Figure:
        if self.scan is None:
            raise ValueError("scan must not be None.")
        """
        Plot the six deviatoric strain components as heatmaps.

        Parameters
        ----------
        scale : str or None
            If provided, overrides any user-defined zmin, zmax, zmid.
            (i.e. these values will be ignored.)
        zmin, zmax, zmid : float or None
            Manual z-range settings (ignored if `scale` is used).

        Returns
        -------
        go.Figure
        """

        # ------------------------------------------------------
        # Validate tensors
        # ------------------------------------------------------
        tensors = np.asarray(self.tensors)
        if tensors.ndim != 3 or tensors.shape[1:] != (3, 3):
            raise ValueError("`tensors` must have shape (N, 3, 3).")
            
        if scale is not None and any(v is not None for v in (zmin, zmax, zmid)):
            warnings.warn(
                "You provided zmin/zmax/zmid, but `scale` was also specified. "
                "The values zmin, zmax and zmid will be ignored",

                UserWarning,
            )

        # Strain components
        components = [
            (tensors[:, 0, 0]+tensors[:, 1, 1,])/2,
            tensors[:, 2, 2],
            tensors[:, 0, 1],
            (tensors[:, 0, 2]+tensors[:, 1, 2])/2,
        ]

        # Mask
        if mask is not None:
            components = [np.where(mask, c, np.nan) for c in components]

        # Convert to spatial grids
        grids = [
            plots._as_grid(c * multiplier, self.scan)
            for c in components
        ]

        # Titles
        default_titles = [
            r"$\varepsilon_{(xx+yy)/2}$",
            r"$\varepsilon_{zz}$",
            r"$\varepsilon_{xy}$",
            r"$\varepsilon_{(xz+yz)/2}$",
        ]
        subplot_titles = subplot_titles or default_titles

        # Hover
        customdata, hovertemplate = scan_hovermenu(self.scan)

        # Axes in µm
        x = self.scan.xpoints * 1e3
        y = self.scan.ypoints * 1e3

        # ------------------------------------------------------
        # Compute z-limits per component
        # Note: if scale is provided → user zmin/zmax/zmid ignored!
        # ------------------------------------------------------
        zlimits = []
        if scale is not None:
            for comp in components:
                zmin_i, zmax_i, zmid_i = compute_zlimits(
                    comp, scale=scale, multiplier=multiplier
                )
                zlimits.append((zmin_i, zmax_i, zmid_i))
        else:
            zlimits = [(zmin, zmax, zmid)] * len(components)

        # ------------------------------------------------------
        # Build main figure using tiles()
        # ------------------------------------------------------
        fig = tiles(
            grids,
            x=x,
            y=y,
            nrows=nrows,
            ncols=ncols,
            colorscale=colorscale,
            width=width,
            height=height,
            hspacing=hspacing,
            vspacing=vspacing,
            customdata=customdata,
            hovertemplate=hovertemplate,
            cbar_title=cbar_title,
            cbar_width=cbar_width,
            cbar_padding=cbar_padding,
            subplot_titles=subplot_titles,
            mask=mask,
        )

        # Load base colorscale from Plotly Express
        base_colorscale = get_colorscale(colorscale)

        # ------------------------------------------------------
        # Apply zmin/zmax/zmid with nonlinear midpoint shift
        # ------------------------------------------------------
        for trace, (zmin_i, zmax_i, zmid_i) in zip(fig.data, zlimits):

            if zmin_i is not None:
                trace.update(zmin=zmin_i)
            if zmax_i is not None:
                trace.update(zmax=zmax_i)

            if zmid_i is not None:
                shifted = nonlinear_colorscale(
                    base_colorscale,
                    zmin=zmin_i,
                    zmax=zmax_i,
                    zmid=zmid_i,
                )
                trace.update(colorscale=shifted)

        # ------------------------------------------------------
        # Final layout
        # ------------------------------------------------------
        fig.update_layout(
            width=width,
            height=height,
            title=title
           )

        return fig





