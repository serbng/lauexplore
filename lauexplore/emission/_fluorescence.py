from pathlib import Path
from dataclasses import dataclass
import numpy as np
import h5py
import plotly.graph_objects as go

from lauexplore.scan import Scan
from lauexplore._parsers import _h5
from lauexplore._plots import (
    _as_grid,
    scan_hovermenu,
    plot_heatmap
)

@dataclass
class Fluorescence:
    data: np.ndarray
    material: str
    scan: Scan | None = None
    
    @classmethod
    def from_h5(cls, 
            filepath: str | Path, 
            material: str, 
            scan_number: int = 1, 
        ):
        with h5py.File(filepath) as h5f:
            data = _h5.get_fluo(h5f, material, scan_number)
            
        scan = Scan.from_h5(filepath, scan_number)
            
        return cls(data, material, scan)
    
    def plot(self,
            width: int = 600,
            height: int = 600,
            zmin: float | None = None,
            zmax: float | None = None,
            title: str | None = None,
            xlabel: str | None = None,
            ylabel: str | None = None,
            colorscale: str = "Viridis",
            log10: bool = False,
            cbartitle: str | None = None
        ) -> go.Figure:
        if self.scan is None:
            raise ValueError("scan must not be None.")
        
        z = _as_grid(self.data, self.scan)
        x = np.arange(self.scan.nbxpoints)
        y = np.arange(self.scan.nbypoints)
        
        customdata, hovertemplate = scan_hovermenu(self.scan)
        
        fluoplot = plot_heatmap(
            z, x, y,
            customdata=customdata,
            hovertemplate=hovertemplate,
            width = width,
            height = height,
            zmin = zmin,
            zmax = zmax,
            title = title or f"Fluorescence plot – {self.material} ({self.scan.nbxpoints} × {self.scan.nbxpoints})",
            xlabel = xlabel,
            ylabel = ylabel,
            colorscale = colorscale,
            log10 = log10,
            cbartitle = cbartitle,
        )
        
        return fluoplot