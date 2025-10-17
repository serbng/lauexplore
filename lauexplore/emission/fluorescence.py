from pathlib import Path
from dataclasses import dataclass
import numpy as np
import h5py
import plotly.graph_objects as go

from ..scan import Scan
from .._parsers import _h5
from ._plots import fluoplot

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
    
    def plot(self, **kwargs) -> go.Figure:
        return fluoplot(self, **kwargs)