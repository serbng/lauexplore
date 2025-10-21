import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from lauexplore.image import ROI

from lauexplore.scan import Scan


def scan_hovermenu(scan: Scan) -> tuple[np.ndarray, str]:
    nx, ny = scan.nbxpoints, scan.nbypoints
    customdata = np.empty((ny, nx, 5), dtype=float)
    for j in range(scan.nbypoints):
        for i in range(scan.nbxpoints):
            index = scan.ij_to_index(i, j)
            x, y  = scan.ij_to_xy(i,j)
            customdata[j, i] = (i, j, index, x, y)
            
    hovertemplate = (
        "(i, j) = (%{customdata[0]}, %{customdata[1]})<br>"
        "(x, y) = (%{customdata[3]}, %{customdata[4]})<br>"
        "image index = %{customdata[2]}<br>"
        "value = %{z}<extra></extra>"
    )
    
    if scan.is_linear:
        # Identify the varying axis
        if nx > 1 and ny == 1:
            # Horizontal line
            hoverdata = hoverdata[0, :, :]
        if ny > 1 and nx == 1:
            hoverdata = hoverdata[:, 0, :]
    
    return customdata, hovertemplate

def base_hovermenu(nx, ny) -> tuple[np.ndarray, str]:
    customdata = np.empty((ny, nx, 2), dtype=int)
    for j in range(ny):
        for i in range(nx):
            customdata[j, i] = (i, j)
            
    hovertemplate = (
        "(i, j) = (%{customdata[0]}, %{customdata[1]})<br>"
        "value = %{z}<extra></extra>"
    )
    
    if nx > 1 and ny == 1:
        # Horizontal line
        hoverdata = hoverdata[0, :, :]
    if ny > 1 and nx == 1:
        hoverdata = hoverdata[:, 0, :]
    
    return customdata, hovertemplate

def mosaic_hovermenu(scan: Scan, roi: "ROI") -> tuple[np.ndarray, str]:
    mosaic_nbxpoints = scan.nbxpoints * roi.xboxsize
    mosaic_nbypoints = scan.nbypoints * roi.yboxsize
    customdata = np.empty((mosaic_nbypoints, mosaic_nbxpoints, 9), dtype=float)
    for mosaic_j in range(mosaic_nbypoints):
        scan_j = mosaic_j // roi.yboxsize
        roi_j  = mosaic_j  % roi.yboxsize + roi.y1
        for mosaic_i in range(mosaic_nbxpoints):
            scan_i = mosaic_i // roi.xboxsize
            roi_i  = mosaic_i  % roi.xboxsize + roi.x1
            scan_index = scan.ij_to_index(scan_i, scan_j)
            scan_x, scan_y = scan.ij_to_xy(scan_i, scan_j)
            customdata[mosaic_j, mosaic_i] = (
                mosaic_i, mosaic_j,
                roi_i, roi_j,
                scan_i, scan_j,
                scan_index,
                scan_x, scan_y
            )
            
    if scan.is_linear:
        # Identify the varying axis
        if scan.nbxpoints > 1 and scan.nbypoints == 1:
            # Horizontal line
            hoverdata = hoverdata[0, :, :]
        if scan.nbypoints > 1 and scan.nbxpoints == 1:
            hoverdata = hoverdata[:, 0, :]
            
    hovertemplate = (
        "Mosaic coordinate = (%{customdata[0]}. %{customdata[1]})<br>"
        "Image coordinate = (%{customdata[2]}. %{customdata[3]})<br>"
        "(i, j) = (%{customdata[4]}, %{customdata[5]})<br>"
        "(x, y) = (%{customdata[7]}, %{customdata[8]})<br>"
        "image index = %{customdata[6]}<br>"
        "value = %{z}<extra></extra>"
    )

    return customdata, hovertemplate