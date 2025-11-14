from pathlib import Path
from dataclasses import dataclass
from typing import Union
from warnings import warn
import numpy as np
import h5py


from lauexplore.constants.motors import AXIS_FROM_MOTOR
from lauexplore._defaults import MAX_DIGITS_SCAN_POS
from lauexplore._parsers import _h5

DirectionError = ValueError("Expected direction to be in {'horizontal', 'vertical'}. Is it correctly set?")

@dataclass
class Scan:
    xsize: float | int
    ysize: float | int
    nbxpoints: int
    nbypoints: int
    direction: str
    count_time: float | None = None
    
    @classmethod
    def from_title(cls, title: str):
        """Initialize Scan class using the title reported in the h5 file._h5
        Typical title:
        "amesh yech -6.16040 -6.12540 70 xech 26.80480 26.83980 70 0.15"

        Args:
            title (str): title of a standard laue scan, as reported in .h5 file.
        """
        (scan_type, 
         motor1,
         motor1_lim1,
         motor1_lim2,
         motor1_nbintervals,
         motor2,
         motor2_lim1,
         motor2_lim2,
         motor2_nbintervals,
         count_time) = title.split()
        
        axis1 = AXIS_FROM_MOTOR.get(motor1)
        axis2 = AXIS_FROM_MOTOR.get(motor2)
        
        motor1_lim1 = float(motor1_lim1)
        motor1_lim2 = float(motor1_lim2)
        motor2_lim1 = float(motor2_lim1)
        motor2_lim2 = float(motor2_lim2)
        
        motor1_nbintervals = int(motor1_nbintervals)
        motor2_nbintervals = int(motor2_nbintervals)
        
        count_time = float(count_time)
        
        if axis1 == "x" and axis2 == "y":
            direction = "horizontal"
            xsize = motor1_lim2 - motor1_lim1
            ysize = motor2_lim2 - motor2_lim1
            nbxpoints = motor1_nbintervals + 1
            nbypoints = motor2_nbintervals + 1

        elif axis1 == "y" and axis2 == "x":
            direction = "vertical"
            xsize = motor2_lim2 - motor2_lim1
            ysize = motor1_lim2 - motor1_lim1
            nbxpoints = motor2_nbintervals + 1
            nbypoints = motor1_nbintervals + 1

        else:
            raise ValueError(f"Unsupported motor combination: {motor1}, {motor2}")

            xsize = round(xsize, MAX_DIGITS_SCAN_POS)
            ysize = round(ysize, MAX_DIGITS_SCAN_POS)

        return cls(xsize, ysize, nbxpoints, nbypoints, direction, count_time)
            
    @classmethod
    def from_h5(cls, filepath: str | Path, scan_number: int = 1):
        with h5py.File(filepath) as h5f:
            scan = cls.from_title(_h5.get_title(h5f, scan_number))
            scan.end_reason = _h5.get_end_reason(h5f, scan_number)
            scan.start_time = _h5.get_start_time(h5f, scan_number)
            scan.end_time = _h5.get_end_time(h5f, scan_number)
            scan.duration = _h5.get_duration(h5f, scan_number)
            end_reason = _h5.get_end_reason(h5f, scan_number)
            scan.monitor_data = _h5.get_monitor(h5f, scan_number)
            
        scan.is_complete = True if end_reason == "SUCCESS" else False
        
        return scan
    
    @property
    def shape(self) -> tuple[int, int]:
        return (self.nbxpoints, self.nbypoints)
    
    @property
    def size(self) -> tuple[float, float]:
        return (self.xsize, self.ysize)
    
    @property
    def is_linear(self) -> bool:
        return self.nbxpoints == 1 or self.nbypoints == 1
    
    @property
    def length(self) -> int:
        return self.nbxpoints * self.nbypoints
    
    @property
    def xpoints(self) -> np.ndarray:
        return np.linspace(0, self.xsize, self.nbxpoints)
    
    @property
    def ypoints(self) -> np.ndarray:
        return np.linspace(0, self.ysize, self.nbypoints)
    
    @property
    def xstepsize(self) -> float:
        if self.is_linear and self.nbxpoints == 1:
            return 0.
        return round(self.xsize/(self.nbxpoints-1), MAX_DIGITS_SCAN_POS)
    
    @property
    def ystepsize(self) -> float:
        if self.is_linear and self.nbypoints == 1:
            return 0.
        return round(self.ysize/(self.nbypoints-1), MAX_DIGITS_SCAN_POS)
    
    @property
    def mesh(self) -> list[np.ndarray, np.ndarray]:
        return np.meshgrid(self.xpoints, self.ypoints)

    
    def index_to_ij(self, index: int) -> tuple[int, int]:
        if index < 0 or index > self.length:
            raise(IndexError(f"Expected index inside bounds (0, {self.length-1}). Got {index}."))
            
        if self.direction == "horizontal":
            i = int(index  % self.nbxpoints)
            j = int(index // self.nbxpoints)
            return i, j
        
        if self.direction == "vertical":
            i = int(index // self.nbypoints)
            j = int(index  % self.nbypoints)
            return i, j
        
        raise(DirectionError)
        
    def index_to_xy(self, index: int) -> tuple[float, float]:
        i, j = self.index_to_ij(index)
        x = round(i * self.xstepsize, MAX_DIGITS_SCAN_POS)
        y = round(j * self.ystepsize, MAX_DIGITS_SCAN_POS)
        
        return x, y
    
    def ij_to_index(self, i: int, j: int) -> int:
        if i < 0 or j < 0:
            raise(IndexError(f"Expected positive indices. Got ({i}, {j})."))
            
        if i > self.nbxpoints or j > self.nbypoints:
            raise(IndexError(f"Expected i < {self.nbxpoints} and j < {self.nbypoints}. Got ({i}, {j})."))
            
        if self.direction == "horizontal":
            return i + j * self.nbxpoints
        
        if self.direction == "vertical":
            return j + i * self.nbypoints
        
        raise DirectionError
        
    def ij_to_xy(self, i: int, j: int) -> tuple[float, float]:
        if i < 0 or j < 0:
            raise(IndexError(f"Expected positive indices. Got ({i}, {j})."))
            
        if i > self.nbxpoints or j > self.nbypoints:
            raise(IndexError(f"Expected i < {self.nbxpoints} and j < {self.nbypoints}. Got ({i}, {j})."))
            
        x = round(i * self.xstepsize, MAX_DIGITS_SCAN_POS)
        y = round(j * self.ystepsize, MAX_DIGITS_SCAN_POS)
        
        return x, y
        
    def xy_to_index(self, x: float, y: float) -> int:
        i, j = self.xy_to_ij(x, y)
        return self.ij_to_index(i, j)
    
    def xy_to_ij(self, x: float, y: float) -> tuple[int, int]:
        if x < 0 or y < 0:
            raise(ValueError(f"Expected positive positions. Got ({x}, {y})."))
            
        if x > self.xsize or y > self.ysize:
            raise(ValueError(f"Expected x < {self.xsize} and y < {self.ysize}. Got ({x}, {y})."))
        
        # Is x or y close to an actual scan points?
        if np.abs(self.xpoints - x).min() > 0.1 * self.xstepsize:
            warn(f"{x} is more than 10% away from a valid x coordinate. Is it a good value?", RuntimeWarning)
        if np.abs(self.ypoints - y).min() > 0.1 * self.ystepsize:
            warn(f"{y} is more than 10% away from a valid y coordinate. Is it a good value?", RuntimeWarning)
        
        i = int(round(x / self.xstepsize))
        j = int(round(y / self.ystepsize))
        
        return i, j
    
    def crop(self, inplace=False):
        pass