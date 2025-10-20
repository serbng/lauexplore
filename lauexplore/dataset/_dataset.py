from typing import Iterable, Literal
from pathlib import Path
from functools import partial
import multiprocessing as mp
import numpy as np
import pandas as pd

from lauexplore._parsers import parsers

def _parse(filepath: str | Path, parser: object):
    try:
        return parser(filepath)
    except Exception:
        return None

class Dataset:
    def __init__(self, 
            filepaths: Iterable[str | Path], 
            parser: Literal["fit", "yaml", "infer"] = "infer",
            workers: int = 8
        ):
        
        filepaths = list(filepaths)
        filepaths = [Path(fpath) for fpath in filepaths]
        
        if parser == "fit":
            parser_obj = parsers["fit"]
        
        elif parser == "yaml":
            parser_obj = parsers["yaml"]
            
        else:
            is_yaml, is_fit = False, False
            i = 0
            while not(is_yaml and is_fit):
                if filepaths[i].suffix == ".fit":
                    is_fit = True
                    parser_obj = parsers["fit"]
                if filepaths[i].suffix == ".yaml":
                    is_yaml = True
                    parser_obj = parsers["yaml"]
                    
        if is_yaml and is_fit:
            raise ValueError("filepaths contains both .fit and .yaml files.")
        
        if not(is_yaml or is_fit):
            raise ValueError("filepaths does not contain recognized extensions.")
        
        worker = partial(_parse, parser=parser_obj)
        
        with mp.Pool(workers) as pool:
            self.files = pool.map(worker, filepaths, chunksize=5)
        
        # It will turn useful when building overall class attributes
        self._first_existing = None
        for file in self.files:
            if file is not None:
                self._first_existing = file
                break
        
        if self._first_existing is None:
            raise ValueError("None of the files could be loaded. Perhaps a wrong name?")
        
        self.length = len(self.files)
    
    def _collect(self, attr, padding=np.nan):
        """
        Gather `attr` from each FitFile in `self.fitfiles`.
    
        Returns
        -------
        np.ndarray
            If values are numeric:
              - scalar attribute  -> shape (N,)
              - array attribute   -> shape (N, *target_shape)
        list
            If values are non-numeric, returns a Python list (with None
            for missing fitfiles).
        """
        # Validate attribute existence on the first valid FitFile
        if attr not in self._first_existing.__dict__:
            raise AttributeError(f"Attribute '{attr}' not in FitFile object.")
    
        sample_val = getattr(self._first_existing, attr)
        sample_arr = np.asarray(sample_val)
        is_numeric = (sample_arr.dtype != object) and np.issubdtype(sample_arr.dtype, np.number)
        
        N = self.length
        
         # --------- non-numeric path -> list with None for missing ---------
        if not is_numeric:
           
            out = []
            for pf in self.files:
                out.append(None if pf is None else getattr(pf, attr))
            return out
    
        # -------------------------- numeric path ---------------------------
        # Scalar case. Return 1D array, shape (N, )
        if sample_arr.shape == ():
            out_arr = np.full((N,), dtype=type(sample_val))
            for i, pf in enumerate(self.files):
                if pf is None:
                    out_arr[i] = padding
                    
                else:
                    arr = np.asarray(getattr(pf, attr))
                    if arr.shape != ():
                        raise ValueError(f"Attribute '{attr}' expected scalar (), got {arr.shape}.")
                    # Extract scalar
                    out_arr[i] = arr.item()
            return out_arr

        # Array case -> Return 3D array, shape (N, *target_shape)
        target_shape = sample_arr.shape
        out_arr = np.full((N,) + target_shape, dtype=sample_arr.dtype)
        for i, pf in enumerate(self.files):
            if pf is None:
                out_arr[i] = np.full(target_shape, padding)
            else:
                arr = np.asarray(getattr(pf, attr))
                if arr.shape != target_shape:
                    raise ValueError(
                        f"Attribute '{attr}' expected shape {target_shape}, "
                        f"got {arr.shape}."
                    )
                out_arr[i] = arr
        return out_arr

        
    @property
    def number_indexed_spots(self):
        return self._collect("number_indexed_spots")

    @property
    def mean_pixel_deviation(self):
        return self._collect("mean_pixel_deviation")

    @property
    def boa(self):
        return self._collect("boa")

    @property
    def coa(self):
        return self._collect("coa")
    
    @property
    def euler_angles(self):
        return self._collect("euler_angles")
    
    @property
    def UB(self):
        return self._collect("UB")
    
    @property
    def B0(self):
        return self._collect("B0")
    
    @property
    def UBB0(self):
        return self._collect("UBB0")
    
    @property
    def deviatoric_strain_crystal_frame(self):
        return self._collect("deviatoric_strain_crystal_frame")
    
    @property
    def deviatoric_strain_sample_frame(self):
        return self._collect("deviatoric_strain_sample_frame")
    
    @property
    def new_lattice_parameters(self):
        return self._collect("new_lattice_parameters")
    
    @property
    def a_prime(self):
        return self._collect("a_prime")
    
    @property
    def b_prime(self):
        return self._collect("b_prime")
    
    @property
    def c_prime(self):
        return self._collect("c_prime")
    
    @property
    def astar_prime(self):
        return self._collect("astar_prime")
    
    @property
    def bstar_prime(self):
        return self._collect("bstar_prime")
    
    @property
    def cstar_prime(self):
        return self._collect("cstar_prime")
    
    @property
    def peaklist(self):
        return self._collect("peaklist")

    def track_hkl(self, h: int, k: int, l: int) -> pd.DataFrame:
        """_summary_

        Args:
            h (int): _description_
            k (int): _description_
            l (int): _description_

        Raises:
            TypeError: _description_

        Returns:
            pd.DataFrame: _description_
        """
        if not all([isinstance(index, int) for index in (h,k,l)]):
            raise TypeError("All Miller indices must be integers")
        
        matches = []
        columns = list(self._first_existing.peaklist.columns)
        columns.append("spot idx")
        M = len(columns)
        
        for i, ff in enumerate(self.fitfiles):
            if ff is not None:
                match = ff.peaklist.query(f"h=={h} and k=={k} and l=={l}")
            
            if len(match)==0 or ff is None:
                match =  pd.DataFrame(
                    data=np.full((1, M), np.nan), 
                    columns=columns, 
                    index=[i]
                )
                # Even though every field is a nan, let's put the correct miller indices values
                # Remember that the index is the image number, i.e. i
                match.loc[i, ["h", "k", "l"]] = h, k, l
                matches.append(match)
                continue
            
            # I expect only one match, so I create one new column called spot idx
            # and I store in row 0 the value of the index of the dataframe (that
            # is the spot idx) and set the current index to the image number
            match.insert(M-1, "spot idx", match.index.values)
            match.index = [i]
            matches.append(match)
            
        return pd.concat(matches).convert_dtypes()

