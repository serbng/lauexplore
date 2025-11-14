from warnings import warn
from datetime import datetime, timedelta
import numpy as np
import h5py

ENTRIES = [
    ""
]

def get_title(h5f: h5py.File, scan_number: int = 1) -> str:
    return h5f[f"{scan_number}.1/title"][()].decode()

def get_end_reason(h5f: h5py.File, scan_number: int = 1) -> str:
    reason = h5f[f"{scan_number}.1/end_reason"][()].decode()
    if reason != "SUCCESS":
        warn(f"Scan did not finish. End reason: {reason}.", RuntimeWarning)
    return reason

def get_start_time(h5f: h5py.File, scan_number: int = 1) -> datetime:
    start_time = h5f[f"{scan_number}.1/start_time"][()].decode()
    return datetime.fromisoformat(start_time)

def get_end_time(h5f: h5py.File, scan_number: int = 1) -> datetime:
    end_time = h5f[f"{scan_number}.1/end_time"][()].decode()
    return datetime.fromisoformat(end_time)

def get_duration(h5f: h5py.File, scan_number: int = 1) -> timedelta:
    start_time = get_start_time(h5f, scan_number)
    end_time = get_end_time(h5f, scan_number)
    return start_time - end_time

def get_fluo(h5f: h5py.File, material: str, scan_number: int = 1) -> np.ndarray:
    key = f"{scan_number}.1/measurement/xia_sum_fluo{material}"
    try:
        fluo = h5f[key][()]
    except KeyError:
        raise ValueError(f"Can't find fluorescence data of {material}.")
    return fluo


def get_xeol(h5f: h5py.File, scan_number: int = 1) -> np.ndarray:
    key = f"{scan_number}.1/measurement/qepro_det0"
    try:
        xeol = h5f[key][::]
    except KeyError:
        raise ValueError(f"Can't find xeol data in h5 file.")
    return xeol
    
def get_monitor(h5f: h5py.File, scan_number: int = 1) -> np.ndarray:
    key = f"{scan_number}.1/measurement/mon"
    try:
        mon = h5f[key][()]
    except KeyError:
        raise ValueError(f"Can't find monitor data in h5 file.")
    return mon