from pathlib import Path
import numpy as np
import fabio

def read(filepath: str | Path) -> np.ndarray:
    with fabio.open(filepath) as f:
        image_data = f.data
    return image_data