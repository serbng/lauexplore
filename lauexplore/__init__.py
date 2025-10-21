from . import (
    _defaults,
    image,
    scan,
    emission,
    dataset,
    peaks,
    plots
)

from ._parsers import (
    FitFile,
    YAMLFile,
)

# For ESRF Jupyter cluster correct rendering
import plotly.io as pio
pio.renderers.default = "notebook"
