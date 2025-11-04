# from ._peaklist import PeakList
# from ._fitfile import FitFile
# from ..dataset._indexed_dataset import IndexedDataset
from ._h5 import (
    get_title,
    get_end_reason,
    get_start_time,
    get_end_time,
    get_duration
)

from ._fitfile import FitFile
from ._yamlfile import YAMLFile

parsers: dict[str, object] = dict(
    fit  = FitFile,
    yaml = YAMLFile
)