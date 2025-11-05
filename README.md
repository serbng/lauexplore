# lauexplore

lauexplore is a Python library for *visualization* and *post-processing* of Laue µdiffraction data.

The data is collected with the LAUEMAX instrument at beamline BM32 of the European Synchrotron Radiation Facility (ESRF) in Grenoble. The raw Laue patterns are analyzed with the in-house developed [lauetools](https://github.com/BM32ESRF/lauetools) software.

The library is designed upon the lauetools ecosystem to manage its outputs for raster scans.

## Installation

The latest version can be installed directly from GitHub, where the source code is hosted

```bash
pip install git+https://github.com/serbng/lauexplore.git
```

The binary installer is also available at the Python Package Index (PyPI) and on Conda.

```bash
# PyPI
pip install lauexplore
```

```bash
# or conda
conda install -c ... lauexplore #TBD
```

## Documentation

Documentation is under development.

For now, refer to the in-code docstrings and the notebooks in the `examples/` folder.

## Contributing

Contributions, issues, and feature requests are welcome!
Feel free to open an [issue](https://github.com/serbng/lauexplore/issues) or submit a [pull request](https://github.com/serbng/lauexplore/pulls).
