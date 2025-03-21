# Goodman Photometry

[![Build Status](https://github.com/soar-telescope/goodman_photometry/actions/workflows/python-publish.yml/badge.svg)](https://github.com/soar-telescope/goodman_photometry/actions/workflows/python-publish.yml)
[![PyPI Version](https://img.shields.io/pypi/v/goodman-photometry.svg?style=flat)](https://pypi.org/project/goodman-photometry/)
[![License](https://img.shields.io/pypi/l/goodman-photometry.svg)](https://pypi.org/project/goodman-photometry/)

Routines to perform automatic astrometry and photometry of Goodman imaging observations.

The codes were initially based on STDPipe (https://github.com/karpov-sv/stdpipe) and adapted for Goodman HST.

## Features
- Performs automatic astrometry to add celestial WCS to FITS files
- Calculates photometric zero points using Gaia-DR2 catalog
- Includes auxiliary functions for data processing
- Provides both command-line and Python API interfaces

## Installation
To install the package, run:
```bash
pip install goodman-photometry
```

## Prerequisites
- Python 3.10+
- Required dependencies:
  - astropy
  - astroplan
  - ccdproc
  - cython
  - matplotlib
  - numpy
  - packaging
  - pandas
  - requests
  - scipy
  - statsmodels
  - astroquery
  - sip_tpv
  - setuptools

## Usage

### Command Line Interface
The package provides command-line scripts for processing observations:
```bash
#Process astrometry
redastrometry -i input.fits -o output.fits

# Process photometry
redphotometry -i input.fits -o output.fits
```

### Python API
You can also use the package as a library in your Python code:
from goodman_photometry import Astrometry, Photometry

# Initialize astrometry processor

```bash
# The values of the parameters are set to the default values. So an empty call will work as well.
astrometry = Astrometry(
    catalog_name='gaiadr2',
    magnitude_threshold=17,
    scamp_flag=1,
    color_map='Blues_r',
    save_plots=False,
    save_scamp_plots=False,
    save_intermediary_files=False,
    debug=False
)

# Process the FITS file
astrometry('input.fits', 'output.fits')

# Initialize photometry processor, it will use the default values for the parameters.
photometry = Photometry()
photometry.process('input.fits', 'output.fits')
```

## Contributing
1. Fork the repository
2. Create a feature branch: git checkout -b feature-name
3. Commit your changes: git commit -m "description"
4. Push to the branch: git push origin feature-name
5. Open a Pull Request

Please make sure to:
- Include tests for new functionality
- Update documentation
- Follow PEP8 style guidelines

## Contact Information
For questions, bug reports, or suggestions, please contact:
- Felipe Navarete - felipe.navarete@noirlab.edu
- Simón Torres - simon.torres@noirlab.edu

## Project Links
- Homepage: https://github.com/soar-telescope/goodman_photometry
- Bug Reports: https://github.com/soar-telescope/goodman_photometry/issues
- Source Code: https://github.com/soar-telescope/goodman_photometry

## License
This project is licensed under the BSD License. See the LICENSE file for details.
