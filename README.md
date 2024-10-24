# goodman_photometry

[![Upload to PYPI](https://github.com/soar-telescope/goodman_photometry/actions/workflows/python-publish.yml/badge.svg)](https://github.com/soar-telescope/goodman_photometry/actions/workflows/python-publish.yml)
[![pypi](https://img.shields.io/pypi/v/goodman_photometry.svg?style=flat)](https://pypi.org/project/goodman-photometry/)

Routines to perform automatic astrometry and photometry of goodman imaging observations

The codes were initially based on STDPipe (https://github.com/karpov-sv/stdpipe) and adapted for Goodman HST.

- goodman_astro.py contains all the auxiliary functions for redastro and redphot routines.
- redastro.py is the astrometric solution routine, which will add a celestial WCS to the fits file processed or not by redccd.
- redphot.py is the photometric solution routine, which will evaluate the photometric zero point of the image based on the Gaia-DR2 catalog.
