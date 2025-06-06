.. Goodman Photometry documentation master file, created by
   sphinx-quickstart on Fri Mar 21 18:53:41 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Goodman Photometry's documentation!
==============================================

.. image:: https://github.com/soar-telescope/goodman_photometry/actions/workflows/python-publish.yml/badge.svg
   :target: https://github.com/soar-telescope/goodman_photometry/actions/workflows/python-publish.yml
.. image:: https://codecov.io/gh/soar-telescope/goodman_photometry/graph/badge.svg?token=wO5r3IRIHT
   :target: https://codecov.io/gh/soar-telescope/goodman_photometry
.. image:: https://img.shields.io/pypi/v/goodman-photometry.svg?style=flat
   :target: https://pypi.org/project/goodman-photometry/
.. image:: https://img.shields.io/pypi/l/goodman-photometry.svg
   :target: https://pypi.org/project/goodman-photometry/

Routines to perform automatic astrometry and photometry of Goodman imaging observations.

The codes were initially based on `STDPipe <https://github.com/karpov-sv/stdpipe>`_ and adapted for Goodman HST.

Features
--------

- Performs automatic astrometry to add celestial WCS to FITS files
- Calculates photometric zero points using Gaia-DR2 catalog
- Includes auxiliary functions for data processing
- Provides both command-line and Python API interfaces

Installation
------------

To install the package, run:

.. code-block:: bash

   pip install goodman-photometry

Prerequisites
-------------
- Operating System: Linux, MacOS

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

Usage
-----

Command Line Interface
~~~~~~~~~~~~~~~~~~~~~~

The package provides command-line scripts for processing observations:

.. code-block:: bash

   # Process astrometry
   redastrometry -i input.fits -o output.fits

   # Process photometry
   redphotometry -i input.fits -o output.fits

Python API
~~~~~~~~~~

You can also use the package as a library in your Python code:

.. code-block:: python

   from goodman_photometry import Astrometry, Photometry

   # Initialize astrometry processor
   astrometry = Astrometry(
       catalog_name='gaiadr2',
       magnitude_threshold=17,
       scamp_flag=1,
       color_map='Blues_r',
       save_plots=False,
       save_scamp_plots=False,
       save_intermediary_files=False,
       reduced_data_path='/where/reduced/data/goes/',
       artifacts_path='/where/plot/files/go/',
       debug=False
   )

   # Process the FITS file
   results = astrometry('input.fits')

   # Results example
   {
       "output_file": "reduced/0274_wd1_r_025_wcs.fits",
       "elapsed_time": 5.358209,
       "data_quality": {
          "fwhm": 6.3553266525268555,
          "fwhm_error": 1.0162534713745117,
          "ellipticity": 0.2321857213973999,
          "ellipticity_error": 0.08647292852401733
       },
       "plots": {
           "image": "artifacts/0274_wd1_r_025.png",
           "bad_pixel_mask": "artifacts/0274_wd1_r_025_BPM.png",
           "detections": "artifacts/0274_wd1_r_025_detections.png",
           "detections_flag_0": "artifacts/0274_wd1_r_025_detections_flag_0.png"
       },
       "intermediary_files": {
           "bad_pixel_mask": "reduced/0274_wd1_r_025_mask.fits",
           "wcs_init": "reduced/0274_wd1_r_025_wcs_init.fits",
           "catalog_filename": "artifacts/0274_wd1_r_025_gaiadr2_cat.csv",
           "scamp_results_filename": "artifacts/0274_wd1_r_025_scamp_results.txt"
       }
   }

   # Initialize photometry processor
   photometry = Photometry()
   results = photometry('input.fits', 'output.fits')

   # example results
   {
       "output_file": "reduced/0274_wd1_r_025_wcs_phot.fits",
       "elapsed_time": 15.794938,
       "data_quality": {
           "fwhm": 6.872805595397949,
           "fwhm_error": 1.1487979888916016,
           "ellipticity": 0.2556902766227722,
           "ellipticity_error": 0.09023219347000122
       },
       "sources_table_html_file": "artifacts/0274_wd1_r_025_wcs_obj_table.html",
       "plots": {
           "photometry_wcs": "artifacts/0274_wd1_r_025_wcs_phot_wcs.png",
           "photometry_detections": "artifacts/0274_wd1_r_025_wcs_phot_detections.png",
           "photometry_detections_flag0": "artifacts/0274_wd1_r_025_wcs_phot_detections_flag0.png",
           "phot_detections_calibrated": "artifacts/0274_wd1_r_025_wcs_phot_detections_calibrated.png",
           "photometry_match": "artifacts/0274_wd1_r_025_wcs_phot_photmatch.png",
           "photometry_match_2": "artifacts/0274_wd1_r_025_wcs_phot_photmatch2.png",
           "photometry_zeropoint": "artifacts/0274_wd1_r_025_wcs_phot_zp.png"
    }
}

Contributing
------------

1. Fork the repository
2. Create a feature branch: ``git checkout -b feature-name``
3. Commit your changes: ``git commit -m "description"``
4. Push to the branch: ``git push origin feature-name``
5. Open a Pull Request

Please make sure to:

- Include tests for new functionality
- Update documentation
- Follow PEP8 style guidelines

Contact Information
-------------------

For questions, bug reports, or suggestions, please contact:

- Felipe Navarete - `felipe.navarete@noirlab.edu <felipe.navarete@noirlab.edu>`_
- Simón Torres - `simon.torres@noirlab.edu <simon.torres@noirlab.edu>`_

Project Links
-------------

- `Homepage <https://github.com/soar-telescope/goodman_photometry>`_
- `Bug Reports <https://github.com/soar-telescope/goodman_photometry/issues>`_
- `Source Code <https://github.com/soar-telescope/goodman_photometry>`_

License
-------

This project is licensed under the BSD License. See the `LICENSE file <LICENSE>`_ for details.



.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
