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
       debug=False
   )

   # Process the FITS file
   astrometry.process('input.fits', 'output.fits')

   # Initialize photometry processor
   photometry = Photometry()
   photometry.process('input.fits', 'output.fits')

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
