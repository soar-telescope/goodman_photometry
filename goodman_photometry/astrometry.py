"""Astrometry Module for Goodman High Throughput Spectrograph Data.

This module provides functionality for performing astrometric calibration on astronomical images
using the Goodman High Throughput Spectrograph (HTS) data. It includes tools for:

- Detecting sources in astronomical images using SExtractor.
- Performing astrometric calibration using SCAMP and a specified catalog (e.g., Gaia DR2).
- Assessing data quality metrics such as FWHM and ellipticity.
- Saving results, including updated FITS headers and plots.

The main class, `Astrometry`, encapsulates the entire astrometric calibration process,
from loading the image data to saving the final calibrated results.

Classes:
    Astrometry: A class for performing astrometric calibration on astronomical images.

Functions:
    goodman_astrometry: Entry point for astrometry calculation using command-line arguments.

Example:
    To perform astrometric calibration on an image:
    >>> from goodman_astrometry import Astrometry
    >>> astrometry = Astrometry(catalog_name='gaiadr2', magnitude_threshold=17, save_plots=True)
    >>> astrometry("observation.fits")

Notes:
    - The module relies on external libraries such as `astropy`, `numpy`, and `matplotlib`.
    - SExtractor is used for source detection, and SCAMP is used for astrometric calibration.
    - Logging is used extensively to provide detailed information about the processing steps.

Version:
    The module version is retrieved from the package metadata using `importlib.metadata`.
"""
import datetime
import logging
import os.path
import sys
import warnings
import matplotlib

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits as fits
from astropy.io.fits.verify import VerifyWarning
from astropy.wcs import FITSFixedWarning

from .goodman_astro import (extract_observation_metadata,
                            create_bad_pixel_mask,
                            check_wcs,
                            clear_wcs,
                            evaluate_data_quality_results,
                            get_filter_set,
                            get_new_file_name,
                            get_vizier_catalog,
                            get_frame_center,
                            get_objects_sextractor,
                            get_pixel_scale,
                            create_goodman_wcs,
                            plot_image,
                            refine_wcs_scamp)
from .utils import get_astrometry_args, setup_logging

warnings.simplefilter(action='ignore', category=FITSFixedWarning)
warnings.simplefilter(action='ignore', category=VerifyWarning)


# Adjust default parameters for imshow
plt.rc('image', origin='lower', cmap='Blues_r')


class Astrometry(object):
    """Performs astrometry operations on a FITS file.

    This class provides functionality for performing astrometric calibration on astronomical images
    using SExtractor for source detection and SCAMP for refining the World Coordinate System (WCS).

    Attributes:
        filename (str): The path to the input FITS file.
        save_plots (bool): If True, saves plots of the processing steps.
        debug (bool): If True, enables debug-level logging.
        save_intermediary_files (bool): If True, saves intermediary files (e.g., masks, catalogs).
        save_scamp_plots (bool): If True, saves SCAMP-generated plots.
        catalog_name (str): The name of the catalog used for astrometric calibration.
        magnitude_threshold (float): The magnitude threshold for source detection.
        scamp_flag (int): The flag threshold for sources used in SCAMP calibration.
        color_map (str): The colormap used for plotting.
        image (numpy.ndarray): The image data array.
        header (astropy.io.fits.Header): The FITS header associated with the image data.
        log (logging.Logger): The logger instance for logging messages.
        bad_pixel_mask (numpy.ndarray): A mask array to exclude bad or saturated pixels.
        wcs_init (astropy.wcs.WCS): The initial WCS solution.
        ra_0 (float): The right ascension of the image center (in degrees).
        dec_0 (float): The declination of the image center (in degrees).
        fov_radius (float): The radius of the field of view (in degrees).
        image_pixel_scale (float): The pixel scale of the image (in arcseconds per pixel).
        sources (astropy.table.Table): The table of detected sources.
        header_with_wcs (astropy.io.fits.Header): The updated header with refined WCS.
        outgoing_header (astropy.io.fits.Header): The final header to be saved.

    Args:
        catalog_name (str, optional): The name of the catalog to use for astrometric calibration.
                                      Defaults to 'gaiadr2'.
        magnitude_threshold (float, optional): The magnitude threshold for source detection.
                                              Defaults to 17.
        scamp_flag (int, optional): The flag threshold for sources used in SCAMP calibration.
                                    Defaults to 1.
        color_map (str, optional): The colormap to use for plotting. Defaults to 'Blues_r'.
        save_plots (bool, optional): If True, saves plots of the processing steps. Defaults to False.
        save_scamp_plots (bool, optional): If True, saves SCAMP-generated plots. Defaults to False.
        save_intermediary_files (bool, optional): If True, saves intermediary files. Defaults to False.
        debug (bool, optional): If True, enables debug-level logging. Defaults to False.

    Example:
        >>> astrometry = Astrometry(catalog_name='gaiadr2', magnitude_threshold=17, save_plots=True)
        >>> astrometry("observation.fits")
    """

    def __init__(self,
                 catalog_name='gaiadr2',
                 magnitude_threshold=17,
                 scamp_flag=1,
                 color_map='Blues_r',
                 save_plots=False,
                 save_scamp_plots=False,
                 save_intermediary_files=False,
                 reduced_data_path=None,
                 artifacts_path=None,
                 use_interactive_mpl_backend=True,
                 debug=False):
        """Initialize the Astrometry class.

        Args:
            catalog_name (str, optional): The name of the catalog to use for astrometric calibration.
                                          Defaults to 'gaiadr2'.
            magnitude_threshold (float, optional): The magnitude threshold for source detection.
                                                  Defaults to 17.
            scamp_flag (int, optional): The flag threshold for sources used in SCAMP calibration.
                                        Defaults to 1.
            color_map (str, optional): The colormap to use for plotting. Defaults to 'Blues_r'.
            save_plots (bool, optional): If True, saves plots of the processing steps. Defaults to False.
            save_scamp_plots (bool, optional): If True, saves SCAMP-generated plots. Defaults to False.
            save_intermediary_files (bool, optional): If True, saves intermediary files. Defaults to False.
            reduced_data_path (str, optional): The reduced data path for the fits file.
            use_interactive_mpl_backend (bool, optional): If True, enables interactive matplotlib backend.
            debug (bool, optional): If True, enables debug-level logging. Defaults to False.
        """
        self.filename = None
        self.save_plots = save_plots
        self.debug = debug
        self.save_intermediary_files = save_intermediary_files
        self.reduced_data_path = reduced_data_path
        self.artifacts_path = artifacts_path
        self.save_scamp_plots = save_scamp_plots
        self.catalog_name = catalog_name
        self.magnitude_threshold = magnitude_threshold
        self.scamp_flag = scamp_flag
        self.color_map = color_map
        self.image = None
        self.header = None
        self.log = logging.getLogger()
        if not use_interactive_mpl_backend:
            matplotlib.use('Agg')

    def __call__(self, filename):
        """Process a FITS file for astrometric calibration.

        Args:
            filename (str): The path to the input FITS file.

        Notes:
            - This method performs the entire astrometric calibration process, including source detection,
              WCS refinement, and saving the results.
        """
        self.filename = filename

        if self.reduced_data_path is None or not os.path.isdir(self.reduced_data_path):
            self.reduced_data_path = os.path.dirname(self.filename)

        if self.artifacts_path is None or not os.path.isdir(self.artifacts_path):
            self.artifacts_path = self.reduced_data_path

        self.start = datetime.datetime.now()
        self.log.info(f"Processing {self.filename}")

        try:

            self.image = fits.getdata(self.filename).astype(np.double)
            self.header = fits.getheader(self.filename)
        except FileNotFoundError:
            self.log.critical(f"File {self.filename} not found!!")
            sys.exit(0)

        # gather required information from the header
        (self.filter_name,
         self.serial_binning,
         parallel_binning,
         time,
         self.gain,
         read_noise,
         self.saturation_threshold,
         exposure_time) = extract_observation_metadata(self.header)

        self.log.debug(f"Processing {self.filename}: "
                       f"filter {self.filter_name}, "
                       f"gain {self.gain:.2f}, "
                       f"saturation_threshold {self.saturation_threshold: .1f} at {time}")

        self.log.info(f"filter={self.filter_name} exposure_time={exposure_time:.2f} binning={self.serial_binning}x{parallel_binning}")

        self.__create_bad_pixel_mask()

        self.__create_basic_wcs_header()

        self.__detect_sources_with_sextractor()

        self.__data_quality_assessment()

        self.__obtain_astrometric_solution_with_scamp()

        self.__update_header()

        output_file, elapsed_time = self.__save_to_fits_file()
        return {
            "output_file": output_file,
            "elapsed_time": elapsed_time
        }

    def __create_bad_pixel_mask(self):
        """Create a bad pixel mask for the image.

        Notes:
            - The mask excludes saturated or defective pixels.
            - If `save_intermediary_files` is True, the mask is saved as a FITS file.
            - Plots of the image and mask are generated and saved if `save_plots` is True.
        """
        self.bad_pixel_mask = create_bad_pixel_mask(self.image, self.saturation_threshold, self.serial_binning)
        self.log.debug('Done masking cosmics')

        if self.save_intermediary_files:
            hdu = fits.PrimaryHDU(data=self.bad_pixel_mask.astype(int), header=self.header)
            hdu_list = fits.HDUList([hdu])
            hdu_list.writeto(self.filename.replace(".fits", "_mask.fits"), overwrite=True)

        plot_image_filename = get_new_file_name(current_file_name=self.filename,
                                                new_path=self.artifacts_path,
                                                new_extension='png')
        plot_image(
            image=self.image,
            title=self.filename.replace(".fits", ""),
            output_file=plot_image_filename,
            quantiles=(0.01, 0.99),
            cmap=self.color_map)
        self.log.info(f"Image - no WCS: {plot_image_filename}")

        plot_bad_pixel_mask_filename = get_new_file_name(current_file_name=self.filename,
                                                         new_path=self.artifacts_path,
                                                         new_extension="_BPM.png")
        plot_image(
            image=self.bad_pixel_mask,
            title="Bad pixel mask",
            output_file=plot_bad_pixel_mask_filename,
            quantiles=(0, 1),
            cmap=self.color_map)
        self.log.info(f"Image - bad pixel mask: {plot_bad_pixel_mask_filename}")

    def __create_basic_wcs_header(self):
        """Create a basic WCS header for the image.

        Notes:
            - The initial WCS solution is derived from the image header.
            - If `save_intermediary_files` is True, the WCS header is saved as a FITS file.
        """
        header_with_basic_wcs = create_goodman_wcs(self.header)
        self.wcs_init = check_wcs(header_with_basic_wcs)

        if self.save_intermediary_files:
            hdu = fits.PrimaryHDU(data=self.image, header=header_with_basic_wcs)
            hdu_list = fits.HDUList([hdu])
            hdu_list.writeto(self.filename.replace(".fits", "_wcs_init.fits"), overwrite=True)

        self.ra_0, self.dec_0, self.fov_radius = get_frame_center(wcs=self.wcs_init, image_width=self.image.shape[1], image_height=self.image.shape[0])
        self.image_pixel_scale = get_pixel_scale(wcs=self.wcs_init)

        self.log.info(f"Initial WCS: RA={self.ra_0} DEC={self.dec_0} SR={self.fov_radius} PIXSCALE={self.image_pixel_scale}")

    def __detect_sources_with_sextractor(self):
        """Detect sources in the image using SExtractor.

        Notes:
            - The detected sources are stored in the `sources` attribute.
            - If `save_plots` is True, a plot of the detected sources is saved.
        """
        self.seeing = 1.0
        self.full_width_at_tenth_maximum_to_fwhm = 1.82
        sextractor_aperture = np.round(self.full_width_at_tenth_maximum_to_fwhm * self.seeing / (self.image_pixel_scale * 3600.))
        self.log.info(f"SExtractor aperture radius={sextractor_aperture:.1f} pixels")

        self.sources = get_objects_sextractor(image=self.image,
                                              mask=self.bad_pixel_mask,
                                              gain=self.gain,
                                              r0=2,
                                              aper=sextractor_aperture,
                                              wcs=self.wcs_init)

        self.log.info(f"SExtractor detections: {len(self.sources)}")

        self.log.info("SExtractor detections per flag")

        sextractor_flags = np.unique(self.sources['flags'])

        for flag in sextractor_flags:
            self.log.info(f"Flag={flag} - {np.sum(self.sources['flags'] == flag)}")

        plot_detections_filename = get_new_file_name(current_file_name=self.filename,
                                                     new_path=self.artifacts_path,
                                                     new_extension="_detections.png")

        plot_image(
            image=self.image,
            x_points=self.sources['x'],
            y_points=self.sources['y'],
            title='Detected objects',
            output_file=plot_detections_filename,
            quantiles=(0.01, 0.99),
            cmap=self.color_map)

        self.log.info(f"Image - SExtractor detections: {plot_detections_filename}")

    def __data_quality_assessment(self):
        """Assess data quality metrics for the detected sources.

        Notes:
            - Metrics include FWHM, ellipticity, and their uncertainties.
            - If `save_plots` is True, a plot of the sources with FLAG=0 is saved.
        """
        data_quality_sources = self.sources[self.sources['flags'] == 0]

        plot_detections_flag_0_filename = get_new_file_name(current_file_name=self.filename,
                                                            new_path=self.artifacts_path,
                                                            new_extension="_detections_flag_0.png")

        plot_image(
            image=self.image,
            x_points=data_quality_sources['x'],
            y_points=data_quality_sources['y'],
            title='Detected objects (FLAG=0)',
            output_file=plot_detections_flag_0_filename,
            quantiles=(0.01, 0.99),
            cmap=self.color_map)

        self.log.info(f"Image - Detected objects (FLAG=0): {plot_detections_flag_0_filename}")

        fwhm, fwhm_error, ellipticity, ellipticity_error = evaluate_data_quality_results(data_quality_sources)

        self.log.info("Data quality results")
        self.log.info(f"Number of objects: {len(data_quality_sources)}/{len(self.sources)}")
        self.log.info(f"Median FWHM: {fwhm:.2f}+/-{fwhm_error:.2f} pixels")
        self.log.info(f"Median FWHM: {fwhm * self.image_pixel_scale * 3600.:.2f}+/-{fwhm_error * self.image_pixel_scale * 3600.:.2f} arcseconds")
        self.log.info(f"Median Ellipticity: {ellipticity:.3f}+/-{ellipticity_error:.3f}")

    def __obtain_astrometric_solution_with_scamp(self):
        """Refine the WCS solution using SCAMP.

        Notes:
            - The refined WCS solution is stored in the `header_with_wcs` attribute.
            - If `save_scamp_plots` is True, SCAMP-generated plots are saved.
        """
        self.log.info(f"Performing astrometry with SCAMP using {self.catalog_name}")
        self.catalog_filter, _ = get_filter_set(self.filter_name)

        self.log.info(f"Querying Vizier for {self.catalog_name} catalog")

        vizier_catalog = get_vizier_catalog(right_ascension=self.ra_0,
                                            declination=self.dec_0,
                                            search_radius=self.fov_radius,
                                            catalog=self.catalog_name,
                                            column_filters={self.catalog_filter: f'<{self.magnitude_threshold}'})

        self.log.info(f"Retrieved {len(vizier_catalog)} stars on {self.catalog_filter} filter (magnitude threshold={self.magnitude_threshold:.2f}).")

        if self.save_intermediary_files:
            catalog_file_filename = self.filename.replace(".fits", f"_{self.catalog_name}_cat.vsv")
            vizier_catalog.write(catalog_file_filename, overwrite=True)
            self.log.info(f"Vizier catalog saved as {catalog_file_filename}")

        if self.save_scamp_plots:
            scamp_plots = ['FGROUPS', 'DISTORTION', 'ASTR_REFERROR2D', 'ASTR_REFERROR1D']
            scamp_names = ','.join([self.filename.replace("./", "").replace(".fits", "") + "_SCAMP_" + item for item in scamp_plots])
            scamp_extra = {
                'CHECKPLOT_TYPE': ','.join(scamp_plots),
                'CHECKPLOT_NAME': scamp_names,
                'CHECKPLOT_RES': '1200'
            }
        else:
            scamp_extra = {}

        scamp_sources = self.sources if not self.scamp_flag else self.sources[self.sources['flags'] <= self.scamp_flag]

        self.log.info("Running SCAMP for refining the WCS solution.")

        self.header_with_wcs = refine_wcs_scamp(
            obj=scamp_sources,
            cat=vizier_catalog,
            sr=5 * self.image_pixel_scale,
            wcs=self.wcs_init,
            order=3,
            cat_col_ra='RAJ2000',
            cat_col_dec='DEJ2000',
            cat_col_ra_err='e_RAJ2000',
            cat_col_dec_err='e_DEJ2000',
            cat_col_mag=self.catalog_filter,
            cat_col_mag_err='e_' + self.catalog_filter,
            update=True,
            verbose=True,
            get_header=True,
            extra=scamp_extra)

        if self.save_intermediary_files:
            scamp_results_filename = self.filename.replace(".fits", "_scamp_results.txt")
            with open(scamp_results_filename, "w") as scamp_results:
                scamp_results.write(repr(self.header_with_wcs))
            self.log.info(f"SCAMP results saved as {scamp_results_filename}")

    def __update_header(self):
        """Update the FITS header with the refined WCS solution.

        Notes:
            - The updated header is stored in the `outgoing_header` attribute.
        """
        wcs = check_wcs(header=self.header_with_wcs)

        self.outgoing_header = clear_wcs(
            header=self.header,
            remove_comments=True,
            remove_underscored=True,
            remove_history=True)

        if not wcs or not wcs.is_celestial:
            self.log.error("WCS refinement failed. Using initial WCS from header information.")
            self.outgoing_header.update(self.wcs_init.to_header(relax=True))
            self.outgoing_header.append(('GSP_ASOL', 'Header information', 'Astrometry solution mode'), end=True)

        else:
            self.log.info("WCS refinement succeeded.")
            self.log.info(f"RMS(x,y) = {self.header_with_wcs['ASTRRMS1'] * 3600.:.2f}\" {self.header_with_wcs['ASTRRMS2'] * 3600.:.2f}\"")

            self.outgoing_header.update(wcs.to_header(relax=True))

            self.outgoing_header.append(
                ('GSP_ASOL', "SCAMP", 'Astrometry solution mode'),
                end=True)
            self.outgoing_header.append(
                ('GSP_ACAT', self.catalog_name.strip(), 'Catalog name used for astrometric calibration'),
                end=True)
            self.outgoing_header.append(
                ('GSP_AFIL', self.catalog_filter.strip(), 'Filter name used for retrieving the catalog list for astrometric calibration'),
                end=True)
            self.outgoing_header.append(
                ('GSP_AMAG', float(self.magnitude_threshold), 'Magnitude threshold for GSP_AFIL used for astrometric calibration'),
                end=True)
            self.outgoing_header.append(
                ('GSP_XRMS', float(self.header_with_wcs['ASTRRMS1'] * 3600.), 'Astrometric rms error on the x-axis (in arcseconds)'),
                end=True)
            self.outgoing_header.append(
                ('GSP_YRMS', float(self.header_with_wcs['ASTRRMS2'] * 3600.), 'Astrometric rms error on the y-axis (in arcseconds)'),
                end=True)

    def __save_to_fits_file(self):
        """Save the calibrated image and updated header to a new FITS file.

        Notes:
            - The output file is saved with the suffix "_wcs.fits".
        """
        new_filename = os.path.basename(self.filename.replace(".fits", "_wcs.fits"))
        outgoing_filename = os.path.join(self.reduced_data_path, new_filename)
        hdu = fits.PrimaryHDU(data=self.image, header=self.outgoing_header)
        hdu_list = fits.HDUList([hdu])
        hdu_list.writeto(outgoing_filename, overwrite=True)

        self.log.info(f"FITS file saved as {outgoing_filename}")

        duration_in_seconds = (datetime.datetime.now() - self.start).total_seconds()

        self.log.info(f"Astrometric calibration executed in {duration_in_seconds:.2f} seconds")

        self.log.info('Astrometric calibration finished.')

        return outgoing_filename, duration_in_seconds


def goodman_astrometry():
    """Entry point for astrometry calculation using command-line arguments.

    Notes:
        - This function parses command-line arguments, sets up logging, and performs astrometric calibration
          using the `Astrometry` class.
    """
    args = get_astrometry_args()

    setup_logging(debug=args.debug, log_filename=args.log_filename)

    astrometry = Astrometry(
        catalog_name=args.catalog_name,
        magnitude_threshold=args.magnitude_threshold,
        scamp_flag=args.scamp_flag,
        color_map=args.color_map,
        save_plots=args.save_plots,
        save_scamp_plots=args.save_scamp_plots,
        save_intermediary_files=args.save_intermediary_files,
        reduced_data_path=args.reduced_data_path,
        artifacts_path=args.artifacts_path,
        debug=args.debug)

    astrometry(filename=args.filename)
