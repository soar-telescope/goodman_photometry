"""Photometric Calibration Module for Astronomical Images.

This module provides a class for performing photometric calibration on astronomical images
using the Goodman High Throughput Spectrograph (HTS) data. It includes functionality for:

- Detecting sources in astronomical images using SExtractor.
- Performing photometric calibration using a specified catalog (e.g., Gaia DR2).
- Assessing data quality metrics such as FWHM and ellipticity.
- Saving results, including calibrated magnitudes, plots, and updated FITS headers.

The main class, `Photometry`, encapsulates the entire photometric calibration process,
from loading the image data to saving the final calibrated results.

Classes:
    Photometry: A class for performing photometric calibration on astronomical images.

Example:
    To perform photometric calibration on an image:
    >>> from photometry import Photometry
    >>> photometry = Photometry(catalog_name='gaiadr2', magnitude_threshold=17, save_plots=True)
    >>> photometry("observation.fits")

Notes:
    - The module relies on external libraries such as `astropy`, `numpy`, and `matplotlib`.
    - SExtractor is used for source detection, and the Gaia DR2 catalog is used for calibration by default.
    - Logging is used extensively to provide detailed information about the processing steps.

Version:
    The module version is retrieved from the package metadata using `importlib.metadata`.
"""
import datetime
import logging
import os.path
import warnings
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.io.fits.verify import VerifyWarning
from astropy.wcs import FITSFixedWarning

from .goodman_astro import (create_bad_pixel_mask,
                            calibrate_photometry,
                            check_photometry_results,
                            check_wcs,
                            convert_match_results_to_table,
                            evaluate_data_quality_results,
                            get_filter_set,
                            get_vizier_catalog,
                            get_frame_center,
                            get_new_file_name,
                            extract_observation_metadata,
                            get_objects_sextractor,
                            get_photometric_zeropoint,
                            get_pixel_scale,
                            plot_image,
                            plot_photometric_match,
                            plot_photcal)

from .utils import get_photometry_args, setup_logging

warnings.simplefilter(action='ignore', category=FITSFixedWarning)
warnings.simplefilter(action='ignore', category=VerifyWarning)


class Photometry(object):
    """A class for performing photometric calibration on astronomical images.

    This class provides functionality for processing astronomical images, detecting sources,
    and performing photometric calibration using a specified catalog. It supports logging,
    plotting, and saving results for further analysis.

    Attributes:
        filename (str): The path to the input FITS file.
        output_filename (str): The path to the output file (if applicable).
        save_plots (bool): If True, saves plots of the processing steps.
        debug (bool): If True, enables debug-level logging.
        catalog_name (str): The name of the catalog used for photometric calibration.
        magnitude_threshold (float): The magnitude threshold for source detection.
        magnitude_error_threshold (float): The maximum allowed magnitude error for calibration.
        magnitude_range (list): The range of magnitudes to consider for calibration.
        plot_file_resolution (int): The resolution (in DPI) for saved plots.
        color_map (str): The colormap used for plotting.
        log (logging.Logger): The logger instance for logging messages.
        sources (astropy.table.Table): The table of detected sources.
        data_quality_sources (astropy.table.Table): The table of sources used for data quality assessment.
        _data_quality (dict): A dictionary containing data quality metrics (FWHM, ellipticity, etc.).

    Args:
        catalog_name (str, optional): The name of the catalog to use for photometric calibration.
                                      Defaults to 'gaiadr2'.
        magnitude_threshold (float, optional): The magnitude threshold for source detection.
                                              Defaults to 17.
        magnitude_error_threshold (float, optional): The maximum allowed magnitude error for calibration.
                                                    Defaults to 0.1.
        color_map (str, optional): The colormap to use for plotting. Defaults to 'Blues_r'.
        plot_file_resolution (int, optional): The resolution (in DPI) for saved plots. Defaults to 600.
        save_plots (bool, optional): If True, saves plots of the processing steps. Defaults to False.
        debug (bool, optional): If True, enables debug-level logging. Defaults to False.

    Example:
        >>> photometry = Photometry(catalog_name='gaiadr2', magnitude_threshold=17, save_plots=True)
        >>> photometry("observation.fits")
    """

    def __init__(self,
                 catalog_name='gaiadr2',
                 magnitude_threshold=17,
                 magnitude_error_threshold=0.1,
                 color_map='Blues_r',
                 plot_file_resolution=600,
                 save_plots=False,
                 reduced_data_path=None,
                 artifacts_path=None,
                 use_interactive_mpl_backend=True,
                 debug=False) -> None:
        """Initialize the Photometry class.

        This method initializes the Photometry class with default or user-specified parameters
        for photometric calibration. It sets up attributes for catalog selection, magnitude thresholds,
        plotting preferences, and logging.

        Args:
            catalog_name (str, optional): The name of the catalog to use for photometric calibration.
                                          Defaults to 'gaiadr2'.
            magnitude_threshold (float, optional): The magnitude threshold for source detection.
                                                  Defaults to 17.
            magnitude_error_threshold (float, optional): The maximum allowed magnitude error for calibration.
                                                        Defaults to 0.1.
            color_map (str, optional): The colormap to use for plotting. Defaults to 'Blues_r'.
            plot_file_resolution (int, optional): The resolution (in DPI) for saved plots. Defaults to 600.
            save_plots (bool, optional): If True, saves plots of the processing steps. Defaults to False.
            reduced_data_path (str, optional): The path to the reduced data directory. Defaults to None.
            use_interactive_mpl_backend (bool, optional): If True, enables interactive matplotlib backend.
            debug (bool, optional): If True, enables debug-level logging. Defaults to False.

        Attributes:
            filename (str): The path to the input FITS file.
            output_filename (str): The path to the output file (if applicable).
            save_plots (bool): If True, saves plots of the processing steps.
            debug (bool): If True, enables debug-level logging.
            catalog_name (str): The name of the catalog used for photometric calibration.
            magnitude_threshold (float): The magnitude threshold for source detection.
            magnitude_error_threshold (float): The maximum allowed magnitude error for calibration.
            magnitude_range (list): The range of magnitudes to consider for calibration.
            plot_file_resolution (int): The resolution (in DPI) for saved plots.
            color_map (str): The colormap used for plotting.
            log (logging.Logger): The logger instance for logging messages.
            sources (astropy.table.Table): The table of detected sources.
            data_quality_sources (astropy.table.Table): The table of sources used for data quality assessment.
            _data_quality (dict): A dictionary containing data quality metrics (FWHM, ellipticity, etc.).

        Example:
            >>> photometry = Photometry(catalog_name='gaiadr2', magnitude_threshold=17, save_plots=True)
        """
        self.filename = None
        self.output_filename = None
        self.save_plots = save_plots
        self.reduced_data_path = reduced_data_path
        self.artifacts_path = artifacts_path
        self.debug = debug
        self.catalog_name = catalog_name
        self.magnitude_threshold = magnitude_threshold
        self.magnitude_error_threshold = magnitude_error_threshold
        self.magnitude_range = [8, 22]

        self.plot_file_resolution = plot_file_resolution
        self.color_map = color_map

        self.log = logging.getLogger()

        self.sources = None
        self.data_quality_sources = None
        self._data_quality = {
            "fwhm": 0.0,
            "fwhm_error": 0.0,
            "ellipticity": 0.0,
            "ellipticity_error": 0.0
        }

        self._plot_artifacts = {

        }
        if not use_interactive_mpl_backend:
            matplotlib.use('Agg')



    def __call__(self, filename) -> None:
        """Process a FITS file for photometric calibration.

        This method processes a FITS file by performing the following steps:
        1. Loads the image data and header.
        2. Extracts metadata such as filter name, exposure time, and pixel scale.
        3. Normalizes the image data by exposure time.
        4. Creates a bad pixel mask to exclude saturated or defective pixels.
        5. Runs SExtractor to detect sources in the image.
        6. Performs data quality assessment on the detected sources.
        7. Executes photometric calibration using the detected sources.

        Args:
            filename (str): The path to the FITS file to be processed.

        Returns:
            None: The method does not return a value but updates internal attributes and saves outputs.

        Notes:
            - The method logs detailed information about the processing steps.
            - If `save_plots` is enabled, plots of the image and detections are saved.
            - The photometric calibration results are saved in the FITS header and output files.

        Example:
            >>> processor = PhotometryProcessor()
            >>> processor("observation.fits")
        """
        self.filename = filename

        if self.reduced_data_path is None or not os.path.isdir(self.reduced_data_path):
            self.reduced_data_path = os.path.dirname(self.filename)

        if self.artifacts_path is None or not os.path.isdir(self.artifacts_path):
            self.artifacts_path = self.reduced_data_path

        self.start = datetime.datetime.now()
        data = fits.getdata(self.filename).astype(np.double)
        header = fits.getheader(self.filename)
        width, height = data.shape

        wcs = check_wcs(header=header)
        center_ra, center_dec, fov_radius = get_frame_center(wcs=wcs,
                                                             image_width=width,
                                                             image_height=height)
        self.pixel_scale = get_pixel_scale(wcs=wcs)

        self.log.info(f"Frame center is {center_ra:.2f} {center_dec:.2f} "
                      f"radius {fov_radius * 60:.2f} arcmin,"
                      f" {self.pixel_scale * 36000:.3f} arcsec/pixel")

        (self.filter_name,
         serial_binning,
         parallel_binning,
         time,
         gain,
         read_noise,
         saturation_threshold,
         exposure_time) = extract_observation_metadata(header)

        self.log.info(f"Filter={self.filter_name}"
                      f" Exposure Time={exposure_time:.2f}"
                      f" Binning={serial_binning}x{parallel_binning}"
                      f"  Pixel Scale={self.pixel_scale * 3600}")

        self.log.info("Normalizing image data by exposure time.")
        data /= exposure_time

        self.log.info(f"Processing {self.filename}: filter: {self.filter_name} gain: {gain:.2f} at {time}")
        if self.save_plots:
            output_filename = get_new_file_name(current_file_name=self.filename,
                                                new_path=self.artifacts_path,
                                                new_extension="_phot_wcs.png")
            self._plot_artifacts["photometry_wcs"] = output_filename
            plot_image(
                image=data,
                wcs=wcs,
                title=self.filename.replace(".fits", ""),
                output_file=output_filename,
                dpi=self.plot_file_resolution,
                cmap=self.color_map,
                save_to_file=self.save_plots)

        bad_pixel_mask = create_bad_pixel_mask(
            image=data,
            saturation_threshold=saturation_threshold,
            binning=serial_binning)
        self.log.debug("Bad pixel mask created")

        self.run_sextractor(
            data=data,
            mask=bad_pixel_mask,
            gain=gain,
            pixel_scale=self.pixel_scale,
            wcs=wcs)

        self.data_quality_assessment(data=data)

        (output_file,
         elapsed_time,
         sources_table_html_file) = self.do_photometry(data=data,
                                                       header=header,
                                                       wcs=wcs,
                                                       center_ra=center_ra,
                                                       center_dec=center_dec,
                                                       fov_radius=fov_radius)

        return {
            "output_file": output_file,
            "elapsed_time": elapsed_time,
            "data_quality": self._data_quality,
            "sources_table_html_file": sources_table_html_file,
            "plots": self._plot_artifacts
        }

    @property
    def dq(self):
        """Get the data quality assessment results.

        This property provides access to the data quality assessment results, which include:
        - FWHM (Full Width at Half Maximum) of detected sources.
        - FWHM error.
        - Ellipticity of detected sources.
        - Ellipticity error.

        Returns:
            dict: A dictionary containing the data quality metrics. The keys are:
                - 'fwhm': The median FWHM of detected sources.
                - 'fwhm_error': The uncertainty in the FWHM measurement.
                - 'ellipticity': The median ellipticity of detected sources.
                - 'ellipticity_error': The uncertainty in the ellipticity measurement.
        """
        return self._data_quality

    @dq.setter
    def dq(self, results):
        """Set the data quality assessment results.

        This setter updates the data quality assessment results. It ensures that only valid
        float values are assigned to the respective keys in the `_data_quality` dictionary.

        Args:
            results (tuple): A tuple containing the following values in order:
                - fwhm (float): The median FWHM of detected sources.
                - fwhm_error (float): The uncertainty in the FWHM measurement.
                - ellipticity (float): The median ellipticity of detected sources.
                - ellipticity_error (float): The uncertainty in the ellipticity measurement.

        Notes:
            - Only float values are accepted for updating the data quality metrics.
            - If any value in the tuple is not a float, it is ignored.
        """
        fwhm, fwhm_error, ellipticity, ellipticity_error = results
        if isinstance(fwhm, float) or isinstance(fwhm, np.float32):
            self._data_quality["fwhm"] = float(fwhm)
        if isinstance(fwhm_error, float) or isinstance(fwhm_error, np.float32):
            self._data_quality["fwhm_error"] = float(fwhm_error)
        if isinstance(ellipticity, float) or isinstance(ellipticity, np.float32):
            self._data_quality["ellipticity"] = float(ellipticity)
        if isinstance(ellipticity_error, float) or isinstance(ellipticity_error, np.float32):
            self._data_quality["ellipticity_error"] = float(ellipticity_error)

    def run_sextractor(self, data, mask, gain, pixel_scale, wcs, seeing=1):
        """Run SExtractor to detect sources in the image.

        This method runs SExtractor to detect sources in the provided image data. It calculates the aperture
        size based on the seeing and pixel scale, detects sources, and optionally saves a plot of the detections.

        Args:
            data (numpy.ndarray): The image data array to be processed.
            mask (numpy.ndarray): A mask array to exclude certain regions from detection.
            gain (float): The gain of the image, used for SExtractor calculations.
            pixel_scale (float): The pixel scale of the image (in arcseconds per pixel).
            wcs (astropy.wcs.WCS): The World Coordinate System (WCS) information for the image.
            seeing (float, optional): The seeing value (in arcseconds) used to calculate the aperture size.
                                      Defaults to 1.

        Returns:
            astropy.table.Table: A table containing the detected sources, including their positions, fluxes,
                                 and other SExtractor measurements.

        Notes:
            - The aperture size is calculated using the seeing and pixel scale.
            - SExtractor flags are logged for diagnostic purposes.
            - If `save_plots` is enabled, a plot of the detected sources is saved.

        Example:
            >>> sources = run_sextractor(data, mask, gain=2.0, pixel_scale=0.2, wcs=wcs, seeing=1.5)
        """
        full_width_at_tenth_maximum_to_fwhm = 1.82
        aperture = np.round(full_width_at_tenth_maximum_to_fwhm * seeing / (pixel_scale * 3600.))
        self.log.info(f"SExtractor aperture radius: {aperture:.1f} pixels.")
        self.sources = get_objects_sextractor(image=data,
                                              mask=mask,
                                              gain=gain,
                                              r0=2,
                                              aper=aperture,
                                              thresh=1.0,
                                              wcs=wcs)
        self.log.info(f"SExtractor detections (1-sigma threshold): {len(self.sources)}")

        sextractor_flags = np.unique(self.sources['flags'])

        for flag in sextractor_flags:
            self.log.info(f"Flag={flag} - {np.sum(self.sources['flags'] == flag)}")

        if self.save_plots:
            output_filename = get_new_file_name(current_file_name=self.filename,
                                                new_path=self.artifacts_path,
                                                new_extension="_phot_detections.png")
            self._plot_artifacts['photometry_detections'] = output_filename
            plot_image(
                image=data,
                wcs=wcs,
                x_points=self.sources['x'],
                y_points=self.sources['y'],
                title="Detected sources",
                output_file=output_filename,
                dpi=self.plot_file_resolution,
                cmap=self.color_map,
                save_to_file=self.save_plots)
            self.log.info(f"SExtractor detections plot saved to: {output_filename}")
        return self.sources

    def data_quality_assessment(self, data):
        """Assess the quality of photometric data by analyzing source detections.

        This method evaluates the quality of detected sources using predefined criteria.
        It filters sources with `flags == 0` (high-quality detections) and, if enabled,
        generates a diagnostic plot highlighting these detections. It then computes
        key quality metrics such as FWHM (Full Width at Half Maximum) and ellipticity.

        Args:
            data (numpy.ndarray): The image data used for photometric analysis.

        Returns:
            dict: A dictionary containing the following data quality metrics:
                - `fwhm` (float): Median Full Width at Half Maximum of sources.
                - `fwhm_error` (float): Uncertainty in FWHM.
                - `ellipticity` (float): Median ellipticity of sources.
                - `ellipticity_error` (float): Uncertainty in ellipticity.

        Notes:
            - Only sources with `flags == 0` are considered for quality assessment.
            - If `save_plots` is enabled, a visualization of detected sources is saved.
            - The computed metrics are logged for further analysis.
        """
        self.data_quality_sources = self.sources[self.sources['flags'] == 0]
        if self.save_plots:
            output_filename = get_new_file_name(current_file_name=self.filename,
                                                new_path=self.artifacts_path,
                                                new_extension="_phot_detections_flag0.png")
            self._plot_artifacts['photometry_detections_flag0'] = output_filename
            plot_image(
                image=data,
                x_points=self.data_quality_sources['x'],
                y_points=self.data_quality_sources['y'],
                title="Detected sources (FLAG=0)",
                output_file=output_filename,
                dpi=self.plot_file_resolution,
                cmap=self.color_map,
                save_to_file=self.save_plots)
            self.log.info(f"SExtractor detections (flag=0) plot saved to: {output_filename}")
        self.dq = evaluate_data_quality_results(source_catalog=self.data_quality_sources)

        self.log.info("--------------------------")
        self.log.info("Data Quality Outputs")
        self.log.info(f"Number of Objects for Data Quality: {len(self.data_quality_sources)}/{len(self.sources)}")
        self.log.info(f"Median FWHM: {self.dq['fwhm']:.2f}+/-{self.dq['fwhm_error']:.2f} pixels")
        self.log.info(f"Median FWHM: {self.dq['fwhm'] * self.pixel_scale * 3600.}+/-{self.dq['fwhm_error'] * self.pixel_scale * 3600} arcsec")
        self.log.info(f"Median ellipticity: {self.dq['ellipticity']:.3f}+/-{self.dq['ellipticity_error']:.3f}")
        self.log.info("--------------------------")
        return self.dq

    def do_photometry(self, data, header, wcs, center_ra, center_dec, fov_radius):
        """Perform photometric calibration on Goodman HST data.

        This method calibrates the photometry of Goodman High Throughput Spectrograph (HST) observations
        using a specified catalog and filter. It retrieves the catalog data, performs photometric calibration,
        and saves the results, including calibrated magnitudes, plots, and updated FITS headers.

        Args:
            data (numpy.ndarray): The image data array to be processed.
            header (astropy.io.fits.Header): The FITS header associated with the image data.
            wcs (astropy.wcs.WCS): The World Coordinate System (WCS) information for the image.
            center_ra (float): Right ascension (RA) of the center of the field of view (in degrees).
            center_dec (float): Declination (Dec) of the center of the field of view (in degrees).
            fov_radius (float): Radius of the field of view (in degrees).

        Notes:
            - The method uses a catalog (e.g., 'gaiadr2') to retrieve reference stars for photometric calibration.
            - Calibration is performed using a specified filter, and results are saved in the FITS header.
            - Plots of the photometric calibration process are generated and saved if `save_plots` is enabled.
            - The calibrated data and updated header are saved to a new FITS file.

        Example:
            >>> do_photometry(data, header, wcs, 123.45, -67.89, 0.1)
        """
        catalog_filter, photometry_filter = get_filter_set(filter_name=self.filter_name)
        default_photometry_filter = 'Gmag'
        self.log.debug(f"Calibrating Goodman HST {self.filter_name} "
                       f"filter observations using {catalog_filter} magnitudes from {self.catalog_name} "
                       f"converted to {photometry_filter} filter.")
        catalog = get_vizier_catalog(right_ascension=center_ra,
                                     declination=center_dec,
                                     search_radius=fov_radius,
                                     catalog=self.catalog_name,
                                     column_filters={catalog_filter: f'<{self.magnitude_threshold}'})

        self.log.debug(f"{len(catalog)} catalogue stars on {catalog_filter} filter")
        self.log.info(f"Photometric calibration using {catalog_filter} "
                      f"magnitudes from {self.catalog_name} converted to {photometry_filter} filter")

        magnitudes = calibrate_photometry(
            object_table=self.sources,
            catalog_table=catalog,
            pixel_scale=self.pixel_scale,
            error_threshold=self.magnitude_error_threshold,
            magnitude_limits=self.magnitude_range,
            catalog_mag_column=catalog_filter,
            catalog_mag1_column=None,
            catalog_mag2_column=None,
            spatial_order=0,
            verbose=True)

        magnitudes_with_default_filter = calibrate_photometry(
            object_table=self.sources,
            catalog_table=catalog,
            pixel_scale=self.pixel_scale,
            error_threshold=self.magnitude_error_threshold,
            magnitude_limits=self.magnitude_range,
            catalog_mag_column=default_photometry_filter,
            catalog_mag1_column=None,
            catalog_mag2_column=None,
            spatial_order=0,
            verbose=True)

        check_photometry_results(magnitudes)

        self.sources['mag_calib'] = self.sources['mag'] + magnitudes['zero_fn'](self.sources['x'], self.sources['y'])
        self.sources['mag_calib_err'] = np.hypot(self.sources['magerr'], magnitudes['zero_fn'](self.sources['x'], self.sources['y'], get_err=True))
        sources_table = convert_match_results_to_table(self.sources,
                                                       pixscale=self.pixel_scale * 3600.,
                                                       columns=['x', 'y', 'xerr', 'yerr', 'flux', 'fluxerr', 'mag', 'magerr',
                                                                'a', 'b', 'theta', 'FLUX_RADIUS', 'fwhm', 'flags', 'bg',
                                                                'ra', 'dec', 'mag_calib', 'mag_calib_err'])

        sources_table_html_filename = get_new_file_name(current_file_name=self.filename,
                                                        new_path=self.artifacts_path,
                                                        new_extension="_obj_table.html")
        sources_table.write(sources_table_html_filename, format='html', overwrite=True)
        self.log.info(f"Table of sources used for photometric calibration is stored as {sources_table_html_filename}")

        # plot calibrated detections over the image

        calibrated_detections_plot_filename = get_new_file_name(current_file_name=self.filename,
                                                                new_path=self.artifacts_path,
                                                                new_extension="_phot_detections_calibrated.png")
        if self.save_plots:
            self._plot_artifacts['phot_detections_calibrated'] = calibrated_detections_plot_filename
        plot_photcal(image=data,
                     phot_table=sources_table,
                     wcs=wcs,
                     column_scale='mag_calib',
                     quantiles=(0.02, 0.98),
                     output_file=calibrated_detections_plot_filename,
                     dpi=self.plot_file_resolution,
                     save_to_file=self.save_plots)
        self.log.info(f"Photometric calibrated detections plotted over the image with WCS solution as "
                      f"{calibrated_detections_plot_filename}")

        plot_bins = np.round(2. * (fov_radius * 3600.) / 60.0)

        plt.figure()
        plot_photometric_match(match_result=magnitudes, mode='dist', bins=plot_bins)
        plt.tight_layout()
        if self.save_plots:
            photometry_match = get_new_file_name(current_file_name=self.filename,
                                                 new_path=self.artifacts_path,
                                                 new_extension="_phot_photmatch.png")
            self._plot_artifacts['photometry_match'] = photometry_match
            plt.savefig(photometry_match)

        plt.figure()
        plot_photometric_match(match_result=magnitudes)
        plt.tight_layout()
        if self.save_plots:
            photometry_match_2 = get_new_file_name(current_file_name=self.filename,
                                                   new_path=self.artifacts_path,
                                                   new_extension="_phot_photmatch2.png")
            self._plot_artifacts['photometry_match_2'] = photometry_match_2
            plt.savefig(photometry_match_2)

        plt.figure()
        plot_photometric_match(match_result=magnitudes,
                               mode='zero',
                               bins=plot_bins,
                               # Whether to show positions of the stars
                               show_points=True,
                               point_color='red',
                               aspect='equal')
        plt.title('Zero point')
        plt.tight_layout()
        if self.save_plots is True:
            photometry_zeropoint = get_new_file_name(current_file_name=self.filename,
                                                     new_path=self.artifacts_path,
                                                     new_extension="_phot_zp.png")
            self._plot_artifacts['photometry_zeropoint'] = photometry_zeropoint
            plt.savefig(photometry_zeropoint)

        # get photometric zero point estimate
        median_zeropoint, median_zeropoint_error = get_photometric_zeropoint(match_results=magnitudes)
        self.log.debug(f"Median empirical ZP: {median_zeropoint:.3f}+/-{median_zeropoint_error:.3f}")

        median_zeropoint_default, median_zeropoint_error_default = get_photometric_zeropoint(match_results=magnitudes_with_default_filter)
        self.log.debug(f"Median empirical ZP on {default_photometry_filter} "
                       f"filter: {median_zeropoint_default:.3f}+/-{median_zeropoint_error_default:.3f}")

        self.log.info(f"Median empirical ZP: {median_zeropoint:.3f}+/-{median_zeropoint_error:.3f}")

        header_out = header
        # results from Data Quality measurements (Sextractor)
        header_out.append(('GSP_NDET', len(self.sources), 'Number of SEXtractor sources'), end=True)
        header_out.append(
            ('GSP_NDDQ', len(self.data_quality_sources), 'Number of SEXtractor sources associated with FLAG=0 (used for DQ)'),
            end=True)
        header_out.append(('GSP_FWHM', float(self.dq['fwhm']),
                           'Median FWHM of SEXtractor sources associated with FLAG=0 (in pixels)'),
                          end=True)
        header_out.append(('GSP_FUNC', float(self.dq['fwhm_error']),
                           'Uncertainty on FWHM of SEXtractor sources associated with FLAG=0 (in pixels)'),
                          end=True)
        header_out.append(
            ('GSP_ELLI', float(self.dq['ellipticity']), 'Ellipticity of SEXtractor sources associated with FLAG=0'), end=True)
        header_out.append(('GSP_EUNC', float(self.dq['ellipticity_error']),
                           'Uncertainty on ellipticity of SEXtractor sources associated with FLAG=0'),
                          end=True)
        header_out.append(('GSP_SEEI', float(self.dq['fwhm'] * self.pixel_scale * 3600.),
                           'Median FWHM of SEXtractor sources associated with FLAG=0 (in arcsec)'),
                          end=True)
        header_out.append(('GSP_SUNC', float(self.dq['fwhm_error'] * self.pixel_scale * 3600.),
                           'Uncertainty on FWHM of SEXtractor sources associated with FLAG=0 (in arcsec)'),
                          end=True)
        # photometric calibration setup
        header_out.append(('GSP_PCAT', self.catalog_name, 'Catalog name used for photometric calibration'), end=True)
        header_out.append(('GSP_CFIL', catalog_filter,
                           'Filter name used for retrieving the catalog list for photometric calibration'),
                          end=True)
        header_out.append(('GSP_CMAG', float(self.magnitude_threshold),
                           'Magnitude threshold for PHOTCFIL used for photometric calibration'), end=True)
        header_out.append(('GSP_CMET', float(self.magnitude_error_threshold),
                           'Use catalog sources with magnitude errors smaller than this threshold for photometric calibration'),
                          end=True)
        header_out.append(('GSP_CMLM', str(self.magnitude_range),
                           'Magnitude range of catalog sources used for photometric calibration'), end=True)
        header_out.append(('GSP_PFIL', photometry_filter, 'Filter used for photometric calibration'), end=True)
        # header_out.append(('GSP_PCOL', color_term,                       'Using color-term for photometric calibration [True/False]'),                    end=True)
        # header_out.append(('GSP_COL1', phot_color_mag1,                  'Filter #1 used for color-term correction on the photometric calibration'),      end=True)
        # header_out.append(('GSP_COL2', phot_color_mag2,                  'Filter #2 used for color-term correction on the photometric calibration'),      end=True)
        # results from photometric calibration
        header_out.append(
            ('GSP_MZPT', float(median_zeropoint), 'Photometric zero point on {} filter'.format(photometry_filter)), end=True)
        header_out.append(('GSP_EZPT', float(median_zeropoint_error),
                           'Error of the photometric zero point on {} filter'.format(photometry_filter)), end=True)
        header_out.append(('GSP_MZPG', float(median_zeropoint_default),
                           'Photometric zero point on {} filter'.format(default_photometry_filter)), end=True)
        header_out.append(('GSP_EZPG', float(median_zeropoint_error_default),
                           'Error of the photometric zero point on {} filter'.format(default_photometry_filter)),
                          end=True)

        hdu = fits.PrimaryHDU(data=data, header=header_out)
        hdul = fits.HDUList([hdu])

        output_file = get_new_file_name(current_file_name=self.filename,
                                        new_path=self.reduced_data_path,
                                        new_extension="_phot.fits")
        hdul.writeto(output_file, overwrite=True)

        self.log.info(f"FITS file saved as {output_file}")

        # set start of the code
        end = datetime.datetime.now()
        elapsed_time = (end - self.start).total_seconds()
        self.log.info(f"Photometric calibration executed in {elapsed_time:.2f} seconds")

        #
        # Exit
        #
        print("")
        print("Photometric calibration was applied.")
        print("")

        return output_file, elapsed_time, sources_table_html_filename




def goodman_photometry():
    """Executes the photometry pipeline using user-defined arguments.

    This function initializes and runs the photometry process based on command-line
    arguments. It sets up logging, processes input parameters, and performs
    photometric analysis using the `Photometry` class.

    The main steps include:
    1. Parsing command-line arguments using `get_photometry_args()`.
    2. Configuring logging based on user preferences.
    3. Initializing a `Photometry` object with the specified parameters.
    4. Running the photometry pipeline on the provided input file.

    Args:
        None (all parameters are retrieved from command-line arguments).

    Returns:
        None

    Notes:
        - The function relies on `get_photometry_args()` to extract parameters.
        - Logging behavior is controlled by `setup_logging()`.
        - The `Photometry` class handles the photometry computation.
    """
    args = get_photometry_args()

    setup_logging(debug=args.debug, log_filename=args.log_filename)

    photometry = Photometry(
        catalog_name=args.catalog_name,
        magnitude_threshold=args.magnitude_threshold,
        magnitude_error_threshold=args.magnitude_error_threshold,
        color_map=args.color_map,
        plot_file_resolution=args.plot_file_resolution,
        save_plots=args.save_plots,
        reduced_data_path=args.reduced_data_path,
        artifacts_path=args.artifacts_path,
        debug=args.debug)
    photometry(filename=args.filename)
