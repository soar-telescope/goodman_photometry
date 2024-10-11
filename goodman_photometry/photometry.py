import logging
import matplotlib.pyplot as plt
import numpy as np
import datetime
import warnings

from astropy.io.fits.verify import VerifyWarning
from astropy.io import fits
from astropy.wcs import FITSFixedWarning

from .goodman_astro import (bpm_mask,
                            calibrate_photometry,
                            check_phot,
                            check_wcs,
                            dq_results,
                            filter_sets,
                            get_cat_vizier,
                            get_frame_center,
                            get_info,
                            get_objects_sextractor,
                            get_pixscale,
                            plot_photometric_match,
                            plot_photcal,
                            phot_table,
                            phot_zeropoint)

from .goodman_astro import imgshow as plot_image

from .utils import get_photometry_args, setup_logging

warnings.simplefilter(action='ignore', category=FITSFixedWarning)
warnings.simplefilter(action='ignore', category=VerifyWarning)


class Photometry(object):

    def __init__(self,
                 catalog_name='gaiadr2',
                 magnitude_threshold=17,
                 magnitude_error_threshold=0.1,
                 color_map='Blues_r',
                 plot_file_resolution=600,
                 save_plots=False,
                 debug=False) -> None:
        self.filename = None
        self.output_filename = None
        self.save_plots = save_plots
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

    @property
    def dq(self):
        return self._data_quality

    @dq.setter
    def dq(self, results):
        fwhm, fwhm_error, ellipticity, ellipticity_error = results
        if isinstance(fwhm, float):
            self._data_quality["fwhm"] = fwhm
        if isinstance(fwhm_error, float):
            self._data_quality["fwhm_error"] = fwhm_error
        if isinstance(ellipticity, float):
            self._data_quality["ellipticity"] = ellipticity
        if isinstance(ellipticity_error, float):
            self._data_quality["ellipticity_error"] = ellipticity_error

    def __call__(self, filename) -> None:
        self.filename = filename

        self.start = datetime.datetime.now()
        data = fits.getdata(self.filename).astype(np.double)
        header = fits.getheader(self.filename)
        width, height = data.shape

        wcs = check_wcs(header=header)
        center_ra, center_dec, fov_radius = get_frame_center(wcs=wcs,
                                                             width=width,
                                                             height=height)
        self.pixel_scale = get_pixscale(wcs=wcs)

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
         exposure_time) = get_info(header)

        self.log.info(f"Filter={self.filter_name}"
                      f" Exposure Time={exposure_time:.2f}"
                      f" Binning={serial_binning}x{parallel_binning}"
                      f"  Pixel Scale={self.pixel_scale * 3600}")

        self.log.info("Normalizing image data by exposure time.")
        data /= exposure_time

        self.log.info(f"Processing {self.filename}: filter: {self.filter_name} gain: {gain:.2f} at {time}")
        if self.save_plots:
            output_filename = self.filename.replace(".fits", "_phot_wcs.png")
            plot_image(image=data,
                       wcs=wcs,
                       title=filename.replace(".fits", ""),
                       output=output_filename,
                       dpi=self.plot_file_resolution,
                       cmap=self.color_map)

        bad_pixel_mask = bpm_mask(image=data,
                                  saturation=saturation_threshold,
                                  binning=serial_binning)
        self.log.debug("Bad pixel mask created")

        self.run_sextractor(
            data=data,
            mask=bad_pixel_mask,
            gain=gain,
            pixel_scale=self.pixel_scale,
            wcs=wcs)

        self.data_quality_assessment(data=data)

        self.do_photometry(data=data,
                           header=header,
                           wcs=wcs,
                           center_ra=center_ra,
                           center_dec=center_dec,
                           fov_radius=fov_radius)

    def run_sextractor(self, data, mask, gain, pixel_scale, wcs, seeing=1):
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
            output_filename = self.filename.replace(".fits", "_phot_detections.png")
            plot_image(image=data,
                       wcs=wcs,
                       px=self.sources['x'],
                       py=self.sources['y'],
                       title="Detected sources",
                       output=output_filename,
                       dpi=self.plot_file_resolution,
                       cmap=self.color_map,
                       pmarker='r.',
                       psize=2,
                       show_grid=False)
            self.log.info(f"SExtractor detections plot saved to: {output_filename}")
        return self.sources

    def data_quality_assessment(self, data):
        self.data_quality_sources = self.sources[self.sources['flags'] == 0]
        if self.save_plots:
            output_filename = self.filename.replace(".fits", "_phot_detections_flag0.png")
            plot_image(image=data,
                       wcs=None,
                       px=self.data_quality_sources['x'],
                       py=self.data_quality_sources['y'],
                       title="Detected sources (FLAG=0)",
                       output=output_filename,
                       dpi=self.plot_file_resolution,
                       cmap=self.color_map,
                       pmarker="r.",
                       psize=2,
                       show_grid=False)
            self.log.info(f"SExtractor detections (flag=0) plot saved to: {output_filename}")
        self.dq = dq_results(dq_obj=self.data_quality_sources)
        # fwhm, fwhm_error, ellipticity, ellipticity_error = dq_results(dq_obj=self.data_quality_sources)
        # self._data_quality["fwhm"] = fwhm
        # self._data_quality["fwhm_error"] = fwhm_error
        # self._data_quality["ellipticity"] = ellipticity
        # self._data_quality["ellipticity_error"] = ellipticity_error

        self.log.info("--------------------------")
        self.log.info("Data Quality Outputs")
        self.log.info(f"Number of Objects for Data Quality: {len(self.data_quality_sources)}/{len(self.sources)}")
        self.log.info(f"Median FWHM: {self.dq['fwhm']:.2f}+/-{self.dq['fwhm_error']:.2f} pixels")
        self.log.info(f"Median FWHM: {self.dq['fwhm'] * self.pixel_scale * 3600.}+/-{self.dq['fwhm_error'] * self.pixel_scale * 3600} arcsec")
        self.log.info(f"Median ellipticity: {self.dq['ellipticity']:.3f}+/-{self.dq['ellipticity_error']:.3f}")
        self.log.info("--------------------------")
        return self.dq

    def do_photometry(self, data, header, wcs, center_ra, center_dec, fov_radius):
        catalog_filter, photometry_filter = filter_sets(filter_name=self.filter_name)
        default_photometry_filter = 'Gmag'
        self.log.debug(f"Calibrating Goodman HST {self.filter_name} "
                       f"filter observations using {catalog_filter} magnitudes from {self.catalog_name} "
                       f"converted to {photometry_filter} filter.")
        catalog = get_cat_vizier(center_ra, center_dec, fov_radius, self.catalog_name,
                                 filters={catalog_filter: f'<{self.magnitude_threshold}'})

        self.log.debug(f"{len(catalog)} catalogue stars on {catalog_filter} filter")
        self.log.info(f"Photometric calibration using {catalog_filter} "
                      f"magnitudes from {self.catalog_name} converted to {photometry_filter} filter")

        magnitudes = calibrate_photometry(
            obj=self.sources,
            cat=catalog,
            pixscale=self.pixel_scale,
            ecmag_thresh=self.magnitude_error_threshold,
            cmag_limits=self.magnitude_range,
            cat_col_mag=photometry_filter,
            cat_col_mag1=None,
            cat_col_mag2=None,
            order=0,
            verbose=True)

        magnitudes_with_default_filter = calibrate_photometry(
            obj=self.sources,
            cat=catalog,
            pixscale=self.pixel_scale,
            ecmag_thresh=self.magnitude_error_threshold,
            cmag_limits=self.magnitude_range,
            cat_col_mag=default_photometry_filter,
            cat_col_mag1=None,
            cat_col_mag2=None,
            order=0,
            verbose=True)

        check_phot(magnitudes)

        self.sources['mag_calib'] = self.sources['mag'] + magnitudes['zero_fn'](self.sources['x'], self.sources['y'])
        self.sources['mag_calib_err'] = np.hypot(self.sources['magerr'], magnitudes['zero_fn'](self.sources['x'], self.sources['y'], get_err=True))
        sources_table = phot_table(self.sources,
                                   pixscale=self.pixel_scale * 3600.,
                                   columns=['x', 'y', 'xerr', 'yerr', 'flux', 'fluxerr', 'mag', 'magerr',
                                            'a', 'b', 'theta', 'FLUX_RADIUS', 'fwhm', 'flags', 'bg',
                                            'ra', 'dec', 'mag_calib', 'mag_calib_err'])

        sources_table_html_filename = self.filename.replace(".fits", "_obj_table.html")
        sources_table.write(sources_table_html_filename, format='html', overwrite=True)
        self.log.info(f"Table of sources used for photometric calibration is stored as {sources_table_html_filename}")

        # plot calibrated detections over the image
        calibrated_detections_plot_filename = self.filename.replace(".fits", "_phot_detections_calibrated.png")
        plot_photcal(image=data,
                     phot_table=sources_table,
                     wcs=wcs,
                     column_scale='mag_calib',
                     qq=(0.02, 0.98),
                     output=calibrated_detections_plot_filename,
                     dpi=self.plot_file_resolution)
        self.log.info(f"Photometric calibrated detections plotted over the image with WCS solution as "
                      f"{calibrated_detections_plot_filename}")

        plot_bins = np.round(2. * (fov_radius * 3600.) / 60.0)

        plt.figure()
        plot_photometric_match(m=magnitudes, mode='dist', bins=plot_bins)
        plt.tight_layout()
        if self.save_plots:
            plt.savefig(self.filename.replace(".fits", "_phot_photmatch.png"))

        plt.figure()
        plot_photometric_match(m=magnitudes)
        plt.tight_layout()
        if self.save_plots:
            plt.savefig(self.filename.replace(".fits", "_phot_photmatch2.png"))

        plt.figure()
        plot_photometric_match(m=magnitudes,
                               mode='zero',
                               bins=plot_bins,
                               # Whether to show positions of the stars
                               show_dots=True,
                               color='red',
                               aspect='equal')
        plt.title('Zero point')
        plt.tight_layout()
        if self.save_plots is True:
            plt.savefig(self.filename.replace(".fits", "_phot_zp.png"))

        # get photometric zero point estimate
        median_zeropoint, median_zeropoint_error = phot_zeropoint(m=magnitudes)
        self.log.debug(f"Median empirical ZP: {median_zeropoint:.3f}+/-{median_zeropoint_error:.3f}")

        median_zeropoint_default, median_zeropoint_error_default = phot_zeropoint(m=magnitudes_with_default_filter)
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
        photometry_filename = self.filename.replace(".fits", "_phot.fits")
        hdul.writeto(photometry_filename, overwrite=True)

        self.log.info(f"FITS file saved as {photometry_filename}")

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


def goodman_photometry():

    args = get_photometry_args()

    setup_logging(debug=args.debug, log_filename=args.log_filename)
    log = logging.getLogger()
    photometry = Photometry(
        catalog_name=args.catalog_name,
        magnitude_threshold=args.magnitude_threshold,
        magnitude_error_threshold=args.magnitude_error_threshold,
        color_map=args.color_map,
        plot_file_resolution=args.plot_file_resolution,
        save_plots=args.save_plots,
        debug=args.debug)
    photometry(filename=args.filename)
