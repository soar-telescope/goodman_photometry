import datetime
import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits as fits
from astropy.io.fits.verify import VerifyWarning
from astropy.wcs import FITSFixedWarning

from .goodman_astro import (get_info,
                            bpm_mask,
                            check_wcs,
                            clear_wcs,
                            dq_results,
                            filter_sets,
                            get_cat_vizier,
                            get_frame_center,
                            get_objects_sextractor,
                            get_pixscale,
                            goodman_wcs,
                            imgshow,
                            refine_wcs_scamp)
from .utils import get_astrometry_args, setup_logging

warnings.simplefilter(action='ignore', category=FITSFixedWarning)
warnings.simplefilter(action='ignore', category=VerifyWarning)


# Adjust default parameters for imshow
plt.rc('image', origin='lower', cmap='Blues_r')


class Astrometry(object):

    def __init__(self,
                 catalog_name='gaiadr2',
                 magnitude_threshold=17,
                 scamp_flag=1,
                 color_map='Blues_r',
                 save_plots=False,
                 save_scamp_plots=False,
                 save_intermediary_files=False,
                 debug=False):
        self.filename = None
        self.save_plots = save_plots
        self.debug = debug
        self.save_intermediary_files = save_intermediary_files
        self.save_scamp_plots = save_scamp_plots
        self.catalog_name = catalog_name
        self.magnitude_threshold = magnitude_threshold
        self.scamp_flag = scamp_flag
        self.color_map = color_map
        self.image = None
        self.header = None
        self.log = logging.getLogger()

    def __call__(self, filename):
        self.filename = filename

        self.start = datetime.datetime.now()
        self.log.info(f"Processing {self.filename}")

        try:

            self.image = fits.getdata(self.filename).astype(np.double)
            self.header = fits.getheader(self.filename)
        except FileNotFoundError:
            self.log.critical(f"File {self.filename} not found!!")
            raise SystemExit(-1)

        # gather required information from the header
        (self.filter_name,
         self.serial_binning,
         parallel_binning,
         time,
         self.gain,
         read_noise,
         self.saturation_threshold,
         exposure_time) = get_info(self.header)

        self.log.debug(f"Processing {self.filename}: "
                       f"filter {self.filter_name}, "
                       f"gain {self.gain:.2f}, "
                       f"saturation_threshold {self.saturation_threshold: .1f} at {time}")

        self.log.info(f"filter={self.filter_name} exposure_time={exposure_time:.2f} binning={self.serial_binning}x{parallel_binning}")

        self.__create_bad_pixel_mask()

        self.__create_basic_wcs_header()

        self.__detect_sources_with_sextractor()

        self.__obtain_astrometric_solution_with_scamp()

        self.__update_header()

        self.__save_to_fits_file()

    def __create_bad_pixel_mask(self):

        self.bad_pixel_mask = bpm_mask(self.image, self.saturation_threshold, self.serial_binning)

        self.log.debug('Done masking cosmics')

        if self.save_intermediary_files:
            hdu = fits.PrimaryHDU(data=self.bad_pixel_mask.astype(int), header=self.header)
            hdu_list = fits.HDUList([hdu])
            hdu_list.writeto(self.filename.replace(".fits", "_mask.fits"), overwrite=True)

        plot_image_filename = self.filename.replace(".fits", ".png")
        imgshow(image=self.image,
                wcs=None,
                title=self.filename.replace(".fits", ""),
                output=plot_image_filename,
                qq=(0.01, 0.99),
                cmap=self.color_map)
        self.log.info(f"Image - no WCS: {plot_image_filename}")

        plot_bad_pixel_mask_filename = self.filename.replace(".fits", "_BPM.png")
        imgshow(image=self.bad_pixel_mask,
                wcs=None,
                title="Bad pixel mask",
                output=plot_bad_pixel_mask_filename,
                qq=(0, 1),
                cmap=self.color_map)
        self.log.info(f"Image - bad pixel mask: {plot_bad_pixel_mask_filename}")

    def __create_basic_wcs_header(self):
        header_with_basic_wcs = goodman_wcs(self.header)
        self.wcs_init = check_wcs(header_with_basic_wcs)

        if self.save_intermediary_files:
            hdu = fits.PrimaryHDU(data=self.image, header=header_with_basic_wcs)
            hdu_list = fits.HDUList([hdu])
            hdu_list.writeto(self.filename.replace(".fits", "_wcs_init.fits"), overwrite=True)

        self.ra_0, self.dec_0, self.fov_radius = get_frame_center(wcs=self.wcs_init, width=self.image.shape[1], height=self.image.shape[0])
        self.image_pixel_scale = get_pixscale(wcs=self.wcs_init)

        self.log.info(f"Initial WCS: RA={self.ra_0} DEC={self.dec_0} SR={self.fov_radius} PIXSCALE={self.image_pixel_scale}")

    def __detect_sources_with_sextractor(self):
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

        plot_detections_filename = self.filename.replace(".fits", "_detections.png")

        imgshow(image=self.image,
                wcs=None,
                px=self.sources['x'],
                py=self.sources['y'],
                title='Detected objects',
                output=plot_detections_filename,
                qq=(0.01, 0.99),
                cmap=self.color_map,
                pmarker='r.',
                psize=2,
                show_grid=False)

        self.log.info(f"Image - SExtractor detections: {plot_detections_filename}")

    def __data_quality_assessment(self):
        data_quality_sources = self.sources[self.sources['flags'] == 0]

        plot_detections_flag_0_filename = self.filename.replace(".fits", "_detections_flag_0.png")

        imgshow(image=self.image,
                wcs=None,
                px=data_quality_sources['x'],
                py=data_quality_sources['y'],
                title='Detected objects (FLAG=0)',
                output=plot_detections_flag_0_filename,
                qq=(0.01, 0.99),
                cmap=self.color_map,
                pmarker='r.',
                psize=2,
                show_grid=False)

        self.log.info(f"Image - Detected objects (FLAG=0): {plot_detections_flag_0_filename}")

        fwhm, fwhm_error, ellipticity, ellipticity_error = dq_results(data_quality_sources)

        self.log.info("Data quality results")
        self.log.info(f"Number of objects: {len(data_quality_sources)}/{len(self.sources)}")
        self.log.info(f"Median FWHM: {fwhm:.2f}+/-{fwhm_error:.2f} pixels")
        self.log.info(f"Median FWHM: {fwhm * self.image_pixel_scale * 3600.:.2f}+/-{fwhm_error * self.image_pixel_scale * 3600.:.2f} arcseconds")
        self.log.info(f"Median Ellipticity: {ellipticity:.3f}+/-{ellipticity_error:.3f}")

    def __obtain_astrometric_solution_with_scamp(self):
        self.log.info(f"Performing astrometry with SCAMP using {self.catalog_name}")
        self.catalog_filter, _ = filter_sets(self.filter_name)

        self.log.info(f"Querying Vizier for {self.catalog_name} catalog")

        vizier_catalog = get_cat_vizier(ra0=self.ra_0,
                                        dec0=self.dec_0,
                                        sr0=self.fov_radius,
                                        catalog=self.catalog_name,
                                        filters={self.catalog_filter: f'<{self.magnitude_threshold}'})

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

        outgoing_filename = self.filename.replace(".fits", "_wcs.fits")
        hdu = fits.PrimaryHDU(data=self.image, header=self.outgoing_header)
        hdu_list = fits.HDUList([hdu])
        hdu_list.writeto(outgoing_filename, overwrite=True)

        self.log.info(f"FITS file saved as {outgoing_filename}")

        end = datetime.datetime.now()

        self.log.info(f"Astrometric calibration executed in {(end - self.start).total_seconds():.2f} seconds")

        self.log.info('Astrometric calibration finished.')


def goodman_astrometry():
    """Entrypoint for astrometry calculation
    """
    args = get_astrometry_args()

    setup_logging(debug=args.debug, log_filename=args.log_filename)
    log = logging.getLogger()
    astrometry = Astrometry(
        catalog_name=args.catalog_name,
        magnitude_threshold=args.magnitude_threshold,
        scamp_flag=args.scamp_flag,
        color_map=args.color_map,
        save_plots=args.save_plots,
        save_scamp_plots=args.save_scamp_plots,
        save_intermediary_files=args.save_intermediary_files,
        debug=args.debug)

    astrometry(filename=args.filename)
