"""A module for Goodman photometry and astrometry data processing.

This module provides functions for processing photometric data from the Goodman telescope,
including FITS header handling, WCS management, object detection, photometric calibration,
and data visualization. It integrates with various astronomical data reduction tools
and libraries like Astropy and SExtractor.

Key Features:
    - FITS header metadata extraction and manipulation
    - WCS (World Coordinate System) handling and conversion
    - Object detection using SExtractor
    - Photometric calibration and zero-point calculation
    - Catalog matching and data quality evaluation
    - Visualization tools for images and photometric results
    - Integration with Vizier catalog services

Main Functions:
    extract_observation_metadata(): Extracts observation metadata from FITS headers.
    calculate_saturation_threshold(): Calculates saturation threshold based on gain/read noise.
    check_wcs(): Validates and returns WCS from a FITS header.
    get_objects_sextractor(): Detects objects in an image using SExtractor.
    calibrate_photometry(): Performs photometric calibration using reference catalogs.
    get_vizier_catalog(): Retrieves catalog data from Vizier services.
    plot_image(): Plots 2D images with optional WCS projection.

The module is designed to work within the Goodman data reduction pipeline,
providing tools for accurate photometric and astrometric calibration.
"""
import logging
import os
import re
import shlex
import shutil
import sys
import tempfile

import astroscrappy
import dateutil
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import sip_tpv

from astropy import units as u
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.io import fits as fits
from astropy.io.fits import Header
from astropy.table import Table
from astropy.time import Time
from astropy.visualization import simple_norm
from astropy.wcs import WCS
from astroquery.vizier import Vizier
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import binned_statistic_2d
from scipy.stats import chi2

log = logging.getLogger()


CATALOGS = {
    'ps1': {
        'vizier': 'II/349/ps1',
        'name': 'PanSTARRS DR1'
    },
    'gaiadr2': {
        'vizier': 'I/345/gaia2',
        'name': 'Gaia DR2',
        'extra': ['E(BR/RP)']
    },
    'gaiaedr3': {
        'vizier': 'I/350/gaiaedr3',
        'name': 'Gaia EDR3'
    },
    'gaiadr3syn': {
        'vizier': 'I/360/syntphot',
        'name': 'Gaia DR3 synthetic photometry',
        'extra': ['**', '_RAJ2000', '_DEJ2000']
    },
    'usnob1': {
        'vizier': 'I/284/out',
        'name': 'USNO-B1'
    },
    'gsc': {
        'vizier': 'I/271/out',
        'name': 'GSC 2.2'
    },
    'skymapper': {
        'vizier': 'II/358/smss',
        'name': 'SkyMapper DR1.1',
        'extra': [
            '_RAJ2000', '_DEJ2000', 'e_uPSF', 'e_vPSF',
            'e_gPSF', 'e_rPSF', 'e_iPSF', 'e_zPSF'
        ]
    },
    'vsx': {
        'vizier': 'B/vsx/vsx',
        'name': 'AAVSO VSX'
    },
    'apass': {
        'vizier': 'II/336/apass9',
        'name': 'APASS DR9'
    },
    'sdss': {
        'vizier': 'V/147/sdss12',
        'name': 'SDSS DR12',
        'extra': ['_RAJ2000', '_DEJ2000']
    },
    'atlas': {
        'vizier': 'J/ApJ/867/105/refcat2',
        'name': 'ATLAS-REFCAT2',
        'extra': [
            '_RAJ2000', '_DEJ2000', 'e_Gmag', 'e_gmag',
            'e_rmag', 'e_imag', 'e_zmag', 'e_Jmag', 'e_Kmag'
        ]
    }
}


def extract_observation_metadata(header):
    """Extract observation metadata from a FITS header and ensure the wavelength mode is IMAGING.

    Args:
        header (astropy.io.fits.Header): The FITS header containing observation metadata.

    Returns:
        tuple: A tuple containing:
            - filter_name (str): The active filter name, determined from the header.
            - serial_binning (int): Binning factor in the serial direction.
            - parallel_binning (int): Binning factor in the parallel direction.
            - observation_time (str): The observation time, extracted from the header.
            - gain (float): The detector gain (e-/ADU).
            - read_noise (float): The detector read noise (e-).
            - saturation_threshold (float): The saturation threshold (in ADU),
              calculated based on the readout mode.
            - exposure_time (float): The exposure time (in seconds).

    Raises:
        SystemExit: If the wavelength mode (`WAVMODE`) is not set to "IMAGING".
    """
    # Ensure observations are in IMAGING mode; exit if not.
    wavelength_mode = header.get('WAVMODE')
    if wavelength_mode != 'IMAGING':
        raise ValueError("Error: WAVMODE is not IMAGING. No data to process.")

    # Determine the active filter name, considering both filter wheels.
    primary_filter = header.get('FILTER')
    secondary_filter = header.get('FILTER2')
    filter_name = primary_filter if primary_filter != "NO_FILTER" else secondary_filter

    # Extract binning information.
    serial_binning, parallel_binning = (int(value) for value in header['CCDSUM'].split())

    # Retrieve observation time.
    observation_time = get_observation_time(header)

    # Get gain, read noise, and calculate saturation threshold.
    gain = header.get('GAIN')
    read_noise = header.get('RDNOISE')
    saturation_threshold = calculate_saturation_threshold(gain, read_noise)

    # Retrieve exposure time.
    exposure_time = header.get('EXPTIME')

    return (filter_name, serial_binning, parallel_binning, observation_time,
            gain, read_noise, saturation_threshold, exposure_time)


def calculate_saturation_threshold(gain_value, read_noise_value):
    """Estimates the saturation threshold based on the readout mode.

    Args:
        gain_value (float): The detector gain (e-/ADU).
        read_noise_value (float): The detector read noise (e-).

    Returns:
        float: The saturation threshold in ADU, estimated based on the provided
        gain and read noise values.
    """
    if gain_value == 1.54 and read_noise_value == 3.45:
        saturation_threshold = 50000  # 100kHzATTN3
    elif gain_value == 3.48 and read_noise_value == 5.88:
        saturation_threshold = 25000  # 100kHzATTN2
    elif gain_value == 1.48 and read_noise_value == 3.89:
        saturation_threshold = 50000  # 344kHzATTN3
    elif gain_value == 3.87 and read_noise_value == 7.05:
        saturation_threshold = 25000  # 344kHzATTN0
    elif gain_value == 1.47 and read_noise_value == 5.27:
        saturation_threshold = 50000  # 750kHzATTN2
    elif gain_value == 3.77 and read_noise_value == 8.99:
        saturation_threshold = 25000  # 750kHzATTN0
    else:
        saturation_threshold = 50000  # Default value

    return saturation_threshold


def check_wcs(header: Header) -> WCS:
    """Check whether a FITS header contains a valid celestial WCS solution.

    Args:
        header (astropy.io.fits.Header): FITS header to check for WCS.

    Returns:
        astropy.wcs.WCS: Parsed WCS object from the header.

    Raises:
        ValueError: If WCS is absent or not celestial.
    """
    wcs = WCS(header)

    if wcs is None or not wcs.is_celestial:
        raise ValueError("WCS is absent or non-celestial. Cannot perform photometry.")

    return wcs


def wcs_sip2pv(header):
    """Convert the WCS header from SIP (Simple Imaging Polynomial) to TPV (Tangent Plane Polynomial) representation.

    This function modifies the input FITS header to replace SIP distortion keywords with TPV distortion keywords.
    It ensures the presence of the CD matrix by converting the PC matrix if necessary.

    Args:
        header (astropy.io.fits.Header or dict): The FITS header containing SIP distortion keywords.

    Returns:
        astropy.io.fits.Header or dict: The modified header with TPV distortion keywords.
    """
    # Create a copy of the header to avoid modifying the original
    header = header.copy()

    # If the CD matrix is not present but the PC matrix is, convert PC to CD
    if 'CD1_1' not in header and all(key in header for key in ['PC1_1', 'PC2_1', 'PC1_2', 'PC2_2', 'CDELT1', 'CDELT2']):
        # Retrieve the CDELT values (scaling factors for the axes)
        cdelt_values = [header.get('CDELT1'), header.get('CDELT2')]

        # Convert PC matrix to CD matrix by multiplying with CDELT
        header['CD1_1'] = header.pop('PC1_1') * cdelt_values[0]
        header['CD2_1'] = header.pop('PC2_1') * cdelt_values[0]
        header['CD1_2'] = header.pop('PC1_2') * cdelt_values[0]
        header['CD2_2'] = header.pop('PC2_2') * cdelt_values[0]

    # Check if the header contains the required SIP keywords for conversion
    if all(key in header for key in ['A_ORDER', 'B_ORDER', 'A_0_2', 'A_2_0', 'B_0_2', 'B_2_0']):
        # Convert SIP distortion keywords to TPV representation
        sip_tpv.sip_to_pv(header)
    else:
        # If SIP keywords are missing, log a warning or skip the conversion
        import warnings
        warnings.warn("SIP keywords are missing. Skipping SIP to TPV conversion.")

    return header


def check_photometry_results(results: dict) -> dict:
    """Check whether photometric calibration results are available.

    Args:
        results (dict): Output dictionary from `calibrate_photometry()`.

    Returns:
        dict: The same results dictionary if not None.

    Raises:
        ValueError: If `results` is None, indicating photometric calibration failed.
    """
    if results is None:
        raise ValueError("Photometric calibration results are missing or invalid.")

    return results


def get_filter_set(filter_name: str) -> tuple[str, str]:
    """Determine the catalog filter and corresponding photometric filter.

    Determine the catalog filter and corresponding photometric filter for calibration,
    based on the Goodman filter in use.

    Args:
        filter_name (str): Goodman filter name (from FITS header FILTER/FILTER2 keywords).

    Returns:
        tuple[str, str]: A tuple containing:
            - catalog_filter (str): The catalog filter (e.g., Gaia Gmag, BPmag) to retrieve.
            - photometry_filter (str): The photometric filter used for calibration (e.g., g_SDSS, r_SDSS).

    Notes:
        Currently supports only SDSS filters (u, g, r, i, z).
        Future improvements:
            - Add support for Bessel UBVRI, Johnson UBV, Stromgren ubvy, and Kron-Cousins Rc filters.
            - Handle narrow-band filters separately.
    """
    if filter_name == "u-SDSS":
        catalog_filter = "BPmag"
        photometry_filter = "u_SDSS"
    elif filter_name == "g-SDSS":
        catalog_filter = "BPmag"
        photometry_filter = "g_SDSS"
    elif filter_name == "r-SDSS":
        catalog_filter = "Gmag"
        photometry_filter = "r_SDSS"
    elif filter_name in ("i-SDSS", "z-SDSS"):
        catalog_filter = "Gmag"
        photometry_filter = "i_SDSS"
    else:
        # Default fallback for unsupported filters
        catalog_filter = "Gmag"
        photometry_filter = "g_SDSS"

    return catalog_filter, photometry_filter


def get_new_file_name(current_file_name: str, new_path: str = "", new_extension: str = "") -> str:
    """
    Generate a new file path with an optional new directory and/or file extension.

    Args:
        current_file_name (str): The original file name with its path.
        new_path (str, optional): The new directory for the file. Defaults to the original directory.
        new_extension (str, optional): The new file extension (can include an underscore, e.g., '_BMP.png').
                                      If provided, it replaces the existing extension.

    Returns:
        str: The updated file path.

    Raises:
        ValueError: If current_file_name is empty.
    """
    file_path = Path(current_file_name)
    if not file_path.name:  # Ensures the filename is not empty
        raise ValueError("File path is empty.")

    # Replace the entire extension with new_extension if provided
    if new_extension:
        if "." not in new_extension:
            new_extension = "." + new_extension
        new_filename = file_path.stem + new_extension  # Removes old extension and appends new_extension
    else:
        new_filename = file_path.name  # Keep the original filename

    # Use new_path if provided, otherwise keep the original directory
    target_directory = Path(new_path) if new_path else file_path.parent

    return str(target_directory / new_filename)


def create_goodman_wcs(header):
    """Create WCS from a Header.

    Creates a WCS (World Coordinate System) guess using telescope coordinates,
    binning, position angle, and plate scale.

    Args:
        header (astropy.io.fits.Header): The FITS header containing necessary metadata.

    Returns:
        astropy.io.fits.Header: Updated FITS header with WCS information.

    Raises:
        ValueError: If neither "RA"/"DEC" nor "TELRA"/"TELDEC" are present in the header.
    """
    # Set default EQUINOX and EPOCH if not provided
    header.setdefault("EQUINOX", 2000.0)
    header.setdefault("EPOCH", 2000.0)

    # Parse CCD binning
    try:
        serial_binning, parallel_binning = (int(b) for b in header["CCDSUM"].split())
    except KeyError:
        raise ValueError("Header missing 'CCDSUM' keyword for binning information.")

    # Calculate pixel scales
    header["PIXSCAL1"] = -serial_binning * 0.15  # arcsec (for Swarp)
    header["PIXSCAL2"] = parallel_binning * 0.15  # arcsec (for Swarp)

    if abs(header["PIXSCAL1"]) != abs(header["PIXSCAL2"]):
        log.warning("Pixel scales for X and Y axes do not match.")

    plate_scale_in_degrees = (abs(header["PIXSCAL1"]) * u.arcsec).to("degree").value
    wcs_instance = WCS(naxis=2)

    # Determine RA and DEC coordinates
    try:
        coordinates = SkyCoord(
            ra=header["RA"], dec=header["DEC"], unit=(u.hourangle, u.deg)
        )
    except KeyError:
        try:
            log.error(
                '"RA" and "DEC" missing. Falling back to "TELRA" and "TELDEC".'
            )
            coordinates = SkyCoord(
                ra=header["TELRA"], dec=header["TELDEC"], unit=(u.hourangle, u.deg)
            )
        except KeyError:
            raise ValueError(
                'Header must contain either "RA"/"DEC" or "TELRA"/"TELDEC".'
            )

    # Set WCS parameters
    wcs_instance.wcs.crpix = [header["NAXIS2"] / 2, header["NAXIS1"] / 2]
    wcs_instance.wcs.cdelt = [+plate_scale_in_degrees, +plate_scale_in_degrees]
    wcs_instance.wcs.crval = [coordinates.ra.deg, coordinates.dec.deg]
    wcs_instance.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    # Update header with WCS information
    wcs_header = wcs_instance.to_header()
    header.update(wcs_header)

    return header


def mask_field_of_view(image, binning):
    """Masks out the edges of the field of view (FOV) in Goodman images.

    Args:
        image (numpy.ndarray): The image array from the FITS file.
        binning (int): The binning factor (e.g., 1, 2, 3).

    Returns:
        numpy.ndarray: A boolean mask with the same dimensions as 'image'.
                       Pixels outside the FOV are masked (True), and others are unmasked (False).
    """
    binning_centers = {
        1: (1520, 1570, 1550),
        2: (770, 800, 775),
        3: (510, 540, 515),
    }
    center_x, center_y, radius = binning_centers.get(
        binning,
        (image.shape[0] / 2, image.shape[1] / 2, image.shape[0] / 2),
    )
    x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    return distance > radius


def create_bad_pixel_mask(image, saturation_threshold, binning):
    """Create a comprehensive bad pixel mask.

    Creates a comprehensive bad pixel mask, identifying and masking saturated pixels, cosmic rays,
    and pixels outside the circular field of view (FOV) of Goodman images.

    Args:
        image (numpy.ndarray): 2D array representing the image data from a FITS file.
        saturation_threshold (int): Saturation threshold derived from `calculate_saturation_threshold`.
        binning (int): Binning factor of the data (1, 2, 3, etc.).

    Returns:
        numpy.ndarray: Boolean mask with the same dimensions as `image`.
                       Pixels are masked (True) if they are bad, otherwise unmasked (False).
    """
    # Mask saturated pixels
    mask = image > saturation_threshold

    # Identify and mask cosmic rays
    cosmic_ray_mask, _ = astroscrappy.detect_cosmics(image, mask)
    mask |= cosmic_ray_mask

    # Mask pixels outside the field of view
    mask |= mask_field_of_view(image, binning)

    return mask


def spherical_distance(ra_deg_1, dec_deg_1, ra_deg_2, dec_deg_2):
    """Calculate the spherical angular distance between two celestial coordinates.

    This function computes the great-circle distance between two points
    on the celestial sphere using a trigonometric approximation formula.

    Args:
        ra_deg_1 (float or np.ndarray): Right ascension of the first point(s) in degrees.
        dec_deg_1 (float or np.ndarray): Declination of the first point(s) in degrees.
        ra_deg_2 (float or np.ndarray): Right ascension of the second point(s) in degrees.
        dec_deg_2 (float or np.ndarray): Declination of the second point(s) in degrees.

    Returns:
        float or np.ndarray: Spherical angular distance(s) in degrees between the coordinate pairs.
    """
    delta_ra_rad = np.deg2rad((ra_deg_1 - ra_deg_2) / 2.0)
    delta_dec_rad = np.deg2rad((dec_deg_1 - dec_deg_2) / 2.0)

    sin_delta_ra_squared = np.sin(delta_ra_rad) ** 2
    sin_delta_dec_squared = np.sin(delta_dec_rad) ** 2
    cos_avg_dec_squared = np.cos(np.deg2rad((dec_deg_1 + dec_deg_2) / 2.0)) ** 2

    angular_distance_rad = 2.0 * np.arcsin(
        np.sqrt(sin_delta_ra_squared * (cos_avg_dec_squared - sin_delta_dec_squared) + sin_delta_dec_squared)
    )

    return np.rad2deg(angular_distance_rad)


def spherical_match(
    ra_deg_1,
    dec_deg_1,
    ra_deg_2,
    dec_deg_2,
    search_radius_deg=1 / 3600
):
    """Perform spherical positional matching between two lists of celestial coordinates.

    Uses `astropy.coordinates.search_around_sky` to identify matches between two
    sets of coordinates within a given angular search radius.

    Args:
        ra_deg_1 (float or np.ndarray): Right ascension of the first list (in degrees).
        dec_deg_1 (float or np.ndarray): Declination of the first list (in degrees).
        ra_deg_2 (float or np.ndarray): Right ascension of the second list (in degrees).
        dec_deg_2 (float or np.ndarray): Declination of the second list (in degrees).
        search_radius_deg (float, optional): Maximum allowed angular separation (in degrees)
            to be considered a match. Default is 1 arcsecond (1/3600 degrees).

    Returns:
        tuple: A tuple of three arrays:
            - matched_indices_1 (np.ndarray): Indices in the first list of matched coordinates.
            - matched_indices_2 (np.ndarray): Indices in the second list of matched coordinates.
            - matched_distances_deg (np.ndarray): Angular distances (in degrees) between matched pairs.
    """
    coordinates_1 = SkyCoord(ra_deg_1, dec_deg_1, unit='deg')
    coordinates_2 = SkyCoord(ra_deg_2, dec_deg_2, unit='deg')

    matched_indices_1, matched_indices_2, matched_distances, _ = search_around_sky(
        coordinates_1,
        coordinates_2,
        search_radius_deg * u.deg
    )

    matched_distances_deg = matched_distances.deg
    return matched_indices_1, matched_indices_2, matched_distances_deg


def get_frame_center(
    filename: str = None,
    header: fits.Header = None,
    wcs: WCS = None,
    image_width: int = None,
    image_height: int = None,
    image_shape: tuple = None
):
    """Calculate the central coordinates (RA, Dec) and field radius of an image.

    The function accepts either a WCS object, a FITS header, or a FITS file path.
    If image dimensions are not provided, they will be extracted from the header
    or the image shape.

    Args:
        filename (str, optional): Path to the FITS file.
        header (astropy.io.fits.Header, optional): FITS header object.
        wcs (astropy.wcs.WCS, optional): World Coordinate System object.
        image_width (int, optional): Width of the image in pixels.
        image_height (int, optional): Height of the image in pixels.
        image_shape (tuple, optional): Image shape as (height, width), used as fallback.

    Returns:
        tuple: A tuple of (ra_center, dec_center, search_radius), in degrees.
            If a valid celestial WCS is not available, all values will be None.
    """
    if wcs is None:
        if header is not None:
            wcs = WCS(header)
        elif filename is not None:
            header = fits.getheader(filename, -1)
            wcs = WCS(header)

    if image_width is None or image_height is None:
        if header is not None:
            image_width = header.get('NAXIS1')
            image_height = header.get('NAXIS2')
        if (image_width is None or image_height is None) and image_shape is not None:
            image_height, image_width = image_shape

    if wcs is None or not wcs.is_celestial or image_width is None or image_height is None:
        return None, None, None

    # Compute RA/Dec at center and top-center pixel for radius estimation
    ra_top, dec_top = wcs.all_pix2world(image_width / 2.0, 0.0, 0)
    ra_center, dec_center = wcs.all_pix2world(image_width / 2.0, image_height / 2.0, 0)

    search_radius = spherical_distance(ra_center, dec_center, ra_top, dec_top)

    return ra_center.item(), dec_center.item(), search_radius.item()


# phot (STDPipe)
def get_objects_sextractor(
        image,
        header=None,
        mask=None,
        err=None,
        thresh=2.0,
        aper=3.0,
        r0=0.0,
        gain=1,
        edge=0,
        minarea=5,
        wcs=None,
        sn=3.0,
        bg_size=None,
        sort=True,
        reject_negative=True,
        checkimages=[],
        extra_params=[],
        extra={},
        psf=None,
        catfile=None,
        _workdir=None,
        _tmpdir=None,
        _exe=None,
        verbose=False):
    """Extract objects from an image using SExtractor.

    This function is a thin wrapper around the SExtractor binary. It processes the image, taking into account optional mask and noise map, and returns a list of detected objects. Optionally, it can also return SExtractor-produced checkimages.

    For more details about SExtractor parameters and principles of operation, refer to the SExtractor documentation at https://sextractor.readthedocs.io/en/latest/. Detection flags (returned in the `flags` column of the results table) are documented at https://sextractor.readthedocs.io/en/latest/Flagging.html#extraction-flags-flags. Additionally, any object with pixels masked by the input `mask` in its footprint will have the `0x100` flag set.

    Args:
        image (numpy.ndarray): Input image as a NumPy array.
        header (astropy.io.fits.Header, optional): Image header.
        mask (numpy.ndarray, optional): Image mask as a boolean array (True values will be masked).
        err (numpy.ndarray, optional): Image noise map as a NumPy array.
        thresh (float, optional): Detection threshold in sigmas above local background. Used for `DETECT_THRESH` parameter. Default is 2.0.
        aper (float or list, optional): Circular aperture radius in pixels for flux measurement. If a list is provided, flux is measured for all apertures. Default is 3.0.
        r0 (float, optional): Smoothing kernel size (sigma, or FWHM/2.355) for improving object detection. Default is 0.0.
        gain (float, optional): Image gain in e/ADU. Default is 1.
        edge (int, optional): Reject objects closer to the image edge than this value. Default is 0.
        minarea (int, optional): Minimum number of pixels for an object to be considered a detection (`DETECT_MINAREA`). Default is 5.
        wcs (astropy.wcs.WCS, optional): Astrometric solution for assigning sky coordinates (`ra`/`dec`) to detected objects.
        sn (float, optional): Minimum S/N ratio for an object to be considered a detection. Default is 3.0.
        bg_size (int, optional): Background grid size in pixels (`BACK_SIZE`). Default is None.
        sort (bool, optional): Whether to sort detections by decreasing brightness. Default is True.
        reject_negative (bool, optional): Whether to reject detections with negative fluxes. Default is True.
        checkimages (list, optional): List of SExtractor checkimages to return. Default is [].
        extra_params (list, optional): List of extra object parameters to return. Default is [].
        extra (dict, optional): Dictionary of extra configuration parameters for SExtractor. Default is {}.
        psf (str, optional): Path to PSFEx-made PSF model file for PSF photometry. Default is None.
        catfile (str, optional): Path to save the output SExtractor catalog. Default is None.
        _workdir (str, optional): Directory for temporary files. If specified, files are not deleted. Default is None.
        _tmpdir (str, optional): Directory for temporary files. Files are deleted after execution. Default is None.
        _exe (str, optional): Path to SExtractor executable. If not provided, the function searches for it in the system PATH. Default is None.
        verbose (bool or callable, optional): Whether to show verbose messages. Can be a boolean or a `print`-like function. Default is False.

    Returns:
        astropy.table.Table or list: Table of detected objects. If checkimages are requested, returns a list with the table as the first element and checkimages as subsequent elements.
    """
    # Find the binary
    binname = None

    if _exe is not None:
        # Check user-provided binary path, and fail if not found
        if os.path.isfile(_exe):
            binname = _exe
    else:
        # Find SExtractor binary in common paths
        for exe in ['sex', 'sextractor', 'source-extractor']:
            binname = shutil.which(exe)
            if binname is not None:
                break
    if binname is None:
        log.critical("Can't find SExtractor binary")
        raise SystemError("Can't find SExtractor binary")
    # else:
    #     log.info("Using SExtractor binary at", binname)

    workdir = (
        _workdir
        if _workdir is not None
        else tempfile.mkdtemp(prefix='sex', dir=_tmpdir)
    )
    obj = None

    if mask is None:
        # Create minimal mask
        mask = ~np.isfinite(image)
    else:
        # Ensure the mask is boolean array
        mask = mask.astype(bool)

    # now mask the bad pixels and region outside FOV
    image = image.copy()
    image[mask] = np.nan

    # Prepare
    if type(image) is str:
        # FIXME: this mode of operation is currently broken!
        imagename = image
    else:
        imagename = os.path.join(workdir, 'image.fits')
        fits.writeto(imagename, image, header, overwrite=True)

    # Dummy config filename, to prevent loading from current dir
    confname = os.path.join(workdir, 'empty.conf')
    file_write(confname)

    opts = {
        'c': confname,
        'VERBOSE_TYPE': 'QUIET',
        'DETECT_MINAREA': minarea,
        'GAIN': gain,
        'DETECT_THRESH': thresh,
        'WEIGHT_TYPE': 'BACKGROUND',
        'MASK_TYPE': 'NONE',  # both 'CORRECT' and 'BLANK' seem to cause systematics?
        'SATUR_LEVEL': np.nanmax(image[~mask]) + 1  # Saturation should be handled in external mask
    }

    if bg_size is not None:
        opts['BACK_SIZE'] = bg_size

    if err is not None:
        # User-provided noise model
        err = err.copy().astype(np.double)
        err[~np.isfinite(err)] = 1e30
        err[err == 0] = 1e30

        errname = os.path.join(workdir, 'errors.fits')
        fits.writeto(errname, err, overwrite=True)
        opts['WEIGHT_IMAGE'] = errname
        opts['WEIGHT_TYPE'] = 'MAP_RMS'

    flagsname = os.path.join(workdir, 'flags.fits')
    fits.writeto(flagsname, mask.astype(np.int16), overwrite=True)
    opts['FLAG_IMAGE'] = flagsname

    if np.isscalar(aper):
        opts['PHOT_APERTURES'] = aper * 2  # SExtractor expects diameters, not radii
        size = ''
    else:
        opts['PHOT_APERTURES'] = ','.join([str(_ * 2) for _ in aper])
        size = '[%d]' % len(aper)

    checknames = [
        os.path.join(workdir, _.replace('-', 'M_') + '.fits') for _ in checkimages
    ]
    if checkimages:
        opts['CHECKIMAGE_TYPE'] = ','.join(checkimages)
        opts['CHECKIMAGE_NAME'] = ','.join(checknames)

    params = [
        'MAG_APER' + size,
        'MAGERR_APER' + size,
        'FLUX_APER' + size,
        'FLUXERR_APER' + size,
        'X_IMAGE',
        'Y_IMAGE',
        'ERRX2_IMAGE',
        'ERRY2_IMAGE',
        'A_IMAGE',
        'B_IMAGE',
        'THETA_IMAGE',
        'FLUX_RADIUS',
        'FWHM_IMAGE',
        'FLAGS',
        'IMAFLAGS_ISO',
        'BACKGROUND',
    ]
    params += extra_params

    if psf is not None:
        opts['PSF_NAME'] = psf
        params += [
            'MAG_PSF',
            'MAGERR_PSF',
            'FLUX_PSF',
            'FLUXERR_PSF',
            'XPSF_IMAGE',
            'YPSF_IMAGE',
            'SPREAD_MODEL',
            'SPREADERR_MODEL',
            'CHI2_PSF',
        ]

    paramname = os.path.join(workdir, 'cfg.param')
    with open(paramname, 'w') as paramfile:
        paramfile.write("\n".join(params))
    opts['PARAMETERS_NAME'] = paramname

    catname = os.path.join(workdir, 'out.cat')
    opts['CATALOG_NAME'] = catname
    opts['CATALOG_TYPE'] = 'FITS_LDAC'

    if not r0:
        opts['FILTER'] = 'N'
    else:
        kernel = make_kernel(core_radius=r0, extent_factor=2.0)
        kernelname = os.path.join(workdir, 'kernel.txt')
        np.savetxt(
            kernelname,
            kernel / np.sum(kernel),
            fmt='%.6f',
            header='CONV NORM',
            comments='',
        )
        opts['FILTER'] = 'Y'
        opts['FILTER_NAME'] = kernelname

    opts.update(extra)

    # Build the command line
    cmd = (binname + ' ' + shlex.quote(imagename) + ' ' + format_astromatic_opts(opts))
    if not verbose:
        cmd += ' > /dev/null 2>/dev/null'
    log.debug("Will run SExtractor like that:")
    log.debug(cmd)

    # Run the command!

    res = os.system(cmd)

    if res == 0 and os.path.exists(catname):
        log.info("SExtractor run succeeded")
        obj = Table.read(catname, hdu=2)
        obj.meta.clear()  # Remove unnecessary entries from the metadata

        idx = (obj['X_IMAGE'] > edge) & (obj['X_IMAGE'] < image.shape[1] - edge)
        idx &= (obj['Y_IMAGE'] > edge) & (obj['Y_IMAGE'] < image.shape[0] - edge)

        if np.isscalar(aper):
            if sn:
                idx &= obj['MAGERR_APER'] < 1.0 / sn
            if reject_negative:
                idx &= obj['FLUX_APER'] > 0
        else:
            if sn:
                idx &= np.all(obj['MAGERR_APER'] < 1.0 / sn, axis=1)
            if reject_negative:
                idx &= np.all(obj['FLUX_APER'] > 0, axis=1)

        obj = obj[idx]

        if wcs is None and header is not None:
            wcs = WCS(header)

        if wcs is not None:
            obj['ra'], obj['dec'] = wcs.all_pix2world(obj['X_IMAGE'], obj['Y_IMAGE'], 1)
        else:
            obj['ra'], obj['dec'] = (
                np.zeros_like(obj['X_IMAGE']),
                np.zeros_like(obj['Y_IMAGE']),
            )

        obj['FLAGS'][obj['IMAFLAGS_ISO'] > 0] |= 0x100  # Masked pixels in the footprint
        obj.remove_column('IMAFLAGS_ISO')  # We do not need this column

        # Convert variances to rms
        obj['ERRX2_IMAGE'] = np.sqrt(obj['ERRX2_IMAGE'])
        obj['ERRY2_IMAGE'] = np.sqrt(obj['ERRY2_IMAGE'])

        for _, __ in [
            ['X_IMAGE', 'x'],
            ['Y_IMAGE', 'y'],
            ['ERRX2_IMAGE', 'xerr'],
            ['ERRY2_IMAGE', 'yerr'],
            ['FLUX_APER', 'flux'],
            ['FLUXERR_APER', 'fluxerr'],
            ['MAG_APER', 'mag'],
            ['MAGERR_APER', 'magerr'],
            ['BACKGROUND', 'bg'],
            ['FLAGS', 'flags'],
            ['FWHM_IMAGE', 'fwhm'],
            ['A_IMAGE', 'a'],
            ['B_IMAGE', 'b'],
            ['THETA_IMAGE', 'theta'],
        ]:
            obj.rename_column(_, __)

        if psf:
            for _, __ in [
                ['XPSF_IMAGE', 'x_psf'],
                ['YPSF_IMAGE', 'y_psf'],
                ['MAG_PSF', 'mag_psf'],
                ['MAGERR_PSF', 'magerr_psf'],
                ['FLUX_PSF', 'flux_psf'],
                ['FLUXERR_PSF', 'fluxerr_psf'],
                ['CHI2_PSF', 'chi2_psf'],
                ['SPREAD_MODEL', 'spread_model'],
                ['SPREADERR_MODEL', 'spreaderr_model'],
            ]:
                if _ in obj.keys():
                    obj.rename_column(_, __)
                    if 'mag' in __:
                        obj[__][obj[__] == 99] = np.nan  # TODO: use masked column here?

        # SExtractor uses 1-based pixel coordinates
        obj['x'] -= 1
        obj['y'] -= 1

        if 'x_psf' in obj.keys():
            obj['x_psf'] -= 1
            obj['y_psf'] -= 1

        obj.meta['aper'] = aper

        if sort:
            if np.isscalar(aper):
                obj.sort('flux', reverse=True)
            else:
                # Table sorting by vector columns seems to be broken?..
                obj = obj[np.argsort(-obj['flux'][:, 0])]

        if catfile is not None:
            shutil.copyfile(catname, catfile)
            log.info(f"Catalogue stored to {catfile}")

    else:
        log.error(f"Error {res} running SExtractor")

    result = obj

    if checkimages:
        result = [result]

        for name in checknames:
            if os.path.exists(name):
                result.append(fits.getdata(name))
            else:
                log.error(f"Cannot find requested output checkimage file {name}")
                result.append(None)

    if _workdir is None:
        shutil.rmtree(workdir)

    return result


def make_kernel(core_radius: float = 1.0, extent_factor: float = 1.0) -> np.ndarray:
    """Generate a 2D Gaussian kernel image.

    The kernel is centered at the origin and defined using a Gaussian profile
    with a specified core radius and extent.

    Args:
        core_radius (float, optional): The Gaussian core radius (standard deviation). Defaults to 1.0.
        extent_factor (float, optional): Extent of the kernel in units of core radius.
            Determines the size of the kernel array. Defaults to 1.0.

    Returns:
        np.ndarray: A 2D numpy array representing the Gaussian kernel.
    """
    x, y = np.mgrid[
        np.floor(-extent_factor * core_radius): np.ceil(extent_factor * core_radius + 1),
        np.floor(-extent_factor * core_radius): np.ceil(extent_factor * core_radius + 1),
    ]

    radial_distance = np.hypot(x, y)
    kernel = np.exp(-radial_distance**2 / (2 * core_radius**2))

    return kernel


def evaluate_data_quality_results(source_catalog: Table):
    """Evaluate data quality metrics from a source detection catalog.

    This function processes the results of `get_objects_sextractor()` or a similar
    source detection catalog, computing median image quality indicators like FWHM
    and ellipticity.

    Args:
        source_catalog (astropy.table.Table): Catalog of detected sources containing
            columns 'fwhm', 'a' (major axis), and 'b' (minor axis).

    Returns:
        tuple:
            - fwhm (float): Median FWHM of the sources in pixels.
            - fwhm_error (float): Median absolute deviation (MAD) of FWHM in pixels.
            - ellipticity (float): Median ellipticity of the sources (1 - b/a).
            - ellipticity_error (float): Propagated error of the ellipticity estimate.
    """
    fwhm = np.median(source_catalog['fwhm'])
    fwhm_error = np.median(np.abs(source_catalog['fwhm'] - fwhm))

    major_axis_median = np.median(source_catalog['a'])
    minor_axis_median = np.median(source_catalog['b'])

    major_axis_error = np.median(np.abs(source_catalog['a'] - major_axis_median))
    minor_axis_error = np.median(np.abs(source_catalog['b'] - minor_axis_median))

    ellipticity = 1.0 - (minor_axis_median / major_axis_median)
    ellipticity_error = ellipticity * np.sqrt(
        (major_axis_error / major_axis_median) ** 2 +
        (minor_axis_error / minor_axis_median) ** 2
    )

    return fwhm, fwhm_error, ellipticity, ellipticity_error


def file_write(filename: str, contents: str = None, append: bool = False) -> None:
    """Write content to a file.

    Opens the file in write or append mode and writes the given content to it.

    Args:
        filename (str): Path to the file to write to.
        contents (str, optional): The content to be written into the file. If None, nothing is written.
        append (bool, optional): If True, content is appended to the file.
            If False (default), the file is overwritten.
    """
    mode = 'a' if append else 'w'
    with open(filename, mode) as file:
        if contents is not None:
            file.write(contents)


def table_get_column(table: Table, column_name: str, default=0):
    """Get a column from an astropy table, or return a default value if it is missing.

    This is a convenience wrapper to safely access a column in a table.

    Args:
        table (astropy.table.Table): Input table.
        column_name (str): Name of the column to retrieve.
        default (scalar or array-like, optional): Default value to return if the column is not found.
            If scalar, it will be broadcasted to match the table length.
            If array-like, it will be returned directly.
            If None, the function returns None.

    Returns:
        np.ndarray or None: The column values if found, otherwise the default value
        broadcasted or returned as-is depending on its type.
    """
    if column_name in table.colnames:
        return table[column_name]
    if default is None:
        return None
    if hasattr(default, '__len__'):
        return default
    return np.full(len(table), default, dtype=int)


def get_observation_time(
    header=None,
    filename: str = None,
    time_string: str = None,
    return_datetime: bool = False,
    verbose=False  # Unused
):
    """Extract observation time from a FITS header, FITS file, or a user-provided string.

    Args:
        header (astropy.io.fits.Header, optional): FITS header with time keywords.
        filename (str, optional): Path to a FITS file (used if header is not provided).
        time_string (str, optional): A user-provided ISO-formatted time string.
        return_datetime (bool, optional): If True, return Python datetime object instead of Astropy Time.
        verbose (bool, optional): Unused, for backward compatibility.

    Returns:
        astropy.time.Time or datetime.datetime or None: Parsed time object or None if parsing fails.
    """

    def _convert_to_time(value):
        """Convert float or string-like value to Time or datetime."""
        try:
            if isinstance(value, float):
                if 0 < value < 100000:
                    log.debug("Assuming float is MJD")
                    time_obj = Time(value, format='mjd')
                elif 2400000 < value < 2500000:
                    log.debug("Assuming float is JD")
                    time_obj = Time(value, format='jd')
                else:
                    log.debug("Assuming float is Unix timestamp")
                    time_obj = Time(value, format='unix')
            else:
                time_obj = Time(value)
            log.info(f"Time parsed as: {time_obj.iso}")
            return time_obj.datetime if return_datetime else time_obj
        except Exception as err:
            log.error(f"Failed to convert time: {value} — {err}")
            return None

    if time_string:
        log.info(f"Parsing time from time_string: {time_string}")
        try:
            parsed = dateutil.parser.parse(time_string)
            return _convert_to_time(parsed)
        except Exception as err:
            log.error(f"Failed to parse time_string: {err}")
            return None

    if header is None and filename:
        log.info(f"Loading FITS header from file: {filename}")
        try:
            header = fits.getheader(filename)
        except Exception as err:
            log.error(f"Failed to read header from file: {err}")
            return None

    if header is None:
        log.error("No header or filename provided for observation time extraction.")
        return None

    # Combined DATE + TIME-OBS or UT
    if 'DATE' in header and ('TIME-OBS' in header or 'UT' in header):
        time_part = header.get('TIME-OBS', header.get('UT'))
        try:
            combined = f"{header['DATE']} {time_part}"
            log.info(f"Trying combined DATE and TIME: {combined}")
            parsed = dateutil.parser.parse(combined)
            return _convert_to_time(parsed)
        except Exception as err:
            log.error(f"Failed to parse combined DATE and TIME-OBS/UT: {err}")

    # Fallback: try standard keys individually
    for key in ['DATE-OBS', 'DATE', 'TIME-OBS', 'UT', 'MJD', 'JD']:
        if key in header:
            log.info(f"Found {key} in header: {header[key]}")
            result = _convert_to_time(header[key])
            if result is not None:
                return result

    log.error("No valid observation time found in header.")
    return None


def format_astromatic_opts(options: dict) -> str:
    """Format a dictionary of options into an Astromatic-compatible command-line string.

    Booleans are converted to Y/N, arrays to comma-separated strings, and strings are quoted when necessary.

    Args:
        options (dict): Dictionary of options to be converted into command-line arguments.

    Returns:
        str: A formatted string with options suitable for Astromatic tools (e.g., SExtractor, SWarp).
    """
    formatted_args = []

    for key, value in options.items():
        if value is None:
            continue

        if isinstance(value, bool):
            formatted_args.append(f"-{key} {'Y' if value else 'N'}")
        else:
            if isinstance(value, str):
                value_str = shlex.quote(value)
            elif hasattr(value, '__len__') and not isinstance(value, str):
                value_str = ','.join(str(item) for item in value)
            else:
                value_str = str(value)

            formatted_args.append(f"-{key} {value_str}")

    return ' '.join(formatted_args)


def plot_image(image, wcs=None, quantiles=(0.01, 0.99), cmap='Blues_r',
               x_points=None, y_points=None, use_wcs_for_points=False,
               point_marker='r.', point_size=2, title=None, figsize=None,
               show_grid=False, output_file=None,  dpi=300, save_to_file=False):
    """Plot a 2D image with optional WCS projection, color scaling, and overlay points.

    Args:
        image (numpy.ndarray): The 2D array representing the image data to be plotted.
        wcs (astropy.wcs.WCS, optional): The WCS object for astronomical projections.
            If None, a standard plot is created.
        quantiles (tuple of float, optional): Percentile values for scaling image brightness
            (default is (0.01, 0.99)).
        cmap (str, optional): Colormap to use for the image (default is 'Blues_r').
        x_points (array-like, optional): X-coordinates of points to overlay on the image.
        y_points (array-like, optional): Y-coordinates of points to overlay on the image.
        use_wcs_for_points (bool, optional): Whether to transform points using WCS
            (default is False).
        point_marker (str, optional): Matplotlib marker style for the overlay points
            (default is 'r.').
        point_size (float, optional): Size of the overlay points (default is 2).
        title (str, optional): Title for the plot.
        figsize (tuple of float, optional): Size of the figure in inches (width, height).
            If None, defaults to Matplotlib's default.
        show_grid (bool, optional): Whether to overlay a grid (default is False).
        output_file (str, optional): File path to save the plot. If None, the plot is
            displayed instead of being saved (default is None).
        dpi (int, optional): Resolution of the saved plot in dots per inch (default is 300).

    Returns:
        None: This function does not return any value. It either displays the plot or
        saves it to a file if `output_file` is provided.

    Raises:
        ValueError: If the quantiles are invalid (e.g., negative or out of range).
    """
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': wcs} if wcs else None)

    # Compute quantiles for color scaling
    brightness_limits = np.nanquantile(image, quantiles)
    brightness_limits[0] = max(brightness_limits[0], 0)  # Ensure vmin is non-negative
    norm = simple_norm(image, 'linear', vmin=brightness_limits[0], vmax=brightness_limits[1])

    # Plot the image
    img = ax.imshow(image, origin='lower', norm=norm, cmap=cmap)

    # Optionally overlay grid
    if show_grid:
        ax.grid(color='white', ls='--')

    # Set labels if WCS is provided
    if wcs:
        ax.set_xlabel('Right Ascension (J2000)')
        ax.set_ylabel('Declination (J2000)')

    # Add a colorbar
    plt.colorbar(img, ax=ax)

    # Plot points if provided
    if x_points is not None and y_points is not None:
        transform = ax.get_transform('fk5') if use_wcs_for_points else None
        ax.plot(x_points, y_points, point_marker, ms=point_size, transform=transform)

    # Set the title
    if title:
        ax.set_title(title)

    # Save or show the plot
    plt.tight_layout()
    if save_to_file:
        if output_file:
            plt.savefig(output_file, dpi=dpi)
        else:
            log.error(f"output file name must be provided")

    else:
        plt.show()


def add_colorbar(mappable=None, ax=None, size="5%", pad=0.1):
    """Add a colorbar to a matplotlib Axes object.

    This function appends a colorbar to the side of the given Axes using a
    custom size and padding. If no Axes is provided, the current active Axes
    is used.

    Args:
        mappable (matplotlib.cm.ScalarMappable, optional): The object to which the colorbar corresponds (e.g., from `imshow()`).
        ax (matplotlib.axes.Axes, optional): The Axes to which the colorbar should be added. If None, uses current Axes.
        size (str, optional): Width of the colorbar as a percentage of the original axes. Default is "5%".
        pad (float, optional): Padding between the colorbar and the plot in inches. Default is 0.1.

    Returns:
        matplotlib.colorbar.Colorbar: The created colorbar instance.
    """
    if mappable is not None:
        ax = mappable.axes
    elif ax is None:
        ax = plt.gca()

    # Create an axes for the colorbar next to the main Axes
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=size, pad=pad)

    # Add colorbar and return it
    colorbar_obj = ax.get_figure().colorbar(mappable, cax=cax)

    # Restore focus to the original Axes
    ax.get_figure().sca(ax)

    return colorbar_obj


def binned_map(
    x,
    y,
    values,
    bins=16,
    statistic='mean',
    quantiles=(0.5, 97.5),
    point_color=None,
    show_colorbar=True,
    show_axes=True,
    show_points=False,
    ax=None,
    data_range=None,
    **imshow_kwargs,
):
    """Plot a binned 2D statistical map from irregular data using `binned_statistic_2d`.

    Args:
        x (array-like): X-coordinates of data points.
        y (array-like): Y-coordinates of data points.
        values (array-like): Data values at each (x, y) location.
        bins (int or [int, int], optional): Number of bins per axis (default is 16).
        statistic (str or callable, optional): Statistic to compute in each bin
            (e.g., 'mean', 'median', or a function).
        quantiles (tuple, optional): Percentiles for image normalization (default: (0.5, 97.5)).
        point_color (str or None): Color for overplotted data points (if shown).
        show_colorbar (bool): Whether to add a colorbar.
        show_axes (bool): Whether to show axis ticks and labels.
        show_points (bool): Whether to overlay the (x, y) points on the plot.
        ax (matplotlib.axes.Axes, optional): Axes object to plot in. Defaults to current axes.
        data_range (list or tuple, optional): [[xmin, xmax], [ymin, ymax]] data range for binning.
        **imshow_kwargs: Additional keyword arguments passed to `imshow`.

    Returns:
        None
    """
    stat_image, x_edges, y_edges, _ = binned_statistic_2d(
        x, y, values, statistic=statistic, bins=bins, range=data_range
    )

    # Handle percentile scaling
    finite_values = stat_image[np.isfinite(stat_image)]
    if len(finite_values) > 0:
        vmin, vmax = np.percentile(finite_values, quantiles)
    else:
        vmin, vmax = None, None

    if 'vmin' not in imshow_kwargs and vmin is not None:
        imshow_kwargs['vmin'] = vmin
    if 'vmax' not in imshow_kwargs and vmax is not None:
        imshow_kwargs['vmax'] = vmax

    if ax is None:
        ax = plt.gca()

    imshow_kwargs.setdefault('aspect', 'auto')

    im = ax.imshow(
        stat_image.T,
        origin='lower',
        extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
        interpolation='nearest',
        **imshow_kwargs
    )

    if show_colorbar:
        add_colorbar(im, ax=ax)

    if not show_axes:
        ax.set_axis_off()

    if show_points:
        ax.plot(x, y, '.', color=point_color or 'red', alpha=0.3)


def plot_photometric_match(
    match_result,
    ax=None,
    mode='mag',
    show_masked=True,
    show_final=True,
    cmag_limits=None,
    **kwargs
):
    """Visualizes photometric match results in various modes.

    This function generates different types of diagnostic plots based on the
    results of a photometric matching routine. The visualization can include
    magnitude residuals, normalized residuals, color dependencies, zero points,
    model predictions, residuals, and distance distributions.

    Args:
        match_result (dict): Dictionary containing the results of photometric matching.
            Expected keys include:
            - `idx0` (ndarray): Boolean mask indicating objects used in initial fitting.
            - `idx` (ndarray): Boolean mask indicating objects used in the final fit.
            - `cmag` (ndarray): Catalogue magnitudes of matched objects.
            - `zero` (ndarray): Empirical zero points (catalogue - instrumental magnitudes).
            - `zero_model` (ndarray): Modeled zero points.
            - `zero_err` (ndarray): Errors of the empirical zero points.
            - `color` (ndarray, optional): Catalogue colors of matched objects.
            - `ox`, `oy` (ndarray): `x` and `y` coordinates of objects on the image.
            - `dist` (ndarray): Pairwise distances between matched objects and catalogue stars (in degrees).
            - `cat_col_mag`, `cat_col_mag1`, `cat_col_mag2` (str, optional): Names of the catalogue magnitudes and colors.
            - `color_term` (float, optional): Fitted color term for magnitude transformation.

        ax (matplotlib.axes.Axes, optional): The matplotlib axis to plot on. If None,
            the current axis (`plt.gca()`) will be used.
        mode (str): Type of plot to generate. Options include:
            - `'mag'`: Residuals vs. catalogue magnitude.
            - `'normed'`: Normalized residuals vs. magnitude.
            - `'color'`: Residuals vs. catalogue color.
            - `'zero'`: Spatial map of the zero point.
            - `'model'`: Spatial map of the model zero point.
            - `'residuals'`: Spatial map of residuals (instrumental - model).
            - `'dist'`: Spatial map of displacement between matched objects and catalogue stars.
        show_masked (bool, optional): Whether to display masked data points
            (those that were rejected from the final fit). Defaults to True.
        show_final (bool, optional): Whether to highlight the final selection of
            matched objects used in the fit. Defaults to True.
        cmag_limits (tuple, optional): Limits for the x-axis in magnitude plots
            (e.g., `(8, 22)`). If None, limits are automatically determined.
        **kwargs: Additional arguments passed to `binned_map`, which is used
            for spatial mapping plots.

    Returns:
        matplotlib.axes.Axes: The axis containing the generated plot.

    Notes:
        - The function includes a `_model_string()` helper to display the
          photometric transformation equation used.
        - `_plot_residual_vs()` is a helper function to generate various
          residual plots.
        - When `binned_map` is used (for `zero`, `model`, `residuals`, and `dist` modes),
          it creates a spatially binned visualization of the respective quantities.
    """
    m = match_result
    ax = ax or plt.gca()

    def _model_string():
        s = f"Instr = {m.get('cat_col_mag', 'Cat')}"
        if all(k in m for k in ['cat_col_mag1', 'cat_col_mag2', 'color_term']) and m['color_term'] is not None:
            sign = '-' if m['color_term'] > 0 else '+'
            s += f" {sign} {abs(m['color_term']):.2f} ({m['cat_col_mag1']} - {m['cat_col_mag2']})"
        return s + " + ZP"

    def _plot_residual_vs(var, xlabel, ylabel, title=None):
        ax.errorbar(m[var][m['idx0']], (m['zero_model'] - m['zero'])[m['idx0']], m['zero_err'][m['idx0']], fmt='.', alpha=0.3, zorder=0)
        if show_masked:
            ax.plot(m[var][~m['idx0']], (m['zero_model'] - m['zero'])[~m['idx0']], 'x', alpha=0.3, color='orange', label='Masked', zorder=5)
        if show_final:
            ax.plot(m[var][m['idx']], (m['zero_model'] - m['zero'])[m['idx']], '.', alpha=1.0, color='red', label='Final fit', zorder=10)
        ax.axhline(0, ls='--', color='black', alpha=0.3)
        ax.legend()
        ax.grid(alpha=0.2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        ax.text(0.02, 0.05, _model_string(), transform=ax.transAxes)

    if mode == 'mag':
        _plot_residual_vs('cmag', f"Catalogue {m.get('cat_col_mag', '')} magnitude", "Instrumental - Model",
                          title=f"{np.sum(m['idx'])} of {np.sum(m['idx0'])} unmasked stars used in final fit")
        if cmag_limits:
            x = m['cmag'][m['idx0']].value if hasattr(m['cmag'], 'value') else m['cmag'][m['idx0']]
            y = (m['zero_model'] - m['zero'])[m['idx0']].value if hasattr(m['zero_model'], 'value') else (m['zero_model'] - m['zero'])[m['idx0']]
            idx = (x > cmag_limits[0]) & (x < cmag_limits[1])
            ylim = (np.min(y[idx]), np.max(y[idx]))
            dy = ylim[1] - ylim[0]
            ax.set_xlim(cmag_limits)
            ax.set_ylim((ylim[0] - 0.05 * dy, ylim[1] + 0.05 * dy))

    elif mode == 'normed':
        ydata = ((m['zero_model'] - m['zero']) / m['zero_err'])
        ax.plot(m['cmag'][m['idx0']], ydata[m['idx0']], '.', alpha=0.3, zorder=0)
        if show_masked:
            ax.plot(m['cmag'][~m['idx0']], ydata[~m['idx0']], 'x', alpha=0.3, color='orange', label='Masked', zorder=5)
        if show_final:
            ax.plot(m['cmag'][m['idx']], ydata[m['idx']], '.', alpha=1.0, color='red', label='Final fit', zorder=10)
        ax.axhline(0, ls='--', color='black', alpha=0.3)
        ax.axhline(-3, ls=':', color='black', alpha=0.3)
        ax.axhline(3, ls=':', color='black', alpha=0.3)
        ax.legend()
        ax.grid(alpha=0.2)
        ax.set_xlabel(f"Catalogue {m.get('cat_col_mag', '')} magnitude")
        ax.set_ylabel("(Instrumental - Model) / Error")
        ax.set_title(f"{np.sum(m['idx'])} of {np.sum(m['idx0'])} unmasked stars used in final fit")
        ax.text(0.02, 0.05, _model_string(), transform=ax.transAxes)

    elif mode == 'color':
        _plot_residual_vs('color', f"Catalogue {m.get('cat_col_mag1', '')}-{m.get('cat_col_mag2', '')} color",
                          "Instrumental - Model",
                          title=f"color term = {m.get('color_term', 0.0):.2f}")

    elif mode == 'zero':
        binned_map(m['ox'][m['idx']] if show_final else m['ox'][m['idx0']],
                   m['oy'][m['idx']] if show_final else m['oy'][m['idx0']],
                   m['zero'][m['idx']] if show_final else m['zero'][m['idx0']],
                   ax=ax, **kwargs)
        ax.set_title("Zero point")

    elif mode == 'model':
        binned_map(m['ox'][m['idx0']], m['oy'][m['idx0']], m['zero_model'][m['idx0']], ax=ax, **kwargs)
        ax.set_title("Model")

    elif mode == 'residuals':
        binned_map(m['ox'][m['idx0']], m['oy'][m['idx0']], (m['zero_model'] - m['zero'])[m['idx0']], ax=ax, **kwargs)
        ax.set_title("Instrumental - model")

    elif mode == 'dist':
        arcsec_dist = m['dist'][m['idx']] * 3600
        binned_map(m['ox'][m['idx']], m['oy'][m['idx']], arcsec_dist, ax=ax, **kwargs)
        ax.set_title(f"{np.sum(m['idx'])} stars: mean displacement {np.mean(arcsec_dist):.1f}" +
                     f" arcsec, median {np.median(arcsec_dist):.1f} arcsec")

    return ax


def plot_photcal(
        image,
        phot_table,
        wcs=None,
        column_scale='mag_calib',
        quantiles=(0.02, 0.98),
        output_file=None,
        save_to_file=False,
        show_plot=False,
        dpi=300
):
    """Plot a calibrated photometric image with source ellipses colored by a photometric quantity.

    Args:
        image (np.ndarray): 2D image array to display.
        phot_table (astropy.table.Table): Table of photometric detections.
        wcs (astropy.wcs.WCS, optional): WCS solution for the image.
        column_scale (str): Column in `phot_table` to use for color mapping.
        quantiles (tuple of float): Lower and upper quantiles for image display scaling.
        output_file (str, optional): Path to save the plot. If None, plot is shown.
        show_plot (bool, optional): Whether to show the plot. Defaults to False.
        dpi (int): Resolution of the saved plot in dots per inch.
    """
    from matplotlib.patches import Ellipse

    fig, ax = plt.subplots(subplot_kw={'projection': wcs} if wcs else {}, figsize=(8, 6))

    # Scale image brightness using quantiles
    vmin, vmax = np.nanquantile(image, quantiles)
    vmin = max(0, vmin)

    ax.imshow(image, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)

    # Normalize color scale for ellipse overlay
    norm = plt.Normalize(vmin=phot_table[column_scale].min(), vmax=phot_table[column_scale].max())
    cmap = plt.cm.viridis.reversed()

    # Overlay ellipses for each detected source
    for source in phot_table:
        ellipse = Ellipse(
            xy=(source['x'], source['y']),
            width=2 * source['a'],
            height=2 * source['b'],
            angle=source['theta'],
            edgecolor=cmap(norm(source[column_scale])),
            facecolor='none',
            linewidth=0.5,
            alpha=0.55,
            transform=ax.transData
        )
        ax.add_patch(ellipse)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Required by older matplotlib versions
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(column_scale)
    cbar.ax.invert_yaxis()

    plt.tight_layout()

    if save_to_file:
        if output_file:
            plt.savefig(output_file, dpi=dpi)
        else:
            log.error(f"output file name must be provided")
    if show_plot:
        plt.show()


# cat (STDPipe)
def get_vizier_catalog(
    right_ascension,
    declination,
    search_radius,
    catalog='gaiadr2',
    row_limit=-1,
    column_filters={},
    extra_columns=[],
    include_distance=False,
    verbose=False,
):
    """Download any catalog from Vizier.

    This function retrieves data from Vizier for a given catalog and field region.
    For popular catalogs, additional photometric data is augmented based on analytical magnitude conversion formulas.

    Supported catalogs with augmentation:
      - ps1: Pan-STARRS DR1
      - gaiadr2: Gaia DR2
      - gaiaedr3: Gaia eDR3
      - gaiadr3syn: Gaia DR3 synthetic photometry
      - skymapper: SkyMapper DR1.1
      - vsx: AAVSO Variable Stars Index
      - apass: AAVSO APASS DR9
      - sdss: SDSS DR12
      - atlas: ATLAS-RefCat2
      - usnob1: USNO-B1
      - gsc: Guide Star Catalogue 2.2

    Args:
        right_ascension (float): Right Ascension of the field center in degrees.
        declination (float): Declination of the field center in degrees.
        search_radius (float): Search radius around the field center in degrees.
        catalog (str): Vizier catalog identifier or a supported catalog short name.
        row_limit (int): Maximum number of rows to return. Default is -1 (no limit).
        column_filters (dict): Dictionary of column filters as documented at Vizier.
        extra_columns (list): Additional column names to include in the output.
        include_distance (bool): If True, adds a column for distances from the field center.
        verbose (bool): [Currently unused] Enable verbose logging.

    Returns:
        astropy.table.Table: Table containing the catalog data augmented with additional columns,
        if applicable.
    """
    # TODO: Add positional error handling

    if catalog in CATALOGS:
        vizier_id = CATALOGS.get(catalog).get('vizier')
        catalog_name = CATALOGS.get(catalog).get('name')
        columns = (
            ['*', 'RAJ2000', 'DEJ2000', 'e_RAJ2000', 'e_DEJ2000'] +
            extra_columns +
            CATALOGS.get(catalog).get('extra', [])
        )
    else:
        vizier_id = catalog
        catalog_name = catalog
        columns = ['*', 'RAJ2000', 'DEJ2000', 'e_RAJ2000', 'e_DEJ2000'] + extra_columns

    log.info(f"Requesting from VizieR: {vizier_id} columns: {columns}")
    log.info(f"Center: {right_ascension:.3f} {declination:.3f} radius:{search_radius:.3f}")
    log.info(f"Filters: {column_filters}")

    vizier = Vizier(row_limit=row_limit, columns=columns, column_filters=column_filters)
    catalog_data = vizier.query_region(
        SkyCoord(right_ascension, declination, unit='deg'),
        radius=search_radius * u.deg,
        catalog=vizier_id
    )

    if not catalog_data or len(catalog_data) != 1:
        catalog_data = vizier.query_region(
            SkyCoord(right_ascension, declination, unit='deg'),
            radius=search_radius * u.deg,
            catalog=vizier_id,
            cache=False,
        )
        if not catalog_data or len(catalog_data) != 1:
            log.info(f"Error requesting catalog {catalog}")
            return None

    catalog_table = catalog_data[0]
    catalog_table.meta['vizier_id'] = vizier_id
    catalog_table.meta['name'] = catalog_name

    log.info(f"Got {len(catalog_table)} entries with {len(catalog_table.colnames)} columns")

    # Rename coordinate columns if necessary
    if '_RAJ2000' in catalog_table.keys() and '_DEJ2000' in catalog_table.keys() and 'RAJ2000' not in catalog_table.keys():
        catalog_table.rename_columns(['_RAJ2000', '_DEJ2000'], ['RAJ2000', 'DEJ2000'])

    if include_distance and 'RAJ2000' in catalog_table.colnames and 'DEJ2000' in catalog_table.colnames:
        log.info("Augmenting the catalog with distances from field center")
        catalog_table['_r'] = spherical_distance(
            right_ascension, declination, catalog_table['RAJ2000'], catalog_table['DEJ2000']
        )

    # Add photometric data augmentation for supported catalogs
    if catalog == 'gaiadr2':
        log.info("Augmenting the catalog with Johnson-Cousins photometry")

        # Coefficients for Gaia DR2 to Johnson-Cousins photometry conversion
        pB = [-0.05927724559795761, 0.4224326324292696, 0.626219707920836, -0.011211539139725953]
        pV = [0.0017624722901609662, 0.15671377090187089, 0.03123927839356175, 0.041448557506784556]
        pR = [0.02045449129406191, 0.054005149296716175, -0.3135475489352255, 0.020545083667168156]
        pI = [0.005092289380850884, 0.07027022935721515, -0.7025553064161775, -0.02747532184796779]
        pCB = [876.4047401692277, 5.114021693079334, -2.7332873314449326, 0]
        pCV = [98.03049528983964, 20.582521666713028, 0.8690079603974803, 0]
        pCR = [347.42190542330945, 39.42482430363565, 0.8626828845232541, 0]
        pCI = [79.4028706486939, 9.176899238787003, -0.7826315256072135, 0]

        # Extract relevant columns from the catalog
        g = catalog_table['Gmag']
        bp = catalog_table['BPmag']
        rp = catalog_table['RPmag']
        bp_rp = bp - rp

        # Perform transformations using the coefficients
        catalog_table['Bmag'] = g + np.polyval(pB, bp_rp) + np.polyval(pCB, bp_rp)
        catalog_table['Vmag'] = g + np.polyval(pV, bp_rp) + np.polyval(pCV, bp_rp)
        catalog_table['Rmag'] = g + np.polyval(pR, bp_rp) + np.polyval(pCR, bp_rp)
        catalog_table['Imag'] = g + np.polyval(pI, bp_rp) + np.polyval(pCI, bp_rp)

        # Add estimated errors based on G band errors
        for band in ['B', 'V', 'R', 'I']:
            catalog_table[f'e_{band}mag'] = catalog_table['e_Gmag']

    return catalog_table


def table_to_ldac(table, header=None, writeto=None):
    """Convert an Astropy Table to an LDAC-style FITS HDU list.

    Args:
        table (astropy.table.Table): The data table to convert.
        header (astropy.io.fits.Header, optional):
            FITS header to include. If None, an empty header is used.
        writeto (str, optional):
            If provided, writes the HDU list to a FITS file.

    Returns:
        astropy.io.fits.HDUList:
            The HDU list containing the primary, header, and data extensions.
    """
    primary_hdu = fits.PrimaryHDU()

    header_str = header.tostring(endcard=True)
    # FIXME: this is a quick and dirty hack to preserve final 'END     ' in the string
    # as astropy.io.fits tends to strip trailing whitespaces from string data, and it breaks at least SCAMP
    header_str += fits.Header().tostring(endcard=True)

    header_col = fits.Column(
        name='Field Header Card', format='%dA' % len(header_str), array=[header_str]
    )
    header_hdu = fits.BinTableHDU.from_columns(fits.ColDefs([header_col]))
    header_hdu.header['EXTNAME'] = 'LDAC_IMHEAD'

    data_hdu = fits.table_to_hdu(table)
    data_hdu.header['EXTNAME'] = 'LDAC_OBJECTS'

    hdulist = fits.HDUList([primary_hdu, header_hdu, data_hdu])

    if writeto is not None:
        hdulist.writeto(writeto, overwrite=True)

    return hdulist


def get_pixel_scale(wcs=None, filename=None, header=None):
    """Return the pixel scale of an image in degrees per pixel.

    This function calculates the pixel scale from a WCS object, FITS header, or a FITS file.

    Args:
        wcs (astropy.wcs.WCS, optional): A precomputed WCS object.
        filename (str, optional): Path to a FITS file. Used if `wcs` and `header` are not provided.
        header (astropy.io.fits.Header, optional): A FITS header object. Used if `wcs` is not provided.

    Returns:
        float: Pixel scale in degrees per pixel.

    Raises:
        ValueError: If none of `wcs`, `header`, or `filename` is provided or WCS cannot be constructed.
    """
    if wcs is None:
        if header is not None:
            wcs = WCS(header=header)
        elif filename is not None:
            header = fits.getheader(filename, -1)
            wcs = WCS(header=header)
        else:
            raise ValueError("Must provide either `wcs`, `header`, or `filename` to calculate pixel scale.")

    pixel_scale = np.hypot(wcs.pixel_scale_matrix[0, 0], wcs.pixel_scale_matrix[0, 1])
    return pixel_scale


# astrometry (STDPipe)
def refine_wcs_scamp(
    obj,
    cat=None,
    wcs=None,
    header=None,
    sr=2 / 3600,
    order=3,
    cat_col_ra='RAJ2000',
    cat_col_dec='DEJ2000',
    cat_col_ra_err='e_RAJ2000',
    cat_col_dec_err='e_DEJ2000',
    cat_col_mag='rmag',
    cat_col_mag_err='e_rmag',
    cat_mag_lim=99,
    sn=None,
    extra={},
    get_header=False,
    update=False,
    _workdir=None,
    _tmpdir=None,
    _exe=None,
    verbose=False,
):
    """Refine the WCS solution using SCAMP.

    This function is a wrapper for running SCAMP on a user-provided object list and reference catalog to refine the astrometric solution. It matches objects in the frame with a reference catalog and computes a refined WCS solution.

    Args:
        obj (astropy.table.Table): List of objects on the frame, containing at least `x`, `y`, and `flux` columns.
        cat (astropy.table.Table or str, optional): Reference astrometric catalog or name of a network catalog. Default is None.
        wcs (astropy.wcs.WCS, optional): Initial WCS solution. Default is None.
        header (astropy.io.fits.Header, optional): FITS header containing the initial astrometric solution. Default is None.
        sr (float, optional): Matching radius in degrees. Default is 2/3600.
        order (int, optional): Polynomial order for PV distortion solution (1 or greater). Default is 3.
        cat_col_ra (str, optional): Catalog column name for Right Ascension. Default is 'RAJ2000'.
        cat_col_dec (str, optional): Catalog column name for Declination. Default is 'DEJ2000'.
        cat_col_ra_err (str, optional): Catalog column name for Right Ascension error. Default is 'e_RAJ2000'.
        cat_col_dec_err (str, optional): Catalog column name for Declination error. Default is 'e_DEJ2000'.
        cat_col_mag (str, optional): Catalog column name for the magnitude in the closest band. Default is 'rmag'.
        cat_col_mag_err (str, optional): Catalog column name for the magnitude error. Default is 'e_rmag'.
        cat_mag_lim (float or list, optional): Magnitude limit for catalog stars. If a list, treated as lower and upper limits. Default is 99.
        sn (float or list, optional): Signal-to-noise ratio threshold for object matching. If a list, treated as lower and upper limits. Default is None.
        extra (dict, optional): Dictionary of additional parameters to pass to SCAMP. Default is {}.
        get_header (bool, optional): If True, return the FITS header instead of the WCS solution. Default is False.
        update (bool, optional): If True, update the object list in-place with refined `ra` and `dec` coordinates. Default is False.
        _workdir (str, optional): Directory for temporary files. If specified, files are not deleted. Default is None.
        _tmpdir (str, optional): Directory for temporary files. Files are deleted after execution. Default is None.
        _exe (str, optional): Path to SCAMP executable. If not provided, the function searches for it in the system PATH. Default is None.
        verbose (bool or callable, optional): Whether to show verbose messages. Can be a boolean or a `print`-like function. Default is False.

    Returns:
        astropy.wcs.WCS or astropy.io.fits.Header: Refined WCS solution or FITS header if `get_header=True`. Returns None if the refinement fails.
    """
    # Find the binary
    binname = None

    if _exe is not None:
        # Check user-provided binary path, and fail if not found
        if os.path.isfile(_exe):
            binname = _exe
    else:
        # Find SExtractor binary in common paths
        for exe in ['scamp']:
            binname = shutil.which(exe)
            if binname is not None:
                break

    if binname is None:
        log.critical("Can't find SCAMP binary")
        raise OSError("Can\'t find SCAMP binary")
    # else:
    #     log.info("Using SCAMP binary at", binname)

    workdir = (
        _workdir
        if _workdir is not None
        else tempfile.mkdtemp(prefix='scamp', dir=_tmpdir)
    )

    if header is None:
        # Construct minimal FITS header covering our data points
        header = fits.Header(
            {
                'NAXIS': 2,
                'NAXIS1': np.max(obj['x'] + 1),
                'NAXIS2': np.max(obj['y'] + 1),
                'BITPIX': -64,
                'EQUINOX': 2000.0,
            }
        )
    else:
        header = header.copy()

    if wcs is not None and wcs.is_celestial:
        # Add WCS information to the header
        header += wcs.to_header(relax=True)

        # Ensure the header is in TPV convention, as SCAMP does not support SIP
        if wcs.sip is not None:
            log.info("Converting the header from SIP to TPV convention")
            header = wcs_sip2pv(header)
    else:
        log.error("Can't operate without initial WCS")
        return None

    # Dummy config filename, to prevent loading from current dir
    confname = os.path.join(workdir, 'empty.conf')
    file_write(confname)

    xmlname = os.path.join(workdir, 'scamp.xml')

    opts = {
        'c': confname,
        'VERBOSE_TYPE': 'QUIET',
        'SOLVE_PHOTOM': 'N',
        'CHECKPLOT_TYPE': 'NONE',
        'WRITE_XML': 'Y',
        'XML_NAME': xmlname,
        'PROJECTION_TYPE': 'TPV',
        'CROSSID_RADIUS': sr * 3600,
        'DISTORT_DEGREES': max(1, order),
    }

    if sn is not None:
        if np.isscalar(sn):
            opts['SN_THRESHOLDS'] = [sn, 10 * sn]
        else:
            opts['SN_THRESHOLDS'] = [sn[0], sn[1]]

    opts.update(extra)

    # Minimal LDAC table with objects
    t_obj = Table(
        data={
            'XWIN_IMAGE': obj['x'] + 1,  # SCAMP uses 1-based coordinates
            'YWIN_IMAGE': obj['y'] + 1,
            'ERRAWIN_IMAGE': obj['xerr'],
            'ERRBWIN_IMAGE': obj['yerr'],
            'FLUX_AUTO': obj['flux'],
            'FLUXERR_AUTO': obj['fluxerr'],
            'MAG_AUTO': obj['mag'],
            'MAGERR_AUTO': obj['magerr'],
            'FLAGS': obj['flags'],
        }
    )

    objname = os.path.join(workdir, 'objects.cat')
    table_to_ldac(t_obj, header, objname)

    hdrname = os.path.join(workdir, 'objects.head')
    opts['HEADER_NAME'] = hdrname
    if os.path.exists(hdrname):
        os.unlink(hdrname)

    if cat:
        if type(cat) is str:
            # Match with network catalogue by name
            opts['ASTREF_CATALOG'] = cat
            log.info(f"Using {cat} as a network catalogue")
        else:
            # Match with user-provided catalogue
            t_cat = Table(
                data={
                    'X_WORLD': cat[cat_col_ra],
                    'Y_WORLD': cat[cat_col_dec],
                    'ERRA_WORLD': table_get_column(cat, cat_col_ra_err, 1 / 3600),
                    'ERRB_WORLD': table_get_column(cat, cat_col_dec_err, 1 / 3600),
                    'MAG': table_get_column(cat, cat_col_mag, 0),
                    'MAGERR': table_get_column(cat, cat_col_mag_err, 0.01),
                    'OBSDATE': np.ones_like(cat[cat_col_ra]) * 2000.0,
                    'FLAGS': np.zeros_like(cat[cat_col_ra], dtype=int),
                }
            )

            # Remove masked values
            for _ in t_cat.colnames:
                if np.ma.is_masked(t_cat[_]):
                    t_cat = t_cat[~t_cat[_].mask]

            # Convert units of err columns to degrees, if any
            for _ in ['ERRA_WORLD', 'ERRB_WORLD']:
                if t_cat[_].unit and t_cat[_].unit != 'deg':
                    t_cat[_] = t_cat[_].to('deg')

            # Limit the catalogue to given magnitude range
            if cat_mag_lim is not None:
                if hasattr(cat_mag_lim, '__len__') and len(cat_mag_lim) == 2:
                    # Two elements provided, treat them as lower and upper limits
                    t_cat = t_cat[
                        (t_cat['MAG'] >= cat_mag_lim[0]) & (t_cat['MAG'] <= cat_mag_lim[1])
                    ]
                else:
                    # One element provided, treat it as upper limit
                    t_cat = t_cat[t_cat['MAG'] <= cat_mag_lim]

            catname = os.path.join(workdir, 'catalogue.cat')
            table_to_ldac(t_cat, header, catname)

            opts['ASTREF_CATALOG'] = 'FILE'
            opts['ASTREFCAT_NAME'] = catname
            log.info("Using user-provided local catalogue")
    else:
        log.info("Using default settings for network catalogue")

    # Build the command line
    command = (
        binname + ' ' + shlex.quote(objname) + ' ' + format_astromatic_opts(opts)
    )
    if not verbose:
        command += ' > /dev/null 2>/dev/null'
    log.info("Will run SCAMP like that:")
    log.info(command)

    # Run the command!

    res = os.system(command)

    wcs = None

    if res == 0 and os.path.exists(hdrname) and os.path.exists(xmlname):
        log.info("SCAMP run successfully")

        # xlsname contains the results from SCAMP
        diag = Table.read(xmlname, table_id=0)[0]

        log.info(f"{diag['NDeg_Reference']:d} matches, chi2 {diag['Chi2_Reference']:.1f}")
        # FIXME: is df correct here?..
        if (
            diag['NDeg_Reference'] < 3 or chi2.sf(diag['Chi2_Reference'], df=diag['NDeg_Reference']) < 1e-3
        ):
            log.info("It seems the fitting failed")
        else:
            with open(hdrname, 'r') as f:
                h1 = fits.Header.fromstring(
                    f.read().encode('ascii', 'ignore'), sep='\n'
                )

                # Sometimes SCAMP returns TAN type solution even despite PV keywords present
                if h1['CTYPE1'] != 'RA---TPV' and 'PV1_0' in h1.keys():
                    log.info(f"Got WCS solution with CTYPE1 = {h1['CTYPE1']} and PV keywords, fixing it")
                    h1['CTYPE1'] = 'RA---TPV'
                    h1['CTYPE2'] = 'DEC--TPV'
                # .. while sometimes it does the opposite
                elif h1['CTYPE1'] == 'RA---TPV' and 'PV1_0' not in h1.keys():
                    log.info(f"Got WCS solution with CTYPE1 = {h1['CTYPE1']} and PV keywords, fixing it")
                    h1['CTYPE1'] = 'RA---TAN'
                    h1['CTYPE2'] = 'DEC--TAN'
                    h1 = WCS(h1).to_header(relax=True)

                if get_header:
                    # FIXME: should we really return raw / unfixed header here?..
                    log.info("Returning raw header instead of WCS solution")
                    wcs = h1
                else:
                    wcs = WCS(h1)

                    if update:
                        obj['ra'], obj['dec'] = wcs.all_pix2world(obj['x'], obj['y'], 0)

                    log.info(f"Astrometric accuracy: {h1.get('ASTRRMS1', 0) * 3600:.2f}\" "
                             f"{h1.get('ASTRRMS2', 0) * 3600:.2f}\"")

    else:
        log.info(f"Error {res} running SCAMP")
        wcs = None

    if _workdir is None:
        shutil.rmtree(workdir)

    return wcs


def clear_wcs(
    header,
    remove_comments=False,
    remove_history=False,
    remove_underscored=False,
    copy=False,
):
    """Clear WCS-related keywords from a FITS header.

    Args:
        header (astropy.io.fits.Header): Header to operate on.
        remove_comments (bool): Remove COMMENT keywords if True.
        remove_history (bool): Remove HISTORY keywords if True.
        remove_underscored (bool): Remove keys starting with '_' (e.g., from Astrometry.Net).
        copy (bool): If True, operate on a copy instead of modifying the original header.

    Returns:
        astropy.io.fits.Header: Modified FITS header.
    """
    if copy:
        header = header.copy()

    wcs_keywords = [
        'WCSAXES', 'CRPIX1', 'CRPIX2', 'PC1_1', 'PC1_2', 'PC2_1', 'PC2_2',
        'CDELT1', 'CDELT2', 'CUNIT1', 'CUNIT2', 'CTYPE1', 'CTYPE2',
        'CRVAL1', 'CRVAL2', 'LONPOLE', 'LATPOLE', 'RADESYS', 'EQUINOX',
        'B_ORDER', 'A_ORDER', 'BP_ORDER', 'AP_ORDER', 'CD1_1', 'CD2_1',
        'CD1_2', 'CD2_2', 'IMAGEW', 'IMAGEH'
    ]

    scamp_keywords = [
        'FGROUPNO', 'ASTIRMS1', 'ASTIRMS2', 'ASTRRMS1', 'ASTRRMS2',
        'ASTINST', 'FLXSCALE', 'MAGZEROP', 'PHOTIRMS', 'PHOTINST', 'PHOTLINK'
    ]

    keys_to_remove = []

    for key in header.keys():
        if not key:
            continue

        if (
            key in wcs_keywords or
                key in scamp_keywords or
                re.match(r'^(A|B|AP|BP)_\d+_\d+$', key) or
                re.match(r'^PV_?\d+_\d+$', key) or
                (key.startswith('_') and remove_underscored) or
                (key == 'COMMENT' and remove_comments) or
                (key == 'HISTORY' and remove_history)
        ):
            keys_to_remove.append(key)

    for key in keys_to_remove:
        header.remove(key, remove_all=True, ignore_missing=True)

    return header


def make_series(multiplier=1.0, x=1.0, y=1.0, order=1, sum=False, zero=True):
    """Generate a polynomial series up to the specified order using x and y.

    Args:
        multiplier (float): Scalar multiplier for each term (default is 1.0).
        x (float or np.ndarray): x-values (can be scalar or array).
        y (float or np.ndarray): y-values (can be scalar or array).
        order (int): Maximum order of the polynomial terms.
        sum (bool): If True, return the sum of all terms; otherwise, return the list of terms.
        zero (bool): If True, include a constant term (zeroth order).

    Returns:
        list[np.ndarray] or np.ndarray: List of polynomial terms or their sum (if sum=True).
    """
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)

    terms = [np.ones_like(x) * multiplier] if zero else []

    for total_order in range(1, order + 1):
        for j in range(total_order + 1):
            term = multiplier * (x ** (total_order - j)) * (y ** j)
            terms.append(term)

    return np.sum(terms, axis=0) if sum else terms


def get_intrinsic_scatter(observed_values, observed_errors, min_scatter=0, max_scatter=None):
    """Calculate the intrinsic scatter of a dataset given observed values and their errors.

    This function estimates the intrinsic scatter by fitting a model that accounts for both
    observational errors and intrinsic scatter. The intrinsic scatter is constrained to be
    between `min_scatter` and `max_scatter`.

    Args:
        observed_values (numpy.ndarray): Array of observed values.
        observed_errors (numpy.ndarray): Array of errors corresponding to the observed values.
        min_scatter (float, optional): Minimum allowed value for the intrinsic scatter. Defaults to 0.
        max_scatter (float, optional): Maximum allowed value for the intrinsic scatter. Defaults to None.

    Returns:
        float: The estimated intrinsic scatter.
    """
    if len(observed_values) == 0 or len(observed_errors) == 0:
        raise ValueError("Input arrays cannot be empty.")
    if len(observed_values) != len(observed_errors):
        raise ValueError("observed_values and observed_errors must have the same length.")
    if np.any(observed_errors < 0):
        raise ValueError("observed_errors cannot contain negative values.")

    def log_likelihood(parameters, values, errors):
        """Compute the log-likelihood for the intrinsic scatter model.

        Args:
            parameters (tuple): A tuple containing the model parameters:
                - scaling_factor (float): Scaling factor for the observed errors.
                - model_offset (float): Offset of the model.
                - intrinsic_scatter (float): Intrinsic scatter of the model.
            values (numpy.ndarray): Array of observed values.
            errors (numpy.ndarray): Array of errors corresponding to the observed values.

        Returns:
            float: The log-likelihood value.
        """
        scaling_factor, model_offset, intrinsic_scatter = parameters
        model = model_offset
        total_variance = scaling_factor * errors ** 2 + intrinsic_scatter ** 2
        return -0.5 * np.sum((values - model) ** 2 / total_variance + np.log(total_variance))

    # Define the negative log-likelihood function
    def negative_log_likelihood(*args):
        return -log_likelihood(*args)

    # Perform optimization to find the best-fit parameters
    optimization_result = minimize(
        negative_log_likelihood,
        [1, 0.0, 0.0],  # Initial guesses for scaling_factor, model_offset, and intrinsic_scatter
        args=(observed_values, observed_errors),
        bounds=[[1, 1], [None, None], [min_scatter, max_scatter]],  # Parameter bounds
        method='Powell',  # Optimization method
    )

    # Return the estimated intrinsic scatter
    return optimization_result.x[2]


def match(
    obj_ra,
    obj_dec,
    obj_mag,
    obj_magerr,
    obj_flags,
    cat_ra,
    cat_dec,
    cat_mag,
    cat_magerr=None,
    cat_color=None,
    sr=3 / 3600,
    obj_x=None,
    obj_y=None,
    spatial_order=0,
    bg_order=None,
    threshold=5.0,
    niter=10,
    accept_flags=0,
    cat_saturation=None,
    max_intrinsic_rms=0,
    sn=None,
    verbose=False,
    robust=True,
    scale_noise=False,
    ecmag_thresh=None,  # FN
    cmag_limits=None,  # FN
    use_color=True,
):
    """Low-level photometric matching routine.

    This function builds the photometric model for objects detected in an image.
    It includes catalogue magnitude, positionally-dependent zero point, a linear
    color term, an optional additive flux term, and considers possible intrinsic
    magnitude scatter on top of measurement errors.

    Args:
        obj_ra (ndarray): Array of Right Ascension values for the objects.
        obj_dec (ndarray): Array of Declination values for the objects.
        obj_mag (ndarray): Array of instrumental magnitude values for the objects.
        obj_magerr (ndarray): Array of instrumental magnitude errors for the objects.
        obj_flags (ndarray, optional): Array of flags for the objects.
        cat_ra (ndarray): Array of catalogue Right Ascension values.
        cat_dec (ndarray): Array of catalogue Declination values.
        cat_mag (ndarray): Array of catalogue magnitudes.
        cat_magerr (ndarray, optional): Array of catalogue magnitude errors.
        cat_color (ndarray, optional): Array of catalogue color values.
        sr (float): Matching radius in degrees.
        obj_x (ndarray, optional): Array of `x` coordinates of objects on the image.
        obj_y (ndarray, optional): Array of `y` coordinates of objects on the image.
        spatial_order (int): Order of zero point spatial polynomial (0 for constant).
        bg_order (int, optional): Order of additive flux term spatial polynomial
            (None to disable this term in the model).
        threshold (float, optional): Rejection threshold (relative to magnitude errors)
            for object-catalogue pair rejection in the fit.
        niter (int): Number of iterations for the fitting.
        accept_flags (int): Bitmask for acceptable object flags. Objects with any
            other flags are excluded.
        cat_saturation (float, optional): Saturation level for the catalogue. Stars
            brighter than this magnitude will be excluded from the fit.
        max_intrinsic_rms (float): Maximum intrinsic RMS for fitting. If set to 0,
            intrinsic scatter is not included in the noise model.
        sn (float, optional): Minimum acceptable signal-to-noise ratio (1/obj_magerr)
            for objects included in the fit.
        verbose (bool or callable): Whether to show verbose messages. Can be a
            boolean or a `print`-like function.
        robust (bool): Whether to use robust least squares fitting instead of weighted
            least squares.
        scale_noise (bool): Whether to re-scale the noise model (object and catalogue
            magnitude errors) to match actual data scatter.
        ecmag_thresh (float, optional): Maximum photometric error threshold for
            calibration (applies to both observed and catalogue magnitudes).
        cmag_limits (tuple, optional): Magnitude range for catalogue magnitudes
            (e.g., (8, 22) for reasonable photometry limits).
        use_color (bool): Whether to use catalogue color in deriving the color term.

    Returns:
        dict: Dictionary containing the results of the photometric matching:

        - `oidx` (ndarray): Indices of matched objects in the object list.
        - `cidx` (ndarray): Indices of matched catalogue stars.
        - `dist` (ndarray): Pairwise distances between matched objects and catalogue stars (in degrees).
        - `omag` (ndarray): Instrumental magnitudes of matched objects.
        - `omag_err` (ndarray): Errors of instrumental magnitudes.
        - `cmag` (ndarray): Catalogue magnitudes of matched objects.
        - `cmag_err` (ndarray): Errors of catalogue magnitudes.
        - `color` (ndarray): Catalogue colors corresponding to the matches (zeros if no color term fitting).
        - `ox` (ndarray): `x` coordinates of matched objects on the image.
        - `oy` (ndarray): `y` coordinates of matched objects on the image.
        - `oflags` (ndarray): Flags of matched objects.
        - `zero` (ndarray): Empirical zero points (catalogue - instrumental magnitudes).
        - `zero_err` (ndarray): Errors of the empirical zero points.
        - `zero_model` (ndarray): Modeled zero points (including color terms) for matched objects.
        - `zero_model_err` (ndarray): Errors of the modeled zero points.
        - `color_term` (float or None): Fitted color term, where instrumental magnitude is defined as
          `obj_mag = cat_mag - color * color_term`. None if color term is not used.
        - `zero_fn` (callable): Function to compute the zero point (without color term) at a given
          position and instrumental magnitude.
        - `obj_zero` (ndarray): Zero points computed for all input objects (not necessarily matched to
          the catalogue) using `zero_fn` (excluding the color term).
        - `params` (ndarray): Internal parameters of the fitted polynomial.
        - `intrinsic_rms` (float): Fitted value of intrinsic scatter.
        - `error_scale` (float): Noise scaling factor.
        - `idx` (ndarray): Boolean mask indicating objects/catalogue stars used in the final fit
          (excluding rejected ones).
        - `idx0` (ndarray): Boolean mask indicating objects/catalogue stars passing only initial
          quality cuts.

    `zero_fn` is a callable function with the signature:

        zero_fn(xx, yy, mag=None, get_err=False, add_intrinsic_rms=False)

    where:
    - `xx`, `yy`: Image coordinates.
    - `mag`: Instrumental magnitude of the object (needed for the additive flux term).
    - `get_err`: If `True`, returns the estimated zero point error instead of zero point value.
    - `add_intrinsic_rms`: If `True`, includes the intrinsic scatter term in the error estimation.

    The computed zero point from `zero_fn` does not include the contribution of the color term.
    To derive the final calibrated magnitude:

        mag_calibrated = mag_instrumental + color * color_term

    where `color` is the object's true color, and `color_term` is the fitted color term in the output dictionary.
    """
    oidx, cidx, dist = spherical_match(obj_ra, obj_dec, cat_ra, cat_dec, sr)

    log.info(f"{len(dist)} initial matches between {len(obj_ra)} objects and {len(cat_ra)} "
             f"catalogue stars, sr = {sr * 3600:.2f} arcsec")
    log.info(f"Median separation is {np.median(dist) * 3600:.2f} arcsec")

    omag = np.ma.filled(obj_mag[oidx], fill_value=np.nan)
    omag_err = np.ma.filled(obj_magerr[oidx], fill_value=np.nan)
    oflags = (
        obj_flags[oidx] if obj_flags is not None else np.zeros_like(omag, dtype=bool)
    )
    cmag = np.ma.filled(cat_mag[cidx], fill_value=np.nan)
    cmag_err = (
        np.ma.filled(cat_magerr[cidx], fill_value=np.nan)
        if cat_magerr is not None
        else np.zeros_like(cmag)
    )

    if obj_x is not None and obj_y is not None:
        x0, y0 = np.mean(obj_x[oidx]), np.mean(obj_y[oidx])
        ox, oy = obj_x[oidx], obj_y[oidx]
        x, y = obj_x[oidx] - x0, obj_y[oidx] - y0
    else:
        x0, y0 = 0, 0
        ox, oy = np.zeros_like(omag), np.zeros_like(omag)
        x, y = np.zeros_like(omag), np.zeros_like(omag)

    # Regressor
    X = make_series(1.0, x, y, order=spatial_order)
    log.info(f"Fitting the model with spatial_order = {spatial_order}")

    if bg_order is not None:
        # Spatially varying additive flux component, linearized in magnitudes
        X += make_series(-2.5 / np.log(10) / 10 ** (-0.4 * omag), x, y, order=bg_order)
        log.info(f"Adjusting background level using polynomial with bg_order = {bg_order}")

    if robust:
        log.info("Using robust fitting")
    else:
        log.info("Using weighted fitting")

    if cat_color is not None:
        ccolor = np.ma.filled(cat_color[cidx], fill_value=np.nan)
        if use_color:
            X += make_series(ccolor, x, y, order=0)
            log.info("Using color term")
    else:
        ccolor = np.zeros_like(cmag)

    Nparams = len(X)  # Number of parameters to be fitted

    X = np.vstack(X).T
    zero = cmag - omag  # We will build a model for this definition of zero point
    zero_err = np.hypot(omag_err, cmag_err)
    # weights = 1.0/zero_err**2

    # filter bad photometry
    idx0 = (np.isfinite(omag) &
            np.isfinite(omag_err) &
            np.isfinite(cmag) &
            np.isfinite(cmag_err) &
            ((oflags & ~accept_flags) == 0))  # initial mask

    # FN remove large errors
    if ecmag_thresh is not None:
        idx0 &= omag_err <= ecmag_thresh
        idx0 &= cmag_err <= ecmag_thresh

    # FN make sure we are including well calibrated data from the catalogue
    if cmag_limits is not None:
        idx0 &= ((cmag >= np.min(cmag_limits)) & (cmag <= np.max(cmag_limits)))

    if cat_color is not None and use_color:
        idx0 &= np.isfinite(ccolor)
    if cat_saturation is not None:
        idx0 &= cmag >= cat_saturation
    if sn is not None:
        idx0 &= omag_err < 1 / sn

    log.info(f"{np.sum(idx0):d} objects pass initial quality cuts")

    idx = idx0.copy()

    intrinsic_rms = 0
    scale_err = 1
    total_err = zero_err

    for iter in range(niter):
        if np.sum(idx) < Nparams + 1:
            log.info(f"Fit failed - {np.sum(idx):d} objects remaining for fitting {Nparams:d} parameters")
            return None

        if robust:
            # Rescale the arguments with weights
            C = sm.RLM(zero[idx] / total_err[idx], (X[idx].T / total_err[idx]).T).fit()
        else:
            C = sm.WLS(zero[idx], X[idx], weights=1 / total_err[idx] ** 2).fit()

        zero_model = np.sum(X * C.params, axis=1)
        zero_model_err = np.sqrt(C.cov_params(X).diagonal())

        intrinsic_rms = (
            get_intrinsic_scatter(
                observed_values=(zero - zero_model)[idx],
                observed_errors=total_err[idx],
                max_scatter=max_intrinsic_rms
            )
            if max_intrinsic_rms > 0
            else 0
        )

        scale_err = 1 if not scale_noise else np.sqrt(C.scale)  # rms
        total_err = np.hypot(zero_err * scale_err, intrinsic_rms)

        if threshold:
            idx1 = np.abs((zero - zero_model) / total_err)[idx] < threshold
        else:
            idx1 = np.ones_like(idx[idx])

        log.info(f"Iteration {iter}:{np.sum(idx)}/{len(idx)} - "
                 f"rms {np.std((zero - zero_model)[idx0]):.2f} {np.std((zero - zero_model)[idx]):.2f} - "
                 f"normed {np.std((zero - zero_model)[idx] / zero_err[idx]):.2f} {np.std((zero - zero_model)[idx] / total_err[idx]):.2f} - "
                 f"scale {np.sqrt(C.scale):.2f} {scale_err:.2f} - rms {intrinsic_rms:.2f}")

        if not np.sum(~idx1):  # and new_intrinsic_rms <= intrinsic_rms:
            log.info("Fitting converged")
            break
        else:
            idx[idx] &= idx1

    log.info(f"{np.sum(idx)} good matches")
    if max_intrinsic_rms > 0:
        log.info(f"Intrinsic scatter is {intrinsic_rms:.2f}")

    # Export the model
    def zero_fn(xx, yy, mag=None, get_err=False, add_intrinsic_rms=False):
        if xx is not None and yy is not None:
            x, y = xx - x0, yy - y0
        else:
            x, y = np.zeros_like(omag), np.zeros_like(omag)

        X = make_series(1.0, x, y, order=spatial_order)

        if bg_order is not None and mag is not None:
            X += make_series(
                -2.5 / np.log(10) / 10 ** (-0.4 * mag), x, y, order=bg_order
            )

        X = np.vstack(X).T

        if get_err:
            # It follows the implementation from https://github.com/statsmodels/statsmodels/blob/081fc6e85868308aa7489ae1b23f6e72f5662799/statsmodels/base/model.py#L1383
            # FIXME: crashes on large numbers of stars?..
            if len(x) < 5000:
                err = np.sqrt(np.dot(X, np.dot(C.cov_params()[0:X.shape[1], 0:X.shape[1]], np.transpose(X))).diagonal())
            else:
                err = np.zeros_like(x)
            if add_intrinsic_rms:
                err = np.hypot(err, intrinsic_rms)
            return err
        else:
            return np.sum(X * C.params[0: X.shape[1]], axis=1)

    if cat_color is not None and use_color:
        X = make_series(order=spatial_order)
        if bg_order is not None:
            X += make_series(order=bg_order)
        color_term = C.params[len(X):][0]
        log.info(f"Color term is {color_term:.2f}")
    else:
        color_term = None

    return {
        'oidx': oidx,
        'cidx': cidx,
        'dist': dist,
        'omag': omag,
        'omag_err': omag_err,
        'cmag': cmag,
        'cmag_err': cmag_err,
        'color': ccolor,
        'color_term': color_term,
        'zero': zero,
        'zero_err': zero_err,
        'zero_model': zero_model,
        'zero_model_err': zero_model_err,
        'zero_fn': zero_fn,
        'params': C.params,
        'error_scale': np.sqrt(C.scale),
        'intrinsic_rms': intrinsic_rms,
        'obj_zero': zero_fn(obj_x, obj_y, mag=obj_mag),
        'ox': ox,
        'oy': oy,
        'oflags': oflags,
        'idx': idx,
        'idx0': idx0,
    }


def calibrate_photometry(
    object_table,
    catalog_table,
    search_radius=None,
    pixel_scale=None,
    spatial_order=0,
    background_order=None,
    object_mag_column='mag',
    object_mag_error_column='magerr',
    object_ra_column='ra',
    object_dec_column='dec',
    object_flags_column='flags',
    object_x_column='x',
    object_y_column='y',
    catalog_mag_column='R',
    catalog_mag_error_column=None,
    catalog_mag1_column=None,
    catalog_mag2_column=None,
    catalog_ra_column='RAJ2000',
    catalog_dec_column='DEJ2000',
    error_threshold=None,
    magnitude_limits=None,
    update_results=True,
    verbose=False,
    **kwargs
):
    """Higher-level photometric calibration routine.

    This function wraps the `goodman_photometry.goodman_astro.match` routine with convenient defaults for typical tabular data.
    It performs photometric calibration by matching objects in the object table to stars in the reference catalog.

    Args:
        object_table (astropy.table.Table): Table of detected objects.
        catalog_table (astropy.table.Table): Reference photometric catalog.
        search_radius (float, optional): Matching radius in degrees. If not provided, it is calculated based on the pixel scale and object FWHM.
        pixel_scale (float, optional): Pixel scale in degrees per pixel. Used to calculate the search radius if not provided.
        spatial_order (int, optional): Order of the spatial polynomial for the zero point (0 for constant). Default is 0.
        background_order (int, optional): Order of the spatial polynomial for the additive flux term. Set to None to disable this term.
        object_mag_column (str, optional): Column name for object instrumental magnitude. Default is 'mag'.
        object_mag_error_column (str, optional): Column name for object magnitude error. Default is 'magerr'.
        object_ra_column (str, optional): Column name for object Right Ascension. Default is 'ra'.
        object_dec_column (str, optional): Column name for object Declination. Default is 'dec'.
        object_flags_column (str, optional): Column name for object flags. Default is 'flags'.
        object_x_column (str, optional): Column name for object x coordinate. Default is 'x'.
        object_y_column (str, optional): Column name for object y coordinate. Default is 'y'.
        catalog_mag_column (str, optional): Column name for catalog magnitude. Default is 'R'.
        catalog_mag_error_column (str, optional): Column name for catalog magnitude error. Default is None.
        catalog_mag1_column (str, optional): Column name for the first catalog magnitude defining the stellar color. Default is None.
        catalog_mag2_column (str, optional): Column name for the second catalog magnitude defining the stellar color. Default is None.
        catalog_ra_column (str, optional): Column name for catalog Right Ascension. Default is 'RAJ2000'.
        catalog_dec_column (str, optional): Column name for catalog Declination. Default is 'DEJ2000'.
        error_threshold (float, optional): Maximum photometric error to consider for calibration (for both observed and catalog magnitudes). Default is None.
        magnitude_limits (list, optional): Magnitude range for catalog magnitudes to avoid outliers (e.g., [8, 22]). Default is None.
        update_results (bool, optional): If True, adds `mag_calib` and `mag_calib_err` columns to the object table. Default is True.
        verbose (bool or callable, optional): Whether to show verbose messages during execution. Default is False.
        **kwargs: Additional keyword arguments passed to `goodman_photometry.goodman_astro.match`.

    Returns:
        dict: A dictionary containing photometric calibration results, as returned by `goodman_photometry.goodman_astro.match`.
    """
    # Calculate search radius if not provided
    if search_radius is None:
        if pixel_scale is not None:
            # Use half of the median FWHM multiplied by the pixel scale
            search_radius = np.median(object_table['fwhm'] * pixel_scale) / 2
        else:
            # Fallback to 1 arcsec (in degrees)
            search_radius = 1.0 / 3600

    # Log calibration details
    log.info(f"Performing photometric calibration of {len(object_table):d} objects vs {len(catalog_table):d} catalog stars")
    log.info(f"Using {search_radius * 3600:.1f} arcsec matching radius, {catalog_mag_column:s} magnitude, and spatial order {spatial_order:d}")

    # Calculate color term if color columns are provided
    if catalog_mag1_column and catalog_mag2_column:
        log.info(f"Using ({catalog_mag1_column:s} - {catalog_mag2_column:s}) color for color term")
        color = catalog_table[catalog_mag1_column] - catalog_table[catalog_mag2_column]
    else:
        color = None

    # Handle catalog magnitude errors
    catalog_mag_error = catalog_table[catalog_mag_error_column] if catalog_mag_error_column else None

    # Perform photometric matching
    match_results = match(
        object_table[object_ra_column],
        object_table[object_dec_column],
        object_table[object_mag_column],
        object_table[object_mag_error_column],
        object_table[object_flags_column],
        catalog_table[catalog_ra_column],
        catalog_table[catalog_dec_column],
        catalog_table[catalog_mag_column],
        cat_magerr=catalog_mag_error,
        sr=search_radius,
        cat_color=color,
        obj_x=object_table[object_x_column] if object_x_column else None,
        obj_y=object_table[object_y_column] if object_y_column else None,
        spatial_order=spatial_order,
        bg_order=background_order,
        ecmag_thresh=error_threshold,
        cmag_limits=magnitude_limits,
        verbose=verbose,
        **kwargs
    )

    if match_results:
        log.info("Photometric calibration finished successfully.")
        # Store catalog column names in the results
        match_results['cat_col_mag'] = catalog_mag_column
        if catalog_mag1_column and catalog_mag2_column:
            match_results['cat_col_mag1'] = catalog_mag1_column
            match_results['cat_col_mag2'] = catalog_mag2_column

        # Update object table with calibrated magnitudes if requested
        if update_results:
            object_table['mag_calib'] = object_table[object_mag_column] + match_results['zero_fn'](
                object_table[object_x_column], object_table[object_y_column], object_table[object_mag_column]
            )
            object_table['mag_calib_err'] = np.hypot(
                object_table[object_mag_error_column],
                match_results['zero_fn'](object_table[object_x_column], object_table[object_y_column], object_table[object_mag_column], get_err=True)
            )
    else:
        log.info("Photometric calibration failed")

    return match_results


def convert_match_results_to_table(match_results, pixscale=None, columns=None):
    """Convert dict returned by calibrate_photometry() to an astropy Table.

    Resulting table includes:
        - `oidx`, `cidx`, `dist`: indices of positionally matched objects and catalogue stars, and pairwise distances in degrees.
        - `omag`, `omagerr`, `cmag`, `cmagerr`: instrumental and catalogue magnitudes with errors.
        - `color`: catalogue colors or zeros if no color term fitting is done.
        - `ox`, `oy`, `oflags`: image coordinates and flags of matched objects.
        - `zero`, `zero_err`: empirical zero points (cat - instr. magnitudes) and errors.
        - `zero_model`, `zero_model_err`: modeled zero points and fit errors.
        - `color_term`: fitted color term used in the calibration.
        - `obj_zero`: zero points for all input objects (not necessarily matched).
        - `idx`: boolean index for final fit objects.
        - `idx0`: boolean index for initial quality cut objects.

    Args:
        match_results (dict): Dictionary output from calibrate_photometry().
        pixscale (float, optional): Pixel scale in arcsec/pixel to add `fwhm_arcsec` and `ell` columns.
        columns (list, optional): List of columns to retain in the resulting table.

    Returns:
        astropy.table.Table: Formatted table of photometric results.
    """
    from astropy.table import Table

    m_table = Table()
    ref_len = None

    for i, key in enumerate(match_results.keys()):
        if key in ('zero_fn', 'error_scale', 'intrinsic_rms'):
            continue

        value = match_results[key]
        if value is None:
            continue

        try:
            is_vector = len(value) > 1 or (len(value) == 1 and not isinstance(value, (float, int)))
        except TypeError:
            is_vector = False

        if is_vector:
            if ref_len is None:
                ref_len = len(value)
            if len(value) == ref_len:
                m_table[key] = value

    if columns is not None:
        m_table = m_table[columns]

    if pixscale is not None:
        if 'fwhm' in m_table.colnames:
            fwhm_index = m_table.colnames.index('fwhm')
            m_table.add_column(m_table['fwhm'] * pixscale, name='fwhm_arcsec', index=fwhm_index + 1)
        if 'b' in m_table.colnames and 'a' in m_table.colnames:
            b_index = m_table.colnames.index('b')
            m_table.add_column(1 - m_table['b'] / m_table['a'], name='ell', index=b_index + 1)

    return m_table


def get_photometric_zeropoint(match_results, use_model=False):
    """Calculate the median photometric zero point from calibration results.

    Args:
        match_results (dict): Output from calibrate_photometry().
        use_model (bool): If True, use modeled zero points instead of empirical ones.

    Returns:
        tuple: (median_zero_point, median_zero_point_error)
    """
    if use_model:
        median_zero_point = np.nanmedian(match_results['zero_model'])
        median_zero_point_error = np.nanmedian(match_results['zero_model_err'])
    else:
        median_zero_point = np.nanmedian(match_results['zero'])
        median_zero_point_error = np.nanmedian(match_results['zero_err'])

    return median_zero_point, median_zero_point_error
