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
from astropy import units as u
from astropy import wcs
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.io import fits as fits
from astropy.table import Table
from astropy.time import Time
from astropy.visualization import PercentileInterval
from astropy.wcs import WCS
from astroquery.vizier import Vizier
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import binned_statistic_2d
from scipy.stats import chi2


log = logging.getLogger()


# (F Navarete)
def get_info(header):
    """
      reads a fits header and returns specific keywords used during the execution of the codes.

        header (astropy.io.fits.Header): FITS header.

      returns:
        fname          (str): filter name
        binning        (int): binning factor (single element). For imaging mode, the full binning information should be 'binning'x'binning'
        time           (str): observing time
        gain         (float): gain value (e-/ADU)
        rdnoise      (float): read noise (e-)
        satur_thresh (float): saturation threshold based on the readout mode (in ADU)
        exptime      (float): exposure time (in sec)

    """

    # check if observations are done in imaging mode. if not, exit the code
    wavmode = header.get('WAVMODE')
    check_wavmode(wavmode)

    # get filter keywords from header
    fname1 = header.get('FILTER')
    fname2 = header.get('FILTER2')
    # set the name of the active filter (deal with both filter wheels in case the first one has no filter)
    fname = fname1 if fname1 != "NO_FILTER" else fname2

    # get binning information
    serial_binning, parallel_binning = [int(b) for b in header['CCDSUM'].split()]
    time = get_obs_time(header, verbose=False)

    gain = header.get('GAIN')
    rdnoise = header.get('RDNOISE')
    satur_thresh = get_saturation(gain, rdnoise)

    exptime = header.get('EXPTIME')

    return fname, serial_binning, parallel_binning, time, gain, rdnoise, satur_thresh, exptime


# (F Navarete)
def get_saturation(gain, rdnoise):
    """
      Simple function to estimate the saturation threshold based on the readout mode.

        gain         (float): gain value (e-/ADU)
        rdnoise      (float): read noise (e-)

      returns:
        satur_thresh (float): saturation threshold based on the readout mode (in ADU)

    """
    if gain == 1.54 and rdnoise == 3.45:
        satur_thresh = 50000  # 100kHzATTN3
    elif gain == 3.48 and rdnoise == 5.88:
        satur_thresh = 25000  # 100kHzATTN2
    elif gain == 1.48 and rdnoise == 3.89:
        satur_thresh = 50000  # 344kHzATTN3
    elif gain == 3.87 and rdnoise == 7.05:
        satur_thresh = 25000  # 344kHzATTN0
    elif gain == 1.47 and rdnoise == 5.27:
        satur_thresh = 50000  # 750kHzATTN2
    elif gain == 3.77 and rdnoise == 8.99:
        satur_thresh = 25000  # 750kHzATTN0
    else:
        satur_thresh = 50000
    return satur_thresh


# (F Navarete)
def check_wavmode(wavmode):
    """
      Simple function to check whether WAVMODE is IMAGING or not.

        wavmode (str): Goodman header's keyword. Should be IMAGING or SPECTROSCOPY.

      returns:
        if wavmode is not IMAGING, halts the code.
    """
    if wavmode != "IMAGING":
        sys.exit("WAVMODE is not IMAGING. No data to process.")
    print("IMAGING data.")


# (F Navarete)
def check_wcs(header):
    """
      Simple function to check whether the header has a WCS solution or not.

        header (astropy.io.fits.Header): FITS header.

      returns:
        if no WCS is present, halts the code.

    """
    wcs = WCS(header)

    if wcs is None or not wcs.is_celestial:
        sys.exit("WCS is absent or non-celestial. Impossible to compute photometry.")

    return wcs


# (F Navarete)
def check_phot(m):
    """
      Simple function to check whether a dictionary is None or not.

        m (dict): output from calibrate_photometry()

      returns:
        if 'm' is None, halts the code.

    """
    if m is None:
        sys.exit("Impossible to retrieve photometric results.")


# (F Navarete)
def filter_sets(filter_name):
    """
      Simple function to define which set of filters will be used based on the Goodman filter in usage.

        filter_name (str): Goodman filter name (from header's FILTER/FILTER2 keywords)

      returns:
        catalog_filter (str): Gaia filter to be retrieved
        photometry_filter   (str): will convert the Gaia filter magnitude to the following filter

      TODO: Right now, the function works for SDSS filters only.
            Needs to add Bessel UBVRI, Johnson UBV, stromgren ubvy, Kron-Cousins Rc.
            Narrow band filters should deliver results in the same filter.

    """

    # photometric filters for deriving the calibration (should be as close as possible as the filter in use.
    # available filters from GaiaDR2 are:
    # "Gmag,BPmag,RPmag (gaia system)
    # Bmag,Vmag,Rmag,Imag,gmag,rmag,g_SDSS,r_SDSS,i_SDSS"
    if filter_name == "u-SDSS":
        catalog_filter = "BPmag"
        photometry_filter = "u_SDSS"
        # phot_color_mag1 = "u_SDSS"
        # phot_color_mag2 = "g_SDSS"
    elif filter_name == "g-SDSS":
        catalog_filter = "BPmag"
        photometry_filter = "g_SDSS"
        # phot_color_mag1 = "g_SDSS"
        # phot_color_mag2 = "r_SDSS"
    elif filter_name == "r-SDSS":
        catalog_filter = "Gmag"
        photometry_filter = "r_SDSS"
        # phot_color_mag1 = "g_SDSS"
        # phot_color_mag2 = "r_SDSS"
    elif filter_name == "i-SDSS" or filter_name == "z-SDSS":
        catalog_filter = "Gmag"
        photometry_filter = "i_SDSS"
        # phot_color_mag1 = "r_SDSS"
        # phot_color_mag2 = "i_SDSS"
    else:
        # for any other filter, use the GaiaDR2 G-band magnitudes
        # TODO: add transformation for the z-SDSS filter
        # TODO: add transformation for Bessel, stromgren
        catalog_filter = "Gmag"
        photometry_filter = "g_SDSS"
        # phot_color_mag1 = "g_SDSS"
        # phot_color_mag2 = "r_SDSS"

    # no need for color term on the photometric calibration of a single filter exposure.
    # return cat_filter, phot_mag, phot_color_mag1, phot_color_mag2
    return catalog_filter, photometry_filter


# astro (F Navarete)
def goodman_wcs(header):
    """
      Creates a first guess of the WCS using the telescope coordinates, the
      CCDSUM (binning), position angle and plate scale.
      Parameters
      ----------
        header (astropy.io.fits.Header): Primary Header to be updated.
      Returns
      -------
        header (astropy.io.fits.Header): Primary Header with updated WCS information.
    """

    if 'EQUINOX' not in header:
        header['EQUINOX'] = 2000.

    if 'EPOCH' not in header:
        header['EPOCH'] = 2000.

    binning = np.array([int(b) for b in header['CCDSUM'].split(' ')])

    header['PIXSCAL1'] = -binning[0] * 0.15  # arcsec (for Swarp)
    header['PIXSCAL2'] = +binning[1] * 0.15  # arcsec  (for Swarp)

    if abs(header['PIXSCAL1']) != abs(header['PIXSCAL2']):
        log.warning("Pixel scales for X and Y do not mach.")

    plate_scale = (abs(header['PIXSCAL1']) * u.arcsec).to('degree')
    p = plate_scale.to('degree').value
    w = wcs.WCS(naxis=2)

    try:
        coordinates = SkyCoord(ra=header['RA'], dec=header['DEC'],
                               unit=(u.hourangle, u.deg))

    except ValueError:

        log.error("\"RA\" and \"DEC\" missing. Using \"TELRA\" and \"TELDEC\" instead.")

        coordinates = SkyCoord(ra=header['TELRA'], dec=header['TELDEC'],
                               unit=(u.hourangle, u.deg))

    ra = coordinates.ra.to('degree').value
    dec = coordinates.dec.to('degree').value

    w.wcs.crpix = [header['NAXIS2'] / 2, header['NAXIS1'] / 2]
    w.wcs.cdelt = [+1. * p, +1. * p]  # * binning
    w.wcs.crval = [ra, dec]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    wcs_header = w.to_header()

    for key in wcs_header.keys():
        header[key] = wcs_header[key]

    return header


# (F Navarete)
def mask_fov(image, binning):
    """
      Mask out the edges of the FOV of the Goodman images.
      Parameters
      ----------
        image    (numpy.ndarray): Image from fits file.
        binning            (int): binning of the data (1, 2, 3...) from get_info()
      Returns
      -------
        mask_fov (numpy.ndarray): Boolean mask with the same dimension as 'image' (1/0 for good/masked pixels, respectively)
    """
    # define center of the FOV for binning 1, 2, and 3
    if binning == 1:
        center_x, center_y, radius = 1520, 1570, 1550
    elif binning == 2:
        center_x, center_y, radius = 770, 800, 775
    elif binning == 3:
        center_x, center_y, radius = 510, 540, 515
    # if any other binning value is provided, it will use the center of the image as a reference
    else:
        center_x, center_y, radius = image.shape[0] / 2., image.shape[1] / 2., image.shape[0] / 2.

    # create a grid of pixel coordinates
    x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))

    # calculate the distance of each pixel from the center of the FOV
    distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    mask_fov = distance > radius

    return mask_fov


# (F Navarete)
def bpm_mask(image, saturation, binning):
    """
      Creates a complete bad pixel mask, masking out saturated sources, cosmic rays and pixels outside the circular FOV.
      Mask out the edges of the FOV of the Goodman images.
      Parameters
      ----------
        image    (numpy.ndarray): Image from fits file.
        saturation         (int): Saturation threshold from get_saturation()
        binning            (int): binning of the data (1, 2, 3...) from get_info()
      Returns
      -------
        mask_fov (numpy.ndarray): Boolean mask with the same dimension as 'image' (1/0 for good/masked pixels, respectively)
    """

    # Masks out saturated pixels
    mask = image > saturation

    # Identify and masks out cosmic rays
    cmask, cimage = astroscrappy.detect_cosmics(image, mask, verbose=False)
    mask |= cmask

    # mask out edge of the fov
    mask |= mask_fov(image, binning)

    return mask


# (STDPipe)
def spherical_distance(ra1, dec1, ra2, dec2):
    """
      Evaluates the spherical distance between two sets of coordinates.

        :param ra1: First point or set of points RA
        :param dec1: First point or set of points Dec
        :param ra2: Second point or set of points RA
        :param dec2: Second point or set of points Dec
        :returns: Spherical distance in degrees

    """

    x = np.sin(np.deg2rad((ra1 - ra2) / 2))
    x *= x
    y = np.sin(np.deg2rad((dec1 - dec2) / 2))
    y *= y

    z = np.cos(np.deg2rad((dec1 + dec2) / 2))
    z *= z

    return np.rad2deg(2 * np.arcsin(np.sqrt(x * (z - y) + y)))


# (STDPipe)
def spherical_match(ra1, dec1, ra2, dec2, sr=1 / 3600):
    """
      Positional match on the sphere for two lists of coordinates.

      Aimed to be a direct replacement for :func:`esutil.htm.HTM.match` method with :code:`maxmatch=0`.

        :param ra1: First set of points RA
        :param dec1: First set of points Dec
        :param ra2: Second set of points RA
        :param dec2: Second set of points Dec
        :param sr: Maximal acceptable pair distance to be considered a match, in degrees
        :returns: Two parallel sets of indices corresponding to matches from first and second lists, along with the pairwise distances in degrees

    """

    # Ensure that inputs are arrays
    ra1 = np.atleast_1d(ra1)
    dec1 = np.atleast_1d(dec1)
    ra2 = np.atleast_1d(ra2)
    dec2 = np.atleast_1d(dec2)

    idx1, idx2, dist, _ = search_around_sky(
        SkyCoord(ra1, dec1, unit='deg'), SkyCoord(ra2, dec2, unit='deg'), sr * u.deg
    )

    dist = dist.deg  # convert to degrees

    return idx1, idx2, dist


# astrometry (STDPipe)
def get_frame_center(filename=None, header=None, wcs=None, width=None, height=None, shape=None):
    """
      Returns image center RA, Dec, and radius in degrees.
      Accepts either filename, or FITS header, or WCS structure
    """
    if not wcs:
        if header:
            wcs = WCS(header)
        elif filename:
            header = fits.getheader(filename, -1)
            wcs = WCS(header)

    if width is None or height is None:
        if header is not None:
            width = header['NAXIS1']
            height = header['NAXIS2']
        elif shape is not None:
            height, width = shape

    if not wcs or not wcs.is_celestial:
        return None, None, None

    # ra1, dec1 = wcs.all_pix2world(0, 0, 0)
    # Goodman has a circular FOV, so we need to set the origin at the center of the x-axis to estimate the radius of the FOV, not x=0.
    ra1, dec1 = wcs.all_pix2world(width / 2, 0, 0)
    ra0, dec0 = wcs.all_pix2world(width / 2, height / 2, 0)

    sr = spherical_distance(ra0, dec0, ra1, dec1)

    return ra0.item(), dec0.item(), sr.item()


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
    """Thin wrapper around SExtractor binary.

    It processes the image taking into account optional mask and noise map, and returns the list of detected objects and optionally a set of SExtractor-produced checkimages.

    You may check the SExtractor documentation at https://sextractor.readthedocs.io/en/latest/ for more details about possible parameters and general principles of its operation.
    E.g. detection flags (returned in `flags` column of results table) are documented at https://sextractor.readthedocs.io/en/latest/Flagging.html#extraction-flags-flags . In addition to these flags, any object having pixels masked by the input `mask` in its footprint will have :code:`0x100` flag set.

    :param image: Input image as a NumPy array
    :param header: Image header, optional
    :param mask: Image mask as a boolean array (True values will be masked), optional
    :param err: Image noise map as a NumPy array, optional
    :param thresh: Detection threshold, in sigmas above local background, to be used for `DETECT_THRESH` parameter of SExtractor call
    :param aper: Circular aperture radius in pixels, to be used for flux measurement. May also be list - then flux will be measured for all apertures from that list.
    :param r0: Smoothing kernel size (sigma, or FWHM/2.355) to be used for improving object detection
    :param gain: Image gain, e/ADU
    :param edge: Reject all detected objects closer to image edge than this parameter
    :param minarea: Minimal number of pixels in the object to be considered a detection (`DETECT_MINAREA` parameter of SExtractor)
    :param wcs: Astrometric solution to be used for assigning sky coordinates (`ra`/`dec`) to detected objects
    :param sn: Minimal S/N ratio for the object to be considered a detection
    :param bg_size: Background grid size in pixels (`BACK_SIZE` SExtractor parameter)
    :param sort: Whether to sort the detections in decreasing brightness or not
    :param reject_negative: Whether to reject the detections with negative fluxes
    :param checkimages: List of SExtractor checkimages to return along with detected objects. Any SExtractor checkimage type may be used here (e.g. `BACKGROUND`, `BACKGROUND_RMS`, `MINIBACKGROUND`,  `MINIBACK_RMS`, `-BACKGROUND`, `FILTERED`, `OBJECTS`, `-OBJECTS`, `SEGMENTATION`, `APERTURES`). Optional.
    :param extra_params: List of extra object parameters to return for the detection. See :code:`sex -dp` for the full list.
    :param extra: Dictionary of extra configuration parameters to be passed to SExtractor call, with keys as parameter names. See :code:`sex -dd` for the full list.
    :param psf: Path to PSFEx-made PSF model file to be used for PSF photometry. If provided, a set of PSF-measured parameters (`FLUX_PSF`, `MAG_PSF` etc) are added to detected objects. Optional
    :param catfile: If provided, output SExtractor catalogue file will be copied to this location, to be reused by external codes. Optional.
    :param _workdir: If specified, all temporary files will be created in this directory, and will be kept intact after running SExtractor. May be used for debugging exact inputs and outputs of the executable. Optional
    :param _tmpdir: If specified, all temporary files will be created in a dedicated directory (that will be deleted after running the executable) inside this path.
    :param _exe: Full path to SExtractor executable. If not provided, the code tries to locate it automatically in your :envvar:`PATH`.
    :param verbose: Whether to show verbose messages during the run of the function or not. Maybe either boolean, or a `print`-like function.
    :returns: Either the astropy.table.Table object with detected objects, or a list with table of objects (first element) and checkimages (consecutive elements), if checkimages are requested.
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
        sys.exit("Can't find SExtractor binary")
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
        kernel = make_kernel(r0, ext=2.0)
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
    cmd = (
        binname
        + ' '
        + shlex.quote(imagename)
        + ' '
        + format_astromatic_opts(opts)
    )
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


# phot (STDPipe)
def make_kernel(r0=1.0, ext=1.0):
    x, y = np.mgrid[
        np.floor(-ext * r0): np.ceil(ext * r0 + 1),
        np.floor(-ext * r0): np.ceil(ext * r0 + 1),
    ]
    r = np.hypot(x, y)
    image = np.exp(-r ** 2 / 2 / r0 ** 2)

    return image


# phot (F Navarete)
def dq_results(dq_obj):
    """
      Reads output from get_objects_sextractor() and evaluates Data Quality results.

        dq_obj (astropy.table.Table): output from get_objects_sextractor()

      Returns:
        fwhm       (float): median FWHM of the sources (in pixels)
        fwhm_error (float): median absolute error of the FWHM values (in pixels)
        ell        (float): median ellipticity of the sources
        ell_error  (float): median absolute error of the ellipticity values
    """

    # get FWHM from detections (using median and median absolute deviation as error)
    fwhm = np.median(dq_obj['fwhm'])
    fwhm_error = np.median(np.absolute(dq_obj['fwhm'] - np.median(dq_obj['fwhm'])))

    # estimate median ellipticity of the sources (ell = 1 - b/a)
    med_a = np.median(dq_obj['a'])  # major axis
    med_b = np.median(dq_obj['b'])  # minor axis
    med_a_error = np.median(np.absolute(dq_obj['a'] - np.median(dq_obj['a'])))
    med_b_error = np.median(np.absolute(dq_obj['b'] - np.median(dq_obj['b'])))
    ell = 1 - med_b / med_a
    ell_error = ell * np.sqrt((med_a_error / med_a) ** 2 + (med_b_error / med_b) ** 2)

    return fwhm, fwhm_error, ell, ell_error


# Utils (STDPipe)
def file_write(filename, contents=None, append=False):
    """
    Simple utility for writing some contents into file.
    """

    with open(filename, 'a' if append else 'w') as f:
        if contents is not None:
            f.write(contents)


# utils (STDPipe)
def table_get(table, colname, default=0):
    """
    Simple wrapper to get table column, or default value if it is not present
    """

    if colname in table.colnames:
        return table[colname]
    elif default is None:
        return None
    elif hasattr(default, '__len__'):
        # default value is array, return it
        return default
    else:
        # Broadcast scalar to proper length
        return default * np.ones(len(table), dtype=int)


# utils (STDPipe)
def get_obs_time(header=None, filename=None, string=None, get_datetime=False, verbose=False):
    """
      Extract date and time of observations from FITS headers of common formats, or from a string.

      Will try various FITS keywords that may contain the time information - `DATE_OBS`, `DATE`, `TIME_OBS`, `UT`, 'MJD', 'JD'.

        :param header: FITS header containing the information on time of observations
        :param filename: If `header` is not set, the FITS header will be loaded from the file with this name
        :param string: If provided, the time will be parsed from the string instead of FITS header
        :param get_datetime: Whether to return the time as a standard Python :class:`datetime.datetime` object instead of Astropy Time
        :param verbose: Whether to show verbose messages during the run of the function or not. May be either boolean, or a `print`-like function.
        :returns: :class:`astropy.time.Time` object corresponding to the time of observations, or a :class:`datetime.datetime` object if :code:`get_datetime=True`

    """

    # Simple wrapper to display parsed value and convert it as necessary
    def convert_time(time):
        if isinstance(time, float):
            # Try to parse floating-point value as MJD or JD, depending on the value
            if time > 0 and time < 100000:
                log.debug("Assuming it is MJD")
                time = Time(time, format='mjd')
            elif time > 2400000 and time < 2500000:
                log.debug("Assuming it is JD")
                time = Time(time, format='jd')
            else:
                # Then it is probably an Unix time?..
                log.debug("Assuming it is Unix time")
                time = Time(time, format='unix')

        else:
            time = Time(time)

        log.info(f"Time parsed as: {time.iso}")
        if get_datetime:
            return time.datetime
        else:
            return time

    if string:
        log.info(f"Parsing user-provided time string: {string}")
        try:
            return convert_time(dateutil.parser.parse(string))
        except dateutil.parser.ParserError as err:
            log.error(f"Could not parse user-provided string: {err}")
            return None

    if header is None:
        log.info(f"Loading FITS header from {filename}")
        header = fits.getheader(filename)

    for dkey in ['DATE-OBS', 'DATE', 'TIME-OBS', 'UT', 'MJD', 'JD']:
        if dkey in header:
            log.debug(f"Found {dkey}: {header[dkey]}")
            # First try to parse standard ISO time
            try:
                return convert_time(header[dkey])
            except:
                log.error(f"Could not parse {dkey} using Astropy parser")

            for tkey in ['TIME-OBS', 'UT']:
                if tkey in header:
                    log.info(f"Found {tkey}: {header[tkey]}")
                    try:
                        return convert_time(
                            dateutil.parser.parse(header[dkey] + ' ' + header[tkey])
                        )
                    except dateutil.parser.ParserError as err:
                        log.error(f"Could not parse {dkey} {tkey}: {err}")

    log.error("Unsupported FITS header time format")

    return None


# Utils (STDPipe)
def format_astromatic_opts(opts):
    """
      Auxiliary function to format dictionary of options into Astromatic compatible command-line string.
      Booleans are converted to Y/N, arrays to comma separated lists, strings are quoted when necessary
    """
    result = []

    for key in opts.keys():
        if opts[key] is None:
            pass
        elif type(opts[key]) is bool:
            result.append('-%s %s' % (key, 'Y' if opts[key] else 'N'))
        else:
            value = opts[key]

            if type(value) is str:
                value = shlex.quote(value)
            elif hasattr(value, '__len__'):
                value = ','.join([str(_) for _ in value])

            result.append('-%s %s' % (key, value))

    result = ' '.join(result)

    return result


# plots (STDPipe)
def imgshow(image, wcs=None, qq=(0.01, 0.99), cmap='Blues_r', px=None, py=None, plot_wcs=False, pmarker='r.', psize=2, title=None, figsize=None, show_grid=False, output=None, dpi=300):
    """
      Wrapper for matplotlib imshow, can plot datapoints and use the available WCS.
    """
    if figsize is None:
        plt.figure()
    else:
        plt.figure(figsize=figsize)

    # show WCS if available
    if wcs is not None:
        ax = plt.subplot(projection=wcs)
    else:
        ax = plt.subplot()
    # define 1 and 99-th percentile for plotting the data

    # percentile = PercentileInterval(99.)
    quant = np.nanquantile(image, qq)
    if quant[0] < 0:
        quant[0] = 0

    # now plot
    img = ax.imshow(image, origin='lower', vmin=quant[0], vmax=quant[1], interpolation='nearest', cmap=cmap)    # STDpipe
    if show_grid:
        ax.grid(color='white', ls='--')
    if wcs is not None:
        ax.set_xlabel('Right Ascension (J2000)')
        ax.set_ylabel('Declination (J2000)')
    # add colorbar
    plt.colorbar(img)
    # add datapoints
    if px is not None and py is not None:
        if plot_wcs:
            ax.plot(px, py, pmarker, ms=psize, transform=ax.get_transform('fk5'))
        else:
            plt.plot(px, py, pmarker, ms=psize)
    # add title
    plt.title(title)
    plt.tight_layout()
    if output is not None:
        plt.savefig(output, dpi=dpi)


# plots (STDPipe)
def colorbar(obj=None, ax=None, size="5%", pad=0.1):
    # should_restore = False

    if obj is not None:
        ax = obj.axes
    elif ax is None:
        ax = plt.gca()
        # should_restore = True

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=size, pad=pad)

    ax.get_figure().colorbar(obj, cax=cax)

    # if should_restore:
    ax.get_figure().sca(ax)


# plots (STDPipe)
def binned_map(
    x,
    y,
    value,
    bins=16,
    statistic='mean',
    qq=[0.5, 97.5],
    color=None,
    show_colorbar=True,
    show_axis=True,
    show_dots=False,
    ax=None,
    range=None,
    **kwargs,
):
    """Plots various statistical estimators binned onto regular grid from the set of irregular data points (`x`, `y`, `value`).

    :param x: Abscissae of the data points
    :param y: Ordinates of the data points
    :param value: Values of the data points
    :param bins: Number of bins per axis
    :param statistic: Statistical estimator to plot, may be `mean`, `median`, or a function
    :param qq: two-element tuple (or list) with quantiles that define lower and upper limits for image intensity normalization. Default is `[0.5, 97.5]`. Will be superseded by manually provided `vmin` and `vmax` arguments.
    :param color: Color to use for plotting the positions of data points, optional
    :param show_colorbar: Whether to show a colorbar alongside the image
    :param show_axis: Whether to show the axes around the image
    :param show_dots: Whether to overlay the positions of data points onto the plot
    :param range: Data range as [[xmin, xmax], [ymin, ymax]]
    :param ax: Matplotlib Axes object to be used for plotting, optional
    :param **kwargs: The rest of parameters will be directly passed to :func:`matplotlib.pyplot.imshow`
    :returns: None

    """
    gmag0, xe, ye, binnumbers = binned_statistic_2d(
        x, y, value, bins=bins, statistic=statistic, range=range
    )

    vmin1, vmax1 = np.percentile(gmag0[np.isfinite(gmag0)], qq)
    if 'vmin' not in kwargs:
        kwargs['vmin'] = vmin1
    if 'vmax' not in kwargs:
        kwargs['vmax'] = vmax1

    if ax is None:
        ax = plt.gca()

    if 'aspect' not in kwargs:
        kwargs['aspect'] = 'auto'

    im = ax.imshow(
        gmag0.T,
        origin='lower',
        extent=[xe[0], xe[-1], ye[0], ye[-1]],
        interpolation='nearest',
        **kwargs,
    )
    if show_colorbar:
        colorbar(im, ax=ax)

    if not show_axis:
        ax.set_axis_off()
    else:
        ax.set_axis_on()

    if show_dots:
        ax.set_autoscale_on(False)
        ax.plot(x, y, '.', color=color, alpha=0.3)


# plots (STDPipe)
def plot_photometric_match(m, ax=None, mode='mag', show_masked=True, show_final=True, cmag_limits=None, **kwargs):
    """Convenience plotting routine for photometric match results.

    It plots various representations of the photometric match results returned by :func:`stdpipe.photometry.match` or :func:`stdpipe.pipeline.calibrate_photometry`, depending on the `mode` parameter:

    -  `mag` - displays photometric residuals as a function of catalogue magnitude
    -  `normed` - displays normalized (i.e. divided by errors) photometric residuals as a function of catalogue magnitude
    -  `color` - displays photometric residuals as a function of catalogue color
    -  `zero` - displays the map of empirical zero point, i.e. difference of catalogue and instrumental magnitudes for all matched objects
    -  `model` - displays the map of zero point model
    -  `residuals` - displays fitting residuals between zero point and its model
    -  `dist` - displays the map of angular separation between matched objects and stars, in arcseconds

    The parameter `show_dots` controls whether to overlay the positions of the matched objects onto the maps, when applicable.

    :param m: Dictionary with photometric match results
    :param ax: Matplotlib Axes object to be used for plotting, optional
    :param mode: plotting mode - one of `mag`, `color`, `zero`, `model`, `residuals`, or `dist`
    :param show_masked: Whether to show masked objects
    :param show_final: Whether to additionally highlight the objects used for the final fit, i.e. not rejected during iterative thresholding
    :param **kwargs: the rest of parameters will be directly passed to :func:`stdpipe.plots.binned_map` when applicable.
    :returns: None

    """
    if ax is None:
        ax = plt.gca()

    # Textual representation of the photometric model
    model_str = 'Instr = %s' % m.get('cat_col_mag', 'Cat')

    if ('cat_col_mag1' in m.keys() and 'cat_col_mag2' in m.keys() and 'color_term' in m.keys() and m['color_term'] is not None):
        sign = '-' if m['color_term'] > 0 else '+'
        model_str += ' %s %.2f (%s - %s)' % (sign, np.abs(m['color_term']), m['cat_col_mag1'], m['cat_col_mag2'])

    model_str += ' + ZP'

    if mode == 'mag':
        ax.errorbar(m['cmag'][m['idx0']], (m['zero_model'] - m['zero'])[m['idx0']], m['zero_err'][m['idx0']], fmt='.', alpha=0.3, zorder=0)
        if show_masked:
            ax.plot(m['cmag'][~m['idx0']], (m['zero_model'] - m['zero'])[~m['idx0']], 'x', alpha=0.3, color='orange', label='Masked', zorder=5)
        if show_final:
            ax.plot(m['cmag'][m['idx']], (m['zero_model'] - m['zero'])[m['idx']], '.', alpha=1.0, color='red', label='Final fit', zorder=10)

        ax.axhline(0, ls='--', color='black', alpha=0.3)
        ax.legend()
        ax.grid(alpha=0.2)

        ax.set_xlabel('Catalogue %s magnitude' % (m['cat_col_mag'] if 'cat_col_mag' in m.keys() else ''))
        ax.set_ylabel('Instrumental - Model')
        ax.set_title('%d of %d unmasked stars used in final fit' % (np.sum(m['idx']), np.sum(m['idx0'])))
        ax.text(0.02, 0.05, model_str, transform=ax.transAxes)
        # limit the plot to a limiting range for catalog magnitudes
        if cmag_limits is not None:
            x = m['cmag'][m['idx0']].value
            y = (m['zero_model'] - m['zero'])[m['idx0']].value
            idx = (x > cmag_limits[0]) * (x < cmag_limits[1])
            ylim0 = (np.min(y[idx]), np.max(y[idx]))
            dy = ylim0[1] - ylim0[0]
            ylim = (ylim0[0] - 0.05 * dy, ylim0[1] + 0.05 * dy)
            ax.set_xlim(cmag_limits)
            ax.set_ylim(ylim)

    elif mode == 'normed':
        ax.plot(m['cmag'][m['idx0']], ((m['zero_model'] - m['zero']) / m['zero_err'])[m['idx0']], '.', alpha=0.3, zorder=0)
        if show_masked:
            ax.plot(m['cmag'][~m['idx0']], ((m['zero_model'] - m['zero']) / m['zero_err'])[~m['idx0']], 'x', alpha=0.3, color='orange', label='Masked', zorder=5)
        if show_final:
            ax.plot(m['cmag'][m['idx']], ((m['zero_model'] - m['zero']) / m['zero_err'])[m['idx']], '.', alpha=1.0, color='red', label='Final fit', zorder=10)

        ax.axhline(0, ls='--', color='black', alpha=0.3)
        ax.axhline(-3, ls=':', color='black', alpha=0.3)
        ax.axhline(3, ls=':', color='black', alpha=0.3)
        ax.legend()
        ax.grid(alpha=0.2)

        ax.set_xlabel('Catalogue %s magnitude' % (m['cat_col_mag'] if 'cat_col_mag' in m.keys() else ''))
        ax.set_ylabel('(Instrumental - Model) / Error')
        ax.set_title('%d of %d unmasked stars used in final fit' % (np.sum(m['idx']), np.sum(m['idx0'])))
        ax.text(0.02, 0.05, model_str, transform=ax.transAxes)

    elif mode == 'color':
        ax.errorbar(m['color'][m['idx0']], (m['zero_model'] - m['zero'])[m['idx0']], m['zero_err'][m['idx0']], fmt='.', alpha=0.3, zorder=0)
        if show_masked:
            ax.plot(m['color'][~m['idx0']], (m['zero_model'] - m['zero'])[~m['idx0']], 'x', alpha=0.3, color='orange', label='Masked', zorder=5)
        if show_final:
            ax.plot(m['color'][m['idx']], (m['zero_model'] - m['zero'])[m['idx']], '.', alpha=1.0, color='red', label='Final fit', zorder=10)

        ax.axhline(0, ls='--', color='black', alpha=0.3)
        ax.legend()
        ax.grid(alpha=0.2)
        ax.set_xlabel('Catalogue %s color' % (m['cat_col_mag1'] + '-' + m['cat_col_mag2'] if 'cat_col_mag1' in m.keys() else ''))
        ax.set_ylabel('Instrumental - Model')
        ax.set_title('color term = %.2f' % (m['color_term'] or 0.0))
        ax.text(0.02, 0.05, model_str, transform=ax.transAxes)

    elif mode == 'zero':
        if show_final:
            binned_map(m['ox'][m['idx']], m['oy'][m['idx']], m['zero'][m['idx']], ax=ax, **kwargs)
        else:
            binned_map(m['ox'][m['idx0']], m['oy'][m['idx0']], m['zero'][m['idx0']], ax=ax, **kwargs)
        ax.set_title('Zero point')

    elif mode == 'model':
        binned_map(m['ox'][m['idx0']], m['oy'][m['idx0']], m['zero_model'][m['idx0']], ax=ax, **kwargs,)
        ax.set_title('Model')

    elif mode == 'residuals':
        binned_map(m['ox'][m['idx0']], m['oy'][m['idx0']], (m['zero_model'] - m['zero'])[m['idx0']], ax=ax, **kwargs)
        ax.set_title('Instrumental - model')

    elif mode == 'dist':
        binned_map(m['ox'][m['idx']], m['oy'][m['idx']], m['dist'][m['idx']] * 3600, ax=ax, **kwargs)
        ax.set_title('%d stars: mean displacement %.1f arcsec, median %.1f arcsec' % (np.sum(m['idx']), np.mean(m['dist'][m['idx']] * 3600), np.median(m['dist'][m['idx']] * 3600)))

    return ax


# plots (F Navarete)
def plot_photcal(image, phot_table, wcs=wcs, column_scale='mag_calib', qq=(0.02, 0.98), output=None, dpi=300):
    """
      Simple function to plot the image and overlay the SExtractor detections using the calibrated magnitudes as color scale.

      image            (numpy.ndarray): image from fits file to be plotted
      phot_table (astropy.table.Table): output from phot_table()
      wcs            (astropy.wcs.WCS): WCS of the input image
      column_scale               (str): column name from 'phot_table' to be used as the color scale of the plot
      qq                  (float list): two-element list contaning the quantiles for ploting the image [default: (0.02,0.98)]
      output                     (str): full path of the filename for saving the plot
      dpi                        (int): dots-per-inches resolution of the plot [default: 300]
    """
    # plots photometric calibrated sources over the image
    from matplotlib.patches import Ellipse
    plt.figure()
    ax = plt.subplot(projection=wcs)

    # define percentiles for plotting the data
    quant = np.nanquantile(image, qq)
    if quant[0] < 0:
        quant[0] = 0

    ax.imshow(image, cmap='gray', origin='lower', vmin=quant[0], vmax=quant[1])

    norm = plt.Normalize(phot_table[column_scale].min(), phot_table[column_scale].max())
    cmap = plt.cm.viridis.reversed()

    # add ellipses to the plot:
    for row in phot_table:
        e = Ellipse((row['x'], row['y']), width=2 * row['a'], height=2 * row['b'], angle=row['theta'],
                    edgecolor=cmap(norm(row[column_scale])), facecolor='none', linewidth=0.5, alpha=0.55, transform=ax.get_transform('pixel'))
        ax.add_patch(e)

    # add color bar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # not required for Matplotlib >= 3.1
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(column_scale)
    cbar.ax.invert_yaxis()
    plt.tight_layout()
    if output is not None:
        plt.savefig(output, dpi=dpi)


# cat (STDPipe)
catalogs = {
    'ps1': {'vizier': 'II/349/ps1', 'name': 'PanSTARRS DR1'},
    'gaiadr2': {'vizier': 'I/345/gaia2', 'name': 'Gaia DR2', 'extra': ['E(BR/RP)']},
    'gaiaedr3': {'vizier': 'I/350/gaiaedr3', 'name': 'Gaia EDR3'},
    'gaiadr3syn': {'vizier': 'I/360/syntphot', 'name': 'Gaia DR3 synthetic photometry', 'extra': ['**', '_RAJ2000', '_DEJ2000']},
    'usnob1': {'vizier': 'I/284/out', 'name': 'USNO-B1'},
    'gsc': {'vizier': 'I/271/out', 'name': 'GSC 2.2'},
    'skymapper': {'vizier': 'II/358/smss', 'name': 'SkyMapper DR1.1', 'extra': ['_RAJ2000', '_DEJ2000', 'e_uPSF', 'e_vPSF', 'e_gPSF', 'e_rPSF', 'e_iPSF', 'e_zPSF']},
    'vsx': {'vizier': 'B/vsx/vsx', 'name': 'AAVSO VSX'},
    'apass': {'vizier': 'II/336/apass9', 'name': 'APASS DR9'},
    'sdss': {'vizier': 'V/147/sdss12', 'name': 'SDSS DR12', 'extra': ['_RAJ2000', '_DEJ2000']},
    'atlas': {'vizier': 'J/ApJ/867/105/refcat2', 'name': 'ATLAS-REFCAT2', 'extra': ['_RAJ2000', '_DEJ2000', 'e_Gmag', 'e_gmag', 'e_rmag', 'e_imag', 'e_zmag', 'e_Jmag', 'e_Kmag']}
}


# cat (STDPipe)
def get_cat_vizier(ra0, dec0, sr0, catalog='gaiadr2', limit=-1, filters={}, extra=[], get_distance=False, verbose=False):
    """Download any catalogue from Vizier.

    The catalogue may be anything recognizable by Vizier. For some most popular ones, we have additional support - we try to augment them with photometric measurements not originally present there, based on some analytical magnitude conversion formulae. These catalogues are:

    -  ps1        - Pan-STARRS DR1. We augment it with Johnson-Cousins B, V, R and I magnitudes
    -  gaiadr2    - Gaia DR2. We augment it  with Johnson-Cousins B, V, R and I magnitudes, as well as Pan-STARRS and SDSS ones
    -  gaiaedr3   - Gaia eDR3
    -  gaiadr3syn - Gaia DR3 synthetic photometry based on XP spectra
    -  skymapper  - SkyMapper DR1.1
    -  vsx        - AAVSO Variable Stars Index
    -  apass      - AAVSO APASS DR9
    -  sdss       - SDSS DR12
    -  atlas      - ATLAS-RefCat2, compilative all-sky reference catalogue with uniform zero-points in Pan-STARRS-like bands. We augment it with Johnson-Cousins B, V, R and I magnitudes the same way as Pan-STARRS.
    -  usnob1     - USNO-B1
    -  gsc        - Guide Star Catalogue 2.2

    :param ra0: Right Ascension of the field center, degrees
    :param dec0: Declination of the field center, degrees
    :param sr0: Field radius, degrees
    :param catalog: Any Vizier catalogue identifier, or a catalogue short name (see above)
    :param limit: Limit for the number of returned rows, optional
    :param filters: Dictionary with column filters to be applied on Vizier side. Dictionary key is the column name, value - filter expression as documented at https://vizier.u-strasbg.fr/vizier/vizHelp/cst.htx
    :param extra: List of extra column names to return in addition to default ones.
    :param get_distance: If set, the distance from the field center will be returned in `_r` column.
    :param verbose: Whether to show verbose messages during the run of the function or not. May be either boolean, or a `print`-like function.
    :returns: astropy.table.Table with catalogue as returned by Vizier, with some additional columns added for supported catalogues.
    """
    # TODO: add positional errors

    if catalog in catalogs:
        # For some catalogs we may have some additional information
        vizier_id = catalogs.get(catalog).get('vizier')
        name = catalogs.get(catalog).get('name')

        columns = (
            ['*', 'RAJ2000', 'DEJ2000', 'e_RAJ2000', 'e_DEJ2000']
            + extra
            + catalogs.get(catalog).get('extra', [])
        )
    else:
        vizier_id = catalog
        name = catalog
        columns = ['*', 'RAJ2000', 'DEJ2000', 'e_RAJ2000', 'e_DEJ2000'] + extra

    log.info(f"Requesting from VizieR: {vizier_id} columns: {columns}")
    log.info(f"Center: {ra0:.3f} {dec0:.3f} radius:{sr0:.3f}")
    log.info(f"Filters: {filters}")

    vizier = Vizier(row_limit=limit, columns=columns, column_filters=filters)

    cats = vizier.query_region(SkyCoord(ra0, dec0, unit='deg'), radius=sr0 * u.deg, catalog=vizier_id)

    if not cats or not len(cats) == 1:
        cats = vizier.query_region(
            SkyCoord(ra0, dec0, unit='deg'),
            radius=sr0 * u.deg,
            catalog=vizier_id,
            cache=False,
        )

        if not cats or not len(cats) == 1:
            log.info(f"Error requesting catalogue {catalog}")
            return None

    cat = cats[0]
    cat.meta['vizier_id'] = vizier_id
    cat.meta['name'] = name

    log.info(f"Got {len(cat)} entries with {len(cat.colnames)} columns")

    # Fix _RAJ2000/_DEJ2000
    if (
        '_RAJ2000' in cat.keys()
        and '_DEJ2000' in cat.keys()
        and 'RAJ2000' not in cat.keys()
    ):
        cat.rename_columns(['_RAJ2000', '_DEJ2000'], ['RAJ2000', 'DEJ2000'])

    if get_distance and 'RAJ2000' in cat.colnames and 'DEJ2000' in cat.colnames:
        log.info("Augmenting the catalogue with distances from field center")
        cat['_r'] = spherical_distance(ra0, dec0, cat['RAJ2000'], cat['DEJ2000'])

    # Augment catalogue with additional bandpasses
    if catalog == 'gaiadr2':
        log.info("Augmenting the catalogue with Johnson-Cousins photometry")

        # My simple Gaia DR2 to Johnson conversion based on Stetson standards (this is from STDPipe - GaiaDR2 transformations do not have B filter)
        pB = [-0.05927724559795761,
              0.4224326324292696,
              0.626219707920836,
              -0.011211539139725953,]
        pV = [0.0017624722901609662,
              0.15671377090187089,
              0.03123927839356175,
              0.041448557506784556,]
        pR = [0.02045449129406191,
              0.054005149296716175,
              -0.3135475489352255,
              0.020545083667168156,]
        pI = [0.005092289380850884,
              0.07027022935721515,
              -0.7025553064161775,
              -0.02747532184796779,]

        pCB = [876.4047401692277, 5.114021693079334, -2.7332873314449326, 0]
        pCV = [98.03049528983964, 20.582521666713028, 0.8690079603974803, 0]
        pCR = [347.42190542330945, 39.42482430363565, 0.8626828845232541, 0]
        pCI = [79.4028706486939, 9.176899238787003, -0.7826315256072135, 0]

        g = cat['Gmag']
        bp = cat['BPmag']
        rp = cat['RPmag']
        bp_rp = cat['BPmag'] - cat['RPmag']

        # phot_bp_rp_excess_factor == E(BR/RP) == E_BR_RP_
        Cstar = cat['E_BR_RP_'] - np.polyval([-0.00445024, 0.0570293, -0.02810592, 1.20477819], bp_rp)

        # https://www.cosmos.esa.int/web/gaia/dr2-known-issues#PhotometrySystematicEffectsAndResponseCurves
        gcorr = g.copy()
        gcorr[(g > 2) & (g < 6)] = (-0.047344
                                    + 1.16405 * g[(g > 2) & (g < 6)]
                                    - 0.046799 * g[(g > 2) & (g < 6)] ** 2
                                    + 0.0035015 * g[(g > 2) & (g < 6)] ** 3)
        gcorr[(g > 6) & (g < 16)] = g[(g > 6) & (g < 16)] - 0.0032 * (g[(g > 6) & (g < 16)] - 6)
        gcorr[g > 16] = g[g > 16] - 0.032
        g = gcorr

        cat['Bmag'] = g + np.polyval(pB, bp_rp) + np.polyval(pCB, Cstar)
        cat['Vmag'] = g + np.polyval(pV, bp_rp) + np.polyval(pCV, Cstar)
        cat['Rmag'] = g + np.polyval(pR, bp_rp) + np.polyval(pCR, Cstar)
        cat['Imag'] = g + np.polyval(pI, bp_rp) + np.polyval(pCI, Cstar)

        # Rough estimation of magnitude error, just from G band
        for _ in ['B', 'V', 'R', 'I', 'g', 'r']:
            cat['e_' + _ + 'mag'] = cat['e_Gmag']

        # Copies of columns for convenience
        for _ in ['B', 'V', 'R', 'I']:
            cat[_] = cat[_ + 'mag']

        # to SDSS, from https://gea.esac.esa.int/archive/documentation/GDR2/Data_processing/chap_cu5pho/sec_cu5pho_calibr/ssec_cu5pho_PhotTransf.html
        cat['r_SDSS'] = g - (-0.12879 + 0.24662 * bp_rp - 0.027464 * bp_rp ** 2 - 0.049465 * bp_rp ** 3)
        cat['i_SDSS'] = g - (-0.29676 + 0.64728 * bp_rp - 0.101410 * bp_rp ** 2)
        cat['g_SDSS'] = g - (+0.13518 - 0.46245 * bp_rp - 0.251710 * bp_rp ** 2 + 0.021349 * bp_rp ** 3)

        # (F Navarete)
        # apply selection criteria for transforming to SDSS magnitudes
        # compute SDSS magnitudes only if sources are not saturated
        mask = (bp > 4.0) * (rp > 4.0) * (g > 6.5)
        # secondary masks for the SDSS filters
        mask_g = (cat['g_SDSS'] > 15.0) * (cat['g_SDSS'] > g + 1.4 * bp_rp - 1.37)
        mask_r = (cat['r_SDSS'] > 15.0) * (cat['r_SDSS'] > g + 0.7 * bp_rp - 1.50)
        mask_i = (cat['i_SDSS'] > 15.0)
        # set to NaN all magnitudes that do not satisfy the photometric criteria
        cat['g_SDSS'][~(mask * mask_g)] = np.nan
        cat['r_SDSS'][~(mask * mask_r)] = np.nan
        cat['i_SDSS'][~(mask * mask_i)] = np.nan

    elif catalog == 'gaiadr3':
        print("Gaia DR3 to be implemented yet...")
    else:
        print("Must select 'gaiadr2' from now...")

    return cat


# astrometry (STDPipe)
def table_to_ldac(table, header=None, writeto=None):

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


# astrometry (STDPipe)
def get_pixscale(wcs=None, filename=None, header=None):
    '''
    Returns pixel scale of an image in degrees per pixel.
    Accepts either WCS structure, or FITS header, or filename
    '''
    if not wcs:
        if header:
            wcs = WCS(header=header)
        elif filename:
            header = fits.getheader(filename, -1)
            wcs = WCS(header=header)

    return np.hypot(wcs.pixel_scale_matrix[0, 0], wcs.pixel_scale_matrix[0, 1])


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
    """Wrapper for running SCAMP on user-provided object list and catalogue to get refined astrometric solution.

    :param obj: List of objects on the frame that should contain at least `x`, `y` and `flux` columns.
    :param cat: Reference astrometric catalogue
    :param wcs: Initial WCS
    :param header: FITS header containing initial astrometric solution, optional.
    :param sr: Matching radius in degrees
    :param order: Polynomial order for PV distortion solution (1 or greater)
    :param cat_col_ra: Catalogue column name for Right Ascension
    :param cat_col_dec: Catalogue column name for Declination
    :param cat_col_ra_err: Catalogue column name for Right Ascension error
    :param cat_col_dec_err: Catalogue column name for Declination error
    :param cat_col_mag: Catalogue column name for the magnitude in closest band
    :param cat_col_mag_err: Catalogue column name for the magnitude error
    :param cat_mag_lim: Magnitude limit for catalogue stars
    :param sn: If provided, only objects with signal to noise ratio exceeding this value will be used for matching.
    :param extra: Dictionary of additional parameters to be passed to SCAMP binary, optional.
    :param get_header: If True, function will return the FITS header object instead of WCS solution
    :param update: If set, the object list will be updated in-place to contain correct `ra` and `dec` sky coordinates
    :param _workdir: If specified, all temporary files will be created in this directory, and will be kept intact after running SCAMP. May be used for debugging exact inputs and outputs of the executable. Optional
    :param _tmpdir: If specified, all temporary files will be created in a dedicated directory (that will be deleted after running the executable) inside this path.
    :param _exe: Full path to SCAMP executable. If not provided, the code tries to locate it automatically in your :envvar:`PATH`.
    :param verbose: Whether to show verbose messages during the run of the function or not. May be either boolean, or a `print`-like function.
    :returns: Refined astrometric solution, or FITS header if :code:`get_header=True`
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
        return None
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
                    'ERRA_WORLD': table_get(cat, cat_col_ra_err, 1 / 3600),
                    'ERRB_WORLD': table_get(cat, cat_col_dec_err, 1 / 3600),
                    'MAG': table_get(cat, cat_col_mag, 0),
                    'MAGERR': table_get(cat, cat_col_mag_err, 0.01),
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
                        (t_cat['MAG'] >= cat_mag_lim[0])
                        & (t_cat['MAG'] <= cat_mag_lim[1])
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
            diag['NDeg_Reference'] < 3
            or chi2.sf(diag['Chi2_Reference'], df=diag['NDeg_Reference']) < 1e-3
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


# astro (STDPipe)
def clear_wcs(header,
              remove_comments=False,
              remove_history=False,
              remove_underscored=False,
              copy=False,):
    """Clears WCS related keywords from FITS header

    :param header: Header to operate on
    :param remove_comments: Whether to also remove COMMENT keywords
    :param remove_history: Whether to also remove HISTORY keywords
    :param remove_underscored: Whether to also remove all keywords starting with underscore (often made by e.g. Astrometry.Net)
    :param copy: If True, do not change original FITS header
    :returns: Modified FITS header

    """
    if copy:
        header = header.copy()

    wcs_keywords = [
        'WCSAXES',
        'CRPIX1',
        'CRPIX2',
        'PC1_1',
        'PC1_2',
        'PC2_1',
        'PC2_2',
        'CDELT1',
        'CDELT2',
        'CUNIT1',
        'CUNIT2',
        'CTYPE1',
        'CTYPE2',
        'CRVAL1',
        'CRVAL2',
        'LONPOLE',
        'LATPOLE',
        'RADESYS',
        'EQUINOX',
        'B_ORDER',
        'A_ORDER',
        'BP_ORDER',
        'AP_ORDER',
        'CD1_1',
        'CD2_1',
        'CD1_2',
        'CD2_2',
        'IMAGEW',
        'IMAGEH',
    ]

    scamp_keywords = [
        'FGROUPNO',
        'ASTIRMS1',
        'ASTIRMS2',
        'ASTRRMS1',
        'ASTRRMS2',
        'ASTINST',
        'FLXSCALE',
        'MAGZEROP',
        'PHOTIRMS',
        'PHOTINST',
        'PHOTLINK',
    ]

    remove = []

    for key in header.keys():
        if key:
            is_delete = False

            if key in wcs_keywords:
                is_delete = True
            if key in scamp_keywords:
                is_delete = True
            if re.match(r'^(A|B|AP|BP)_\d+_\d+$', key):
                # SIP
                is_delete = True
            if re.match(r'^PV_?\d+_\d+$', key):
                # PV
                is_delete = True
            if key[0] == '_' and remove_underscored:
                is_delete = True
            if key == 'COMMENT' and remove_comments:
                is_delete = True
            if key == 'HISTORY' and remove_history:
                is_delete = True

            if is_delete:
                remove.append(key)

    for key in remove:
        header.remove(key, remove_all=True, ignore_missing=True)

    return header


# photometry (STDPipe)
def make_series(mul=1.0, x=1.0, y=1.0, order=1, sum=False, zero=True):
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)

    if zero:
        res = [np.ones_like(x) * mul]
    else:
        res = []

    for i in range(1, order + 1):
        maxr = i + 1

        for j in range(maxr):
            res.append(mul * x ** (i - j) * y ** j)
    if sum:
        return np.sum(res, axis=0)
    else:
        return res


# photometry (STDPipe + F Navarete)
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

    It tries to build the photometric model for objects detected on the image that includes catalogue magnitude, positionally-dependent zero point, linear color term, optional additive flux term, and also takes into account possible intrinsic magnitude scatter on top of measurement errors.

    :param obj_ra: Array of Right Ascension values for the objects
    :param obj_dec: Array of Declination values for the objects
    :param obj_mag: Array of instrumental magnitude values for the objects
    :param obj_magerr: Array of instrumental magnitude errors for the objects
    :param obj_flags: Array of flags for the objects
    :param cat_ra: Array of catalogue Right Ascension values
    :param cat_dec: Array of catalogue Declination values
    :param cat_mag: Array of catalogue magnitudes
    :param cat_magerr: Array of catalogue magnitude errors
    :param cat_color: Array of catalogue color values, optional
    :param sr: Matching radius, degrees
    :param obj_x: Array of `x` coordinates of objects on the image, optional
    :param obj_y: Array of `y` coordinates of objects on the image, optional
    :param spatial_order: Order of zero point spatial polynomial (0 for constant).
    :param bg_order: Order of additive flux term spatial polynomial (None to disable this term in the model)
    :param threshold: Rejection threshold (relative to magnitude errors) for object-catalogue pair to be rejected from the fit
    :param niter: Number of iterations for the fitting
    :param accept_flags: Bitmask for acceptable object flags. Objects having any other
    :param cat_saturation: Saturation level for the catalogue - stars brighter than this magnitude will be excluded from the fit
    :param max_intrinsic_rms: Maximal intrinsic RMS to use during the fitting. If set to 0, no intrinsic scatter is included in the noise model.
    :param sn: Minimal acceptable signal to noise ratio (1/obj_magerr) for the objects to be included in the fit
    :param verbose: Whether to show verbose messages during the run of the function or not. May be either boolean, or a `print`-like function.
    :param robust: Whether to use robust least squares fitting routine instead of weighted least squares
    :param scale_noise: Whether to re-scale the noise model (object and catalogue magnitude errors) to match actual scatter of the data points or not. Intrinsic scatter term is not being scaled this way.
    :param ecmag_thresh: set maximum photometri error to be considered for the photometric calibration (for both observed and catalogue magnitudes)
    :param cmag_limits: set magnitude range for the catalog magnitudes to avoid weird values ([8,22] should work for most of the cases)
    :param use_color: Whether to use catalogue color for deriving the color term.
    :returns: The dictionary with photometric results, as described below.

    The results of photometric matching are returned in a dictionary with the following fields:

    -  `oidx`, `cidx`, `dist` - indices of positionally matched objects and catalogue stars, as well as their pairwise distances in degrees
    -  `omag`, `omagerr`, `cmag`, `cmagerr` - arrays of object instrumental magnitudes of matched objects, corresponding catalogue magnitudes, and their errors. Array lengths are equal to the number of positional matches.
    -  `color` - catalogue colors corresponding to the matches, or zeros if no color term fitting is requested
    -  `ox`, `oy`, `oflags` - coordinates of matched objects on the image, and their flags
    -  `zero`, `zero_err` - empirical zero points (catalogue - instrumental magnitudes) for every matched object, as well as its errors, derived as a hypotenuse of their corresponding errors.
    -  `zero_model`, `zero_model_err` - modeled "full" zero points (including color terms) for matched objects, and their corresponding errors from the fit
    -  `color_term` - fitted color term. Instrumental photometric system is defined as :code:`obj_mag = cat_mag - color*color_term`
    -  `zero_fn` - function to compute the zero point (without color term) at a given position and for a given instrumental magnitude of object, and optionally its error.
    -  `obj_zero` - zero points for all input objects (not necessarily matched to the catalogue) computed through aforementioned function, i.e. without color term
    -  `params` - Internal parameters of the fittting polynomial
    -  `intrinsic_rms`, `error_scale` - fitted values of intrinsic scatter and noise scaling
    -  `idx` - boolean index of matched objects/catalogue stars used in the final fit (i.e. not rejected during iterative thresholding, and passing initial quality cuts
    -  `idx0` - the same but with just initial quality cuts taken into account

    Returned zero point computation function has the following signature:

    :obj:`zero_fn(xx, yy, mag=None, get_err=False, add_intrinsic_rms=False)`

    where `xx` and `yy` are coordinates on the image, `mag` is object instrumental magnitude (needed to compute additive flux term). If :code:`get_err=True`, the function returns estimated zero point error instead of zero point, and `add_intrinsic_rms` controls whether this error estimation should also include intrinsic scatter term or not.

    The zero point returned by this function does not include the contribution of color term. Therefore, in order to derive the final calibrated magnitude for the object, you will need to manually add the color contribution: :code:`mag_calibrated = mag_instrumental + color*color_term`, where `color` is a true object color, and `color_term` is reported in the photometric results.

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
    idx0 = (np.isfinite(omag)
            & np.isfinite(omag_err)
            & np.isfinite(cmag)
            & np.isfinite(cmag_err) & ((oflags & ~accept_flags) == 0))  # initial mask

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
                (zero - zero_model)[idx], total_err[idx], max=max_intrinsic_rms
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


# photometry - pipeline (STDPipe)
def calibrate_photometry(
    obj,
    cat,
    sr=None,
    pixscale=None,
    order=0,
    bg_order=None,
    obj_col_mag='mag',
    obj_col_mag_err='magerr',
    obj_col_ra='ra',
    obj_col_dec='dec',
    obj_col_flags='flags',
    obj_col_x='x',
    obj_col_y='y',
    cat_col_mag='R',
    cat_col_mag_err=None,
    cat_col_mag1=None,
    cat_col_mag2=None,
    cat_col_ra='RAJ2000',
    cat_col_dec='DEJ2000',
    ecmag_thresh=None,  # FN
    cmag_limits=None,  # FN
    update=True,
    verbose=False,
    **kwargs
):

    """Higher-level photometric calibration routine.

    It wraps :func:`stdpipe.photometry.match` routine with some convenient defaults so that it is easier to use with typical tabular data.

    :param obj: Table of detected objects
    :param cat: Reference photometric catalogue
    :param sr: Matching radius in degrees, optional
    :param pixscale: Pixel scale, degrees per pixel. If specified, and `sr` is not set, then median value of half of FWHM, multiplied by pixel scale, is used as a matching radius.
    :param order: Order of zero point spatial polynomial (0 for constant).
    :param bg_order: Order of additive flux term spatial polynomial (None to disable this term in the model)
    :param obj_col_mag: Column name for object instrumental magnitude
    :param obj_col_mag_err: Column name for object magnitude error
    :param obj_col_ra: Column name for object Right Ascension
    :param obj_col_dec: Column name for object Declination
    :param obj_col_flags: Column name for object flags
    :param obj_col_x: Column name for object x coordinate
    :param obj_col_y: Column name for object y coordinate
    :param cat_col_mag: Column name for catalogue magnitude
    :param cat_col_mag_err: Column name for catalogue magnitude error
    :param cat_col_mag1: Column name for the first catalogue magnitude defining the stellar color
    :param cat_col_mag2: Column name for the second catalogue magnitude defining the stellar color
    :param cat_col_ra: Column name for catalogue Right Ascension
    :param cat_col_dec: Column name for catalogue Declination
    :param ecmag_thresh: set maximum photometri error to be considered for the photometric calibration (for both observed and catalogue magnitudes)
    :param cmag_limits: set magnitude range for the catalog magnitudes to avoid weird values ([8,22] should work for most of the cases)
    :param update: If True, `mag_calib` and `mag_calib_err` columns with calibrated magnitude (without color term) and its error will be added to the object table
    :param verbose: Whether to show verbose messages during the run of the function or not. May be either boolean, or a `print`-like function
    :param **kwargs: The rest of keyword arguments will be directly passed to :func:`stdpipe.photometry.match`.
    :returns: The dictionary with photometric results, as returned by :func:`stdpipe.photometry.match`.

    """
    if sr is None:
        if pixscale is not None:
            # Matching radius of half FWHM
            sr = np.median(obj['fwhm'] * pixscale) / 2
        else:
            # Fallback value of 1 arcsec, should be sensible for most catalogues
            sr = 1. / 3600

    log.info(f"Performing photometric calibration of {len(obj):d} objects vs {len(cat):d} catalogue stars")
    log.info(f"Using {sr * 3600:.1f} arcsec matching radius, {cat_col_mag:s} magnitude and spatial order {order:d}")
    if cat_col_mag1 and cat_col_mag2:
        log.info(f"Using ({cat_col_mag1:s} - {cat_col_mag2:s}) color for color term")
        color = cat[cat_col_mag1] - cat[cat_col_mag2]
    else:
        color = None

    if cat_col_mag_err:
        cat_magerr = cat[cat_col_mag_err]
    else:
        cat_magerr = None

    m = match(obj[obj_col_ra],
              obj[obj_col_dec],
              obj[obj_col_mag],
              obj[obj_col_mag_err],
              obj[obj_col_flags],
              cat[cat_col_ra],
              cat[cat_col_dec],
              cat[cat_col_mag],
              cat_magerr=cat_magerr,
              sr=sr,
              cat_color=color,
              obj_x=obj[obj_col_x] if obj_col_x else None,
              obj_y=obj[obj_col_y] if obj_col_y else None,
              spatial_order=order,
              bg_order=bg_order,
              ecmag_thresh=ecmag_thresh,  # FN
              cmag_limits=cmag_limits,   # FN
              verbose=verbose,
              **kwargs)

    if m:
        log.info("Photometric calibration finished successfully.")
        # if m['color_term']:
        #     log.info("Color term is .2f' % m['color_term'])

        m['cat_col_mag'] = cat_col_mag
        if cat_col_mag1 and cat_col_mag2:
            m['cat_col_mag1'] = cat_col_mag1
            m['cat_col_mag2'] = cat_col_mag2

        if update:
            obj['mag_calib'] = obj[obj_col_mag] + m['zero_fn'](obj['x'], obj['y'], obj['mag'])
            obj['mag_calib_err'] = np.hypot(obj[obj_col_mag_err],
                                            m['zero_fn'](obj['x'], obj['y'], obj['mag'], get_err=True))
    else:
        log.info("Photometric calibration failed")

    return m


# phot (F Navarete)
def phot_table(m, pixscale=None, columns=None):
    """
      Convert dict returned by calibrate_photometry() to an astropy Table.
      Result table:
        -  `oidx`, `cidx`, `dist` - indices of positionally matched objects and catalogue stars, as well as their pairwise distances in degrees
        -  `omag`, `omagerr`, `cmag`, `cmagerr` - arrays of object instrumental magnitudes of matched objects, corresponding catalogue magnitudes, and their errors. Array lengths are equal to the number of positional matches.
        -  `color` - catalogue colors corresponding to the matches, or zeros if no color term fitting is requested
        -  `ox`, `oy`, `oflags` - coordinates of matched objects on the image, and their flags
        -  `zero`, `zero_err` - empirical zero points (catalogue - instrumental magnitudes) for every matched object, as well as its errors, derived as a hypotenuse of their corresponding errors.
        -  `zero_model`, `zero_model_err` - modeled "full" zero points (including color terms) for matched objects, and their corresponding errors from the fit
        -  `color_term` - fitted color term. Instrumental photometric system is defined as :code:`obj_mag = cat_mag - color*color_termude of object, and optionally its error.
        -  `obj_zero` - zero points for all input objects (not necessarily matched to the catalogue) computed through aforementioned function, i.e. without color term
        -  `idx` - boolean index of matched objects/catalogue stars used in the final fit (i.e. not rejected during iterative thresholding, and passing initial quality cuts
        -  `idx0` - the same but with just initial quality cuts taken into account`
    """
    from astropy.table import Table

    m_table = Table()

    for c, column in enumerate(m.keys()):
        # avoid three specific columns with different formats
        if (column != 'zero_fn') and (column != 'error_scale') and (column != 'intrinsic_rms'):
            l = len(m[column]) if m[column] is not None else 0
            if c == 0:
                l0 = l
            if l == l0:
                m_table[column] = m[column]
        # else:
        #    print(column)
        #    print(m[column])
        #    print("")

    if columns is not None:
        m_table = m_table[columns]

    if pixscale is not None:
        # convert FWHM from pixel to arcseconds
        fwhm_index = m_table.colnames.index('fwhm')
        m_table.add_column(m_table['fwhm'] * pixscale, name='fwhm_arcsec', index=fwhm_index + 1)
        # evaluate ellipticity
        b_index = m_table.colnames.index('b')
        m_table.add_column(1 - m_table['b'] / m_table['a'], name='ell', index=b_index + 1)

    return m_table


# phot (F Navarete)
def phot_zeropoint(m, model=False):
    """
      Reads the output from calibrate_photometry() and returns the photometric zero point.
    """
    # estimate the median photometric zero point of the image
    med_zp = np.nanmedian(m['zero'])
    med_ezp = np.nanmedian(m['zero_err'])
    if model:
        med_zp = np.nanmedian(m['zero_model'])
        med_ezp = np.nanmedian(m['zero_model_err'])

    return med_zp, med_ezp
