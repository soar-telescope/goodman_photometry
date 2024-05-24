# Generic imports
import matplotlib.pyplot as plt
import numpy as np
import glob, datetime, os

from astropy import wcs
from astropy.wcs import WCS
from astropy.io import fits as fits

from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.time import Time

import astroscrappy

import pandas as pd

# Disable some annoying warnings from astropy
import warnings
from astropy.wcs import FITSFixedWarning
from astropy.io.fits.verify import VerifyWarning
warnings.simplefilter(action='ignore', category=FITSFixedWarning)
warnings.simplefilter(action='ignore', category=VerifyWarning)

# Load (most of) our sub-modules
from stdpipe import astrometry, photometry, catalogs, cutouts, templates, subtraction, plots, psf, pipeline, utils

# Adjust default parameters for imshow
plt.rc('image', origin='lower', cmap='Blues_r')

def goodman_wcs(header):
    """
    Creates a first guess of the WCS using the telescope coordinates, the
    CCDSUM (binning), position angle and plate scale.
    Parameters
    ----------
        data : numpy.ndarray
            2D array with the data.
        header : astropy.io.fits.Header
            Primary Header to be updated.
    Returns
    -------
        header : astropy.io.fits.Header
            Primary Header with updated WCS information.
    """

    if 'EQUINOX' not in header:
        header['EQUINOX'] = 2000.

    if 'EPOCH' not in header:
        header['EPOCH'] = 2000.
        
    binning = np.array([int(b) for b in header['CCDSUM'].split(' ')])
    
    header['PIXSCAL1'] =  -binning[0] * 0.15  # arcsec (for Swarp)
    header['PIXSCAL2'] =  +binning[1] * 0.15  # arcsec  (for Swarp)

    if abs(header['PIXSCAL1']) != abs(header['PIXSCAL2']):
        logger.warning('Pixel scales for X and Y do not mach.')
        
    plate_scale = (abs(header['PIXSCAL1'])*u.arcsec).to('degree')
    p = plate_scale.to('degree').value
    w = wcs.WCS(naxis=2)
    
    try:
        coordinates = SkyCoord(ra=header['RA'], dec=header['DEC'],
                               unit=(u.hourangle, u.deg))
    
    except ValueError:
    
        logger.error(
            '"RA" and "DEC" missing. Using "TELRA" and "TELDEC" instead.')
   
        coordinates = SkyCoord(ra=header['TELRA'], dec=header['TELDEC'],
                               unit=(u.hourangle, u.deg))
    
    ra  = coordinates.ra.to('degree').value
    dec = coordinates.dec.to('degree').value
    
    w.wcs.crpix = [header['NAXIS2'] / 2, header['NAXIS1'] / 2]
    w.wcs.cdelt = [+1.*p,+1.*p] #* binning 
    w.wcs.crval = [ra, dec]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    
    wcs_header = w.to_header()
    
    for key in wcs_header.keys():
        header[key] = wcs_header[key]
    
    return header

###########################################################
filename = './0277_wd1_r_5.fits'


image  = fits.getdata(filename).astype(np.double)
# add WCS
header = goodman_wcs(fits.getheader(filename))

hdu = fits.PrimaryHDU(data=image, header=header)
hdul = fits.HDUList([hdu])
hdul.writeto('./wcs_init.fits',overwrite=True)

fname = header.get('FILTER')
gain = header.get('GAIN')

time  = utils.get_obs_time(header, verbose=False)

print('Processing %s: filter %s gain %.2f at %s' 
      % (filename, fname, gain, time))

# Create mask of bad pixels
mask = image > 50000 # Rough saturation level

# Cosmics
cmask, cimage = astroscrappy.detect_cosmics(image, mask, verbose=False)
mask |= cmask
print('Done masking cosmics')

plt.figure()
plots.imshow(image, [0.5, 99.75], cmap='Blues_r')
plt.title(os.path.split(filename)[1])
plt.tight_layout()
plt.savefig(filename.replace(".fits",".png"))

plt.figure()
plots.imshow(mask)
plt.title('Mask')
plt.tight_layout()
plt.savefig(filename.replace(".fits","_mask.png"))

###########################
# Initial WCS
wcs = WCS(header)

# get center position and pixscale
ra0,dec0,sr0 = astrometry.get_frame_center(wcs=wcs, width=image.shape[1], height=image.shape[0])
pixscale = astrometry.get_pixscale(wcs=wcs)

print(f"RA={ra0} DEC={dec0} SR={sr0} PIXSCALE={pixscale}")

###########################
# Extract objects
obj = photometry.get_objects_sextractor(image, mask=mask, gain=gain, r0=2, aper=5.0, wcs=wcs)
print(len(obj), 'objects found')

plt.figure()
plots.imshow(image, interpolation='nearest')
plt.plot(obj['x'], obj['y'], 'r.', ms=2)
plt.title('Detected objects')
plt.tight_layout()
plt.savefig(filename.replace(".fits","_detections.png"))

# get FWHM from detections
fwhm = np.median(obj['fwhm'][obj['flags'] == 0])
med_a = np.median(obj['a'][obj['flags'] == 0])
med_b = np.median(obj['b'][obj['flags'] == 0])
ell = 1 - med_b / med_a
print('Average FWHM is %.1f pixels' % fwhm)
print('Average ellipticity is %.3f' % ell)
print('')

################################
# Prepare for astrometry

mag_thresh = 18

cat = catalogs.get_cat_vizier(ra0, dec0, sr0, 'gaiadr2', filters={'rmag':f'<{mag_thresh}'})
print(len(cat), 'catalogue stars')

cat_col_mag = 'rmag'
cat_color_mag1 = 'gmag'
cat_color_mag2 = 'rmag'

# WCS refinement
#wcs = pipeline.refine_astrometry(obj, cat, 5*pixscale, wcs=wcs, order=1, cat_col_mag=cat_col_mag, method='scamp', verbose=True)

# use SCAMP for astrometry
wcs = astrometry.refine_wcs_scamp(obj, cat, sr=5*pixscale, wcs=wcs, order=3, 
                                  cat_col_ra='RAJ2000', cat_col_dec='DEJ2000', cat_col_ra_err='e_RAJ2000', cat_col_dec_err='e_DEJ2000', cat_col_mag=cat_col_mag, cat_col_mag_err='e_'+cat_col_mag, 
                                  update=True, verbose=True)

if wcs is None or not wcs.is_celestial:
    print('WCS refinement failed')

# Update WCS info in the header
astrometry.clear_wcs(header, remove_comments=True, remove_underscored=True, remove_history=True)
header.update(wcs.to_header(relax=True))

# Save fits file with correct WCS information
hdu = fits.PrimaryHDU(data=image, header=header)
hdul = fits.HDUList([hdu])
hdul.writeto('./wcs_scamp.fits',overwrite=True)

print("Astrometry done")
print("")

print("DONE")
