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

def mask_fov(image, binning):
    """
        Mask out the edges of the FOV of the Goodman images
    """
    # define center of the FOV for binning 1, 2, and 3 
    if binning == 1:
        center_x, center_y, radius = 1520, 1570, 1550
    if binning == 2:
        center_x, center_y, radius = 770, 800, 775
    if binning == 3:
        center_x, center_y, radius = 510, 540, 515
    else:
        center_x, center_y, radius = image.shape[0]/2., image.shape[1]/2., image.shape[0]/2.

    # create a grid of pixel coordinates
    x, y = np.meshgrid( np.arange(image.shape[1]), np.arange(image.shape[0]) )

    # calculate the distance of each pixel from the center of the FOV
    distance = np.sqrt( (x - center_x)**2 + (y - center_y)**2 )
    mask_fov = distance > radius
    return mask_fov

###########################################################
filename = './0277_wd1_r_5.fits'
filename = './0274_wd1_r_025.fits'
filename = './0280_wd1_r_60.fits'

image  = fits.getdata(filename).astype(np.double)
# add WCS
header = goodman_wcs(fits.getheader(filename))

hdu = fits.PrimaryHDU(data=image, header=header)
hdul = fits.HDUList([hdu])
hdul.writeto(filename.replace(".fits","_wcs_init.fits"),overwrite=True)

# get keywords from header
fname = header.get('FILTER')
binning = np.array([int(b) for b in header['CCDSUM'].split(' ')])[0]
time  = utils.get_obs_time(header, verbose=False)
gain = header.get('GAIN')
# TODO define Saturation threshould based on GAIN from header (if gain==1.48, saturation == 50000,...)
saturation = 50000

print('Processing %s: filter %s gain %.2f at %s' 
      % (filename, fname, gain, time))

# Create mask of bad pixels
mask = image > saturation # Rough saturation level

# Cosmics
cmask, cimage = astroscrappy.detect_cosmics(image, mask, verbose=False)
mask |= cmask

# mask out edge of the fov
mask |= mask_fov(image, binning)

print('Done masking cosmics')

# write mask
hdu = fits.PrimaryHDU(data=mask.astype(int), header=header)
hdul = fits.HDUList([hdu])
hdul.writeto(filename.replace(".fits","_mask.fits"),overwrite=True)

# plot image
plt.figure()
plots.imshow(image, [0.5, 99.75], cmap='Blues_r')
plt.title(os.path.split(filename)[1])
plt.tight_layout()
plt.savefig(filename.replace(".fits",".png"))

# plot mask
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
## TODO: the code above is not properly using the mask information. need to understand how to implement it better.

# plot detections
plt.figure()
plots.imshow(image, interpolation='nearest')
plt.plot(obj['x'], obj['y'], 'r.', ms=2)
plt.title('Detected objects')
plt.tight_layout()
plt.savefig(filename.replace(".fits","_detections.png"))

##############################
# Data Quality output
#
dq_flag = obj['flags'] == 0
dq_obj = obj[dq_flag]
# get FWHM from detections (using median and median absolute deviation as error)
fwhm = np.median(dq_obj['fwhm'])
fwhm_error = np.median(np.absolute(dq_obj['fwhm'] - np.median(dq_obj['fwhm'])))
fwhm_arcsec = fwhm * pixscale * 3600.0
fwhm_error_arcsec = fwhm * pixscale * 3600.0
# estimate median ellipticity of the sources (ell = 1 - b/a)
med_a = np.median(dq_obj['a']) # major axis
med_b = np.median(dq_obj['b']) # minor axis
med_a_error = np.median(np.absolute(dq_obj['a'] - np.median(dq_obj['a'])))
med_b_error = np.median(np.absolute(dq_obj['b'] - np.median(dq_obj['b'])))
ell = 1 - med_b / med_a
ell_error = ell * np.sqrt( (med_a_error/med_a)**2 + (med_b_error/med_b)**2 )
print('------------------------')
print('  Data Quality outputs')
print('           Nobj for DQ: {}/{}'.format(len(dq_obj),len(obj)))
print('           Median FWHM: {:.2f}+/-{:.2f} pixels'.format(fwhm, fwhm_error))
print('           Median FWHM: {:.2f}+/-{:.2f} arcsec'.format(fwhm_arcsec,fwhm_error_arcsec))
print('    Median ellipticity: {:.3f}+/-{:.3f}'.format(ell, ell_error))
print('------------------------')
print('')

################################
# Prepare for astrometry

# set up Vizier catalog, filter to be used and magnitude limit.
cat_name = 'gaiadr2'
cat_filter = 'rmag'
cat_magthresh = 18

print('Performing astrometry with SCAMP using',cat_name)

cat = catalogs.get_cat_vizier(ra0, dec0, sr0, cat_name, filters={'rmag':f'<18'})
print('   ',len(cat), 'catalogue stars')
## filter is not working on the code above
cat = cat[cat[cat_filter] <= cat_magthresh]
print('    {} filtered catalogue stars [{} <= {}]'.format(len(cat),cat_filter,cat_magthresh))
print('')

cat_col_mag = 'rmag'
cat_color_mag1 = 'gmag'
cat_color_mag2 = 'rmag'

# WCS refinement
#wcs = pipeline.refine_astrometry(obj, cat, 5*pixscale, wcs=wcs, order=1, cat_col_mag=cat_col_mag, method='scamp', verbose=True)

# use SCAMP for astrometry

# set up useful plots for astrometry
SCAMP_plots = 'FGROUPS,DISTORTION,ASTR_REFERROR2D,ASTR_REFERROR1D'
SCAMP_names = ','.join([filename.replace("./","").replace(".fits","") + "_SCAMP_" + item for item in SCAMP_plots.split(',')])

header_wcs = astrometry.refine_wcs_scamp(obj, cat, sr=5*pixscale, wcs=wcs, order=3, 
                                  cat_col_ra='RAJ2000', cat_col_dec='DEJ2000', cat_col_ra_err='e_RAJ2000', cat_col_dec_err='e_DEJ2000', cat_col_mag=cat_col_mag, cat_col_mag_err='e_'+cat_col_mag, 
                                  update=True, verbose=True, get_header=True, extra={'CHECKPLOT_TYPE': SCAMP_plots, 'CHECKPLOT_NAME': SCAMP_names, 'CHECKPLOT_RES': '1200'})

# write SCAMP results as a txt file
with open(filename.replace(".fits","_scamp_results.txt"), 'w') as txt_file:
    txt_file.write(repr(header_wcs))

# get wcs from header
wcs = WCS(header_wcs)

if wcs is None or not wcs.is_celestial:
    print('WCS refinement failed')

# Update WCS info in the header
astrometry.clear_wcs(header, remove_comments=True, remove_underscored=True, remove_history=True)
header.update(wcs.to_header(relax=True))

# Save fits file with correct WCS information
hdu = fits.PrimaryHDU(data=image, header=header)
hdul = fits.HDUList([hdu])
hdul.writeto(filename.replace(".fits","_wcs.fits"),overwrite=True)

print("Astrometry done")
print('RMS(x,y) = {:.2f}" {:.2f}"'.format(header_wcs['ASTRRMS1']*3600.,header_wcs['ASTRRMS2']*3600.))

print("")

print("DONE")
