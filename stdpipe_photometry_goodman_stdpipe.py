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

import goodman_astro as gtools

###########################################################
filename = './0277_wd1_r_5_wcs.fits'
filename = './0274_wd1_r_025_wcs.fits'
filename = './0280_wd1_r_60_wcs.fits'

image  = fits.getdata(filename).astype(np.double)
header = fits.getheader(filename)
wcs = WCS(header)

# get center position and pixscale
center_ra, center_dec, center_sr = astrometry.get_frame_center(wcs=wcs, width=image.shape[1], height=image.shape[0])      # STDpipe
pixscale = astrometry.get_pixscale(wcs=wcs)

print('Frame center is %.2f %.2f radius %.2f arcmin, %.3f arcsec/pixel' 
      % (center_ra, center_dec, center_sr * 60., pixscale * 3600.))
print('')


# get keywords from header
fname = header.get('FILTER')
binning = np.array([int(b) for b in header['CCDSUM'].split(' ')])[0]
time  = utils.get_obs_time(header, verbose=False)
gain = header.get('GAIN')
exptime = header.get('EXPTIME')
# TODO define Saturation threshould based on GAIN from header (if gain==1.48, saturation == 50000,...)
saturation = 50000

# we need counts/sec for estimating the photometric ZP
image /= exptime

print('Processing %s: filter %s gain %.2f at %s' 
      % (filename, fname, gain, time))

# Create mask of bad pixels
mask = image > saturation # Rough saturation level

# Cosmics
cmask, cimage = astroscrappy.detect_cosmics(image, mask, verbose=False)
mask |= cmask

# mask out edge of the fov
mask |= gtools.mask_fov(image, binning)

print('Done masking cosmics')

plt.figure()
plt.subplot(projection=wcs)
plots.imshow(image, [0.5, 99.75], cmap='Blues_r')     # STDpipe
plt.title(os.path.split(filename)[1])
plt.tight_layout()
plt.savefig(filename.replace(".fits","_phot_wcs.png"))

###########################
# Extract objects (now go as deeper as 1-sigma threshould)
obj = photometry.get_objects_sextractor(image, mask=mask, gain=gain, r0=2, aper=5.0, thresh=1.0, wcs=wcs)      # STDpipe
print(len(obj), 'objects found')
img = image * ~mask
obj = photometry.get_objects_sextractor(img, mask=None, gain=gain, r0=2, aper=5.0, thresh=1.0, wcs=wcs)
print(len(obj), 'objects found (masked)')

## TODO: the code above is not properly using the mask information. need to understand how to implement it better.

plt.figure()
plt.subplot(projection=wcs)
plots.imshow(image, interpolation='nearest')    # STDpipe
plt.plot(obj['x'], obj['y'], 'r.', ms=2)
plt.title('Detected objects')
plt.tight_layout()
plt.savefig(filename.replace(".fits","_phot_detections.png"))

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
# Prepare for photometry

# set up Vizier catalog, filter to be used and magnitude limit.
cat_name = 'gaiadr2'
cat_filter = 'rmag'
cat_magthresh = 18

print('Performing astrometry with SCAMP using',cat_name)

cat = catalogs.get_cat_vizier(center_ra, center_dec, center_sr, cat_name, filters={'rmag':'<18'})     # STDpipe
print('   ',len(cat), 'catalogue stars')
## filter is not working on the code above
cat = cat[cat[cat_filter] <= cat_magthresh]
print('    {} filtered catalogue stars [{} <= {}]'.format(len(cat),cat_filter,cat_magthresh))      # STDpipe
print('')

cat_col_mag = 'rmag'
cat_color_mag1 = 'gmag'
cat_color_mag2 = 'rmag'

# Photometric calibration
m = pipeline.calibrate_photometry(obj, cat, pixscale=pixscale, cat_col_mag=cat_col_mag, cat_col_mag1=cat_color_mag1, cat_col_mag2=cat_color_mag2, order=0, verbose=True)     # STDpipe

#f = open('./phot.txt','w')
#f.write(str(m))
#f.close()

plt.figure()
plots.plot_photometric_match(m, mode='dist', bins=6)     # STDpipe
plt.tight_layout()
plt.savefig(filename.replace(".fits","_phot_photmatch.png"))

plt.figure()
plt.subplot(211)
plots.plot_photometric_match(m)     # STDpipe
#plt.ylim(-1., 1.)

plt.subplot(212)
plots.plot_photometric_match(m, mode='color')     # STDpipe
#plt.ylim(-1., 1.)
#plt.xlim(-1., 1.)
plt.tight_layout()
plt.savefig(filename.replace(".fits","_phot_photmatch2.png"))

# Zero point (difference between catalogue and instrumental magnitudes for every star) map
plt.figure()
plots.plot_photometric_match(               # STDpipe
    m, 
    mode='zero', 
    bins=6, 
    # Whether to show positions of the stars
    show_dots=True, 
    color='red', 
    aspect='equal'
)
plt.title('Zero point')
plt.tight_layout()
plt.savefig(filename.replace(".fits","_phot_zp.png"))

# estimate the median photometric zero point of the field
med_zp   = np.nanmedian(m['zero'])
med_ezp  = np.nanmedian(m['zero_err'])
med_mzp  = np.nanmedian(m['zero_model'])
med_emzp = np.nanmedian(m['zero_model_err'])

print("Median empirical ZP: {:.3f}+/-{:.3f}".format(med_zp, med_ezp))
print("    Median model ZP: {:.3f}+/-{:.3f}".format(med_mzp, med_emzp))

## TODO understand the difference betweem empirical and model ZP

print("")


## t=0.25sec
## Median empirical ZP: 25.851+/-0.036
##  Median model ZP: 25.814+/-0.003
## t=5sec
## Median empirical ZP: 26.078+/-0.014
##  Median model ZP: 26.068+/-0.005
## t=60sec
## Median empirical ZP: 26.073+/-0.013
##  Median model ZP: 26.052+/-0.006

##

print("DONE")
