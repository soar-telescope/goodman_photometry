# Generic imports
import matplotlib.pyplot as plt
import numpy as np
import datetime
from astropy.io import fits as fits
# import local tasks
import goodman_astro as gtools

# Disable some annoying warnings from astropy
import warnings
from astropy.wcs import FITSFixedWarning
from astropy.io.fits.verify import VerifyWarning
warnings.simplefilter(action='ignore', category=FITSFixedWarning)
warnings.simplefilter(action='ignore', category=VerifyWarning)

# Adjust default parameters for imshow
plt.rc('image', origin='lower', cmap='Blues_r')

###########################################################
#filename = './0277_wd1_r_5_wcs.fits'
#filename = './0274_wd1_r_025_wcs.fits'
#filename = './0280_wd1_r_60_wcs.fits'
#filename = './processed/cfzt_0277_wd1_r_5_wcs.fits'
#filename = './new_tests/0061_N24A-383097.fits'
#filename = './new_tests/0177_EP240305a_z.fits'
#filename = './new_tests/0277_wd1_r_5.fits'
#filename = './new_tests/cfzst_0456_VFTS682_r.fits'
#filename = './new_tests/cfzt_0277_wd1_r_5.fits'
#filename = './new_tests/0175_EP240305a_r.fits'
#filename = './new_tests/0178_EP240305a_g.fits'
#filename = './new_tests/0280_wd1_r_60.fits'
#filename = './new_tests/cfzst_0463_VFTS682_ii.fits'
#filename = './new_tests/cfzt_0280_wd1_r_60.fits'
#filename = './new_tests/0176_EP240305a_i.fits'
#filename = './new_tests/0274_wd1_r_025.fits'
#filename = './new_tests/cfzst_0450_VFTS682_g.fits'
#
filename = './tests_last/0274_wd1_r_025.fits'
filename = './tests_last/cfzt_0274_wd1_r_025.fits'

# add suffix for astrometric calibrated frames
filename = filename.replace(".fits","_wcs.fits")

# save intermediary fits files and plots
print_messages = True
save_intermediary_files = False
save_plots = True

color_term = False # set as True for using color term for photometric calibration

# set up Vizier catalog, filter to be used and magnitude limit.
cat_name = 'gaiadr2'  # set up the catalog name
cat_magthresh = 17    # use only sources brighter than this limit for deriving astrometric solution
# parameters for photometric calibration 
ecmag_thresh = 0.1    # use only magnitude measurements with error smaller than this limit
cmag_limits = [8,22]  # use only magnitude measurements within the provided range for catalog stars (avoid weird sources)

## TODO: save 'm' dictionary as a table containing all the photometric results - done!
## TODO: generate a log file - done
## TODO: understand the difference betweem empirical and model ZP

# START THE CODE ----------

# set start time of the code
start = datetime.datetime.now()

# set up log file
logfile = filename.replace(".fits","_photometry_log.txt")
gtools.log_message(logfile,"Start the log", init=True, print_time=True)
gtools.log_message(logfile,"File: {}".format(filename), print_time=True)

##########################################################
#
# 1) load the fits file and retrieve header information
#

# reads the FITS file
image  = fits.getdata(filename).astype(np.double)
header = fits.getheader(filename)

# check if WCS is present
wcs = gtools.check_wcs(header)


# get center position and pixscale
center_ra, center_dec, fov_radius = gtools.get_frame_center(wcs=wcs, width=image.shape[1], height=image.shape[0])      # STDpipe
pixscale = gtools.get_pixscale(wcs=wcs)

# print information on screen
if print_messages:
    print('Frame center is %.2f %.2f radius %.2f arcmin, %.3f arcsec/pixel' 
          % (center_ra, center_dec, fov_radius * 60., pixscale * 3600.))
    print('')

# log 
gtools.log_message(logfile,"RA={:.5f} Dec={:.5f}".format(center_ra, center_dec), print_time=True)

# gather required information from the header
fname, binning, time, gain, rdnoise, satur_thresh, exptime = gtools.get_info(header)

# log 
gtools.log_message(logfile,"filter={} exptime={:.2f} binning={}x{} pixscale={:.3f}".format(fname, exptime, binning, binning, 3600*pixscale), print_time=True)



# we need counts/sec for estimating the photometric ZP
image /= exptime

# print information on screen
if print_messages:
    print('Processing %s: filter %s gain %.2f at %s' 
          % (filename, fname, gain, time))

# plot image
file_out = filename.replace(".fits","_phot_wcs.png") if save_plots is True else None
gtools.imgshow(image, wcs=wcs, title=filename.replace(".fits",""), output=file_out, qq=(0.01,0.99), cmap='Blues_r')
if save_plots: gtools.log_message(logfile,"Image - Astrometric calibrated, ready for photometry: {}".format(file_out), print_time=True)



##########################################################
#
# 2) create bad pixel mask (BPM)
#    masks the outside of the FOV and identify bad pixels.
#    should work for raw and processed data
#    (even if the data is already processed by Goodman Live Pipeline, we need to mask the region outside the FOV)
#
mask = gtools.bpm_mask(image, satur_thresh, binning)
if print_messages:
    print('Done masking cosmics')

##########################################################
#
# 3) Source detection with SExtractor (now goes as deep as 1-sigma)
#
# Define extraction aperture: assuming a mean seeing of 1.0 arcseconds and an aperture based on a Gaussian Full Width at Tenth Maximum (FWTM).
seeing    = 1.0 # arcseconds (fixed)
fwtm2fwhm = 1.82
sextractor_aperture = np.round(fwtm2fwhm * seeing / ( pixscale * 3600. ) )

# log 
gtools.log_message(logfile,"SExtractor aperture radius={:.1f} pixels".format(sextractor_aperture), print_time=True)

# run SExtractor to detect the sources
obj = gtools.get_objects_sextractor(image, mask=mask, gain=gain, r0=2, aper=sextractor_aperture, thresh=1.0, wcs=wcs)

# print information on screen
if print_messages:
    print(len(obj), 'objects found')

# log 
gtools.log_message(logfile,"SExtractor detections (1-sigma threshold): {}".format(len(obj)), print_time=True)

# write number of detections per SExtractor flag.
gtools.log_message(logfile,"SExtractor detections (per flag)", print_time=True)
sex_flags = np.unique(obj['flags'])
for sflag in sex_flags:
    gtools.log_message(logfile,"flag={} - {}".format(sflag,np.sum(obj['flags']==sflag)), print_time=True)
    if print_messages:
        print("flag={} - {}".format(sflag,np.sum(obj['flags']==sflag)))

# plot detections
file_out = filename.replace(".fits","_phot_detections.png") if save_plots is True else None
gtools.imgshow(image, wcs=wcs, px=obj['x'], py=obj['y'], title='Detected objects', output=file_out, qq=(0.01,0.99), cmap='Blues_r', pmarker='r.', psize=2, show_grid=False)
if save_plots: gtools.log_message(logfile,"Image - SExtractor detections: {}".format(file_out), print_time=True)

##############################
#
# 4) Data Quality results
#
# use FLAG=0 objects to derive DQ results
dq_obj  = obj[obj['flags'] == 0]

# plot detections
file_out = filename.replace(".fits","_phot_detections_flag0.png") if save_plots is True else None
gtools.imgshow(image, wcs=None, px=dq_obj['x'], py=dq_obj['y'], title='Detected objects (FLAG=0)', output=file_out, qq=(0.01,0.99), cmap='Blues_r', pmarker='r.', psize=2, show_grid=False)
if save_plots: gtools.log_message(logfile,"Image - SExtractor detections (flag=0): {}".format(file_out), print_time=True)

# get median FWHM and ellipcity of the point sources
fwhm, fwhm_error, ell, ell_error  = gtools.dq_results(dq_obj)

# print results on screen
if print_messages:
    print('------------------------')
    print('  Data Quality outputs')
    print('           Nobj for DQ: {}/{}'.format(len(dq_obj),len(obj)))
    print('           Median FWHM: {:.2f}+/-{:.2f} pixels'.format(fwhm, fwhm_error))
    print('           Median FWHM: {:.2f}+/-{:.2f} arcsec'.format(fwhm * pixscale *3600.,fwhm_error * pixscale *3600.))
    print('    Median ellipticity: {:.3f}+/-{:.3f}'.format(ell, ell_error))
    print('------------------------')
    print('')

gtools.log_message(logfile,"Data Quality outputs", print_time=True)
gtools.log_message(logfile,"Nobj for DQ: {}/{}".format(len(dq_obj),len(obj)), print_time=True)
gtools.log_message(logfile,"Median FWHM: {:.2f}+/-{:.2f} pixels".format(fwhm, fwhm_error), print_time=True)
gtools.log_message(logfile,"Median FWHM: {:.2f}+/-{:.2f} arcsec".format(fwhm * pixscale *3600.,fwhm_error * pixscale *3600.), print_time=True)
gtools.log_message(logfile,"Median ellipticity: {:.3f}+/-{:.3f}".format(ell, ell_error), print_time=True)

##############################
#
# 5) Run photometry
#
# retrieve the filters to be used on the external catalog based on the Goodman filter information
cat_filter, phot_mag, phot_color_mag1, phot_color_mag2 = gtools.filter_sets(fname)

if print_messages:
    print("Calibrating Goodman HST {} filter observations using {} magnitudes from {} converted to {} filter.".format(fname,cat_filter,cat_name,phot_mag))

# Query Vizier
cat = gtools.get_cat_vizier(center_ra, center_dec, fov_radius, cat_name, filters={cat_filter:f'<{cat_magthresh}'})     # gtools
if print_messages:
    print('   {} catalogue stars on {} filter'.format(len(cat),cat_filter))

gtools.log_message(logfile,"Photometric calibration using {} magnitudes from {} converted to {} filter".format(cat_filter,cat_name,phot_mag), print_time=True)
if color_term:
    gtools.log_message(logfile,"using photometric color term based on {}-{} color index".format(phot_color_mag1, phot_color_mag2), print_time=True)

# photometric filters for deriving the calibration (should be as close as possible as the filter in use. available filters from GaiaDR2 are:
# "Gmag,BPmag,RPmag,Bmag,Vmag,Rmag,Imag,gmag,rmag,g_SDSS,r_SDSS,i_SDSS"

# Use color term for photometric calibration or not
phot_color_mag1 = phot_color_mag1 if color_term else None
phot_color_mag2 = phot_color_mag1 if color_term else None

# Photometric calibration
m = gtools.calibrate_photometry(obj, cat, pixscale=pixscale, ecmag_thresh = ecmag_thresh, cmag_limits = cmag_limits,  
                                cat_col_mag=phot_mag, cat_col_mag1=phot_color_mag1, cat_col_mag2=phot_color_mag2, order=0, verbose=True)

# check if there are photometric results to proceed (in case of no results, the WCS might be wrong)
gtools.check_phot(m)

# convert the dict 'm' to an astropy table (not required)
#m_table = gtools.phot_table(m)
#m_table.write(filename.replace(".fits","_phot_table.html"), format='html', overwrite=True)
#gtools.log_message(logfile,"Table of sources used for photometric calibration is stored as {}".format(filename.replace(".fits","_phot_table.html")), print_time=True)

# calibrate all detections
if color_term :
    obj['mag_calib'] = obj['mag'] + obj['color'] * m['color_term'] + m['zero_fn'](obj['x'], obj['y'])
else:
    obj['mag_calib'] = obj['mag'] + m['zero_fn'](obj['x'], obj['y'])

obj['mag_calib_err'] = np.hypot(obj['magerr'],m['zero_fn'](obj['x'], obj['y'], get_err=True)) # TODO: add color-term uncertainty

# convert photometric table to astropy table
obj_table = gtools.phot_table(obj, pixscale=pixscale*3600., columns=['x','y','xerr','yerr','flux','fluxerr','mag','magerr','a','b','theta','FLUX_RADIUS','fwhm','flags','bg','ra','dec','mag_calib','mag_calib_err'])
obj_table.write(filename.replace(".fits","_obj_table.html"), format='html', overwrite=True)
gtools.log_message(logfile,"Table of sources used for photometric calibration is stored as {}".format(filename.replace(".fits","_obj_table.html")), print_time=True)

# plot calibrated detections over the image 
gtools.plot_photcal(image, obj_table, wcs=wcs, column_scale='mag_calib', qq=(0.02,0.98))
plt.savefig(filename.replace(".fits","_phot_detections_calibrated.png"))
gtools.log_message(logfile,"Photometric calibrated detections plotted over the image with WCS solution as {}".format(filename.replace(".fits","_phot_detections_calibrated.png")), print_time=True)

# define a map in a 60"x60" binning.
plot_bins = np.round( 2. * ( fov_radius * 3600. ) / 60.0 )

plt.figure()
gtools.plot_photometric_match(m, mode='dist', bins=plot_bins)     # gtools
plt.tight_layout()
if save_plots is True:
    plt.savefig(filename.replace(".fits","_phot_photmatch.png"))

plt.figure()
plt.subplot(211)
gtools.plot_photometric_match(m)     # gtools

plt.subplot(212)
gtools.plot_photometric_match(m, mode='color')     # gtools
plt.tight_layout()
if save_plots is True:
    plt.savefig(filename.replace(".fits","_phot_photmatch2.png"))

# Zero point (difference between catalogue and instrumental magnitudes for every star) map
plt.figure()
gtools.plot_photometric_match(m, 
                                mode='zero', 
                                bins=plot_bins, 
                                # Whether to show positions of the stars
                                show_dots=True, 
                                color='red', 
                                aspect='equal')
plt.title('Zero point')
plt.tight_layout()
if save_plots is True:
    plt.savefig(filename.replace(".fits","_phot_zp.png"))

# get photometric zero point estimate
med_zp, med_ezp = gtools.phot_zeropoint(m)
if print_messages:
    print("Median empirical ZP: {:.3f}+/-{:.3f}".format(med_zp, med_ezp))
    #print("    Median model ZP: {:.3f}+/-{:.3f}".format(med_mzp, med_emzp))


gtools.log_message(logfile,"Median empirical ZP: {:.3f}+/-{:.3f}".format(med_zp, med_ezp), print_time=True)

# Update the header information
header_out = header
# results from Data Quality measurements (Sextractor)
header_out.append(('SNDET',len(obj),                           'Number of SEXtractor sources'),                                                 end=True)
header_out.append(('SNDET-DQ',len(dq_obj),                     'Number of SEXtractor sources associated with FLAG=0 (used for DQ)'),            end=True)
header_out.append(('SFWHM',float(fwhm),                        'Median FWHM of SEXtractor sources associated with FLAG=0 (in pixels)'),         end=True)
header_out.append(('SFWHMUNC',float(fwhm_error),               'Uncertainty on FWHM of SEXtractor sources associated with FLAG=0 (in pixels)'), end=True) 
header_out.append(('SELL',float(ell),                          'Ellipticity of SEXtractor sources associated with FLAG=0'),                     end=True)
header_out.append(('SELLUNC',float(ell_error),                 'Uncertainty on ellipticity of SEXtractor sources associated with FLAG=0'),      end=True) 
header_out.append(('SSEEING',float(fwhm * pixscale *3600.),    'Median FWHM of SEXtractor sources associated with FLAG=0 (in arcsec)'),         end=True)
header_out.append(('SSEEINGU',float(fwhm_error*pixscale*3600.),'Uncertainty on FWHM of SEXtractor sources associated with FLAG=0 (in arcsec)'), end=True) 
# photometric calibration setup
header_out.append(('PHOTCAT',cat_name,                         'Catalog name used for photometric calibration'),                                end=True)
header_out.append(('PHOTCFIL',cat_filter,                      'Filter name used for retrieving the catalog list for photometric calibration'), end=True)
header_out.append(('PHOTCMAG',float(cat_magthresh),            'Magnitude threshold for PHOTCFIL used for photometric calibration'),            end=True)
header_out.append(('PHOTCTHR',float(ecmag_thresh),             'Use catalog sources with magnitude errors smaller than this threshold for photometric calibration'), end=True)
header_out.append(('PHOTCLMB',str(cmag_limits),                'Magnitude range of catalog sources used for photometric calibration'),          end=True)
header_out.append(('PHOTFILT',phot_mag,                        'Filter used for photometric calibration'),                                      end=True)
header_out.append(('PHOTCOL1',phot_color_mag1,                 'Filter #1 used for color correction on the photometric calibration'),           end=True)
header_out.append(('PHOTCOL2',phot_color_mag2,                 'Filter #2 used for color correction on the photometric calibration'),           end=True)
# results from photometric calibration
header_out.append(('MAGZPT',float(med_zp),                     'Filter #2 used for color correction on the photometric calibration'),           end=True)
header_out.append(('MAGZPTER',float(med_ezp),                  'Filter #2 used for color correction on the photometric calibration'),           end=True)

##############################
#
# 6) Save FITS file with correct astrometric solution and photometric information
#   
hdu  = fits.PrimaryHDU(data=image, header=header_out)
hdul = fits.HDUList([hdu])
hdul.writeto(filename.replace(".fits","_phot.fits"),overwrite=True)

gtools.log_message(logfile,'FITS file saved as {}'.format(filename.replace(".fits","_phot.fits")), print_time=True)

# set start of the code
end = datetime.datetime.now()
gtools.log_message(logfile,'Photometric calibration executed in {:.2f} seconds'.format((end-start).total_seconds()), print_time=True)

#
# Exit
#
print("")
print("Photometric calibration was applied.")
print("")
