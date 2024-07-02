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
filename = './0277_wd1_r_5.fits'
filename = './0274_wd1_r_025.fits'
filename = './0280_wd1_r_60.fits'
filename = './processed/cfzt_0277_wd1_r_5.fits'
#filename = './processed/cfzt_0274_wd1_r_025.fits'
#filename = './processed/cfzt_0280_wd1_r_60.fits'

#
filename = './new_tests/0061_N24A-383097.fits'
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
filename = './new_tests/cfzt_0274_wd1_r_025.fits'

# 
print_messages          = True  # print messages on screen during execution
save_intermediary_files = False # save intermediary fits files
save_scamp_plots        = True  # save scamp astrometric plots as png files
save_plots              = True  # save plots as png files

# set up Vizier catalog, filter to be used and magnitude threshold limit.
cat_name = 'gaiadr2'
cat_magthresh = 17

scamp_flag = 1 # set maximum FLAG for performing astrometry using SCAMP, set to None for using all detections

# START THE CODE ----------

# set start of the code
start = datetime.datetime.now()

# set up log file
logfile = filename.replace(".fits","_astrometry_log.txt")
gtools.log_message(logfile,"Start the log", init=True, print_time=True)
gtools.log_message(logfile,"File: {}".format(filename), print_time=True)

##########################################################
#
# 1) load the fits file and retrieve header information
#

# reads the FITS file
image       = fits.getdata(filename).astype(np.double)
header_init = fits.getheader(filename)

# gather required information from the header
fname, binning, time, gain, rdnoise, satur_thresh, exptime = gtools.get_info(header_init)


# print information on screen
if print_messages:
    print('Processing %s: filter %s, gain %.2f, satur_thresh %.1f at %s' 
        % (filename, fname, gain, satur_thresh, time))
# log 
gtools.log_message(logfile,"filter={} exptime={:.2f} binning={}x{}".format(fname, exptime, binning, binning), print_time=True)

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

# creates HDU to save the BPM
if save_intermediary_files is True:
    hdu  = fits.PrimaryHDU(data=mask.astype(int), header=header_init)
    hdul = fits.HDUList([hdu])
    hdul.writeto(filename.replace(".fits","_mask.fits"),overwrite=True)

# plot image
file_out = filename.replace(".fits",".png") if save_plots is True else None
gtools.imgshow(image, wcs=None, title=filename.replace(".fits",""), output=file_out, qq=(0.01,0.99), cmap='Blues_r')
if save_plots: gtools.log_message(logfile,"Image - no WCS: {}".format(file_out), print_time=True)

# plot mask
file_out = filename.replace(".fits","_BPM.png") if save_plots is True else None
gtools.imgshow(mask, wcs=None, title="Bad Pixel Mask", output=file_out, qq=(0,1), cmap='Blues_r')
if save_plots: gtools.log_message(logfile,"Image - Bad pixel mask: {}".format(file_out), print_time=True)

##########################################################
#
# 3) add initial WCS based on header (RA and DEC) information
#    No correction for Instrument Position Angle is added, assuming N is up (y > 0)and E to the right (x < 0).
#
header   = gtools.goodman_wcs(header_init)
wcs_init = gtools.check_wcs(header)

# creates HDU to save intermediary files
if save_intermediary_files is True:
    hdu  = fits.PrimaryHDU(data=image, header=header)
    hdul = fits.HDUList([hdu])
    hdul.writeto(filename.replace(".fits","_wcs_init.fits"),overwrite=True)

# get center position and pixscale
ra0,dec0,fov_radius = gtools.get_frame_center(wcs=wcs_init, width=image.shape[1], height=image.shape[0])
pixscale     = gtools.get_pixscale(wcs=wcs_init)

if print_messages:
    print(f"Initial WCS: RA={ra0} DEC={dec0} SR={fov_radius} PIXSCALE={pixscale}")

# log 
gtools.log_message(logfile,"Center coordinates: RA={:.5f} Dec={:.5f} (initial WCS from fits header)".format(ra0, dec0), print_time=True)


##########################################################
#
# 4) Source detection with SExtractor
#
# Define extraction aperture: assuming a mean seeing of 1.0 arcseconds and an aperture based on a Gaussian Full Width at Tenth Maximum (FWTM).
seeing    = 1.0 # arcseconds (fixed)
fwtm2fwhm = 1.82
sextractor_aperture = np.round(fwtm2fwhm * seeing / ( pixscale * 3600. ) )

# run SExtractor to detect the sources
obj = gtools.get_objects_sextractor(image, mask=mask, gain=gain, r0=2, aper=sextractor_aperture, wcs=wcs_init)

# log 
gtools.log_message(logfile,"SExtractor aperture radius={:.1f} pixels".format(sextractor_aperture), print_time=True)

if print_messages:
    print(len(obj), 'objects found')

gtools.log_message(logfile,"SExtractor detections: {}".format(len(obj)), print_time=True)

# write number of detections per SExtractor flag.
gtools.log_message(logfile,"SExtractor detections (per flag)", print_time=True)
sex_flags = np.unique(obj['flags'])
for sflag in sex_flags:
    gtools.log_message(logfile,"flag={} - {}".format(sflag,np.sum(obj['flags']==sflag)), print_time=True)
    if print_messages:
        print("flag={} - {}".format(sflag,np.sum(obj['flags']==sflag)))
print("")

# plot detections
file_out = filename.replace(".fits","_detections.png") if save_plots is True else None
gtools.imgshow(image, wcs=None, px=obj['x'], py=obj['y'], title='Detected objects', output=file_out, qq=(0.01,0.99), cmap='Blues_r', pmarker='r.', psize=2, show_grid=False)    
if save_plots: gtools.log_message(logfile,"Image - SExtractor detections: {}".format(file_out), print_time=True)

##############################
#
# 5) Data Quality results
#
# use FLAG=0 objects to derive DQ results
dq_obj  = obj[obj['flags'] == 0]

# plot detections
file_out = filename.replace(".fits","_detections_flag0.png") if save_plots is True else None
gtools.imgshow(image, wcs=None, px=dq_obj['x'], py=dq_obj['y'], title='Detected objects (FLAG=0)', output=file_out,
               qq=(0.01,0.99), cmap='Blues_r', pmarker='r.', psize=2, show_grid=False)
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
# 6) Run SCAMP for astrometric solution
#
#
if print_messages:
    print('Performing astrometry with SCAMP using',cat_name)

# retrieve the filters to be used on the external catalog based on the Goodman filter information
cat_filter, _, _, _ = gtools.filter_sets(fname)

# log
gtools.log_message(logfile,"Querying Vizier for {} catalog.".format(cat_name), print_time=True)

# query the vizier catalog
cat = gtools.get_cat_vizier(ra0, dec0, fov_radius, cat_name, filters={cat_filter:f'<{cat_magthresh}'})     # gtools
if print_messages:
    print('   {} catalogue stars on {} filter'.format(len(cat),cat_filter))

# log
gtools.log_message(logfile,"Retrieved {} stars on {} filter (magnitude threshold={:.2f})".format(len(cat), cat_filter, cat_magthresh), print_time=True)

# if necessary, save the catalog
if save_intermediary_files is True:
    cat.write(filename.replace(".fits",f"_{cat_name}_cat.csv"), format='csv', overwrite=True)
    gtools.log_message(logfile,"Vizier catalog saved as {}.".format(filename.replace(".fits",f"_{cat_name}_cat.csv")), print_time=True)

# set up useful SCAMP plots for astrometry - this will not work when running python from Anaconda
if save_scamp_plots:
    SCAMP_plots = 'FGROUPS,DISTORTION,ASTR_REFERROR2D,ASTR_REFERROR1D'
    SCAMP_names = ','.join([filename.replace("./","").replace(".fits","") + "_SCAMP_" + item for item in SCAMP_plots.split(',')])
    scamp_extra = {'CHECKPLOT_TYPE': SCAMP_plots, 'CHECKPLOT_NAME': SCAMP_names, 'CHECKPLOT_RES': '1200'}
else:
    scamp_extra = {}

# apply filter on SExtractor detections (remove bad detections including saturated, blended, sources at the edge of the CCD, etc.)
obj_scamp = obj if scamp_flag is None else obj[obj['flags'] <= scamp_flag]

# log
gtools.log_message(logfile,"Running SCAMP for refining the WCS solution.", print_time=True)

print(scamp_extra)

# now run SCAMP to refine the WCS
header_wcs = gtools.refine_wcs_scamp(obj, cat, sr=5*pixscale, wcs=wcs_init, order=3, 
                                     cat_col_ra='RAJ2000', cat_col_dec='DEJ2000', cat_col_ra_err='e_RAJ2000', cat_col_dec_err='e_DEJ2000',
                                     cat_col_mag=cat_filter, cat_col_mag_err='e_'+cat_filter, 
                                     update=True, verbose=True, get_header=True, extra=scamp_extra)

# write SCAMP results as a txt file
if save_intermediary_files is True:
    with open(filename.replace(".fits","_scamp_results.txt"), 'w') as txt_file:
        txt_file.write(repr(header_wcs))
    gtools.log_message(logfile,"SCAMP results saved as {}".format(filename.replace(".fits","_scamp_results.txt")), print_time=True)

# get new WCS from header
wcs = gtools.check_wcs(header_wcs)

# Update WCS info in the header
header_out = gtools.clear_wcs(header, remove_comments=True, remove_underscored=True, remove_history=True)

if wcs is None or not wcs.is_celestial:
    print('WCS refinement failed. Using initial WCS from header information.')
    gtools.log_message(logfile,"WCS refinement failed. Using initial WCS from header information.", print_time=True)
    header_out.update(wcs_init.to_header(relax=True))
    header_out.append(('ASTR_SOL',  "Header information",               'Astrometry solution'),                                                          end=True)

else:
    print("WCS refinement was successful.")
    print('RMS(x,y) = {:.2f}" {:.2f}"'.format(header_wcs['ASTRRMS1']*3600.,header_wcs['ASTRRMS2']*3600.))
    gtools.log_message(logfile,'WCS refinement was successful. RMS(x,y) = {:.2f}" {:.2f}"'.format(header_wcs['ASTRRMS1']*3600.,header_wcs['ASTRRMS2']*3600.), print_time=True)

    header_out.update(wcs.to_header(relax=True))
    # Astrometric calibration setup
    header_out.append(('ASTR_SOL',  "SCAMP",                            'Astrometry solution'),                                                          end=True)
    header_out.append(('ASTRCAT',  cat_name.strip(),                    'Catalog name used for astrometric calibration'),                                end=True)
    header_out.append(('ASTRCFIL', cat_filter.strip(),                  'Filter name used for retrieving the catalog list for astrometric calibration'), end=True)
    header_out.append(('ASTRCMAG', float(cat_magthresh),                'Magnitude threshold for ASTRCFIL used for astrometric calibration'),            end=True)
    # results from photometric calibration
    header_out.append(('ASTRXRMS', float(header_wcs['ASTRRMS1']*3600.), 'Astrometric rms error on the x-axis (in arcseconds)'),                          end=True)
    header_out.append(('ASTRYRMS', float(header_wcs['ASTRRMS2']*3600.), 'Astrometric rms error on the y-axis (in arcseconds)'),                          end=True)

##############################
#
# 7) Save FITS file with correct astrometric solution
#
hdu  = fits.PrimaryHDU(data=image, header=header_out)
hdul = fits.HDUList([hdu])
hdul.writeto(filename.replace(".fits","_wcs.fits"),overwrite=True)
gtools.log_message(logfile,'FITS file saved as {}'.format(filename.replace(".fits","_wcs.fits")), print_time=True)


# set start of the code
end = datetime.datetime.now()
gtools.log_message(logfile,'Astrometric calibration executed in {:.2f} seconds'.format((end-start).total_seconds()), print_time=True)

#
# Exit
#
print("Astrometric solution was applied.")
print("")

