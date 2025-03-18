import shlex
import tempfile
import unittest
import os
import warnings
from unittest.mock import patch

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from astropy.visualization import simple_norm
from astropy.wcs import WCS
from matplotlib import pyplot as plt

from ..goodman_astro import (
    calculate_saturation_threshold,
    create_bad_pixel_mask,
    create_goodman_wcs,
    extract_observation_metadata,
    get_vizier_catalog,
    mask_field_of_view,
    table_to_ldac,
    get_pixel_scale,
    get_photometric_zeropoint,
    spherical_distance,
    spherical_match,
    get_frame_center,
    make_kernel,
    evaluate_data_quality_results,
    file_write,
    table_get_column,
    get_observation_time,
    format_astromatic_opts,
    check_wcs,
    check_photometry_results,
    get_filter_set,
    plot_image,
    add_colorbar,
    binned_map,
    plot_photometric_match,
    plot_photcal,
    clear_wcs,
    make_series,
    convert_match_results_to_table,
    # match_photometric_objects,
    wcs_sip2pv,
    get_intrinsic_scatter,
    calibrate_photometry)


class TestExtractObservationMetadata(unittest.TestCase):

    @patch("goodman_photometry.goodman_astro.sys.exit")
    def test_invalid_wavelength_mode(self, mock_exit):
        """
        Test that the function exits when WAVMODE is not 'IMAGING'.
        """
        # Create a mock FITS header with a non-'IMAGING' WAVMODE
        header = fits.Header()
        header['WAVMODE'] = 'SPECTROSCOPY'  # Invalid mode
        header['FILTER'] = 'R'
        header['FILTER2'] = 'B'
        header['CCDSUM'] = '2 2'
        header['GAIN'] = 1.5
        header['RDNOISE'] = 5.0
        header['EXPTIME'] = 100.0

        # Call the function with the mock header
        extract_observation_metadata(header)

        # Assert sys.exit was called
        mock_exit.assert_called_with("Error: WAVMODE is not IMAGING. No data to process.")

    @patch('goodman_photometry.goodman_astro.get_observation_time', return_value='2025-01-01T00:00:00')
    @patch('goodman_photometry.goodman_astro.calculate_saturation_threshold', return_value=50000)
    def test_valid_header(self, mock_saturation, mock_time):
        """
        Test that the function correctly extracts metadata when WAVMODE is 'IMAGING'.
        """
        # Create a mock FITS header with valid data
        header = fits.Header()
        header['WAVMODE'] = 'IMAGING'
        header['FILTER'] = 'R'
        header['FILTER2'] = 'B'
        header['CCDSUM'] = '2 2'
        header['GAIN'] = 1.5
        header['RDNOISE'] = 5.0
        header['EXPTIME'] = 100.0

        # Call the function with the mock header
        metadata = extract_observation_metadata(header)

        # Check that the metadata was correctly extracted
        expected_metadata = ('R', 2, 2, '2025-01-01T00:00:00', 1.5, 5.0, 50000, 100.0)
        self.assertEqual(metadata, expected_metadata)


# from goodman_astro import calculate_saturation_threshold

class TestCalculateSaturationThreshold(unittest.TestCase):

    def test_known_gain_and_read_noise(self):
        """
        Test known combinations of gain and read noise that have specific thresholds.
        """
        test_cases = [
            (1.54, 3.45, 50000),  # 100kHzATTN3
            (3.48, 5.88, 25000),  # 100kHzATTN2
            (1.48, 3.89, 50000),  # 344kHzATTN3
            (3.87, 7.05, 25000),  # 344kHzATTN0
            (1.47, 5.27, 50000),  # 750kHzATTN2
            (3.77, 8.99, 25000),  # 750kHzATTN0
        ]
        for gain, rdnoise, expected in test_cases:
            with self.subTest(gain=gain, rdnoise=rdnoise):
                result = calculate_saturation_threshold(gain, rdnoise)
                self.assertEqual(result, expected)

    def test_unknown_gain_and_read_noise(self):
        """
        Test that unknown combinations of gain and read noise default to 50000.
        """
        unknown_cases = [
            (2.0, 4.0),
            (1.0, 1.0),
            (3.0, 6.0),
        ]
        for gain, rdnoise in unknown_cases:
            with self.subTest(gain=gain, rdnoise=rdnoise):
                result = calculate_saturation_threshold(gain, rdnoise)
                self.assertEqual(result, 50000)


class TestGoodmanWCS(unittest.TestCase):

    def setUp(self):
        """Set up common headers for tests."""
        self.header_with_valid_coords = fits.Header()
        self.header_with_valid_coords['RA'] = '10:00:00'
        self.header_with_valid_coords['DEC'] = '-10:00:00'
        self.header_with_valid_coords['CCDSUM'] = '2 2'
        self.header_with_valid_coords['NAXIS1'] = 2048
        self.header_with_valid_coords['NAXIS2'] = 2048

        self.header_with_tel_coords = fits.Header()
        self.header_with_tel_coords['TELRA'] = '10:00:00'
        self.header_with_tel_coords['TELDEC'] = '-10:00:00'
        self.header_with_tel_coords['CCDSUM'] = '2 2'
        self.header_with_tel_coords['NAXIS1'] = 2048
        self.header_with_tel_coords['NAXIS2'] = 2048

    def test_create_goodman_wcs_with_valid_header(self):
        updated_header = create_goodman_wcs(self.header_with_valid_coords)

        # Verify WCS keywords are present
        wcs_keys = ['CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2', 'CDELT1', 'CDELT2', 'CTYPE1', 'CTYPE2']
        for key in wcs_keys:
            self.assertIn(key, updated_header)

        # Validate pixel scales and center coordinates
        binning = 2
        expected_pixel_scale = (binning * 0.15 / 3600)  # Convert arcsec to degrees
        self.assertAlmostEqual(updated_header['CDELT1'], expected_pixel_scale, places=6)
        self.assertAlmostEqual(updated_header['CDELT2'], expected_pixel_scale, places=6)

        coordinates = SkyCoord(ra='10:00:00', dec='-10:00:00', unit=(u.hourangle, u.deg))
        self.assertAlmostEqual(updated_header['CRVAL1'], coordinates.ra.degree, places=6)
        self.assertAlmostEqual(updated_header['CRVAL2'], coordinates.dec.degree, places=6)

    def test_create_goodman_wcs_fallback_to_tel_coords(self):
        updated_header = create_goodman_wcs(self.header_with_tel_coords)

        # Verify WCS keywords are present
        wcs_keys = ['CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2', 'CDELT1', 'CDELT2', 'CTYPE1', 'CTYPE2']
        for key in wcs_keys:
            self.assertIn(key, updated_header)

        # Validate pixel scales and center coordinates
        binning = 2
        expected_pixel_scale = (binning * 0.15 / 3600)  # Convert arcsec to degrees
        self.assertAlmostEqual(updated_header['CDELT1'], expected_pixel_scale, places=6)
        self.assertAlmostEqual(updated_header['CDELT2'], expected_pixel_scale, places=6)

        coordinates = SkyCoord(ra='10:00:00', dec='-10:00:00', unit=(u.hourangle, u.deg))
        self.assertAlmostEqual(updated_header['CRVAL1'], coordinates.ra.degree, places=6)
        self.assertAlmostEqual(updated_header['CRVAL2'], coordinates.dec.degree, places=6)

    def test_create_goodman_wcs_missing_coordinates(self):
        header = fits.Header()
        header['CCDSUM'] = '2 2'
        header['NAXIS1'] = 2048
        header['NAXIS2'] = 2048

        with self.assertRaises(ValueError) as context:
            create_goodman_wcs(header)

        self.assertIn('Header must contain either "RA"/"DEC" or "TELRA"/"TELDEC".', str(context.exception))


class TestMaskFieldOfView(unittest.TestCase):

    def test_mask_field_of_view_binning_1(self):
        image = np.zeros((3100, 3100))  # Simulated image with binning 1 dimensions
        binning = 1
        mask = mask_field_of_view(image, binning)

        center_x, center_y, radius = 1520, 1570, 1550
        x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
        expected_mask = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) > radius

        np.testing.assert_array_equal(mask, expected_mask)

    def test_mask_field_of_view_binning_2(self):
        image = np.zeros((1600, 1600))  # Simulated image with binning 2 dimensions
        binning = 2
        mask = mask_field_of_view(image, binning)

        center_x, center_y, radius = 770, 800, 775
        x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
        expected_mask = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) > radius

        np.testing.assert_array_equal(mask, expected_mask)

    def test_mask_field_of_view_binning_3(self):
        image = np.zeros((1080, 1080))  # Simulated image with binning 3 dimensions
        binning = 3
        mask = mask_field_of_view(image, binning)

        center_x, center_y, radius = 510, 540, 515
        x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
        expected_mask = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) > radius

        np.testing.assert_array_equal(mask, expected_mask)

    def test_mask_field_of_view_unsupported_binning(self):
        image = np.zeros((2000, 2000))  # Simulated image
        binning = 4
        mask = mask_field_of_view(image, binning)

        center_x, center_y, radius = image.shape[0] / 2, image.shape[1] / 2, image.shape[0] / 2
        x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
        expected_mask = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) > radius

        np.testing.assert_array_equal(mask, expected_mask)


class TestCreateBadPixelMask(unittest.TestCase):

    @patch('goodman_photometry.goodman_astro.mask_field_of_view')
    @patch('astroscrappy.detect_cosmics')
    def test_create_bad_pixel_mask(self, mock_detect_cosmics, mock_mask_field_of_view):
        # Create a mock image
        image = np.array([
            [10, 20, 30],
            [40, 50, 60],
            [70, 80, 90]
        ])

        # Set the saturation threshold
        saturation_threshold = 50

        # Set the binning factor
        binning = 1

        # Mock the return value of detect_cosmics
        cosmic_ray_mask = np.array([
            [False, True, False],
            [False, False, False],
            [True, False, False]
        ])
        mock_detect_cosmics.return_value = (cosmic_ray_mask, None)

        # Mock the return value of mask_field_of_view
        fov_mask = np.array([
            [False, False, True],
            [False, False, False],
            [False, False, False]
        ])
        mock_mask_field_of_view.return_value = fov_mask

        # Call the function
        result = create_bad_pixel_mask(image, saturation_threshold, binning)

        # Expected mask
        expected_mask = np.array([
            [False, True, True],
            [False, False, True],
            [True, True, True]
        ])

        # Assert the result matches the expected mask
        np.testing.assert_array_equal(result, expected_mask)


class TestGetVizierCatalog(unittest.TestCase):

    @patch('astroquery.vizier.core.VizierClass.query_region')
    def test_get_vizier_catalog(self, mock_query_region):
        """Test the get_vizier_catalog function with precise mocking of query_region."""

        # Mock Vizier response
        mock_table = Table(
            {
                'RAJ2000': [10.684, 10.685],
                'DEJ2000': [41.269, 41.270],
                'Gmag': [15.3, 16.1],
                'BPmag': [15.7, 16.5],
                'RPmag': [14.9, 15.8],
                'E_BR_RP_': [1.1, 1.2],
                'e_Gmag': [0.01, 0.02]
            }
        )
        mock_query_region.return_value = [mock_table]

        # Test parameters
        ra_center = 10.684
        dec_center = 41.269
        search_radius = 0.1
        catalog = 'gaiadr2'
        catalog_mapped = 'I/345/gaia2'
        row_limit = 10
        column_filters = {}
        additional_columns = []
        include_distance = True

        # Call the function
        result = get_vizier_catalog(
            right_ascension=ra_center,
            declination=dec_center,
            search_radius=search_radius,
            catalog=catalog,
            row_limit=row_limit,
            column_filters=column_filters,
            extra_columns=additional_columns,
            include_distance=include_distance
        )

        # Construct expected call arguments
        expected_coord = SkyCoord(ra_center, dec_center, unit='deg')
        mock_query_region.assert_called_once_with(
            expected_coord,
            radius=search_radius * u.deg,
            catalog=catalog_mapped
        )

        # Assertions for the result
        self.assertIsNotNone(result)
        self.assertIn('RAJ2000', result.colnames)
        self.assertIn('DEJ2000', result.colnames)

        if include_distance:
            self.assertIn('_r', result.colnames)
            self.assertTrue(all(result['_r'] >= 0))

        # Verify augmentation
        self.assertTrue('Gmag' in result.colnames)
        self.assertTrue('BPmag' in result.colnames)
        self.assertTrue('RPmag' in result.colnames)

        # Optional: Check specific augmentation values
        self.assertTrue('Bmag' in result.colnames)
        self.assertTrue('Vmag' in result.colnames)
        self.assertTrue('Rmag' in result.colnames)


class TestTableToLDAC(unittest.TestCase):
    """Unit tests for the table_to_ldac function."""

    def setUp(self):
        """Set up a sample Astropy table and header for testing."""
        self.table = Table({'col1': [1, 2, 3], 'col2': [4.5, 5.5, 6.5]})
        self.header = fits.Header()
        self.header.set(keyword='TESTKEY', value='TESTVALUE')
        self.test_filename = "test_ldac.fits"

    def tearDown(self):
        """Clean up test files if they were created."""
        if os.path.exists(self.test_filename):
            os.remove(self.test_filename)

    def test_ldac_structure(self):
        """Test that the function returns a properly structured LDAC HDU list."""
        hdulist = table_to_ldac(self.table, self.header)

        # Check the number of HDUs
        self.assertEqual(len(hdulist), 3)

        # Check EXTNAMEs
        self.assertEqual(hdulist[1].header['EXTNAME'], 'LDAC_IMHEAD')
        self.assertEqual(hdulist[2].header['EXTNAME'], 'LDAC_OBJECTS')

        # Check that header information is stored
        header_data = hdulist[1].data['Field Header Card'][0]

        self.assertIn('TESTKEY', header_data)

        # Check table data
        ldac_table = Table(hdulist[2].data)
        self.assertTrue(all(ldac_table['col1'] == [1, 2, 3]))
        self.assertTrue(all(ldac_table['col2'] == [4.5, 5.5, 6.5]))

    def test_ldac_writing(self):
        """Test that the function correctly writes to a FITS file."""
        table_to_ldac(self.table, self.header, writeto=self.test_filename)

        # Ensure file is created
        self.assertTrue(os.path.exists(self.test_filename))

        # Check the file contents
        with fits.open(self.test_filename) as hdulist:
            self.assertEqual(len(hdulist), 3)
            self.assertEqual(hdulist[1].header['EXTNAME'], 'LDAC_IMHEAD')
            self.assertEqual(hdulist[2].header['EXTNAME'], 'LDAC_OBJECTS')


class TestGetPixelScale(unittest.TestCase):
    def setUp(self):
        # Create a minimal WCS header with a known pixel scale
        self.header = fits.Header()
        self.header['NAXIS'] = 2
        self.header['CTYPE1'] = 'RA---TAN'
        self.header['CTYPE2'] = 'DEC--TAN'
        self.header['CRPIX1'] = 100.0
        self.header['CRPIX2'] = 100.0
        self.header['CRVAL1'] = 0.0
        self.header['CRVAL2'] = 0.0
        self.header['CD1_1'] = -0.000277778
        self.header['CD1_2'] = 0.0
        self.header['CD2_1'] = 0.0
        self.header['CD2_2'] = 0.000277778

        # Create temporary FITS file with this header
        self.temp_fits = tempfile.NamedTemporaryFile(suffix='.fits', delete=False)
        hdu = fits.PrimaryHDU(data=np.zeros((100, 100)), header=self.header)
        hdu.writeto(self.temp_fits.name, overwrite=True)

    def tearDown(self):
        # Clean up the temp file
        if os.path.exists(self.temp_fits.name):
            os.remove(self.temp_fits.name)

    def test_get_pixel_scale_from_file(self):
        expected_scale = np.hypot(self.header['CD1_1'], self.header['CD2_1'])
        result = get_pixel_scale(filename=self.temp_fits.name)
        self.assertAlmostEqual(result, expected_scale, places=8)


class TestSphericalDistance(unittest.TestCase):
    def test_zero_distance(self):
        """Test distance between identical coordinates (should be 0)."""
        ra = 120.0
        dec = -45.0
        result = spherical_distance(ra, dec, ra, dec)
        self.assertAlmostEqual(result, 0.0, places=10)

    def test_known_distance(self):
        """Test distance between two known points on the celestial sphere."""
        ra1 = 0.0
        dec1 = 0.0
        ra2 = 0.0
        dec2 = 90.0
        result = spherical_distance(ra1, dec1, ra2, dec2)
        self.assertAlmostEqual(result, 90.0, places=6)

    def test_vectorized_inputs(self):
        """Test array inputs for coordinate pairs."""
        ra1 = np.array([0.0, 10.0])
        dec1 = np.array([0.0, 10.0])
        ra2 = np.array([0.0, 20.0])
        dec2 = np.array([90.0, 10.0])

        result = spherical_distance(ra1, dec1, ra2, dec2)

        self.assertEqual(result.shape, (2,))
        expected_distance = np.rad2deg(
            np.arccos(
                np.sin(np.deg2rad(dec1[1])) * np.sin(np.deg2rad(dec2[1])) +
                np.cos(np.deg2rad(dec1[1])) * np.cos(np.deg2rad(dec2[1])) *
                np.cos(np.deg2rad(ra1[1] - ra2[1]))
            )
        )
        self.assertAlmostEqual(result[0], 90.0, places=6)  # (0,0) to (0,90)
        self.assertAlmostEqual(result[1], expected_distance, places=6)

    def test_antipodal_points(self):
        """Test distance between antipodal points (should be 180 deg)."""
        ra1 = 0.0
        dec1 = 0.0
        ra2 = 180.0
        dec2 = 0.0
        result = spherical_distance(ra1, dec1, ra2, dec2)
        self.assertAlmostEqual(result, 180.0, places=6)


class TestSphericalMatch(unittest.TestCase):
    def test_exact_match(self):
        """Test exact positional match between identical coordinate lists."""
        ra = np.array([10.0, 20.0, 30.0])
        dec = np.array([0.0, -10.0, +5.0])

        idx1, idx2, dist = spherical_match(ra, dec, ra, dec, search_radius_deg=1 / 3600)

        np.testing.assert_array_equal(idx1, np.array([0, 1, 2]))
        np.testing.assert_array_equal(idx2, np.array([0, 1, 2]))
        np.testing.assert_array_almost_equal(dist, 0.0, decimal=6)

    def test_single_match_within_radius(self):
        """Test single match just within search radius."""
        ra1 = np.array([10.0])
        dec1 = np.array([0.0])
        ra2 = np.array([10.0001])  # ~0.36 arcsec separation
        dec2 = np.array([0.0])

        idx1, idx2, dist = spherical_match(ra1, dec1, ra2, dec2, search_radius_deg=1 / 3600)  # 1 arcsec

        self.assertEqual(len(idx1), 1)
        self.assertEqual(idx1[0], 0)
        self.assertEqual(idx2[0], 0)
        self.assertTrue(dist[0] < 1 / 3600)

    def test_no_match_outside_radius(self):
        """Test when no match should be found due to distance."""
        ra1 = np.array([10.0])
        dec1 = np.array([0.0])
        ra2 = np.array([10.01])  # ~36 arcsec separation
        dec2 = np.array([0.0])

        idx1, idx2, dist = spherical_match(ra1, dec1, ra2, dec2, search_radius_deg=1 / 3600)  # 1 arcsec

        self.assertEqual(len(idx1), 0)
        self.assertEqual(len(idx2), 0)
        self.assertEqual(len(dist), 0)

    def test_multiple_matches(self):
        """Test matching multiple objects within radius."""
        ra1 = np.array([10.0, 20.0])
        dec1 = np.array([0.0, 0.0])
        ra2 = np.array([10.0, 20.0, 10.0002])
        dec2 = np.array([0.0, 0.0, 0.0])

        idx1, idx2, dist = spherical_match(ra1, dec1, ra2, dec2, search_radius_deg=1 / 3600)  # 1 arcsec

        self.assertGreaterEqual(len(idx1), 2)
        for d in dist:
            self.assertTrue(d <= 1 / 3600)

    def test_vectorized_inputs(self):
        """Ensure function supports array inputs of different sizes."""
        ra1 = np.linspace(0, 1, 10)
        dec1 = np.zeros(10)
        ra2 = np.linspace(0, 1, 100)
        dec2 = np.zeros(100)

        idx1, idx2, dist = spherical_match(ra1, dec1, ra2, dec2, search_radius_deg=5 / 3600)  # 5 arcsec

        self.assertTrue(len(idx1) > 0)
        self.assertTrue(np.all(dist >= 0.0))


class TestGetFrameCenter(unittest.TestCase):
    def setUp(self):
        self.header = fits.Header()
        self.header['NAXIS'] = 2
        self.header['NAXIS1'] = 200
        self.header['NAXIS2'] = 100
        self.header['CTYPE1'] = 'RA---TAN'
        self.header['CTYPE2'] = 'DEC--TAN'
        self.header['CRPIX1'] = 100.0
        self.header['CRPIX2'] = 50.0
        self.header['CRVAL1'] = 180.0
        self.header['CRVAL2'] = -30.0
        self.header['CD1_1'] = -0.00027
        self.header['CD1_2'] = 0.0
        self.header['CD2_1'] = 0.0
        self.header['CD2_2'] = 0.00027

        self.wcs = WCS(self.header)

    def test_center_from_wcs(self):
        """Test center calculation directly from WCS."""
        ra, dec, radius = get_frame_center(wcs=self.wcs, image_width=200, image_height=100)
        self.assertAlmostEqual(ra, 180.0, places=3)
        self.assertAlmostEqual(dec, -30.0, places=3)
        self.assertTrue(radius > 0)

    def test_center_from_header(self):
        """Test center calculation from FITS header."""
        ra, dec, radius = get_frame_center(header=self.header)
        self.assertAlmostEqual(ra, 180.0, places=3)
        self.assertAlmostEqual(dec, -30.0, places=3)
        self.assertTrue(radius > 0)

    def test_center_from_file(self):
        """Test center calculation from FITS file."""
        with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as tmpfile:
            hdu = fits.PrimaryHDU(data=np.zeros((100, 200)), header=self.header)
            hdu.writeto(tmpfile.name, overwrite=True)
            ra, dec, radius = get_frame_center(filename=tmpfile.name)
            self.assertAlmostEqual(ra, 180.0, places=3)
            self.assertAlmostEqual(dec, -30.0, places=3)
            self.assertTrue(radius > 0)
            os.remove(tmpfile.name)

    def test_center_with_shape_fallback(self):
        """Test fallback shape-based width/height when header is incomplete."""
        header = self.header.copy()
        del header['NAXIS1']
        del header['NAXIS2']
        ra, dec, radius = get_frame_center(wcs=self.wcs, image_shape=(100, 200))
        self.assertAlmostEqual(ra, 180.0, places=3)
        self.assertAlmostEqual(dec, -30.0, places=3)
        self.assertTrue(radius > 0)

    def test_no_wcs_available(self):
        """Test return when no WCS can be constructed."""
        ra, dec, radius = get_frame_center(header=fits.Header())
        self.assertIsNone(ra)
        self.assertIsNone(dec)
        self.assertIsNone(radius)


class TestMakeKernel(unittest.TestCase):
    def test_kernel_shape_default(self):
        """Test default kernel shape with core_radius=1.0, extent=1.0."""
        kernel = make_kernel()
        expected_size = int(np.ceil(1.0 * 1.0 + 1) - np.floor(-1.0 * 1.0))
        self.assertEqual(kernel.shape, (expected_size, expected_size))

    def test_kernel_shape_custom(self):
        """Test kernel shape for various input values."""
        core_radius = 2.0
        extent = 1.5
        kernel = make_kernel(core_radius, extent)
        expected_size = int(np.ceil(extent * core_radius + 1) - np.floor(-extent * core_radius))
        self.assertEqual(kernel.shape, (expected_size, expected_size))

    def test_kernel_peak_at_center(self):
        """Test that the maximum value is at the center of the kernel."""
        kernel = make_kernel(core_radius=1.0, extent_factor=3.0)
        center = tuple(s // 2 for s in kernel.shape)
        self.assertEqual(np.argmax(kernel), np.ravel_multi_index(center, kernel.shape))

    def test_kernel_symmetry(self):
        """Test that the kernel is symmetric along both axes."""
        kernel = make_kernel()
        self.assertTrue(np.allclose(kernel, kernel[::-1, :]))  # vertical symmetry
        self.assertTrue(np.allclose(kernel, kernel[:, ::-1]))  # horizontal symmetry

    def test_kernel_positive_values(self):
        """Test that all kernel values are positive and finite."""
        kernel = make_kernel()
        self.assertTrue(np.all(kernel >= 0))
        self.assertTrue(np.all(np.isfinite(kernel)))


class TestEvaluateDataQualityResults(unittest.TestCase):
    def test_dq_results_typical_case(self):
        """Test data quality evaluation with a typical mock catalog."""
        catalog = Table({
            'fwhm': [2.0, 2.2, 2.1, 1.9, 2.0],
            'a': [3.0, 3.1, 2.9, 3.0, 3.2],
            'b': [2.5, 2.4, 2.6, 2.3, 2.7]
        })

        fwhm, fwhm_error, ellipticity, ellipticity_error = evaluate_data_quality_results(catalog)

        expected_fwhm = np.median(catalog['fwhm'])
        expected_fwhm_error = np.median(np.abs(catalog['fwhm'] - expected_fwhm))

        expected_a = np.median(catalog['a'])
        expected_b = np.median(catalog['b'])
        expected_ellipticity = 1.0 - (expected_b / expected_a)

        expected_a_err = np.median(np.abs(catalog['a'] - expected_a))
        expected_b_err = np.median(np.abs(catalog['b'] - expected_b))
        expected_ell_error = expected_ellipticity * np.sqrt(
            (expected_a_err / expected_a) ** 2 + (expected_b_err / expected_b) ** 2
        )

        self.assertAlmostEqual(fwhm, expected_fwhm, places=6)
        self.assertAlmostEqual(fwhm_error, expected_fwhm_error, places=6)
        self.assertAlmostEqual(ellipticity, expected_ellipticity, places=6)
        self.assertAlmostEqual(ellipticity_error, expected_ell_error, places=6)

    def test_dq_results_all_equal(self):
        """Test case where all values are identical (no dispersion)."""
        catalog = Table({
            'fwhm': [2.0] * 5,
            'a': [3.0] * 5,
            'b': [2.0] * 5
        })

        fwhm, fwhm_error, ellipticity, ellipticity_error = evaluate_data_quality_results(catalog)

        self.assertEqual(fwhm, 2.0)
        self.assertEqual(fwhm_error, 0.0)
        self.assertAlmostEqual(ellipticity, 1.0 - 2.0 / 3.0, places=6)
        self.assertEqual(ellipticity_error, 0.0)

    def test_dq_results_small_catalog(self):
        """Test function with a small catalog (2 entries)."""
        catalog = Table({
            'fwhm': [1.5, 2.5],
            'a': [2.0, 4.0],
            'b': [1.0, 2.0]
        })

        fwhm, fwhm_error, ellipticity, ellipticity_error = evaluate_data_quality_results(catalog)
        self.assertGreater(fwhm, 0)
        self.assertGreaterEqual(fwhm_error, 0)
        self.assertGreater(ellipticity, 0)
        self.assertGreaterEqual(ellipticity_error, 0)


class TestFileWrite(unittest.TestCase):

    def test_write_mode(self):
        """Test writing content to file (overwrite mode)."""
        with tempfile.NamedTemporaryFile(mode='r+', delete=False) as tmp:
            file_write(tmp.name, contents="Hello world!", append=False)

            tmp.seek(0)
            content = tmp.read()
            self.assertEqual(content, "Hello world!")

        os.remove(tmp.name)

    def test_append_mode(self):
        """Test appending content to an existing file."""
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmp:
            tmp.write("First line.\n")
            tmp.flush()

            file_write(tmp.name, contents="Second line.\n", append=True)

            tmp.seek(0)
            content = tmp.read()
            self.assertEqual(content, "First line.\nSecond line.\n")

        os.remove(tmp.name)

    def test_no_content(self):
        """Test that file is created but empty when no contents are passed."""
        with tempfile.NamedTemporaryFile(mode='r+', delete=False) as tmp:
            file_write(tmp.name, contents=None, append=False)

            tmp.seek(0)
            content = tmp.read()
            self.assertEqual(content, "")

        os.remove(tmp.name)


class TestTableGetColumn(unittest.TestCase):

    def setUp(self):
        self.table = Table({
            'fwhm': [1.2, 1.5, 1.4],
            'flux': [100, 120, 110]
        })

    def test_column_exists(self):
        """Return actual column if it exists."""
        result = table_get_column(self.table, 'fwhm')
        np.testing.assert_array_equal(result, self.table['fwhm'])

    def test_column_missing_scalar_default(self):
        """Return scalar default broadcasted if column is missing."""
        result = table_get_column(self.table, 'ellipticity', default=42)
        expected = np.full(len(self.table), 42, dtype=int)
        np.testing.assert_array_equal(result, expected)

    def test_column_missing_array_default(self):
        """Return array default directly if column is missing."""
        default_array = [0.1, 0.2, 0.3]
        result = table_get_column(self.table, 'ellipticity', default=default_array)
        np.testing.assert_array_equal(result, default_array)

    def test_column_missing_none_default(self):
        """Return None if column is missing and default is None."""
        result = table_get_column(self.table, 'ellipticity', default=None)
        self.assertIsNone(result)


class TestGetObservationTime(unittest.TestCase):

    def test_valid_time_string(self):
        """Parse a valid time string."""
        time = get_observation_time(time_string="2023-04-10T22:45:00")
        self.assertIsInstance(time, Time)
        self.assertEqual(time.iso, "2023-04-10 22:45:00.000")

    def test_invalid_time_string(self):
        """Return None for invalid time string."""
        time = get_observation_time(time_string="invalid-time")
        self.assertIsNone(time)

    def test_date_obs_in_header(self):
        """Parse DATE-OBS from FITS header."""
        header = fits.Header()
        header['DATE-OBS'] = '2023-04-10T22:45:00'
        time = get_observation_time(header=header)
        self.assertIsInstance(time, Time)
        self.assertEqual(time.iso, "2023-04-10 22:45:00.000")

    def test_combined_date_time_obs(self):
        """Parse combined DATE and TIME-OBS."""
        header = fits.Header()
        header['DATE'] = '2023-04-10'
        header['TIME-OBS'] = '22:45:00'
        time = get_observation_time(header=header)
        self.assertIsInstance(time, Time)
        self.assertEqual(time.iso, "2023-04-10 22:45:00.000")

    def test_mjd_from_header(self):
        """Parse MJD float from header."""
        header = fits.Header()
        header['MJD'] = 60000.0
        time = get_observation_time(header=header)
        self.assertIsInstance(time, Time)
        self.assertAlmostEqual(time.mjd, 60000.0, places=6)

    def test_jd_from_header(self):
        """Parse JD float from header."""
        header = fits.Header()
        header['JD'] = 2459360.5
        time = get_observation_time(header=header)
        self.assertIsInstance(time, Time)
        self.assertAlmostEqual(time.jd, 2459360.5, places=6)

    def test_unix_time_from_header(self):
        """Parse Unix time from header."""
        header = fits.Header()
        header['DATE-OBS'] = 1609459200.0  # Jan 1, 2021 in Unix time
        time = get_observation_time(header=header)
        self.assertIsInstance(time, Time)
        self.assertEqual(time.datetime.year, 2021)

    @patch("astropy.io.fits.getheader")
    def test_load_from_file(self, mock_getheader):
        """Test loading header from a FITS file."""
        header = fits.Header()
        header['DATE-OBS'] = '2023-04-10T22:45:00'
        mock_getheader.return_value = header

        time = get_observation_time(filename="mockfile.fits")
        self.assertIsInstance(time, Time)
        self.assertEqual(time.iso, "2023-04-10 22:45:00.000")
        mock_getheader.assert_called_once_with("mockfile.fits")

    def test_no_header_or_filename(self):
        """Return None if neither header nor filename is provided."""
        result = get_observation_time()
        self.assertIsNone(result)


class TestFormatAstromaticOpts(unittest.TestCase):

    def test_boolean_values(self):
        options = {'CHECKIMAGE_TYPE': True, 'VERBOSE_TYPE': False}
        result = format_astromatic_opts(options)
        self.assertIn('-CHECKIMAGE_TYPE Y', result)
        self.assertIn('-VERBOSE_TYPE N', result)

    def test_string_quoting(self):
        options = {'CATALOG_NAME': 'my catalog.cat'}
        result = format_astromatic_opts(options)
        expected = f"-CATALOG_NAME {shlex.quote('my catalog.cat')}"
        self.assertIn(expected, result)

    def test_numeric_value(self):
        options = {'DETECT_MINAREA': 5}
        result = format_astromatic_opts(options)
        self.assertIn('-DETECT_MINAREA 5', result)

    def test_array_value(self):
        options = {'FILTER_NAME': ['gauss_2.0_3x3.conv', 'gauss_3.0_5x5.conv']}
        result = format_astromatic_opts(options)
        expected = '-FILTER_NAME gauss_2.0_3x3.conv,gauss_3.0_5x5.conv'
        self.assertIn(expected, result)

    def test_none_is_skipped(self):
        options = {'CHECKIMAGE_NAME': None, 'DETECT_THRESH': 2.5}
        result = format_astromatic_opts(options)
        self.assertIn('-DETECT_THRESH 2.5', result)
        self.assertNotIn('CHECKIMAGE_NAME', result)

    def test_mixed_arguments(self):
        options = {
            'CATALOG_NAME': 'output.cat',
            'CHECKIMAGE_TYPE': True,
            'DETECT_THRESH': 1.5,
            'FILTER_NAME': ['gauss.conv', 'custom.conv'],
            'SKY_TYPE': None
        }
        result = format_astromatic_opts(options)
        self.assertIn('-CATALOG_NAME', result)
        self.assertIn('-CHECKIMAGE_TYPE Y', result)
        self.assertIn('-DETECT_THRESH 1.5', result)
        self.assertIn('-FILTER_NAME gauss.conv,custom.conv', result)
        self.assertNotIn('-SKY_TYPE', result)


class TestCheckWCS(unittest.TestCase):

    def test_valid_wcs(self):
        """Test that a valid celestial WCS is returned."""
        header = fits.Header()
        header["CTYPE1"] = "RA---TAN"
        header["CTYPE2"] = "DEC--TAN"
        header["CRPIX1"] = 100.0
        header["CRPIX2"] = 100.0
        header["CRVAL1"] = 180.0
        header["CRVAL2"] = 45.0
        header["CD1_1"] = -0.00027778
        header["CD1_2"] = 0.0
        header["CD2_1"] = 0.0
        header["CD2_2"] = 0.00027778

        result = check_wcs(header)
        self.assertIsInstance(result, WCS)
        self.assertTrue(result.is_celestial)

    def test_missing_wcs(self):
        """Test that a header with no WCS raises ValueError."""
        header = fits.Header()
        with self.assertRaises(ValueError) as context:
            check_wcs(header)
        self.assertIn("WCS is absent or non-celestial", str(context.exception))

    def test_non_celestial_wcs(self):
        """Test that a non-celestial WCS raises ValueError."""
        header = fits.Header()
        header["CTYPE1"] = "LINEAR"
        header["CTYPE2"] = "LINEAR"
        header["CRPIX1"] = 100.0
        header["CRPIX2"] = 100.0
        header["CRVAL1"] = 0.0
        header["CRVAL2"] = 0.0

        with self.assertRaises(ValueError) as context:
            check_wcs(header)
        self.assertIn("WCS is absent or non-celestial", str(context.exception))


class TestCheckPhotometryResults(unittest.TestCase):

    def test_valid_photometry_results(self):
        """Should return the dictionary if not None."""
        mock_results = {'zero_point': 25.1, 'extinction': 0.15}
        result = check_photometry_results(mock_results)
        self.assertEqual(result, mock_results)

    def test_none_photometry_results(self):
        """Should raise ValueError if results are None."""
        with self.assertRaises(ValueError) as context:
            check_photometry_results(None)

        self.assertIn("Photometric calibration results are missing", str(context.exception))


class TestGetFilterSet(unittest.TestCase):

    def test_u_sdss(self):
        catalog_filter, photometry_filter = get_filter_set("u-SDSS")
        self.assertEqual(catalog_filter, "BPmag")
        self.assertEqual(photometry_filter, "u_SDSS")

    def test_g_sdss(self):
        catalog_filter, photometry_filter = get_filter_set("g-SDSS")
        self.assertEqual(catalog_filter, "BPmag")
        self.assertEqual(photometry_filter, "g_SDSS")

    def test_r_sdss(self):
        catalog_filter, photometry_filter = get_filter_set("r-SDSS")
        self.assertEqual(catalog_filter, "Gmag")
        self.assertEqual(photometry_filter, "r_SDSS")

    def test_i_sdss(self):
        catalog_filter, photometry_filter = get_filter_set("i-SDSS")
        self.assertEqual(catalog_filter, "Gmag")
        self.assertEqual(photometry_filter, "i_SDSS")

    def test_z_sdss(self):
        catalog_filter, photometry_filter = get_filter_set("z-SDSS")
        self.assertEqual(catalog_filter, "Gmag")
        self.assertEqual(photometry_filter, "i_SDSS")

    def test_unknown_filter(self):
        catalog_filter, photometry_filter = get_filter_set("Ha-NB")
        self.assertEqual(catalog_filter, "Gmag")
        self.assertEqual(photometry_filter, "g_SDSS")


class TestPlotImage(unittest.TestCase):
    def setUp(self):
        # Create a simple synthetic image and WCS
        self.image = np.random.random((100, 100))
        self.wcs = WCS(naxis=2)
        self.wcs.wcs.crpix = [50, 50]
        self.wcs.wcs.cdelt = np.array([-0.000277, 0.000277])
        self.wcs.wcs.crval = [180, 0]
        self.wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    def test_plot_with_wcs(self):
        """Test image plot with WCS projection and saving to file."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            output_path = tmpfile.name

        plot_image(self.image, wcs=self.wcs, title="Test WCS Plot", output_file=output_path)
        self.assertTrue(os.path.exists(output_path))
        os.remove(output_path)

    def test_plot_without_wcs(self):
        """Test image plot without WCS and saving to file."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            output_path = tmpfile.name

        plot_image(self.image, wcs=None, title="Test No WCS", output_file=output_path)
        self.assertTrue(os.path.exists(output_path))
        os.remove(output_path)

    def test_plot_with_points(self):
        """Test plot with overlay points."""
        x_points = [10, 20, 30]
        y_points = [40, 50, 60]

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            output_path = tmpfile.name

        plot_image(self.image, x_points=x_points, y_points=y_points, output_file=output_path)
        self.assertTrue(os.path.exists(output_path))
        os.remove(output_path)

    def test_invalid_quantiles(self):
        """Test plot with invalid quantiles raises ValueError."""
        with self.assertRaises(ValueError):
            plot_image(self.image, quantiles=(-0.1, 1.1))


class TestAddColorbar(unittest.TestCase):
    def test_add_colorbar_to_image(self):
        """Test that a colorbar is correctly added to a plot."""
        image = np.random.normal(loc=100, scale=10, size=(100, 100))

        fig, ax = plt.subplots()
        norm = simple_norm(image, 'linear', vmin=np.percentile(image, 1), vmax=np.percentile(image, 99))
        img = ax.imshow(image, origin='lower', cmap='viridis', norm=norm)

        # Call the function under test
        colorbar_obj = add_colorbar(mappable=img, ax=ax)

        self.assertIsNotNone(colorbar_obj)
        self.assertEqual(colorbar_obj.mappable, img)

        plt.close(fig)


class TestBinnedMap(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.x = np.random.uniform(0, 100, 500)
        self.y = np.random.uniform(0, 100, 500)
        self.values = np.random.normal(loc=10.0, scale=2.0, size=500)

    def test_basic_plot(self):
        """Test basic plotting without errors."""
        fig, ax = plt.subplots()
        try:
            binned_map(self.x, self.y, self.values, ax=ax)
        except Exception as e:
            self.fail(f"binned_map raised an exception: {e}")

    def test_hide_axes(self):
        """Test binned map with axes hidden."""
        fig, ax = plt.subplots()
        binned_map(self.x, self.y, self.values, ax=ax, show_axes=False)
        # We can't assert visibility directly, but we check it runs without error

    def test_show_points_overlay(self):
        """Test binned map with point overlay."""
        fig, ax = plt.subplots()
        try:
            binned_map(self.x, self.y, self.values, ax=ax, show_points=True)
        except Exception as e:
            self.fail(f"Point overlay raised exception: {e}")

    def test_custom_statistic(self):
        """Test binned map with custom statistic."""
        fig, ax = plt.subplots()
        try:
            binned_map(self.x, self.y, self.values, ax=ax, statistic='median')
        except Exception as e:
            self.fail(f"Custom statistic raised exception: {e}")

    def test_outlier_quantile_handling(self):
        """Test binned map with extreme quantiles."""
        fig, ax = plt.subplots()
        try:
            binned_map(self.x, self.y, self.values, ax=ax, quantiles=(1, 99))
        except Exception as e:
            self.fail(f"Quantile edge case raised exception: {e}")


class TestPlotPhotometricMatch(unittest.TestCase):

    def setUp(self):
        # Simulated data dictionary (m)
        size = 100
        self.m = {
            'cmag': np.random.uniform(15, 20, size),
            'zero_model': np.random.normal(0, 0.05, size),
            'zero': np.random.normal(0, 0.05, size),
            'zero_err': np.random.uniform(0.01, 0.1, size),
            'idx': np.random.choice([True, False], size=size, p=[0.7, 0.3]),
            'idx0': np.ones(size, dtype=bool),
            'color': np.random.uniform(-0.5, 1.5, size),
            'cat_col_mag': 'g',
            'cat_col_mag1': 'g',
            'cat_col_mag2': 'r',
            'color_term': 0.2,
            'ox': np.random.uniform(0, 1000, size),
            'oy': np.random.uniform(0, 1000, size),
            'dist': np.random.uniform(0, 0.001, size),
        }

    def test_plot_mag_mode(self):
        fig, ax = plt.subplots()
        plot_photometric_match(self.m, ax=ax, mode='mag')
        plt.close(fig)

    def test_plot_normed_mode(self):
        fig, ax = plt.subplots()
        plot_photometric_match(self.m, ax=ax, mode='normed')
        plt.close(fig)

    def test_plot_color_mode(self):
        fig, ax = plt.subplots()
        plot_photometric_match(self.m, ax=ax, mode='color')
        plt.close(fig)

    def test_plot_zero_mode(self):
        fig, ax = plt.subplots()
        plot_photometric_match(self.m, ax=ax, mode='zero')
        plt.close(fig)

    def test_plot_model_mode(self):
        fig, ax = plt.subplots()
        plot_photometric_match(self.m, ax=ax, mode='model')
        plt.close(fig)

    def test_plot_residuals_mode(self):
        fig, ax = plt.subplots()
        plot_photometric_match(self.m, ax=ax, mode='residuals')
        plt.close(fig)

    def test_plot_dist_mode(self):
        fig, ax = plt.subplots()
        plot_photometric_match(self.m, ax=ax, mode='dist')
        plt.close(fig)

    def test_plot_with_custom_cmag_limits(self):
        fig, ax = plt.subplots()
        plot_photometric_match(self.m, ax=ax, mode='mag', cmag_limits=(16, 19))
        plt.close(fig)

    def test_plot_without_final_fit(self):
        fig, ax = plt.subplots()
        plot_photometric_match(self.m, ax=ax, mode='mag', show_final=False)
        plt.close(fig)

    def test_plot_without_masked_points(self):
        fig, ax = plt.subplots()
        plot_photometric_match(self.m, ax=ax, mode='mag', show_masked=False)
        plt.close(fig)


class TestPlotPhotcal(unittest.TestCase):
    def setUp(self):
        self.image = np.random.normal(loc=1000, scale=50, size=(100, 100))
        self.phot_table = Table({
            'x': [20, 50, 80],
            'y': [20, 50, 80],
            'a': [3, 2.5, 4],
            'b': [2, 2.0, 3],
            'theta': [0, 30, 60],
            'mag_calib': [18.5, 17.2, 19.1]
        })

        self.wcs = WCS(naxis=2)
        self.wcs.wcs.crpix = [50, 50]
        self.wcs.wcs.cdelt = np.array([-0.000277, 0.000277])
        self.wcs.wcs.crval = [150, 2.0]
        self.wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    def test_plot_with_wcs(self):
        # Should plot without errors
        plot_photcal(
            image=self.image,
            phot_table=self.phot_table,
            wcs=self.wcs,
            column_scale='mag_calib',
            quantiles=(0.01, 0.99)
        )
        plt.close('all')

    def test_plot_without_wcs(self):
        plot_photcal(
            image=self.image,
            phot_table=self.phot_table,
            wcs=None,
            column_scale='mag_calib'
        )
        plt.close('all')

    def test_plot_saves_output(self):
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            output_path = tmp.name
        try:
            plot_photcal(
                image=self.image,
                phot_table=self.phot_table,
                wcs=self.wcs,
                column_scale='mag_calib',
                output_file=output_path,
                dpi=150
            )
            self.assertTrue(os.path.exists(output_path))
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)
        plt.close('all')


class TestClearWCS(unittest.TestCase):
    def setUp(self):
        self.header = fits.Header()
        self.header['CRPIX1'] = 1024
        self.header['CRPIX2'] = 1024
        self.header['CTYPE1'] = 'RA---TAN'
        self.header['CTYPE2'] = 'DEC--TAN'
        self.header['CD1_1'] = 0.0001
        self.header['CD1_2'] = 0.0
        self.header['CD2_1'] = 0.0
        self.header['CD2_2'] = 0.0001
        self.header['COMMENT'] = "This is a comment"
        self.header['HISTORY'] = "Some history"
        self.header['_ASTROM'] = 'Astrometry.Net'
        self.header['MAGZEROP'] = 25.0
        self.header['A_1_1'] = 0.0
        self.header['PV1_1'] = 0.0

    def test_clear_basic_wcs_keywords(self):
        cleared = clear_wcs(self.header, copy=True)
        self.assertNotIn('CRPIX1', cleared)
        self.assertNotIn('CRPIX2', cleared)
        self.assertNotIn('CTYPE1', cleared)
        self.assertNotIn('CTYPE2', cleared)

    def test_clear_scamp_and_sip_keywords(self):
        cleared = clear_wcs(self.header, copy=True)
        self.assertNotIn('MAGZEROP', cleared)
        self.assertNotIn('A_1_1', cleared)
        self.assertNotIn('PV1_1', cleared)

    def test_keep_comments_and_history_by_default(self):
        cleared = clear_wcs(self.header, copy=True)
        self.assertIn('COMMENT', cleared)
        self.assertIn('HISTORY', cleared)

    def test_remove_comments_and_history(self):
        cleared = clear_wcs(self.header, remove_comments=True, remove_history=True, copy=True)
        self.assertNotIn('COMMENT', cleared)
        self.assertNotIn('HISTORY', cleared)

    def test_remove_underscored_keys(self):
        cleared = clear_wcs(self.header, remove_underscored=True, copy=True)
        self.assertNotIn('_ASTROM', cleared)


class TestMakeSeries:
    def test_default_behavior(self):
        x = 2.0
        y = 3.0
        order = 2
        result = make_series(multiplier=1.0, x=x, y=y, order=order, sum=False, zero=True)
        expected_terms = 1 + sum(i + 1 for i in range(1, order + 1))  # 1 constant + poly terms
        assert len(result) == expected_terms
        np.testing.assert_array_equal(result[0], np.ones_like(np.atleast_1d(x)))  # constant term

    def test_zero_false(self):
        x = 2.0
        y = 3.0
        order = 2
        result = make_series(multiplier=1.0, x=x, y=y, order=order, sum=False, zero=False)
        expected_terms = sum(i + 1 for i in range(1, order + 1))
        assert len(result) == expected_terms
        assert not np.all(result[0] == 1.0)

    def test_sum_output(self):
        x = np.array([1.0, 2.0])
        y = np.array([2.0, 3.0])
        result = make_series(multiplier=2.0, x=x, y=y, order=2, sum=True, zero=True)
        assert isinstance(result, np.ndarray)
        assert result.shape == x.shape

    def test_order_zero(self):
        x = np.array([1.0, 2.0])
        y = np.array([2.0, 3.0])
        result = make_series(multiplier=2.0, x=x, y=y, order=0, sum=False, zero=True)
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], np.ones_like(x) * 2.0)

    def test_custom_multiplier(self):
        x = np.array([1.0, 2.0])
        y = np.array([0.0, 1.0])
        result = make_series(multiplier=3.0, x=x, y=y, order=1, sum=False, zero=True)
        expected = [np.ones_like(x) * 3.0, 3.0 * x, 3.0 * y]
        for r, e in zip(result, expected):
            np.testing.assert_array_almost_equal(r, e)


class TestConvertMatchResultsToTable(unittest.TestCase):
    def setUp(self):
        self.match_result = {
            'oidx': np.array([0, 1, 2]),
            'cidx': np.array([0, 1, 2]),
            'dist': np.array([0.1, 0.2, 0.3]),
            'omag': np.array([20.1, 19.8, 20.5]),
            'omagerr': np.array([0.05, 0.05, 0.07]),
            'cmag': np.array([18.2, 18.5, 18.9]),
            'cmagerr': np.array([0.03, 0.04, 0.03]),
            'color': np.array([0.3, 0.5, 0.2]),
            'ox': np.array([10, 20, 30]),
            'oy': np.array([15, 25, 35]),
            'oflags': np.array([0, 1, 0]),
            'zero': np.array([1.8, 1.3, 1.6]),
            'zero_err': np.array([0.1, 0.1, 0.1]),
            'zero_model': np.array([1.9, 1.4, 1.7]),
            'zero_model_err': np.array([0.1, 0.1, 0.1]),
            'color_term': 0.05,
            'obj_zero': np.array([1.8, 1.3, 1.6]),
            'idx': np.array([True, True, False]),
            'idx0': np.array([True, True, True]),
            'fwhm': np.array([3.0, 2.8, 3.2]),
            'a': np.array([2.5, 2.3, 2.6]),
            'b': np.array([2.0, 2.1, 2.2]),
        }

    def test_basic_convert_match_results_to_table(self):
        table = convert_match_results_to_table(self.match_result)
        self.assertIsInstance(table, Table)
        self.assertTrue('omag' in table.colnames)
        self.assertEqual(len(table), 3)

    def test_with_pixscale(self):
        table = convert_match_results_to_table(self.match_result, pixscale=0.3)
        self.assertIn('fwhm_arcsec', table.colnames)
        self.assertIn('ell', table.colnames)
        self.assertAlmostEqual(table['fwhm_arcsec'][0], self.match_result['fwhm'][0] * 0.3, places=5)

    def test_select_columns(self):
        selected_cols = ['oidx', 'cmag', 'zero']
        table = convert_match_results_to_table(self.match_result, columns=selected_cols)
        self.assertEqual(table.colnames, selected_cols)


class TestGetPhotometricZeroPoint(unittest.TestCase):

    def setUp(self):
        # Define synthetic photometric match data
        self.match_data = {
            'zero': np.array([24.1, 24.2, 24.3, 24.4]),
            'zero_err': np.array([0.05, 0.06, 0.04, 0.07]),
            'zero_model': np.array([24.15, 24.25, 24.35, 24.45]),
            'zero_model_err': np.array([0.06, 0.05, 0.06, 0.07])
        }

    def test_default_zero_point(self):
        zp, zp_err = get_photometric_zeropoint(self.match_data)
        self.assertAlmostEqual(zp, np.median(self.match_data['zero']))
        self.assertAlmostEqual(zp_err, np.median(self.match_data['zero_err']))

    def test_model_zero_point(self):
        zp, zp_err = get_photometric_zeropoint(self.match_data, use_model=True)
        self.assertAlmostEqual(zp, np.median(self.match_data['zero_model']))
        self.assertAlmostEqual(zp_err, np.median(self.match_data['zero_model_err']))

    def test_nan_handling(self):
        # Add NaNs
        self.match_data['zero'][1] = np.nan
        self.match_data['zero_err'][2] = np.nan
        zp, zp_err = get_photometric_zeropoint(self.match_data)
        self.assertTrue(np.isfinite(zp))
        self.assertTrue(np.isfinite(zp_err))

    def test_missing_keys(self):
        bad_data = {
            'zero': np.array([24.1, 24.2]),
            'zero_err': np.array([0.1, 0.2])
        }
        # Missing model fields: should still work with use_model=False
        zp, zp_err = get_photometric_zeropoint(bad_data)
        self.assertEqual(zp, np.median(bad_data['zero']))
        self.assertEqual(zp_err, np.median(bad_data['zero_err']))

        # If use_model=True but model fields missing, should raise KeyError
        with self.assertRaises(KeyError):
            get_photometric_zeropoint(bad_data, use_model=True)


class TestWcsSip2Pv(unittest.TestCase):
    """
    Test suite for the wcs_sip2pv function.
    """

    def test_header_with_pc_matrix(self):
        """
        Test the function with a header containing a PC matrix but no CD matrix.
        """
        # Create a header with PC matrix and CDELT values
        header = fits.Header({
            'PC1_1': 1.0,
            'PC2_1': 0.0,
            'PC1_2': 0.0,
            'PC2_2': 1.0,
            'CDELT1': 0.1,
            'CDELT2': 0.1,
            'A_ORDER': 2,
            'B_ORDER': 2,
            'A_0_2': 0.01,
            'A_2_0': 0.01,
            'B_0_2': 0.01,
            'B_2_0': 0.01
        })

        # Call the function
        result = wcs_sip2pv(header)

        # Verify that the PC matrix is converted to CD matrix
        self.assertIn('CD1_1', result)
        self.assertIn('CD2_1', result)
        self.assertIn('CD1_2', result)
        self.assertIn('CD2_2', result)
        self.assertNotIn('PC1_1', result)
        self.assertNotIn('PC2_1', result)
        self.assertNotIn('PC1_2', result)
        self.assertNotIn('PC2_2', result)

        # Verify the values of the CD matrix
        self.assertAlmostEqual(result['CD1_1'], 0.1)
        self.assertAlmostEqual(result['CD2_1'], 0.0)
        self.assertAlmostEqual(result['CD1_2'], 0.0)
        self.assertAlmostEqual(result['CD2_2'], 0.1)

    def test_header_with_cd_matrix(self):
        """
        Test the function with a header already containing a CD matrix.
        """
        # Create a header with CD matrix
        header = fits.Header({
            'CD1_1': 0.1,
            'CD2_1': 0.0,
            'CD1_2': 0.0,
            'CD2_2': 0.1,
            'A_ORDER': 2,
            'B_ORDER': 2,
            'A_0_2': 0.01,
            'A_2_0': 0.01,
            'B_0_2': 0.01,
            'B_2_0': 0.01
        })

        # Call the function
        result = wcs_sip2pv(header)

        # Verify that the CD matrix remains unchanged
        self.assertIn('CD1_1', result)
        self.assertIn('CD2_1', result)
        self.assertIn('CD1_2', result)
        self.assertIn('CD2_2', result)
        self.assertAlmostEqual(result['CD1_1'], 0.1)
        self.assertAlmostEqual(result['CD2_1'], 0.0)
        self.assertAlmostEqual(result['CD1_2'], 0.0)
        self.assertAlmostEqual(result['CD2_2'], 0.1)

    def test_header_without_sip_keywords(self):
        """
        Test the function with a header missing SIP keywords.
        """
        # Create a header without SIP keywords
        header = fits.Header({
            'CD1_1': 0.1,
            'CD2_1': 0.0,
            'CD1_2': 0.0,
            'CD2_2': 0.1
        })

        # Call the function and expect a warning
        with warnings.catch_warnings(record=True) as w:
            result = wcs_sip2pv(header)
            self.assertEqual(len(w), 1)
            self.assertIn("SIP keywords are missing", str(w[0].message))

        # Verify that the CD matrix remains unchanged
        self.assertIn('CD1_1', result)
        self.assertIn('CD2_1', result)
        self.assertIn('CD1_2', result)
        self.assertIn('CD2_2', result)
        self.assertAlmostEqual(result['CD1_1'], 0.1)
        self.assertAlmostEqual(result['CD2_1'], 0.0)
        self.assertAlmostEqual(result['CD1_2'], 0.0)
        self.assertAlmostEqual(result['CD2_2'], 0.1)


class TestGetIntrinsicScatter(unittest.TestCase):
    """
    Test suite for the `get_intrinsic_scatter` function.
    """

    def test_basic_case(self):
        """
        Test the function with a basic case where intrinsic scatter is expected.
        """
        observed_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        observed_errors = np.array([0.1, 0.2, 0.15, 0.3, 0.25])
        intrinsic_scatter = get_intrinsic_scatter(observed_values, observed_errors)
        self.assertIsInstance(intrinsic_scatter, float)
        self.assertGreaterEqual(intrinsic_scatter, 0)

    def test_zero_scatter_case(self):
        """
        Test the function with data that has no intrinsic scatter.
        """
        observed_values = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        observed_errors = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        intrinsic_scatter = get_intrinsic_scatter(observed_values, observed_errors)
        self.assertAlmostEqual(intrinsic_scatter, 0, delta=1e-4)  # Relaxed tolerance

    def test_with_bounds(self):
        """
        Test the function with bounds on the intrinsic scatter.
        """
        observed_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        observed_errors = np.array([0.1, 0.2, 0.15, 0.3, 0.25])
        intrinsic_scatter = get_intrinsic_scatter(observed_values, observed_errors, min_scatter=0.5, max_scatter=1.0)
        self.assertGreaterEqual(intrinsic_scatter, 0.5)
        self.assertLessEqual(intrinsic_scatter, 1.0)

    def test_invalid_input(self):
        """
        Test the function with invalid input (empty arrays).
        """
        observed_values = np.array([])
        observed_errors = np.array([])
        with self.assertRaises(ValueError):
            get_intrinsic_scatter(observed_values, observed_errors)

    def test_mismatched_input_lengths(self):
        """
        Test the function with mismatched lengths of observed_values and observed_errors.
        """
        observed_values = np.array([1.0, 2.0, 3.0])
        observed_errors = np.array([0.1, 0.2])
        with self.assertRaises(ValueError):
            get_intrinsic_scatter(observed_values, observed_errors)

    def test_negative_errors(self):
        """
        Test the function with negative observed errors.
        """
        observed_values = np.array([1.0, 2.0, 3.0])
        observed_errors = np.array([0.1, -0.2, 0.3])
        with self.assertRaises(ValueError):
            get_intrinsic_scatter(observed_values, observed_errors)


class TestCalibratePhotometry(unittest.TestCase):
    """
    Test suite for the `calibrate_photometry` function.
    """

    def setUp(self):
        """
        Set up synthetic test data for the test cases.
        """
        # Synthetic object table
        self.object_table = Table({
            'ra': [10.0, 10.1, 10.2],  # Right Ascension (degrees)
            'dec': [20.0, 20.1, 20.2],  # Declination (degrees)
            'mag': [18.0, 18.1, 18.3],  # Instrumental magnitude
            'magerr': [0.03, 0.04, 0.05],  # Magnitude error
            'flags': [0, 0, 0],  # Flags
            'x': [100.0, 200.0, 300.0],  # X coordinates (pixels)
            'y': [150.0, 250.0, 350.0],  # Y coordinates (pixels)
            'fwhm': [2.0, 2.1, 2.2]  # FWHM (pixels)
        })

        # Synthetic catalog table
        self.catalog_table = Table({
            'RAJ2000': [10.0, 10.1, 10.2],  # Right Ascension (degrees)
            'DEJ2000': [20.0, 20.1, 20.2],  # Declination (degrees)
            'R': [18.1, 18.0, 18.4],  # Catalog magnitude
            'B': [19.0, 18.9, 19.3],  # First magnitude for color term
            'V': [18.5, 18.4, 18.8]  # Second magnitude for color term
        })

    @patch('goodman_photometry.goodman_astro.match')
    def test_basic_calibration(self, mock_match):
        """
        Test basic photometric calibration with default parameters.
        """
        # Mock the match function to return a dummy result
        def zero_fn(x, y, mag, get_err=False):
            return 0.01 if get_err else 0.1

        mock_match.return_value = {
            'zero_fn': zero_fn,  # Dummy zero-point function
            'color_term': 0.05  # Dummy color term
        }

        # Call the function
        results = calibrate_photometry(
            object_table=self.object_table,
            catalog_table=self.catalog_table,
            pixel_scale=0.1 / 3600,  # 0.1 arcsec per pixel
            verbose=False
        )

        # Verify the results
        self.assertIsInstance(results, dict)
        self.assertIn('zero_fn', results)
        self.assertIn('color_term', results)

    @patch('goodman_photometry.goodman_astro.match')
    def test_calibration_with_color_term(self, mock_match):
        """
        Test photometric calibration with a color term.
        """
        # Mock the match function to return a dummy result
        def zero_fn(x, y, mag, get_err=False):
            return 0.01 if get_err else 0.1

        mock_match.return_value = {
            'zero_fn': zero_fn,  # Dummy zero-point function
            'color_term': 0.05  # Dummy color term
        }

        # Call the function with color term columns
        results = calibrate_photometry(
            object_table=self.object_table,
            catalog_table=self.catalog_table,
            pixel_scale=0.1 / 3600,  # 0.1 arcsec per pixel
            catalog_mag1_column='B',
            catalog_mag2_column='V',
            verbose=False
        )

        # Verify the results
        self.assertIsInstance(results, dict)
        self.assertIn('zero_fn', results)
        self.assertIn('color_term', results)
        self.assertIn('cat_col_mag1', results)
        self.assertIn('cat_col_mag2', results)

    @patch('goodman_photometry.goodman_astro.match')
    def test_calibration_with_magnitude_limits(self, mock_match):
        """
        Test photometric calibration with magnitude limits.
        """
        # Mock the match function to return a dummy result
        def zero_fn(x, y, mag, get_err=False):
            return 0.01 if get_err else 0.1

        mock_match.return_value = {
            'zero_fn': zero_fn,  # Dummy zero-point function
            'color_term': 0.05  # Dummy color term
        }

        # Call the function with magnitude limits
        results = calibrate_photometry(
            object_table=self.object_table,
            catalog_table=self.catalog_table,
            pixel_scale=0.1 / 3600,  # 0.1 arcsec per pixel
            magnitude_limits=[8, 22],  # Magnitude limits
            verbose=False
        )

        # Verify the results
        self.assertIsInstance(results, dict)
        self.assertIn('zero_fn', results)

    @patch('goodman_photometry.goodman_astro.match')
    def test_calibration_with_error_threshold(self, mock_match):
        """
        Test photometric calibration with an error threshold.
        """
        # Mock the match function to return a dummy result
        def zero_fn(x, y, mag, get_err=False):
            return 0.01 if get_err else 0.1

        mock_match.return_value = {
            'zero_fn': zero_fn,  # Dummy zero-point function
            'color_term': 0.05  # Dummy color term
        }

        # Call the function with an error threshold
        results = calibrate_photometry(
            object_table=self.object_table,
            catalog_table=self.catalog_table,
            pixel_scale=0.1 / 3600,  # 0.1 arcsec per pixel
            error_threshold=0.1,  # Error threshold
            verbose=False
        )

        # Verify the results
        self.assertIsInstance(results, dict)
        self.assertIn('zero_fn', results)

    @patch('goodman_photometry.goodman_astro.match')
    def test_calibration_failure(self, mock_match):
        """
        Test photometric calibration failure.
        """
        # Mock the match function to return None (simulating failure)
        mock_match.return_value = None

        # Call the function
        results = calibrate_photometry(
            object_table=self.object_table,
            catalog_table=self.catalog_table,
            pixel_scale=0.1 / 3600,  # 0.1 arcsec per pixel
            verbose=False
        )

        # Verify the results
        self.assertIsNone(results)
if __name__ == "__main__":
    unittest.main()
