import tempfile
import unittest
import os
from unittest.mock import patch

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from ..goodman_astro import (
    calculate_saturation_threshold,
    create_bad_pixel_mask,
    create_goodman_wcs,
    extract_observation_metadata,
    get_vizier_catalog,
    mask_field_of_view,
    table_to_ldac,
    get_pixel_scale,
    spherical_distance,
    spherical_match,
    get_frame_center,
    make_kernel,
    evaluate_data_quality_results,
    file_write)


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

        idx1, idx2, dist = spherical_match(ra, dec, ra, dec, search_radius_deg=1/3600)

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


if __name__ == "__main__":
    unittest.main()
