import unittest
from unittest.mock import patch

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits

from ..goodman_astro import (
    calculate_saturation_threshold,
    create_bad_pixel_mask,
    create_goodman_wcs,
    extract_observation_metadata,
    mask_field_of_view)


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


if __name__ == "__main__":
    unittest.main()
