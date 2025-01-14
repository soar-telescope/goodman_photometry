import unittest
from unittest.mock import patch

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits

from ..goodman_astro import (calculate_saturation_threshold, create_goodman_wcs, extract_observation_metadata)


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


if __name__ == "__main__":
    unittest.main()
