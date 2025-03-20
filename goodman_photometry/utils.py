"""Goodman Photometry and Astrometry Processing Module.

This module provides functionality for processing photometry and astrometry data
from the Goodman High Throughput Spectrograph. It includes utilities for setting up
logging, parsing command-line arguments, and performing photometry and astrometry
analysis.

The module contains the following main functions:
- `setup_logging`: Configures logging for the application.
- `get_astrometry_args`: Parses command-line arguments for astrometry processing.
- `get_photometry_args`: Parses command-line arguments for photometry processing.

Example:
    To use this module, import it and call the relevant functions:
    >>> from goodman_photometry import setup_logging, get_astrometry_args
    >>> setup_logging(debug=True)
    >>> args = get_astrometry_args()

Version:
    The module version is retrieved from the package metadata using `importlib.metadata`.

Notes:
    - Logging is configured to write to a file (`goodman_photometry_log.txt` by default).
    - Command-line arguments are parsed using `argparse`.
"""
import argparse
import sys
import logging

from importlib.metadata import version
__version__ = version('goodman_photometry')


def setup_logging(debug=False, generic=False, log_filename='goodman_photometry_log.txt'):  # pragma: no cover
    """Configure logging for the application.

    This function sets up logging with a specified format and logging level. If debugging is enabled,
    the log format includes additional details such as the module, function name, and line number.
    Logs are written to a file specified by `log_filename`.

    Args:
        debug (bool, optional): If True, enables debug mode with detailed logging. Defaults to False.
        generic (bool, optional): If True, skips adding generic log entries (e.g., pipeline start time).
                                  Defaults to False.
        log_filename (str, optional): The name of the file where logs will be saved. Defaults to
                                      'goodman_photometry_log.txt'.

    Notes:
        The default log file name is 'goodman_photometry_log.txt'. If `--debug` is activated or `debug=True`,
        the log format includes additional details for debugging purposes.

    Example:
        >>> setup_logging(debug=True)
        >>> setup_logging(log_filename='custom_log.txt')
    """
    if '--debug' in sys.argv or debug:
        log_format = '[%(asctime)s][%(levelname)8s]: %(message)s ' \
                     '[%(module)s.%(funcName)s:%(lineno)d]'
        logging_level = logging.DEBUG
    else:
        log_format = '[%(asctime)s][%(levelname).1s]: %(message)s'
        logging_level = logging.INFO

    date_format = '%H:%M:%S'

    formatter = logging.Formatter(fmt=log_format,
                                  datefmt=date_format)

    logging.basicConfig(level=logging_level,
                        format=log_format,
                        datefmt=date_format)

    log = logging.getLogger()

    file_handler = logging.FileHandler(filename=log_filename)
    file_handler.setFormatter(fmt=formatter)
    file_handler.setLevel(level=logging_level)
    log.addHandler(file_handler)

    print(logging_level)

    # if not generic:
    #     log.info("Starting Goodman HTS Pipeline Log")
    #     log.info("Local Time    : {:}".format(
    #         datetime.datetime.now()))
    #     log.info("Universal Time: {:}".format(
    #         datetime.datetime.utcnow()))


def get_astrometry_args(arguments=None):
    """Parse command-line arguments for astrometry solution processing.

    This function defines and processes command-line arguments required
    for performing an astrometry solution using the Goodman High Throughput Spectrograph data.

    Args:
        arguments (list, optional): A list of command-line arguments. If None,
            arguments are taken from `sys.argv`. Defaults to None.

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.

    Parsed Arguments:
        filename (str, optional): Path to the input FITS file for astrometry processing.
        catalog_name (str): Name of the reference star catalog (default: 'gaiadr2').
        magnitude_threshold (float): Threshold for source magnitudes (default: 12).
        scamp_flag (int): SCAMP (Software for Calibrating AstroMetry and Photometry) flag (default: 1).
        color_map (str): Matplotlib colormap for plots (default: 'Blues_r').
        log_filename (str): Name of the log file (default: 'goodman_astrometry_log.txt').
        save_plots (bool): Flag to enable saving astrometry-related plots.
        save_scamp_plots (bool): Flag to enable saving SCAMP-generated plots.
        save_intermediary_files (bool): Flag to save intermediary processing files.
        debug (bool): Flag to enable debug mode.
        version (bool): Flag to print version information and exit.

    Raises:
        SystemExit: If no filename is provided, the function prints help information
            and exits the program.

    Notes:
        - If `--version` is provided, the script prints the version and exits.
        - If no filename is provided, the function prints usage information and exits.
    """
    parser = argparse.ArgumentParser(
        description=f"Does astrometry solution of Goodman High Throughput Spectrograph data.\n\nVersion: {__version__}")

    parser.add_argument('filename', nargs='?', help="Path to file to process.")
    parser.add_argument(
        '--catalog-name',
        default='gaiadr2',
        type=str,
        action='store',
        dest='catalog_name',
        help='Catalog name')
    parser.add_argument(
        '--magnitude-threshold',
        default=12,
        type=float,
        action='store',
        dest='magnitude_threshold',
        help='Magnitude threshold')
    parser.add_argument(
        '--scamp-flag',
        default=1,
        type=int,
        action='store',
        dest='scamp_flag',
        help='Scamp flag')
    parser.add_argument(
        '--color-map',
        default='Blues_r',
        type=str,
        action='store',
        dest='color_map',
        help='Color map')
    parser.add_argument(
        '--log-filename',
        default='goodman_astrometry_log.txt',
        type=str,
        action='store',
        dest='log_filename',
        help='Defines the filename of the log file.')

    parser.add_argument('-p', '--save-plots', action='store_true')
    parser.add_argument('-s', '--save-scamp-plots', action='store_true')
    parser.add_argument('-i', '--save-intermediary-files', action='store_true')
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-v', '--version', action='store_true')

    args = parser.parse_args(args=arguments)
    if args.version:
        print(__version__)
        sys.exit(0)

    if not args.filename:
        parser.print_help()
        parser.exit(0, "\nPlease specify a filename to process.\n")

    return args


def get_photometry_args(arguments=None):
    """Parse command-line arguments for Goodman photometry processing.

    This function defines and processes command-line arguments required
    for photometry analysis using the Goodman High Throughput Spectrograph data.

    Args:
        arguments (list, optional): A list of command-line arguments. If None,
            arguments are taken from `sys.argv`. Defaults to None.

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.

    Parsed Arguments:
        filename (str, optional): Path to the input FITS file for photometry processing.
        catalog_name (str): Name of the reference star catalog (default: 'gaiadr2').
        magnitude_threshold (float): Threshold for source magnitudes (default: 17).
        magnitude_error_threshold (float): Maximum allowed magnitude error (default: 0.1).
        color_map (str): Matplotlib colormap for plots (default: 'Blues_r').
        plot_file_resolution (int): Resolution of output plots in DPI (default: 600).
        log_filename (str): Name of the log file (default: 'goodman_photometry_log.txt').
        save_plots (bool): Flag to enable saving plots.
        debug (bool): Flag to enable debug mode.
        version (bool): Flag to print version information and exit.

    Raises:
        SystemExit: If no filename is provided, the function prints help information
            and exits the program.

    Notes:
        - If `--version` is provided, the script prints the version and exits.
        - If no filename is provided, the function prints usage information and exits.
    """
    parser = argparse.ArgumentParser(
        description=f"Obtains photometry of Goodman High Throughput Spectrograph data.\n\nVersion: {__version__}")

    parser.add_argument('filename', nargs='?', help="Path to file to process.")
    parser.add_argument(
        '--catalog-name',
        default='gaiadr2',
        type=str,
        action='store',
        dest='catalog_name',
        help='Catalog name')
    parser.add_argument(
        '--magnitude-threshold',
        default=17,
        type=float,
        action='store',
        dest='magnitude_threshold',
        help='Magnitude threshold')
    parser.add_argument(
        '--magnitude-error-threshold',
        default=0.1,
        type=float,
        action='store',
        dest='magnitude_error_threshold',
        help='Magnitude error threshold')
    parser.add_argument(
        '--color-map',
        default='Blues_r',
        type=str,
        action='store',
        dest='color_map',
        help='Color map')
    parser.add_argument(
        '--plot-file-resolution',
        default=600,
        type=int,
        action='store',
        dest='plot_file_resolution',
        help='Plot file resolution')
    parser.add_argument(
        '--log-filename',
        default='goodman_photometry_log.txt',
        type=str,
        action='store',
        dest='log_filename',
        help='Defines the filename of the log file.')

    parser.add_argument('-p', '--save-plots', action='store_true')
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-v', '--version', action='store_true')

    args = parser.parse_args(args=arguments)
    if args.version:
        print(__version__)
        sys.exit(0)

    if not args.filename:
        parser.print_help()
        parser.exit(0, "\nPlease provide a filename to process.\n")

    return args
