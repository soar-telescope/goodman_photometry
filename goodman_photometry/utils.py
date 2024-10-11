import argparse
import sys
import logging

from importlib.metadata import version
__version__ = version('goodman_photometry')


def setup_logging(debug=False, generic=False, log_filename='goodman_photometry_log.txt'):  # pragma: no cover
    """configures logging

    Notes:
        Logging file name is set to default 'goodman_photometry_log.txt'.
        If --debug is activated then the format of the message is different.
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
    """Does astrometry solution of goodman data
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

    return args


def get_photometry_args(arguments=None):
    """Obtains photometry of goodman data
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

    return args
