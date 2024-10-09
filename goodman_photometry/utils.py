import sys
import logging


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

    log = logging.getLogger(__name__)

    file_handler = logging.FileHandler(filename=log_filename)
    file_handler.setFormatter(fmt=formatter)
    file_handler.setLevel(level=logging_level)
    log.addHandler(file_handler)

    # if not generic:
    #     log.info("Starting Goodman HTS Pipeline Log")
    #     log.info("Local Time    : {:}".format(
    #         datetime.datetime.now()))
    #     log.info("Universal Time: {:}".format(
    #         datetime.datetime.utcnow()))
