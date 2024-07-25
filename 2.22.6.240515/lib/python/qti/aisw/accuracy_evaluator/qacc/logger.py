import sys
import logging


class QaccLogger:
    file_format = '%(asctime)s - %(levelname)-8s [%(module)-14s] - %(message)s'
    console_format = '%(levelname)s: %(message)s'

    @classmethod
    def setup_logger(cls, log_file, log_level):
        """Set loggers for Accuracy Evaluator.

        Sets qacc_logger and qacc_file_logger loggers.
        qacc_logger logger logs on console and given log_file.
        qacc_file_logger logs only in log file.

        Args:
            log_file: Path to log file.
            log_level: Logging level to be used.
        """
        logging.debug('Enabling loggers')
        console_logger = logging.getLogger('qacc')
        file_logger = logging.getLogger('qacc_logfile')
        console_logger.handlers = []
        file_logger.handlers = []
        console_logger.propagate = file_logger.propagate = False

        # create file handlers
        fh_file = logging.FileHandler(log_file, mode='w+')
        fh_file.setFormatter(logging.Formatter(cls.file_format))
        fh_file.setLevel(log_level)

        fh_console = logging.StreamHandler(sys.stdout)
        fh_console.setFormatter(logging.Formatter(cls.console_format))
        fh_console.setLevel(log_level)

        # By default we set root logging level to error. Since root is an ancestor and root level is greater,
        # root level is considered as effective level. so removing root from ancestors.
        console_logger.parent = []
        file_logger.parent = []

        # Add handlers
        console_logger.addHandler(fh_console)
        console_logger.addHandler(fh_file)
        file_logger.addHandler(fh_file)

        # set log level
        console_logger.setLevel(log_level)
        file_logger.setLevel(log_level)
