# =============================================================================
#
#  Copyright (c) 2023-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from . import get_generator_cls
from . import get_extractor_cls
from qti.aisw.accuracy_debugger.lib.utils.nd_logger import setup_logger
from qti.aisw.accuracy_debugger.lib.utils.nd_symlink import symlink
from .nd_processor import Processor
from .nd_formatter import DataFormatter
from .nd_histogram_visualizer import visualize


class QuantChecker:
    """
    Base class for all QuantChecker functions
    """

    def __init__(self, args, logger=None) -> None:
        generator_class = get_generator_cls(args.engine)
        self._args = args
        self._logger = setup_logger(args.verbose, args.output_dir) if logger is None else logger
        self._generator = generator_class(args, self._logger)
        self._extractor = None
        self._processor = None
        self._formator = None

    def run(self):
        # Generate and dump all the qunat models w.r.t each quantization schemes like tf_cle etc.
        self._generator.generate_all_quant_models()
        # Get the map of quant_scheme and corresponding path where the generated model is dumped
        quant_schemes_dir_map = self._generator.get_quantized_variation_model_dir_map()

        # Get the extractor class for QNN/SNPE and extract all encodings from json files
        extractor_cls = get_extractor_cls(self._args.engine)
        self._extractor = extractor_cls(self._args, quant_schemes_dir_map, self._logger)
        self._extractor.extract()

        # Process the extracted encodings and perform sensitivity calculation based
        # on comparision algorithms
        self._processor = Processor(list(quant_schemes_dir_map.keys()),
                                    self._args.comparison_algorithms, self._logger)
        processed_results = self._processor.processResults(self._extractor)

        # Take the processed results and dump them in three formats: csv, html, histogram
        self._formator = DataFormatter(self._args, list(quant_schemes_dir_map.keys()),
                                       self._extractor.get_input_files(), self._logger)
        self._formator.formatResults(processed_results)
        if self._args.generate_plots:
            visualize(list(quant_schemes_dir_map.keys()), self._extractor.getAllOps(),
                    self._args.output_dir, self._args.generate_plots, self._args.per_channel_plots, self._logger)
