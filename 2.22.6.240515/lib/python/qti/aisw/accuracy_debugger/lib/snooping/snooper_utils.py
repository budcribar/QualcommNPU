# =============================================================================
#
#  Copyright (c) 2021-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import sys
import re
from qti.aisw.accuracy_debugger.lib.utils.nd_namespace import Namespace
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import ConfigError
from qti.aisw.accuracy_debugger.lib.verifier.nd_verifier_factory import VerifierFactory
from qti.aisw.accuracy_debugger.lib.utils.nd_errors import get_message
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import VerifierError


def replace_special_chars(name):
    """
        This method replaces special characters with underscore in supplied name
        Args:
            name : name of node
        Returns:
            name : modified name after replacing special characters
        """
    kdict = {':': '_', '/': '_', '-': '_'}
    for key in kdict:
        name = name.replace(key, kdict[key])
    return name


def show_progress(total_count, cur_count, info='', key='='):
    """Displays the progress bar."""
    completed = int(round(80 * cur_count / float(total_count)))
    percent = round(100.0 * cur_count / float(total_count), 1)
    bar = key * completed + '-' * (80 - completed)

    sys.stdout.write('[%s] %s%s (%s)\r' % (bar, percent, '%', info))
    sys.stdout.flush()


class SnooperUtils:
    """
    SnooperUtils class contains all configuration parameters supplied by user
    To use:
    >>> config = SnooperUtils.getInstance()
    """
    __instance = None

    def __init__(self, args):
        if SnooperUtils.__instance is not None:
            raise ConfigError('instance of SnooperUtils already exists')
        else:
            SnooperUtils.__instance = self

        self._config = args
        self._transformer = None
        self._traverser = None
        self.updated_name_map = None
        self._framework_ins = None
        self._comparator = None

    @classmethod
    def clear(cls):
        if cls.__instance is not None:
            cls.__instance = None

    def __str__(self):
        return str(self._config)

    @classmethod
    def getInstance(cls, args=None):
        if cls.__instance is None:
            cls.__instance = SnooperUtils(args)
        if args is not None:
            cls.__instance.args = args
        return cls.__instance

    def getStartLayer(self):
        if not self._config.start_layer:
            return None
        start_layer = replace_special_chars(self._config.start_layer)
        if not self.updated_name_map or start_layer not in self.updated_name_map:
            return start_layer
        else:
            #This is used only for Caffe due to output name change made by caffe_transform
            return self.updated_name_map[start_layer]

    def getEndLayer(self):
        if not self._config.end_layer:
            return None
        end_layer = replace_special_chars(self._config.end_layer)
        if not self.updated_name_map or end_layer not in self.updated_name_map:
            return end_layer
        else:
            #This is used only for Caffe due to output name change made by caffe_transform
            return self.updated_name_map[end_layer]

    def getModelTraverserInstance(self):
        """This method returns the appropriate ModelTraverser class instance
        Returns the same instance each time."""
        return self._framework_ins

    def setModelTraverserInstance(self, logger, args, model_path=None, add_layer_outputs=[],
                                  add_layer_types=[], skip_layer_outputs=[], skip_layer_types=[]):
        """This method returns the appropriate ModelTraverser class instance
        Returns the same instance each time."""
        from qti.aisw.accuracy_debugger.lib.framework_diagnosis.nd_framework_runner import ModelTraverser
        if model_path:
            args.model_path = model_path
        framework_args = Namespace(framework=args.framework, version=args.framework_version,
                                   model_path=args.model_path, output_dir=args.output_dir,
                                   engine=args.engine)
        self._framework_ins = ModelTraverser(logger, framework_args,
                                             add_layer_outputs=add_layer_outputs,
                                             add_layer_types=add_layer_types,
                                             skip_layer_outputs=skip_layer_outputs,
                                             skip_layer_types=skip_layer_types)
        return self._framework_ins

    def getFrameworkInstance(self):
        """This method returns the appropriate FrameworkRunner class instance
        Returns the same instance each time."""
        return self._framework_ins

    def setFrameworkInstance(self, logger, args, model_path=None):
        """This method returns the appropriate FrameworkRunner class instance
        Returns the same instance each time."""
        from qti.aisw.accuracy_debugger.lib.framework_diagnosis.nd_framework_runner import FrameworkRunner
        if model_path:
            args.model_path = model_path
        framework_args = Namespace(framework=args.framework, version=args.framework_version,
                                   model_path=args.model_path, output_dir=args.output_dir,
                                   engine=args.engine)
        self._framework_ins = FrameworkRunner(logger, framework_args)
        self._framework_ins.load_framework()
        return self._framework_ins

    def getComparator(self, tol_thresolds=None):
        """Returns the list of configured verifiers."""

        verifier_objects = []
        for verifier in self._config.default_verifier:
            verifier = verifier[0].split(',')
            verifier_name = verifier[0]
            try:
                verifier_config = {}
                ret, verifier_config = VerifierFactory().validate_configs(
                    verifier_name, verifier[1:])
                if not ret:
                    errormsg = str(verifier_config['error']) if 'error' in verifier_config else ''
                    raise VerifierError("VerifierFactory config_verify error: " + errormsg)

                verifier_obj = VerifierFactory().factory(verifier_name, verifier_config)
                if verifier_obj is None:
                    raise VerifierError(
                        get_message('ERROR_VERIFIER_INVALID_VERIFIER_NAME')(verifier_name))

                verifier_objects.append(verifier_obj)
            except Exception as err:
                raise Exception(
                    f"Error occurred while configuring {verifier_name} verifier. Reason: {err}")

        self._comparator = verifier_objects
        return self._comparator
