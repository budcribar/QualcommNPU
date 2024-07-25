# =============================================================================
#
#  Copyright (c) 2019-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import subprocess
import sys


class ToolConfig(object):
    def __init__(self):
        self.interpreter_path  = "python"

    def run_framework_diagnosis(self, args):
        """Runs the framework diagnosis tool in the specified environment
        """
        process_path = os.path.join(os.path.dirname(__file__), '..', '..', 'bin', 'nd_run_framework_diagnosis.py')
        framework_subprocess = subprocess.Popen([self.interpreter_path, process_path] + args,
                                                stdout=subprocess.PIPE,
                                                stderr=subprocess.STDOUT,
                                                universal_newlines=True,
                                                env=os.environ.copy())

        for line in iter(framework_subprocess.stdout.readline, ""):
            print(line.strip())

        return framework_subprocess.wait()

    def run_qnn_inference_engine(self,args):
        process_path = os.path.join(os.path.dirname(__file__), '..', '..', 'bin', 'nd_run_qnn_inference_engine.py')
        engine_subprocess = subprocess.Popen([self.interpreter_path, process_path] + args,
                                             stdout=subprocess.PIPE,
                                             stderr=subprocess.STDOUT,
                                             universal_newlines=True,
                                             env=os.environ.copy())

        for line in iter(engine_subprocess.stdout.readline, ""):
            print(line.strip())

        return engine_subprocess.wait()

    def run_snpe_inference_engine(self,args):
        process_path = os.path.join(os.path.dirname(__file__), '..', '..', 'bin', 'nd_run_snpe_inference_engine.py')
        engine_subprocess = subprocess.Popen([self.interpreter_path, process_path] + args,
                                             stdout=subprocess.PIPE,
                                             stderr=subprocess.STDOUT,
                                             universal_newlines=True,
                                             env=os.environ.copy())

        for line in iter(engine_subprocess.stdout.readline, ""):
            print(line.strip())

        return engine_subprocess.wait()

    def run_verifier(self, args):
        process_path = os.path.join(os.path.dirname(__file__), '..', '..', 'bin', 'nd_run_verification.py')

        verifier_subprocess = subprocess.Popen([self.interpreter_path, process_path] + args,
                                               stdout=subprocess.PIPE,
                                               stderr=subprocess.STDOUT,
                                               universal_newlines=True,
                                               env=os.environ.copy())

        for line in iter(verifier_subprocess.stdout.readline, ""):
            print(line.strip())

        return verifier_subprocess.wait()

    def run_accuracy_deep_analyzer(self, args):
        """Runs the deep_analyzer tool in the specified environment"""

        process_path = os.path.join(os.path.dirname(__file__), '..', '..', 'bin', 'nd_run_accuracy_deep_analyzer.py')

        deep_analyzer_subprocess = subprocess.Popen([self.interpreter_path, process_path] + args,
                                                    stdout=subprocess.PIPE,
                                                    stderr=subprocess.STDOUT,
                                                    universal_newlines=True,
                                                    env=os.environ.copy())

        for line in iter(deep_analyzer_subprocess.stdout.readline, ""):
            print(line.strip())

        return deep_analyzer_subprocess.wait()
