# =============================================================================
#
#  Copyright (c) 2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import json

from GoldenIWrapper import GoldenIWrapper


class PythonServer:
    """
    This PythonServer is the entry point of the GraphVisualizer backend. It receives parameters from the frontend
    as system arguments, and it branches execution based on OS (Linux or Windows).
    """

    def __init__(self):
        """
        Instantiates a PythonServer object, using parameters from the front end.
        """
        self.os_args = sys.argv[1]
        self.isWin = len(self.os_args) != 0
        if self.isWin:
            self.os_args = json.loads(self.os_args)

        self.workspace = sys.argv[2]
        self.case = sys.argv[3]
        self.inf1_args = sys.argv[4].split(sep=",") if sys.argv[4] is not None and sys.argv[4] != 'null' else None
        self.inf2_args = sys.argv[5].split(sep=",") if sys.argv[5] is not None and sys.argv[5] != 'null' else None
        self.verif_args = sys.argv[6].split(sep=",")
        self.verif_param_json = sys.argv[7] if sys.argv[7] is not None and sys.argv[7] != '' else None

    def windows_execution(self):
        """
        This is the entry point of execution on a Windows machine.
        """
        pass

    def linux_execution(self):
        """
        This is the entry point of execution on a Linux machine. It simply runs GoldenIWrapper.py, passing in the
        parameters from the frontend.
        """
        goldenIWrapper = GoldenIWrapper(workspace=self.workspace)

        if self.verif_param_json:
            jsonObj =json.load(open(self.verif_param_json,'r'))
            del jsonObj['default_verifier']
            goldenIWrapper.run(self.case, self.inf1_args, self.inf2_args, self.verif_args,jsonObj)
        else:
            goldenIWrapper.run(self.case, self.inf1_args, self.inf2_args, self.verif_args)


if __name__ == '__main__':
    pyserver = PythonServer()

    if pyserver.isWin:
        pyserver.windows_execution()
    else:
        pyserver.linux_execution()
