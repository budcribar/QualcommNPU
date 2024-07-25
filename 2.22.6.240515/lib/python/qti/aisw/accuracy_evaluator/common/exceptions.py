##############################################################################
#
# Copyright (c) 2022-2023 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# All data and information contained in or disclosed by this document are
# confidential and proprietary information of Qualcomm Technologies, Inc., and
# all rights therein are expressly reserved. By accepting this material, the
# recipient agrees that this material and the information contained therein
# are held in confidence and in trust and will not be used, copied, reproduced
# in whole or in part, nor its contents revealed in any manner to others
# without the express written permission of Qualcomm Technologies, Inc.
#
##############################################################################


class ConfigurationException(Exception):
    """Exceptions encountered in Configuration class."""

    def __init__(self, msg, exc=None):
        self.exc = exc
        self.msg = msg
        super().__init__(self.msg)

    def __str__(self):
        if self.exc == None:
            return 'Reason: {}'.format(self.msg)
        else:
            return 'Exc: {} Reason: {}'.format(self.exc, self.msg)


class InferenceEngineException(Exception):
    """Exceptions encountered in InferenceEngine class."""

    def __init__(self, msg, exc=None):
        self.exc = exc
        self.msg = msg
        super().__init__(self.msg)

    def __str__(self):
        if self.exc == None:
            return 'Reason: {}'.format(self.msg)
        else:
            return 'Exc: {} Reason: {}'.format(self.exc, self.msg)


class UserInterruptException(Exception):
    """User interrupted the application."""

    def __init__(self, msg, exc=None):
        self.exc = exc
        self.msg = msg
        super().__init__(self.msg)

    def __str__(self):
        if self.exc == None:
            return 'Reason: {}'.format(self.msg)
        else:
            return 'Exc: {} Reason: {}'.format(self.exc, self.msg)


class UnsupportedException(Exception):
    """Unsupported Feature exception."""

    def __init__(self, msg, exc=None):
        self.exc = exc
        self.msg = msg
        super().__init__(self.msg)

    def __str__(self):
        if self.exc == None:
            return 'Reason: {}'.format(self.msg)
        else:
            return 'Exc: {} Reason: {}'.format(self.exc, self.msg)


class DefaultsException(Exception):
    """Exceptions encountered in Defaults class."""

    def __init__(self, msg, exc=None):
        self.exc = exc
        self.msg = msg
        super().__init__(self.msg)

    def __str__(self):
        if self.exc == None:
            return 'Reason: {}'.format(self.msg)
        else:
            return 'Exc: {} Reason: {}'.format(self.exc, self.msg)


class ModelTransformationException(Exception):
    """Handles exceptions encountered in ModelTransformation class."""

    def __init__(self, msg, exc=None):
        self.exc = exc
        self.msg = msg
        super().__init__(self.msg)

    def __str__(self):
        if self.exc == None:
            return 'Reason: {}'.format(self.msg)
        else:
            return 'Exc: {} Reason: {}'.format(self.exc, self.msg)


class FileComparatorException(Exception):
    """Handles exceptions encountered in FileComparator class."""

    def __init__(self, msg, exc=None):
        self.exc = exc
        self.msg = msg
        super().__init__(self.msg)

    def __str__(self):
        if self.exc == None:
            return 'Reason: {}'.format(self.msg)
        else:
            return 'Exc: {} Reason: {}'.format(self.exc, self.msg)


class CompilationException(Exception):
    """Exceptions encountered in InferenceEngine class."""

    def __init__(self, msg, exc=None):
        self.exc = exc
        self.msg = msg
        super().__init__(self.msg)

    def __str__(self):
        if self.exc == None:
            return 'Reason: {}'.format(self.msg)
        else:
            return 'Exc: {} Reason: {}'.format(self.exc, self.msg)


class ExecutionException(Exception):
    """Exceptions encountered in InferenceEngine class."""

    def __init__(self, msg, exc=None):
        self.exc = exc
        self.msg = msg
        super().__init__(self.msg)

    def __str__(self):
        if self.exc == None:
            return 'Reason: {}'.format(self.msg)
        else:
            return 'Exc: {} Reason: {}'.format(self.exc, self.msg)


class QnnConverterException(Exception):
    """Exceptions encountered in QNN Converter."""

    def __init__(self, msg, exc=None):
        self.exc = exc
        self.msg = msg
        super().__init__(self.msg)

    def __str__(self):
        if self.exc == None:
            return 'Reason: {}'.format(self.msg)
        else:
            return 'Exc: {} Reason: {}'.format(self.exc, self.msg)


class QnnModelLibGeneratorException(Exception):
    """Exceptions encountered in QNN Converter."""

    def __init__(self, msg, exc=None):
        self.exc = exc
        self.msg = msg
        super().__init__(self.msg)

    def __str__(self):
        if self.exc == None:
            return 'Reason: {}'.format(self.msg)
        else:
            return 'Exc: {} Reason: {}'.format(self.exc, self.msg)


class QnnContextBinaryGeneratorException(Exception):
    """Exceptions encountered in QNN Converter."""

    def __init__(self, msg, exc=None):
        self.exc = exc
        self.msg = msg
        super().__init__(self.msg)

    def __str__(self):
        if self.exc == None:
            return 'Reason: {}'.format(self.msg)
        else:
            return 'Exc: {} Reason: {}'.format(self.exc, self.msg)


class QnnNetRunException(Exception):
    """Exceptions encountered in QNN Converter."""

    def __init__(self, msg, exc=None):
        self.exc = exc
        self.msg = msg
        super().__init__(self.msg)

    def __str__(self):
        if self.exc == None:
            return 'Reason: {}'.format(self.msg)
        else:
            return 'Exc: {} Reason: {}'.format(self.exc, self.msg)
