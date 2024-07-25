# =============================================================================
#
#  Copyright (c) 2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================


class FrameworkError(Exception):
    """
    Defines a generic error for any framework errors.
    """

    def __init__(self, message=''):
        super(FrameworkError, self).__init__(message)


class ConfigError(Exception):
    """
    Defines a generic error for any config errors.
    """

    def __init__(self, message=''):
        super(ConfigError, self).__init__(message)


class DependencyError(Exception):
    """
    Defines a generic error for any dependency errors
    """

    def __init__(self, message=''):
        super(DependencyError, self).__init__(message)


class DeviceError(Exception):
    """
    Defines a generic error for any device errors
    """

    def __init__(self, message=''):
        super(DeviceError, self).__init__(message)


class InferenceEngineError(Exception):
    """
    Defines a generic error for any inference engine errors.
    """

    def __init__(self, message=''):
        super(InferenceEngineError, self).__init__(message)


class VerifierError(Exception):
    """
    Defines a generic error for any verifier errors.
    """

    def __init__(self, message=''):
        super(VerifierError, self).__init__(message)


class VerifierInputError(Exception):
    """
    Defines an error for invalid input to a verifier.
    """

    def __init__(self, message=''):
        super(VerifierInputError, self).__init__(message)


class DeepAnalyzerError(Exception):
    """
    Defines a generic error for any deep analyzer errors.
    """

    def __init__(self, message=''):
        super(DeepAnalyzerError, self).__init__(message)


class ProfilingError(Exception):
    """
    Defines a generic error for any profile viewer errors.
    """

    def __init__(self, message=''):
        super(ProfilingError, self).__init__(message)

class ParameterError(Exception):
    """
    Defines a generic error for any parameter errors.
    """

    def __init__(self, message=''):
        super(ParameterError, self).__init__(message)