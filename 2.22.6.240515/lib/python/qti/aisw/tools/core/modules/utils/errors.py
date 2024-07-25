# ==============================================================================
#
#  Copyright (c) 2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
# ==============================================================================

class SchemaVersionError(AttributeError):
    """ Intended for version related errors"""
    pass


class SchemaFieldTypeError(TypeError):
    """ Intended for schema field errors"""
    pass


class SchemaFieldValueError(ValueError):
    """ Intended for schema field value errors"""
    pass


class ModuleComplianceError(AttributeError):
    pass
