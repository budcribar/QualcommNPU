# ==============================================================================
#
#  Copyright (c) 2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
# ==============================================================================

from pydantic import BaseModel, Field, model_validator, ConfigDict

"""
This module contains the definition for AISWBaseModel class which is a pydantic class derived from BaseModel,
and AISWVersion which is a pydantic class that stores fields that categorize a semantic versioning scheme.
"""


class AISWBaseModel(BaseModel):
    """ Internal variation of a BaseModel"""

    model_config = ConfigDict(extra='forbid', validate_assignment=True)


class AISWVersion(AISWBaseModel):
    """
    A dataclass that conveys when modifications are made to a module's interface
    or its properties.
    """

    _MAJOR_VERSION_MAX = 15
    _MINOR_VERSION_MAX = 40
    _PATCH_VERSION_MAX = 15
    _PRE_RELEASE_MAX_LENGTH = 26

    major: int = Field(ge=0, le=_MAJOR_VERSION_MAX)  # Backwards incompatible changes to a module
    minor: int = Field(ge=0, le=_MINOR_VERSION_MAX)  # Backwards compatible changes
    patch: int = Field(ge=0, le=_PATCH_VERSION_MAX)  # Bug fixes are made in a backwards compatible manner
    pre_release: str = Field(default="", max_length=_PRE_RELEASE_MAX_LENGTH)

    @model_validator(mode='after')
    def check_allowed_sem_ver(self):
        """
        Sanity checks a version to ensure it is not all zeros

        Raises:
            ValueError if no version is set
        """
        if self.major == self.minor == self.patch == 0:
            raise ValueError(f'Version: {self.__repr__()} is not allowed')
        return self

    def __str__(self):
        """
         Formats the version as a string value: "major.minor.patch"
         or "major.minor.patch" if the release tag is set
        """
        if not self.pre_release:
            return f'{self.major}.{self.minor}.{self.patch}'
        return f'{self.major}.{self.minor}.{self.patch}-{self.pre_release}'
