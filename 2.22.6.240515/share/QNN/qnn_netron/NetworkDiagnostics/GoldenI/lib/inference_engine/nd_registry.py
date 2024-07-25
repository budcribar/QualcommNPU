# =============================================================================
#
#  Copyright (c) 2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from distutils.version import LooseVersion, Version

from lib.utils.nd_constants import ComponentType, Framework, Engine
from lib.utils.nd_exceptions import InferenceEngineError


class Registry(object):
    def __init__(self):
        self.registry = {}

    def _set(self, cls_type, framework, engine, engine_version, cls):
        # type: (ComponentType, Framework, Engine, str, 'cls') -> None
        self.registry.setdefault((cls_type, framework, engine), {})[str(engine_version)] = cls

    def _get(self, cls_type, framework, engine, engine_version):
        # type: (ComponentType, Framework, Engine, str) -> None
        versions = self.registry.get((cls_type, framework, engine), None)

        if versions is None:
            # Check for a framework-less version
            versions = self.registry.get((cls_type, None, engine), None)

            if versions is None:
                raise InferenceEngineError

        if engine_version is None:
            # select the latest version
            engine_version = max(map(LooseVersion, versions.keys()))
        elif str(engine_version) not in versions:
            if not isinstance(engine_version, Version):
                engine_version = LooseVersion(engine_version)

            # select the closest version to the one requested
            engine_version = sorted([version for version in versions.keys()
                                     if LooseVersion(version) <= engine_version]).pop()

        return versions[str(engine_version)]

    def register(self, cls_type, engine, engine_version, framework=None):
        def cls_wrapper(cls):
            self._set(cls_type, framework, engine, engine_version, cls)
            return cls
        return cls_wrapper

    def get_converter_class(self, framework, engine, engine_version):
        return self._get(ComponentType.converter, framework, engine, engine_version)

    def get_executor_class(self, framework, engine, engine_version):
        return self._get(ComponentType.executor, framework, engine, engine_version)

    def get_inference_engine_class(self, framework, engine, engine_version):
        return self._get(ComponentType.inference_engine, framework, engine, engine_version)

    def get_quantizer_class(self, framework, engine, engine_version):
        return self._get(ComponentType.quantizer, None, engine, engine_version)
