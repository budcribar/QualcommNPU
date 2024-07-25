# ==============================================================================
#
#  Copyright (c) 2020-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from .core import *
from qti.aisw.converters.common.custom_ops.op_factory import *


class QnnCustomOpFactory(CustomOpFactory, metaclass=ABCMeta):
    package_resolver = dict()
    op_collection = CustomOpCollection()
    default_op_collection = []
    custom_opdefs = []

    def __init__(self):
        super(QnnCustomOpFactory, self).__init__()

    @staticmethod
    def create_op(op_type, inputs, outputs, *args, **kwargs):
        return QnnCustomOp(op_type, inputs, outputs, *args, **kwargs)

    @classmethod
    def get_package_name(cls, op_type):
        package_names = [None]
        for package_name, node_types in cls.package_resolver.items():
            if op_type.lower() in node_types:
                package_names.append(package_name)
        return package_names[-1]

    def parse_config(self, config_path, model, converter_type, **kwargs):
        # Import config and parse using XML translator
        try:
            from qti.aisw.converters.backend.custom_ops.snpe_udo_config import UdoGenerator
            package_generator = UdoGenerator()
        except:
            from qti.aisw.op_package_generator.generator import QnnOpPackageGenerator
            package_generator = QnnOpPackageGenerator()
        package_generator.parse_config([config_path])
        if "converter_op_package_libs" in kwargs:
            op_package_libs = kwargs['converter_op_package_libs']
            for i in range(len(package_generator.package_infos)):
                if "converter_op_package_libs" not in self.package_resolver:
                    self.package_resolver["converter_op_package_libs"] = {package_generator.package_infos[i].name: op_package_libs[i]}
                else:
                    self.package_resolver["converter_op_package_libs"][package_generator.package_infos[i].name] = op_package_libs[i]
                op_package_libs.pop(i)
                kwargs["converter_op_package_libs"] = op_package_libs
        for package_info in package_generator.package_infos:
            for operator in package_info.operators:
                # Store all the operator defined in the XML config, this may be used in creating an dynamic onnx op.
                self.custom_opdefs.append(operator)

                if package_info.name not in self.package_resolver:
                    self.package_resolver.update(
                        {package_info.name: [operator.type_name.lower()]})
                else:
                    self.package_resolver[package_info.name].append(operator.type_name.lower())

                # if the op requires default translation, we mark that here and skip.
                # The default op_collection list exists purely for tracking purposes
                if operator.use_default_translation:
                    self.default_op_collection.append(operator.type_name.lower())
                else:
                    try:
                        custom_ops = self.create_ops_from_operator(operator, model=model,
                                                                   converter_type=converter_type,
                                                                   **kwargs)
                        self.op_collection[custom_ops[0].op_type] = custom_ops
                    except CustomOpNotFoundError:  # if an op is not found then it is skipped
                        log_warning(
                            "Custom Op: {} was defined in the config but was not found in the "
                            "model".
                                format(operator.type_name))
                        continue
                    except Exception as e:
                        raise e


OpFactory = QnnCustomOpFactory
