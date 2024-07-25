# ==============================================================================
#
#  Copyright (c) 2020 - 2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from qti.aisw.converters.common.utils.converter_utils import *
import os
from pathlib import Path
import shutil

QNN_SDK_ROOT = os.getenv("QNN_SDK_ROOT")


from qti.aisw.op_package_generator.core import *
from qti.aisw.op_package_generator.generator import *
from .core import *

if QNN_SDK_ROOT is not None:
    OP_PACKAGE_GENERATOR_LIB = os.path.join(os.path.abspath(QNN_SDK_ROOT),
                                            'lib', 'python', 'qti', 'aisw', 'op_package_generator')
    SNPE_OP_PACKAGE_GENERATOR_PATH = os.path.join(Path(__file__).parents[3], 'op_package_generator')
    # overriding the QNN SHARE_LOC_PREFIX
    SHARE_LOC_PREFIX = os.path.join(os.path.abspath(QNN_SDK_ROOT), 'share', 'QNN', 'OpPackageGenerator')
    CUSTOM_OP_DIR = os.path.join(SHARE_LOC_PREFIX, 'CustomOp')


class UDOTemplateFileReader:
    """ Enum class that stores template file names and their corresponding types"""

    template_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'templates')

    class TemplateFileTypes(Enum):
        COMMON = 0,
        CPU = 1,
        GPU = 2,
        DSP = 3,
        HTP = 4,
        MAKEFILE = 5,
        USER_DEFINED = 6,
        UNKNOWN = 7,
        REG = 8

        def describe(self):
            return self.name, self.value

        @classmethod
        def default(cls):
            return cls.CPU

        @classmethod
        def is_backend_type(cls, template_type):
            return template_type == cls.CPU or template_type == cls.GPU or template_type == cls.DSP or \
                template_type == cls.DSP_V68 or template_type == cls.DSP_V69 or template_type == cls.DSP_V73

    # Note: All backend types must have at least one source file, which should be listed
    # first, and one interface file which must be listed second.
    # e.x ["source.mako", "interface.mako"]
    DEFAULT_TEMPLATE_FILES = {
        #COMMON here is Cmake file type
        TemplateFileTypes.COMMON: ['root_cmakelists_template.mako', 'lib_cmakelists_template.mako'],
        TemplateFileTypes.CPU: ['cpu_source.mako', "cpu_interface.mako"],
        TemplateFileTypes.DSP: ["dsp_source.mako", 'dsp_interface.mako', 'dsp_header.mako'],
        TemplateFileTypes.HTP: ['htp_source.mako', 'htp_interface.mako'],
        TemplateFileTypes.GPU: ['gpu_source.mako', 'gpu_interface.mako', 'gpu_operation.mako'],
        TemplateFileTypes.MAKEFILE: ['reg_makefile_template.mako', 'android_makefile_template.mako',
                                     'main_makefile_template.mako', 'android_makefile.mako'],
        TemplateFileTypes.REG: ['reg_lib_template.mako']}

    @classmethod
    def get_template_type_by_backend(cls, backend):
        if backend == 'CPU':
            return cls.DEFAULT_TEMPLATE_FILES[cls.TemplateFileTypes.CPU]
        elif backend == 'GPU':
            return cls.DEFAULT_TEMPLATE_FILES[cls.TemplateFileTypes.GPU]
        elif backend == 'DSP':
            return cls.DEFAULT_TEMPLATE_FILES[cls.TemplateFileTypes.DSP]
        elif backend == 'HTP':
            return cls.DEFAULT_TEMPLATE_FILES[cls.TemplateFileTypes.DSP_V68]
        raise LookupError("Cannot retrieve mako template for unknown backend: {}".format(backend))


# ------------------------------------------------------------------------------
#   Udo config Core Classes
# ------------------------------------------------------------------------------
class UdoGenerator(QnnOpPackageGenerator):
    """
    This class is the main point of entry for the package-generator. It handles the parsing of the user provided
    config and creates a udo package object. It then sets up file paths according to information gathered in the
    package in the parsing step. Finally, implements the package by auto-generating header and source files using a
    UdoFileGenerator object.

    :udo_packages: This is a list of all udo package registered in a single generator instance
    :UdoFileGenerator: This object handles the auto-generation of code using Mako templates.
    """

    def __init__(self):
        super(UdoGenerator, self).__init__()

    def create_package_info(self, op_package_collection, output_path, config_path):
        from qti.aisw.converters.snpe_backend.custom_ops.helpers.udo_module_helpers import get_internal_core_types
        for package_name, package in op_package_collection.items():
            for backend, operators in package.items():
                package_info = QnnPackageInfo(package_name, output_path, backend, operators)
                package_info.SNPE_TEMPLATES_PATH = UDOTemplateFileReader.template_path

                # add extra members required by SNPE-UDO
                package_info.SNPE_UDO_ROOT = os.getenv("SNPE_UDO_ROOT")
                package_info.dsp_arch_types = []
                dsp_arch_types = ['V65', 'V66', 'V68', 'V69', 'V73', 'V75', 'V79']
                if "DSP" in package_info.backend:
                    package_info.backend = 'DSP'
                    for dsp_arch_type in dsp_arch_types:
                        if dsp_arch_type in backend:
                            package_info.dsp_arch_types.append(dsp_arch_type.lower())
                package_info.core_types = get_internal_core_types([package_info.backend])
                package_info.supported_runtimes = [package_info.backend]
                package_info.op_catalog_info = list()
                package_info.calculation_types = []
                for operator in package_info.operators:
                    operator.core_types = []
                    operator.core_types += package_info.core_types
                    operator.dsp_arch_types = []
                    operator.dsp_arch_types += package_info.dsp_arch_types
                    operator.scalar_param = list(param for param in operator.param if param.shape == 'scalar')
                    operator.tensor_param = list(param for param in operator.param if param.shape != 'scalar')
                    for inp in operator.input:
                        inp.data_type = []
                    for out in operator.output:
                        out.data_type = []
                self.register_package_info(package_info, config_paths=[config_path])

    def setup_file_paths(self, force_generation=False, gen_cmakelists=False, **files):
        """
         This sets up file paths and makes the directory structure for the package. It makes the top level directory,
         followed by the src, lib, include and makefile directories. This method will not overwrite directories if they
         already exist.
        :param force_generation:  if set to true, any package directory in the generator instance will be overwritten
                                  if it exists.
        :param ignore_includes: setting this flag to false means files will be copied from SnpeUdo API into
                              the package. If this flag is set to ignore, the user gets a warning as the files will be needed
                              during compilation
        :param files: These are files the user may want to copy into their created package. The current user is to copy
                      the config into this directory.
        :return: the package paths that have been successfully set up
        """
        # for udo package in udo_packages
        # setup root-> lib, src, include, *.json/*.xml
        self.force_generation = force_generation
        self.gen_cmakelists = gen_cmakelists
        package_paths = []

        for i, package_info in enumerate(self.package_infos):
            SNPE_UDO_ROOT = package_info.SNPE_UDO_ROOT
            if not SNPE_UDO_ROOT:
                raise IOError("Files cannot be copied as SNPE_UDO_ROOT variable is not set.")

            # each package must have its own code generator object
            package_generator = QnnOpPackageCodeGenerator()
            if package_info.name in self.file_generators:
                if self.file_generators[package_info.name]. \
                        resource_handler.destination == package_info.root:
                    log_warning("Attempting to create package with duplicate name: "
                                "{} for backend: {} in the same location: {}".format(package_info.name,
                                                                                     package_info.backend,
                                                                                     package_info.root))
                    package_info.name = package_info.name + package_info.backend  # revisit : handle multiple backends
                    log_info("Package name will be changed to {}".format(package_info.name))

            # Make runtime specific checks and set warnings as needed
            if package_info.backend == "DSP" or package_info.backend == "DSP_V68" or package_info.backend == "DSP_V69" or \
                    package_info.backend == "CPU" or package_info.backend == "DSP_V73" or package_info.backend == "GPU":
                if package_info.backend == "DSP_V68" or package_info.backend == "DSP_V69" or package_info.backend == "DSP_V73" \
                        or package_info.backend == "DSP" and not os.getenv("HEXAGON_SDK_ROOT", None):
                    log_warning('DSP Operation detected but HEXAGON_SDK_ROOT is not set. Please '
                                'note HEXAGON_SDK_ROOT needs to be set to compile the package.')
            else:
                raise RuntimeError("The snpe-udo-package-generator only supports "
                                   "generation for the CPU, GPU, DSP, DSP_V68, DSP_V69 and DSP_V73 backends")

            # this defines the top-level directory using the package name and root
            package_handler = DirectoryHandler(package_info.name, package_info.root)

            # Initialize directories
            config_dir_handler = DirectoryHandler('config', package_handler.dir_abs_path)
            include_dir_handler = DirectoryHandler('include', package_handler.dir_abs_path)
            jni_dir_handler = DirectoryHandler('jni', package_handler.dir_abs_path)
            src_dir_handler = DirectoryHandler('src', jni_dir_handler.dir_abs_path)
            reg_dir_handler = DirectoryHandler('reg', src_dir_handler.dir_abs_path)
            util_src_dir_handler = DirectoryHandler('utils', src_dir_handler.dir_abs_path)
            utils_include_dir_handler = DirectoryHandler('utils', include_dir_handler.dir_abs_path)

            # setup util include and source file paths
            self.setup_util_file_paths(package_info, utils_include_dir_handler, util_src_dir_handler)
            include_dir_handler.dir_handlers.extend([utils_include_dir_handler])

            # setup backend specific file paths
            self.setup_backend_specific_file_paths(package_info, package_generator, src_dir_handler,
                                                   gen_cmakelists=False)

            # setup registration file paths
            self.setup_registration_file_paths(package_generator, package_info, reg_dir_handler)
            src_dir_handler.dir_handlers.extend([util_src_dir_handler, reg_dir_handler])
            jni_dir_handler.dir_handlers.extend([src_dir_handler])

            # setup makefile file paths
            self.setup_makefile_file_paths(package_generator, package_handler, jni_dir_handler)

            # setup backend specific implementation files
            self.setup_implementation_file_paths(package_info, src_dir_handler, package_generator)

            # setup config file paths
            for config_path in self.config_paths[package_info.name]:
                config_abs_path = os.path.abspath(config_path)
                config_file_handler = FileHandler(os.path.basename(config_path), config_dir_handler.dir_abs_path,
                                                  copy_source_location=config_abs_path)
                config_dir_handler.file_handlers.append(config_file_handler)

            # nest handlers for resource management
            package_handler.dir_handlers.extend([jni_dir_handler, include_dir_handler, config_dir_handler])
            package_generator.resource_handler = package_handler
            self.file_generators[package_info.name] = package_generator
            package_paths.append(package_info.root)
        return package_paths

    def implement_packages(self):
        """
         This class handles the implementation of the each provided package by following the following stages:
        - makefiles are generated
        - implementation files are auto-generated
        """

        log_debug_msg_as_status("Auto-generating package code")
        for i, package_generator in enumerate(self.file_generators.values()):
            package_handler = package_generator.resource_handler
            writable_content_map = package_generator.substitute_templates(self.package_infos[i])
            src_dir_handler = package_handler.dir_handlers[0].dir_handlers[0]

            if self.gen_cmakelists:
                self.__implement_cmake_package(self.package_infos[i], package_handler, src_dir_handler,
                                               package_generator)
                for file_handler in self.package_infos[i].cmakelist_handlers:
                    per_operator_args = dict()
                    template = ''
                    if 'reg' in file_handler.resource_name:
                        per_operator_args['lib_name_suffix'] = 'Reg'
                        template = package_generator.handler_to_template_map[file_handler.resource_name]
                    elif 'CPU' in file_handler.resource_name:
                        per_operator_args['lib_name_suffix'] = 'ImplCpu'
                        template = package_generator.handler_to_template_map[file_handler.resource_name]
                    else:
                        template = 'root_cmakelists_template.mako'
                    template_path = os.path.join(UDOTemplateFileReader.template_path, template)
                    template_type = UDOTemplateFileReader.TemplateFileTypes.COMMON
                    content = package_generator.substitute_template(self.package_infos[i],
                                                                    template_type,
                                                                    template_path=template_path,
                                                                    **per_operator_args)
                    writable_content_map[file_handler.resource_name] = content
            with package_handler as p:
                log_debug_msg_as_status("Implementing templates for: {} at {}",
                                        package_handler.resource_name,
                                        package_handler.dir_abs_path)
                p.set_writable_content(writable_content_map)
                p.render(force_generation=self.force_generation)
            self.package_infos[i].status = QnnPackageStatus.TEMPLATES_IMPLEMENTED

    def setup_backend_specific_file_paths(self, package_info, package_generator, src_dir_handler,
                                          gen_cmakelists=False):
        android_template = UDOTemplateFileReader.DEFAULT_TEMPLATE_FILES[
            UDOTemplateFileReader.TemplateFileTypes.MAKEFILE][3]
        if "CPU" in package_info.supported_runtimes:
            cpu_dir_handler = DirectoryHandler('CPU', src_dir_handler.dir_abs_path)
            cpu_src_dir_handler = DirectoryHandler('src', cpu_dir_handler.dir_abs_path)
            cpu_makefiles_dir_handler = DirectoryHandler('makefiles', cpu_dir_handler.dir_abs_path)
            cpu_src_ops_dir_handler = DirectoryHandler('ops', cpu_src_dir_handler.dir_abs_path)
            # create and copy utils dir
            cpu_utils_dir_handler = DirectoryHandler('utils', cpu_src_dir_handler.dir_abs_path,
                                                     copy_source_location=CUSTOM_OP_DIR)
            # copy makefiles
            # copy Application.mk
            cpu_application_file = os.path.join(SHARE_LOC_PREFIX, 'makefiles', 'Application.mk')
            cpu_application_file_handler = FileHandler('Application.mk', cpu_makefiles_dir_handler.dir_abs_path,
                                                       copy_source_location=cpu_application_file)
            cpu_makefiles_dir_handler.file_handlers.append(cpu_application_file_handler)
            # copy platform makefiles
            for compiler in ['linux-x86_64', 'qnx-aarch64']:
                makefile = os.path.join(os.path.join(SHARE_LOC_PREFIX, 'makefiles', 'CPU', 'Makefile.%s' % compiler))
                if os.path.isfile(makefile):
                    compiler_makefile_handler = FileHandler('Makefile.'+str(compiler),
                                                            cpu_makefiles_dir_handler.dir_abs_path,
                                                            copy_source_location=makefile)
                    cpu_makefiles_dir_handler.file_handlers.append(compiler_makefile_handler)

            cpu_android_make_handler = FileHandler('Android.mk', os.path.join(src_dir_handler.dir_abs_path,
                                                                              'CPU', 'makefiles'))
            cpu_makefiles_dir_handler.file_handlers.append(cpu_android_make_handler)
            package_generator.register_file_handler_with_template(cpu_android_make_handler.resource_name,
                                                                  android_template)
            op_name = package_info.operators[0].type_name + "CPU"
            package_generator.register_operator_file_handler_with_template(cpu_android_make_handler.resource_name,
                                                                           android_template, op_name)
            # copy CPU Makefile
            src_make_file_path = os.path.join(os.path.join(SHARE_LOC_PREFIX, 'makefiles', 'CPU', 'Makefile'))
            src_makefile_handler = FileHandler('Makefile', cpu_dir_handler.dir_abs_path,
                                               copy_source_location=src_make_file_path)
            cpu_dir_handler.file_handlers.append(src_makefile_handler)
            cpu_src_dir_handler.dir_handlers.extend([cpu_src_ops_dir_handler, cpu_utils_dir_handler])
            cpu_dir_handler.dir_handlers.extend([cpu_makefiles_dir_handler, cpu_src_dir_handler])
            src_dir_handler.dir_handlers.extend([cpu_dir_handler])

        if "GPU" in package_info.supported_runtimes:
            gpu_dir_handler = DirectoryHandler('GPU', src_dir_handler.dir_abs_path)
            gpu_src_dir_handler = DirectoryHandler('src', gpu_dir_handler.dir_abs_path)
            gpu_makefiles_dir_handler = DirectoryHandler('makefiles', gpu_dir_handler.dir_abs_path)
            gpu_src_ops_dir_handler = DirectoryHandler('ops', gpu_src_dir_handler.dir_abs_path)
            # create and copy utils dir
            gpu_include_dir_handler = DirectoryHandler('include', gpu_dir_handler.dir_abs_path)
            # copy source files
            gpu_custom_op_package_src = os.path.join(SHARE_LOC_PREFIX, 'CustomOp', 'GPU',
                                                     'GpuCustomOpPackage.cpp')
            gpu_src_file_handler = FileHandler('GpuCustomOpPackage.cpp', gpu_src_dir_handler.dir_abs_path,
                                               copy_source_location=gpu_custom_op_package_src)
            gpu_src_dir_handler.file_handlers.append(gpu_src_file_handler)
            gpu_custom_op_package_header = os.path.join(SHARE_LOC_PREFIX, 'CustomOp', 'GPU',
                                                        'GpuCustomOpPackage.hpp')
            gpu_header_file_handler = FileHandler('GpuCustomOpPackage.hpp', gpu_include_dir_handler.dir_abs_path,
                                                  copy_source_location=gpu_custom_op_package_header)
            gpu_include_dir_handler.file_handlers.append(gpu_header_file_handler)
            # copy makefiles
            # copy Application.mk
            gpu_application_file = os.path.join(SHARE_LOC_PREFIX, 'makefiles', 'Application.mk')
            gpu_application_handler = FileHandler('Application.mk', gpu_makefiles_dir_handler.dir_abs_path,
                                                  copy_source_location=gpu_application_file)
            gpu_makefiles_dir_handler.file_handlers.append(gpu_application_handler)
            gpu_android_make_handler = FileHandler('Android.mk', os.path.join(src_dir_handler.dir_abs_path,
                                                                              'GPU', 'makefiles'))
            gpu_makefiles_dir_handler.file_handlers.append(gpu_android_make_handler)
            package_generator.register_file_handler_with_template(gpu_android_make_handler.resource_name,
                                                                  android_template)
            op_name = package_info.operators[0].type_name + "GPU"
            package_generator.register_operator_file_handler_with_template(gpu_android_make_handler.resource_name,
                                                                           android_template, op_name)
            # copy GPU Makefile
            src_make_file_path = os.path.join(os.path.join(SHARE_LOC_PREFIX, 'makefiles', 'GPU','Makefile'))
            gpu_dir_handler.file_handlers.append(FileHandler('Makefile',
                                                             gpu_dir_handler.dir_abs_path,
                                                             copy_source_location=src_make_file_path))
            gpu_src_dir_handler.dir_handlers.extend([gpu_src_ops_dir_handler])
            gpu_dir_handler.dir_handlers.extend([gpu_makefiles_dir_handler, gpu_src_dir_handler,
                                                 gpu_include_dir_handler])
            src_dir_handler.dir_handlers.extend([gpu_dir_handler])

        if "DSP" in package_info.supported_runtimes and len(package_info.dsp_arch_types) == 0:
            runtime = 'DSP'
            dsp_dir_handler = DirectoryHandler(runtime, src_dir_handler.dir_abs_path)
            dsp_src_dir_handler = DirectoryHandler('src', dsp_dir_handler.dir_abs_path)
            dsp_src_ops_dir_handler = DirectoryHandler('ops', dsp_src_dir_handler.dir_abs_path)
            # create and copy utils dir
            dsp_include_dir_handler = DirectoryHandler('include', dsp_dir_handler.dir_abs_path)
            # copy DSP Makefile
            src_make_file_path = os.path.join(os.path.join(SHARE_LOC_PREFIX, 'makefiles', 'DSP', 'Makefile'))
            dsp_dir_handler.file_handlers.append(FileHandler('Makefile',
                                                             dsp_dir_handler.dir_abs_path,
                                                             copy_source_location=src_make_file_path))
            dsp_src_dir_handler.dir_handlers.extend([dsp_src_ops_dir_handler])
            dsp_dir_handler.dir_handlers.extend([dsp_src_dir_handler, dsp_include_dir_handler])
            src_dir_handler.dir_handlers.extend([dsp_dir_handler])

        if "DSP" in package_info.supported_runtimes and len(package_info.dsp_arch_types) != 0:
            for arch_type in package_info.dsp_arch_types:
                runtime = 'DSP_' + arch_type.upper()
                dsp_dir_handler = DirectoryHandler(runtime, src_dir_handler.dir_abs_path)
                dsp_src_dir_handler = DirectoryHandler('src', dsp_dir_handler.dir_abs_path)
                dsp_src_ops_dir_handler = DirectoryHandler('ops', dsp_src_dir_handler.dir_abs_path)
                if arch_type < 'v68':
                    dsp_include_dir_handler = DirectoryHandler('include', dsp_dir_handler.dir_abs_path)
                    src_make_file_path = os.path.join(os.path.join(SHARE_LOC_PREFIX, 'makefiles',
                                                                   'DSP', 'Makefile'))
                    dsp_makefile_handler = FileHandler('Makefile', dsp_dir_handler.dir_abs_path,
                                                       copy_source_location=src_make_file_path)
                    dsp_dir_handler.dir_handlers.extend([dsp_include_dir_handler])
                else:
                    # copy DSP Makefile
                    src_make_file_path = os.path.join(os.path.join(SHARE_LOC_PREFIX, 'makefiles',
                                                                   'HTP', 'Makefile'))
                    dsp_makefile_handler = FileHandler('Makefile', dsp_dir_handler.dir_abs_path,
                                                       copy_source_location=src_make_file_path)

                    with open(src_make_file_path, 'r') as file:
                        data = file.read()
                        data = data.replace("-lQnnHtpPrepare",
                                            "-L$(SNPE_ROOT)/lib/aarch64-android -lSnpeHtpPrepare")
                        data = data.replace("-lQnnHtp", "-lSNPE")
                    with open(src_make_file_path, 'w') as file:
                        file.write(data)
                dsp_dir_handler.file_handlers.append(dsp_makefile_handler)

                dsp_src_dir_handler.dir_handlers.extend([dsp_src_ops_dir_handler])
                dsp_dir_handler.dir_handlers.extend([dsp_src_dir_handler])
                src_dir_handler.dir_handlers.extend([dsp_dir_handler])

    def setup_implementation_file_paths(self, package_info, src_dir_handler, package_generator):
        for operator in package_info.operators:
            op_name = operator.type_name
            op_runtimes = list(
                map(title_case,
                    [SnpeUdoConstants.snpe_udo_coretypes[x] for x in operator.core_types]))
            i = 0
            for runtime in op_runtimes:
                dsp_detected = False
                gpu_detected = False
                source_template = None
                interface_template = None
                if runtime.lower() == 'cpu':
                    source_template = UDOTemplateFileReader.DEFAULT_TEMPLATE_FILES[
                        UDOTemplateFileReader.TemplateFileTypes.CPU][0]
                    interface_template = UDOTemplateFileReader.DEFAULT_TEMPLATE_FILES[
                        UDOTemplateFileReader.TemplateFileTypes.CPU][1]
                if runtime.lower() == 'gpu':
                    gpu_detected = True
                    source_template = UDOTemplateFileReader.DEFAULT_TEMPLATE_FILES[
                        UDOTemplateFileReader.TemplateFileTypes.GPU][0]
                    interface_template = UDOTemplateFileReader.DEFAULT_TEMPLATE_FILES[
                        UDOTemplateFileReader.TemplateFileTypes.GPU][1]
                if runtime.lower() == 'dsp':
                    if len(package_info.dsp_arch_types) == 0 or operator.dsp_arch_types[i].lower() < 'v68':
                        if len(package_info.dsp_arch_types):
                            runtime = 'DSP_' + package_info.dsp_arch_types[i].upper()
                            i += 1
                        dsp_detected = True
                        source_template = UDOTemplateFileReader.DEFAULT_TEMPLATE_FILES[
                            UDOTemplateFileReader.TemplateFileTypes.DSP][0]
                        interface_template = UDOTemplateFileReader.DEFAULT_TEMPLATE_FILES[
                            UDOTemplateFileReader.TemplateFileTypes.DSP][1]
                    else:
                        runtime = 'DSP_' + package_info.dsp_arch_types[i].upper()
                        i += 1
                        source_template = UDOTemplateFileReader.DEFAULT_TEMPLATE_FILES[
                            UDOTemplateFileReader.TemplateFileTypes.HTP][0]
                        interface_template = UDOTemplateFileReader.DEFAULT_TEMPLATE_FILES[
                            UDOTemplateFileReader.TemplateFileTypes.HTP][1]
                runtime = runtime.upper()
                src_file_handler = FileHandler('{}.cpp'.format(op_name), os.path.join(src_dir_handler.dir_abs_path,
                                                                                      runtime, 'src', 'ops'))
                interface_file_handler = FileHandler('{}Interface.cpp'.format(package_info.name),
                                                     os.path.join(src_dir_handler.dir_abs_path, runtime, 'src'))
                src_dir_handler.file_handlers.extend([src_file_handler, interface_file_handler])

                # register backend source template substitution
                package_generator.register_file_handler_with_template(src_file_handler.resource_name,
                                                                      source_template)
                # register backend interface template substitution
                package_generator.register_file_handler_with_template(interface_file_handler.resource_name,
                                                                      interface_template)

                package_generator.register_operator_file_handler_with_template(src_file_handler.resource_name,
                                                                               source_template, op_name)
                if gpu_detected:
                    operation_file_handler = FileHandler('Operation.hpp', os.path.join(src_dir_handler.dir_abs_path,
                                                                                       'GPU', 'include'))
                    src_dir_handler.file_handlers.append(operation_file_handler)
                    # register gpu operation template substitution
                    package_generator.register_file_handler_with_template(
                        operation_file_handler.resource_name,
                        UDOTemplateFileReader.DEFAULT_TEMPLATE_FILES[
                            UDOTemplateFileReader.TemplateFileTypes.GPU][2])
                elif dsp_detected:
                    header_filer_handler = FileHandler('DspOps.hpp', os.path.join(src_dir_handler.dir_abs_path,
                                                                                  runtime, 'include'))
                    src_dir_handler.file_handlers.append(header_filer_handler)
                    # register dsp header template substitution
                    package_generator.register_file_handler_with_template(
                        header_filer_handler.resource_name,
                        UDOTemplateFileReader.DEFAULT_TEMPLATE_FILES[
                            UDOTemplateFileReader.TemplateFileTypes.DSP][2])
            operator.core_types = list(set(operator.core_types))

    def __implement_cmake_package(self, package_info, package_handler, src_dir_handler, package_generator):
        package_info.cmakelists = True
        # Preparing output file paths
        src_root = src_dir_handler.dir_abs_path
        cmakelist_handlers = []
        root_cmakelist_handler = FileHandler('CMakeLists.txt', package_handler.dir_abs_path)
        reg_cmakelist_handler = FileHandler('CMakeLists.txt', os.path.join(src_root, 'reg'))
        package_handler.file_handlers.append(root_cmakelist_handler)
        src_dir_handler.file_handlers.append(reg_cmakelist_handler)
        cmakelist_handlers.extend([root_cmakelist_handler, reg_cmakelist_handler])

        package_generator.register_file_handler_with_template(
            root_cmakelist_handler.resource_name,
            UDOTemplateFileReader.DEFAULT_TEMPLATE_FILES[
                UDOTemplateFileReader.TemplateFileTypes.COMMON][0])

        package_generator.register_file_handler_with_template(
            reg_cmakelist_handler.resource_name,
            UDOTemplateFileReader.DEFAULT_TEMPLATE_FILES[
                UDOTemplateFileReader.TemplateFileTypes.COMMON][1])

        output_file_paths = [os.path.join(package_info.root, 'CMakeLists.txt'),
                             os.path.join(src_root, 'reg', 'CMakeLists.txt')]

        src_cmakelist_handlers = []
        for runtime in package_info.supported_runtimes:
            if str(runtime).upper() == 'CPU':
                src_cmakelist_handlers.extend([FileHandler('CMakeLists.txt', os.path.join(src_root, str(runtime).upper()))])
                output_file_paths.append(os.path.join(src_root, str(runtime).upper(), 'CMakeLists.txt'))

        for file_handler in src_cmakelist_handlers:
            package_generator.register_file_handler_with_template(
                file_handler.resource_name,
                UDOTemplateFileReader.DEFAULT_TEMPLATE_FILES[
                    UDOTemplateFileReader.TemplateFileTypes.COMMON][1])
        src_dir_handler.file_handlers.extend(src_cmakelist_handlers)
        cmakelist_handlers.extend(src_cmakelist_handlers)
        package_info.cmakelist_handlers = cmakelist_handlers

        log_debug("CMakeLists.txt generation complete")

    def setup_util_file_paths(self, package_info, utils_include_dir_handler, util_src_dir_handler):
        macro_header = os.path.join(package_info.SNPE_UDO_ROOT, 'utils', 'UdoMacros.hpp')
        macro_file_handler = FileHandler('UdoMacros.hpp', utils_include_dir_handler.dir_abs_path,
                                         copy_source_location=macro_header)
        utils_include_dir_handler.file_handlers.append(macro_file_handler)
        util_header = os.path.join(package_info.SNPE_UDO_ROOT, 'utils', 'UdoUtil.hpp')
        util_file_handler = FileHandler('UdoUtil.hpp', utils_include_dir_handler.dir_abs_path,
                                        copy_source_location=util_header)
        utils_include_dir_handler.file_handlers.append(util_file_handler)
        op_def_header = os.path.join(package_info.SNPE_UDO_ROOT, 'utils', 'IUdoOpDefinition.hpp')
        op_def_file_handler = FileHandler('IUdoOpDefinition.hpp', utils_include_dir_handler.dir_abs_path,
                                          copy_source_location=op_def_header)
        utils_include_dir_handler.file_handlers.append(op_def_file_handler)

        # copy util src files
        util_source = os.path.join(package_info.SNPE_UDO_ROOT, 'utils', 'UdoUtil.cpp')
        util_source_file_handler = FileHandler('UdoUtil.cpp', util_src_dir_handler.dir_abs_path,
                                               copy_source_location=util_source)
        util_src_dir_handler.file_handlers.append(util_source_file_handler)

    def setup_registration_file_paths(self, package_generator, package_info, reg_dir_handler):
        reg_dir_handler.file_handlers.append(FileHandler('Makefile', reg_dir_handler.dir_abs_path))
        reg_dir_handler.file_handlers.append(FileHandler('{}RegLib.cpp'.format(package_info.name),
                                                         reg_dir_handler.dir_abs_path))
        # register reg makefile template substitution
        package_generator.register_file_handler_with_template(
            reg_dir_handler.file_handlers[0].resource_name,
            UDOTemplateFileReader.DEFAULT_TEMPLATE_FILES[
                UDOTemplateFileReader.TemplateFileTypes.MAKEFILE][0])
        # register reg lib template substitution
        package_generator.register_file_handler_with_template(
            reg_dir_handler.file_handlers[1].resource_name,
            UDOTemplateFileReader.DEFAULT_TEMPLATE_FILES[
                UDOTemplateFileReader.TemplateFileTypes.REG][0])

    def setup_makefile_file_paths(self, package_generator, package_handler, jni_dir_handler):
        # copy application make file
        application_makefile = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'makefiles',
                                            'Application.mk')
        jni_dir_handler.file_handlers.append(FileHandler('Application.mk', jni_dir_handler.dir_abs_path,
                                                         copy_source_location=application_makefile))
        # setup android makefile
        android_makefile_file = FileHandler('Android.mk', jni_dir_handler.dir_abs_path)
        jni_dir_handler.file_handlers.append(android_makefile_file)
        package_generator.register_file_handler_with_template(
            android_makefile_file.resource_name,
            UDOTemplateFileReader.DEFAULT_TEMPLATE_FILES[
                UDOTemplateFileReader.TemplateFileTypes.MAKEFILE][1])

        # copy common makefile
        common_makefile = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'makefiles', 'common.mk')
        package_handler.file_handlers.append(FileHandler('common.mk', package_handler.dir_abs_path,
                                                         copy_source_location=common_makefile))
        # setup root Makefile
        makefile_handler = FileHandler('Makefile', package_handler.dir_abs_path)
        package_generator.register_file_handler_with_template(
            makefile_handler.resource_name,
            UDOTemplateFileReader.DEFAULT_TEMPLATE_FILES[
                UDOTemplateFileReader.TemplateFileTypes.MAKEFILE][2])
        package_handler.file_handlers.extend([makefile_handler])


class UdoPackageInfo:
    """
    UdoPackageInfo contains information gleaned from the user provided config that will constitute a package.
    It is freely editable, meaning users can add and remove information as needed. It is also the main reference point
    for constructing a package.
    """

    def __init__(self, package_name, package_root, package_core_types, package_dsp_arch_types,
                 operators=None, snpe_udo_root=""):
        self.name = package_name
        self.root = os.path.join(package_root, package_name)
        self.core_types = package_core_types
        self.dsp_arch_types = package_dsp_arch_types
        self.operators = operators if operators else list()
        self.SNPE_UDO_ROOT = snpe_udo_root

    def get(self, attr, default=None):
        return getattr(self, attr, default)

    def add_operators(self, operators):
        for operator in operators:
            if isinstance(operator, Operator):
                self.operators.append(operator)
            else:
                raise TypeError('Operator must be a valid object of type {}'.format(
                    Operator.__class__.__name__))

    @staticmethod
    def from_dict(udo_package_dict):
        package_name = udo_package_dict.get("UDO_PACKAGE_NAME")
        root = os.path.abspath(udo_package_dict.get("UDO_PACKAGE_PATH", os.getcwd()))
        operators_list = udo_package_dict.get('Operators', list())
        operators = list(map(Operator.from_dict, operators_list))
        core_types = udo_package_dict.get("UDO_PACKAGE_CORETYPES",
                                          set(chain.from_iterable((operator_dict.get("core_types")
                                                                   for operator_dict in
                                                                   operators_list))))
        dsp_arch_types = udo_package_dict.get("UDO_PACKAGE_DSP_ARCH_TYPES",
                                              set(chain.from_iterable(
                                                  (operator_dict.get("dsp_arch_types", [])
                                                   for operator_dict in operators_list))))
        snpe_udo_root = os.environ.get('SNPE_UDO_ROOT', udo_package_dict.get('SNPE_UDO_ROOT', None))
        new_udo_package_info = UdoPackageInfo(package_name, root, list(core_types),
                                              list(dsp_arch_types), snpe_udo_root=snpe_udo_root)
        new_udo_package_info.add_operators(operators)

        return new_udo_package_info

    def __getattr__(self, item):
        return self.__getattribute__(item)

    def __getitem__(self, item):
        return getattr(self, item)

    @property
    def value_info(self):
        return {'package_name': self.root,
                'package_core_types': self.core_types,
                'operators': self.operators}


class UdoPackage:
    """
    The UdoPackage object is the core class used by the UdoGenerator and UdoFileGenerator objects. It contains a
    description of the package's operations, a catalog of op_names, their respective core_types and supported
    calculation types. Some of its members are intended to be set only once when a package is added,
    in contrast to the UdoPackageInfo. A package is expected to be created from a well defined package info only.

    """
    package_info = property_type('package_info', UdoPackageInfo)
    root = property_type('root', str)
    op_catalog_info = property_type('op_catalog_info', list)
    core_types = property_type('core_types', list)
    supported_runtimes = property_type('supported_runtimes',list)
    calculation_types = property_type('calculation_types', list)

    def __init__(self, package_name):
        self.name = package_name

    def add_package_info(self, udo_package_info):
        """
        Add a package info to a package which formalizes and makes sure that relevant fields are mapped correctly to the
        SNPE_UDO API.
        :param udo_package_info: The information needed to define a package object.
        """
        self.package_info = udo_package_info
        self.name = self.package_info.get("udo_package_name", self.name)
        self.root = self.package_info.root
        self.op_catalog_info = [(str(operator['type_name']), operator['core_types']) for operator in
                                self.package_info.operators]
        self.core_types = get_internal_core_types(udo_package_info.core_types)
        self.dsp_arch_types = self.package_info.dsp_arch_types
        self.supported_runtimes = udo_package_info.core_types
        self.calculation_types = list(
            map(lambda x: SnpeUdoConstants.SNPE_CALCULATION_TYPES[x], udo_package_info.core_types))
