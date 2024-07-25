# ==============================================================================
#
#  Copyright (c) 2020-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from collections import defaultdict
from itertools import chain

import qti.aisw.op_package_generator.translator.op_def_translator as xml_package_translator
from qti.aisw.op_package_generator.core import *
from qti.aisw.converters.common.utils import io_utils

# global variables
SDK_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', '..', '..', '..'))
SHARE_LOC_PREFIX = os.path.join(SDK_ROOT, 'share', 'QNN', 'OpPackageGenerator')

CUSTOM_OP_DIR = os.path.join(SHARE_LOC_PREFIX, 'CustomOp')
KNOWN_BACKENDS = ['HTP', 'HTPMCP', 'CPU', 'DSP', 'GPU', 'HTP_FP16']

# makefile is in a different location in the src and SDK.
try:
    MAKEFILE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'makefiles')
    if not os.path.exists(MAKEFILE_PATH):
        raise IOError
except IOError:
    MAKEFILE_PATH = os.path.join(SHARE_LOC_PREFIX, 'makefiles')

# ------------------------------------------------------------------------------
#   Qnn Op Package Generator Classes
# ------------------------------------------------------------------------------
class QnnOpPackageGenerator:
    SCHEMA_PATH = os.path.abspath(os.path.dirname(xml_package_translator.__file__))
    SCHEMA = os.path.join(SCHEMA_PATH, xml_package_translator.OpDefTranslator.DEFAULT_SCHEMA)
    force_generation = False
    converter_op_package = False

    """
    This class is the main point of entry for the package-generator. It handles the parsing of the user provided
    config and creates a qnn package object. It then sets up file paths according to information gathered in the
    package in the parsing step. Finally, it implements the package by auto-generating header and source files using a
    QnnFileGenerator object.

    :package_infos: This is a list of all qnn packages infos registered in a single generator instance
    :file_generator: This object handles the auto-generation of code using Mako templates.
    """
    package_infos = aggregate_property('package_infos', QnnPackageInfo)

    def __init__(self, package_infos=None):
        self.__translator = xml_package_translator.OpDefTranslator
        self.file_generators = dict()
        self.config_paths = dict()  # list in order of packages seen
        if package_infos:
            self.package_infos = package_infos

    def get_package_info_by_name(self, package_name: str) -> Optional[QnnPackageInfo]:
        for package in self._package_infos:
            if package.name == package_name:
                return package

    def register_package_info(self, package_info: QnnPackageInfo, meld_if_present=True,
                              config_paths: Optional[List[str]] = None):
        """
        Registers the package info with the generator instance's dictionary of package infos.

        :param package_info: A QnnPackageInfo objects
        :param meld_if_present: A boolean flag indicating how duplicate package infos should be handled. Note that if
                                this flag is set, then the argument package info and the current instance are melded.
                                If false, then a new package info entry is always created.
        :param config_paths: An optional list of config paths belonging to this package. If the package info
                             is a duplicate, then the argument config path is added to an existing entry.
        :raises: Errors from meld if meld_if_present is set, otherwise noexcept.
        """
        if not hasattr(self, 'package_infos'):
            self.package_infos = [package_info]
            self.config_paths[package_info.name] = set()
        else:
            if self.get_package_info_by_name(package_info.name) is None:
                self.package_infos.append(package_info)
                self.config_paths[package_info.name] = set()
            else:
                log_debug("Package info: {} has already been registered".format(package_info.name))
                existing_package_info = self.get_package_info_by_name(package_info.name)
                if meld_if_present and existing_package_info is not None:
                    log_debug("Attempting to meld package_info(s)")
                    existing_package_info.meld(package_info)
                    log_debug("Meld complete!")
        if config_paths is not None:
            for config_path in config_paths:
                self.config_paths[package_info.name].add(config_path)

    def parse_config(self, config_paths: List[str], output_path: Optional[str] = None, converter_op_package = False):
        """
        Parses a user provided json config into a qnn package object. The config is expected to contain information
        about a user's operation, as well as additional fields about the package.
        :param config_paths: The file path to the user's json config file, which must be a valid XML file.
        :param output_path: The output path for where the package will be saved. Defaults to the first config path base
               directory in config_paths.
        """

        self.converter_op_package = converter_op_package
        # check output_path
        if output_path is None:
            output_path = os.path.abspath(os.path.dirname(config_paths[0]))
            log_debug("Output path not set for the command line, using path of config file: {}: ",
                      output_path,
                      config_paths[0])
        else:
            output_path = os.path.abspath(output_path)
            if not os.path.exists(output_path):
                log_debug("Creating output directory: {}".format(output_path))
                os.makedirs(output_path)
            io_utils.check_validity(output_path, is_directory=True) # sanity check directory

        try:
            from qti.aisw.converters.snpe_backend.custom_ops.helpers.json_to_xml import json_to_xml
        except:
            pass
        for config_path in config_paths:
            extension = config_path.split('.')[-1]
            # if json config is passed, convert to xml config
            if 'json' in extension:
                xml_path = '.'
                config_path = json_to_xml(config_path, xml_path)[0]
            # check config path
            io_utils.check_validity(config_path, extensions=[".xml"])

            # Import config and parse using XML translator
            log_debug_msg_as_status("Parsing config: {}", config_path)
            xml_instance = self.__translator(config_path, self.SCHEMA)
            op_collection = xml_instance.translate_ops()

            op_package_collection = self.__get_qnn_collection_from_op_def_collection(op_collection)
            # now resolve packages into individual packages per backend, per package name
            # TODO: version and domain should be cached before this step
            self.create_package_info(op_package_collection, output_path, config_path)

            log_debug("Config parsed.")

    def create_package_info(self, op_package_collection, output_path, config_path):
        for package_name, package in op_package_collection.items():
            for backend, operators in package.items():
                # here there are two cases that must be handled in the creation of Qnn package info
                # case 1: QNN backends without converter_op_package option
                #         - only Qnn package info is created for qnn backends
                # case 2: QNN backends with converter_op_package option
                #         - Converter op package info is created in addition to the Qnn package info
                package_info = QnnPackageInfo(package_name, output_path, backend, operators)
                self.register_package_info(package_info, config_paths=[config_path])
                if self.converter_op_package:
                    # case 2 - QNN backends with converter_op_package option
                    converter_package_name = package_name + '_Converter_Op_Package'
                    converter_package_info = QnnPackageInfo(converter_package_name, output_path, backend, operators)
                    self.register_package_info(converter_package_info, config_paths=[config_path])

    def setup_file_paths(self, force_generation=False, gen_cmakelists=False):
        """
         This sets up file paths and makes the directory structure for the package. It creates handlers which hold the
         top level directory structure which src, config, src/ops and include. The aforementioned directories contain
         file handlers for each file to be created or copied.

         Note the handlers are nested in a tree-like directory structure i.e:
         package_dir_handler -> src_dir_handler ->src_dir/ops_handler -> op_file_handler(s)
                                                ->utils_dir_handler
                                android_makefile_handler
                                application_makefile_handler
                                include_dir_handler
                                makefile
                                config - > config_file_handler(s)
        :param force_generation:  if set to true, any package directory in the generator instance will be overwritten
                                  if it exists.
                                  if false, only new directories will be created. Note that existing files and
                                  directories will be appended to.
        :return: the package paths that have been successfully set up
        :raises: Indirect errors from IO Handler API calls
        """
        self.force_generation = force_generation
        package_paths = []

        for i, package_info in enumerate(self.package_infos):
            package_generator = QnnOpPackageCodeGenerator()  # each package must have its own code generator object

            if package_info.name in self.file_generators:
                if self.file_generators[package_info.name]. \
                        resource_handler.destination == package_info.root:
                    log_warning("Attempting to create package with duplicate name: "
                                "{} for backend: {} in the same location: {}".format(package_info.name,
                                                                                     package_info.backend,
                                                                                     package_info.root))
                    package_info.name = package_info.name + package_info.backend
                    log_info("Package name will be changed to {}".format(package_info.name))

            # this defines the top-level directory using the package name and root
            package_handler = DirectoryHandler(package_info.name, package_info.root)
            if package_info.backend == "HTP" or package_info.backend == "HTPMCP" or package_info.backend == "HTP_FP16" or \
                    package_info.backend == "CPU" or package_info.backend == "DSP" or package_info.backend == "GPU" or \
                    package_info.backend == "AIC":
                if package_info.backend == "HTP" or package_info.backend == "HTPMCP" or package_info.backend == "HTP_FP16" \
                        or package_info.backend == "DSP" and not os.getenv("HEXAGON_SDK_ROOT", None):
                    log_warning('HTP/DSP Operation detected but HEXAGON_SDK_ROOT is not set. Please '
                                'note HEXAGON_SDK_ROOT needs to be set to compile the package.')
            else:
                raise RuntimeError("The qnn-op-package-generator only supports "
                                   "generation for the HTP, CPU and DSP backends")

            if "_Converter_Op_Package" in package_info.name and self.converter_op_package:
                converter_templates = QnnTemplateFileReader.get_template_type_by_backend("CONVERTER")

                converter_package_handler = DirectoryHandler(package_info.name, package_info.root)
                converter_op_package_dir_handler = DirectoryHandler('ConverterOpPackage', converter_package_handler.dir_abs_path)
                converter_op_package_file_handler = FileHandler('ConverterOpPackage.cpp',
                                                                converter_op_package_dir_handler.dir_abs_path)
                converter_op_package_dir_handler.file_handlers.append(converter_op_package_file_handler)

                package_generator.register_operator_file_handler_with_template(
                    converter_op_package_file_handler.resource_name,
                    converter_templates[0], package_info.operators[0].type_name)

                converter_package_handler.dir_handlers.append(converter_op_package_dir_handler)

                # get the makefile
                makefile_handler = FileHandler('Makefile', converter_package_handler.dir_abs_path,
                                               copy_source_location=os.path.join(MAKEFILE_PATH,
                                                                                 'CONVERTER',
                                                                                 'Makefile'))
                converter_package_handler.file_handlers.extend([makefile_handler])  # add makefile

                make_file_src_location = os.path.join(MAKEFILE_PATH, 'CONVERTER', 'Makefile.linux-x86_64')
                converter_op_package_dir_handler.file_handlers.append(FileHandler('Makefile.linux-x86_64',
                                                                                  converter_op_package_dir_handler.dir_abs_path,
                                                                                  copy_source_location=make_file_src_location))

                package_generator.resource_handler = converter_package_handler

                self.file_generators[package_info.name] = package_generator
                package_paths.append(package_info.root)
                # when the converter_op_package option is set for qnn backends, qnn package info must to be setup and
                # generated in addition to the converter package info. this is handled by the if block
                # when the converter_op_package is set for AIC backend and multiple converter package infos are present,
                # next package info must be setup and generated. this is handled by the elif block
                # when all the package infos are setup, package paths are returned
                if package_info.backend != 'AIC' or i != len(self.package_infos) - 1:
                    # handles qnn and aic backends with multiple package infos
                    continue
                # returns the package paths when all the package infos are processed for aic backend
                return package_paths

            # get templates for qnn backends
            templates = QnnTemplateFileReader.get_template_type_by_backend(package_info.backend)

            # Initialize directories
            src_dir_handler = DirectoryHandler('src', package_handler.dir_abs_path)
            ops_src_dir_handler = DirectoryHandler('ops', src_dir_handler.dir_abs_path)
            include_dir_handler = DirectoryHandler('include', package_handler.dir_abs_path)
            config_dir_handler = DirectoryHandler('config', package_handler.dir_abs_path)

            makefile_handler = self.get_backend_makefiles_handler(package_handler, package_info)
            interface_file_handler = FileHandler('{}Interface.cpp'.format(package_info.name),
                                                 src_dir_handler.dir_abs_path)

            # Implementation file handler setup
            # setup file handlers per operation
            for operator in package_info.operators:
                op_name = operator.type_name
                op_src_file_handler = FileHandler('{}.cpp'.format(op_name),
                                                  ops_src_dir_handler.dir_abs_path)
                ops_src_dir_handler.file_handlers.append(op_src_file_handler)

                # set source file which must exist for all backends
                package_generator.register_operator_file_handler_with_template(
                    op_src_file_handler.resource_name,
                    templates[0], op_name)

            # setup config files to be copied
            for config_path in self.config_paths[package_info.name]:
                config_dir_handler.file_handlers.append(FileHandler(os.path.basename(config_path),
                                                                    config_dir_handler.dir_abs_path,
                                                                    copy_source_location=os.path.abspath(
                                                                        config_path)))

            # nest handlers for resource management
            src_dir_handler.file_handlers.extend([interface_file_handler])
            src_dir_handler.dir_handlers.extend([ops_src_dir_handler])  # add operation source files
            package_handler.dir_handlers.extend([src_dir_handler,
                                                 include_dir_handler,
                                                 config_dir_handler])
            if not gen_cmakelists:
                package_handler.file_handlers.extend([makefile_handler])  # add makefile
            else:
                cmakelists_handler = FileHandler('CMakelists.txt', package_handler.dir_abs_path)
                package_handler.file_handlers.extend([cmakelists_handler])
                package_generator.register_file_handler_with_template(
                    cmakelists_handler.resource_name,
                    QnnTemplateFileReader.DEFAULT_TEMPLATE_FILES[
                        QnnTemplateFileReader.TemplateFileTypes.COMMON][0])

            package_generator.resource_handler = package_handler

            # register remaining handlers that need template substitution
            package_generator.register_file_handler_with_template(
                interface_file_handler.resource_name,
                templates[1])

            self.file_generators[package_info.name] = package_generator

            # backend specific registration
            self.setup_backend_specific_file_paths(package_info, package_handler, gen_cmakelists)

            package_paths.append(package_info.root)
        return package_paths

    def setup_backend_specific_file_paths(self, package_info, package_handler, gen_cmakelists=False):
        # Only CPU and GPU need android makefiles
        if package_info.backend == "CPU" or package_info.backend == "GPU":
            make_dir_handler = DirectoryHandler('makefiles', package_handler.dir_abs_path)
            android_mk_file_handler = FileHandler('Android.mk', make_dir_handler.dir_abs_path)
            application_mk_file_handler = FileHandler('Application.mk',
                                                      make_dir_handler.dir_abs_path,
                                                      copy_source_location=os.path.join(MAKEFILE_PATH,
                                                                                        "Application.mk"))
            make_dir_handler.dir_handlers.extend([android_mk_file_handler,
                                                  application_mk_file_handler])

            self.file_generators[package_info.name].register_file_handler_with_template(
                android_mk_file_handler.resource_name,
                QnnTemplateFileReader.DEFAULT_TEMPLATE_FILES[
                    QnnTemplateFileReader.TemplateFileTypes.MAKEFILE][0])

            # Add other backend specific files such as utils, additional makefiles here
            src_dir_handler = package_handler.get_handler('src')
            if package_info.backend == "CPU":
                makefile_handler = FileHandler('Makefile.linux-x86_64',
                                               make_dir_handler.dir_abs_path,
                                               copy_source_location=os.path.join(MAKEFILE_PATH,
                                                                                 package_info.backend.upper(),
                                                                                 'Makefile.linux-x86_64'))
                make_dir_handler.file_handlers.extend([makefile_handler])

                if os.path.isfile(os.path.join(MAKEFILE_PATH, package_info.backend.upper(), 'Makefile.qnx-aarch64')):
                    makefile_handler = FileHandler('Makefile.qnx-aarch64',
                                                   make_dir_handler.dir_abs_path,
                                                   copy_source_location=os.path.join(MAKEFILE_PATH,
                                                                                     package_info.backend.upper(),
                                                                                     'Makefile.qnx-aarch64'))
                    make_dir_handler.file_handlers.extend([makefile_handler])

                if os.path.isfile(os.path.join(MAKEFILE_PATH, package_info.backend.upper(), 'Makefile.qos224-aarch64')):
                    makefile_handler = FileHandler('Makefile.qos224-aarch64',
                                                   make_dir_handler.dir_abs_path,
                                                   copy_source_location=os.path.join(MAKEFILE_PATH,
                                                                                     package_info.backend.upper(),
                                                                                     'Makefile.qos224-aarch64'))
                    make_dir_handler.file_handlers.extend([makefile_handler])

                cpu_customop_package_src_handler = FileHandler('CpuCustomOpPackage.cpp',
                                                               src_dir_handler.dir_abs_path,
                                                               copy_source_location=os.path.join(CUSTOM_OP_DIR,
                                                                                                 package_info.backend.upper(),
                                                                                                 'CpuCustomOpPackage.cpp'))
                src_dir_handler.dir_handlers.extend([cpu_customop_package_src_handler])

                util_src_dir_handler = DirectoryHandler('utils', src_dir_handler.dir_abs_path,
                                                        copy_source_location=CUSTOM_OP_DIR)

                src_dir_handler.dir_handlers.extend([util_src_dir_handler])

            elif package_info.backend == "GPU":
                # needs Operation file
                templates = QnnTemplateFileReader.get_template_type_by_backend(package_info.backend)
                include_dir_handler = package_handler.get_handler('include')
                operation_file_handler = FileHandler('Operation.hpp', include_dir_handler.dir_abs_path)
                self.file_generators[package_info.name].register_file_handler_with_template(
                    operation_file_handler.resource_name, templates[2])
                include_dir_handler.dir_handlers.extend([operation_file_handler])

                # CustomOpPackage files
                gpu_opPkg_header = FileHandler('GpuCustomOpPackage.hpp',
                                               include_dir_handler.dir_abs_path,
                                               copy_source_location=os.path.join(CUSTOM_OP_DIR,
                                                                                 package_info.backend.upper(),
                                                                                 'GpuCustomOpPackage.hpp'))
                include_dir_handler.file_handlers.extend([gpu_opPkg_header])

                gpu_opPkg_source = FileHandler('GpuCustomOpPackage.cpp',
                                               src_dir_handler.dir_abs_path,
                                               copy_source_location=os.path.join(CUSTOM_OP_DIR,
                                                                                 package_info.backend.upper(),
                                                                                 'GpuCustomOpPackage.cpp'))
                src_dir_handler.file_handlers.extend([gpu_opPkg_source])

            if not gen_cmakelists:
                package_handler.dir_handlers.append(make_dir_handler)

        elif package_info.backend == "DSP": # DSP backend need ops header.
            templates = QnnTemplateFileReader.get_template_type_by_backend(package_info.backend)
            include_dir_handler = package_handler.get_handler('include')
            header_file_handler = FileHandler('DspOps.hpp', include_dir_handler.dir_abs_path)
            self.file_generators[package_info.name].register_file_handler_with_template(
                header_file_handler.resource_name, templates[2])
            include_dir_handler.dir_handlers.extend([header_file_handler])

        elif package_info.backend == "AIC":
            templates = QnnTemplateFileReader.get_template_type_by_backend(package_info.backend)
            src_dir_handler = package_handler.get_handler("src")
            util_src_dir_handler = DirectoryHandler("utils", src_dir_handler.dir_abs_path)
            src_dir_handler.dir_handlers.append(util_src_dir_handler)

            # yaml config
            yaml_file_handler = FileHandler("QnnAicOpPackageConfig.yaml",
                                            package_handler.dir_abs_path)
            self.file_generators[package_info.name].register_file_handler_with_template(
                yaml_file_handler.resource_name, templates[3])
            package_handler.file_handlers.append(yaml_file_handler)

            for operator in package_info.operators:
                # util functions
                functions_file_handler = FileHandler("{}Utils.cpp".format(operator.type_name),
                                                     util_src_dir_handler.dir_abs_path)
                self.file_generators[package_info.name].register_operator_file_handler_with_template(
                    functions_file_handler.resource_name, templates[2], operator.type_name)
                util_src_dir_handler.file_handlers.append(functions_file_handler)

    def implement_packages(self):
        """
         This class handles the implementation of the each provided package by following the following stages:
        - makefiles are copied
        - implementation files are auto-generated
        """

        log_debug_msg_as_status("Auto-generating package code")
        for i, package_generator in enumerate(self.file_generators.values()):
            package_handler = package_generator.resource_handler
            writable_content_map = package_generator.substitute_templates(self.package_infos[i])
            with package_handler as p:
                log_debug_msg_as_status("Implementing templates for: {} at {}",
                                        package_handler.resource_name,
                                        package_handler.dir_abs_path)
                p.set_writable_content(writable_content_map)
                p.render(force_generation=self.force_generation)
            self.package_infos[i].status = QnnPackageStatus.TEMPLATES_IMPLEMENTED

    @classmethod
    def __get_qnn_collection_from_op_def_collection(cls, op_def_collection):
        """
        This transforms an op_def_collection of Opdef objects into a QnnPackage collection
        consisting of Operator instances
        :param op_def_collection: A collection of opDefs categorized by backend
        :return: The newly created op_def_collection
        """
        supported_backends = [key for key, value in op_def_collection.supported_ops.items() if
                              value]

        # create op_collection for easy look-up
        op_package_collection = QnnPackageCollection()
        for backend in supported_backends:
            backend_op_defs = op_def_collection.get_backend_op_defs(backend).values()
            for op_def in backend_op_defs:
                resolved_op_def = QnnPackageInfo.get_operator_from_translator_op_def(op_def)
                op_package_collection[resolved_op_def.package_name] = {backend: [resolved_op_def]}
        return op_package_collection

    def generation_is_complete(self):
        """
        Performs a final check of the package status to ensure it is in the right stage. if the package status is not
        IMPLEMENTED then a debug message will be returned, in addition to boolean false. Note this is mostly useful
        for testing purposes
        :return: returns True if the package is in the right stage
        """
        for package in self.package_infos:
            if package.status == QnnPackageStatus.STRUCTURE_IS_SET:
                log_debug(
                    "Package files for {} have been setup but code could not be auto-generated from templates",
                    package.name)
                return False
            elif package.status == QnnPackageStatus.NOT_GENERATED:
                log_debug("Package files could not been created for package: ", package.name)
                return False
            log_info("Code generation is complete for package: {} at {}", package.name,
                     package.root)
            log_debug("All packages files have been created at: {}",
                      os.path.join(os.path.abspath(package.root), package.name))
            package.status = QnnPackageStatus.PACKAGE_CAN_COMPILE

        return True

    def get_backend_makefiles_handler(self, package_handler, package_info):
        """"
        Returns a FileHandler for the backend makefiles
        :param package_handler: package handler
        :param package_info: package info
        :return: returns makefile handler
        """
        # Makefiles are identical for HTP and HTP FP16 backends
        if package_info.backend == "HTP" or package_info.backend == "HTP_FP16":
            return FileHandler('Makefile', package_handler.dir_abs_path,
                               copy_source_location=os.path.join(MAKEFILE_PATH,
                                                                 'HTP',
                                                                 'Makefile'))
        else:
            return FileHandler('Makefile', package_handler.dir_abs_path,
                               copy_source_location=os.path.join(MAKEFILE_PATH,
                                                                 package_info.backend.upper(),
                                                                 'Makefile'))


class QnnOpPackageCodeGenerator:
    """
    Handles the code generation of files by performing template substitution using user provided information. Its member
    fields are explained below:
    The resource handler is an object of type IOHandler which may be directory or file handler
    The reader contains the dictionary of all known template types
    The handler to template map, creates an association between a handler name and any templates it may use to generate
    writable content for its file handlers.
    The operator to resource name map contains a mapping for each operator to its corresponding file handler.
    """

    def __init__(self, handler: Optional[IOHandler] = None):
        self.reader = QnnTemplateFileReader
        self.resource_handler = handler
        self.handler_to_template_map = dict()
        self.operator_to_resource_name = defaultdict(list)

    def register_file_handler_with_template(self, file_handler_name, template):
        """
        This associates a file handler with a template, which will ultimately be used to generate writable content
        for that file. Note that the handler must be nested within the instance's top_level resource handler.
        :param file_handler_name: A name of a file handler object.
        :param template_type: The corresponding template type to be used
        :return:
        """
        self.handler_to_template_map[file_handler_name] = template

    def register_operator_file_handler_with_template(self, file_handler_name, template,
                                                     operator_name):
        """
        This calls register_with_template, and then associates the registered file handler with the operator.
        This enables the file_handler to be retrieved specifically in order to generate its writable content.

        Note this is intended solely for generating operator specific source files.
        :param file_handler_name: The name of the file handler
        :param template: The corresponding template type to be used
        :param operator_name: The name of the operator the registered handler is associated
        :return:
        """
        self.register_file_handler_with_template(file_handler_name, template)
        self.operator_to_resource_name[operator_name].append(file_handler_name)

    def substitute_templates(self, package_info: QnnPackageInfo,
                             file_to_template_map: Optional[Dict[str, str]] = None):
        """
        Substitute templates strings in a template file with user provided information.
        :param file_to_template_map: An optional argument through which users can provided a mapping from
        file_name -> template_path. Using this option, a customized template can be used to generate a file instead
         of the default
        :param package_info: the package which contain the files that will be auto-generated
        """
        if not self.handler_to_template_map:
            if file_to_template_map is None:
                raise RuntimeError("Template file types must be defined to enable substitution")

        if self.resource_handler is None:
            raise RuntimeError("Package Handler must be defined to enable substitution")

        writable_content_map = dict()
        per_operator_args = dict()

        for file_handler_name, template in self.handler_to_template_map.items():
            if file_handler_name in chain(*self.operator_to_resource_name.values()):
                operator_name = [name for name, r_names in self.operator_to_resource_name.items()
                                 if file_handler_name in r_names][0]
                operator_match = [operator for operator in
                                  package_info.operators if operator.type_name == operator_name]
                if "CPU" in operator_name:
                    per_operator_args["backend"] = "CPU"
                elif "GPU" in operator_name:
                    per_operator_args["backend"] = "GPU"
                elif operator_match:
                    per_operator_args["operator"] = operator_match[0]
                else:
                    # this is not an error as the operator may not match this package info but will
                    # match another since it has been found in the operator to resource name mapping
                    log_debug3("Operator: {} was not found in package: {}. Skipping operation.",
                               operator_name, package_info.name)
                    continue
            template_file_path = os.path.join(QnnTemplateFileReader.template_path, template)
            if not os.path.exists(template_file_path):
                template_file_path = os.path.join(package_info.SNPE_TEMPLATES_PATH, template)
            writable_content_map[file_handler_name] = self.render_templates(package_info,
                                                                            template_file_path,
                                                                            **per_operator_args)
        if file_to_template_map is not None:
            for resource_name, template_path in file_to_template_map.items():
                io_utils.check_validity(template_path)
                writable_content_map[resource_name] = self.substitute_template(package_info,
                                                                               self.reader.
                                                                               TemplateFileTypes.USER_DEFINED,
                                                                               template_path=template_path,
                                                                               **per_operator_args)

        return writable_content_map

    def substitute_template(self, package_info: QnnPackageInfo,
                            template_type: QnnTemplateFileReader.TemplateFileTypes,
                            *, template_path=None,
                            **per_operator_args):
        """
        Substitutes the package info and operator arguments into a given template. The template path is either located at
        template path or is determined by the template type internally.
        :return: The string content as a result of the template substiution by Mako
        :raises: IOError if no writable content is created, or indirect errors from the call to render_templates
        """

        writable_content = self.render_templates(package_info, template_path, **per_operator_args)
        if not writable_content:
            raise IOError("Unknown Error: Template substitution failed for package: {}".format(
                package_info.name))
        return writable_content

    def get_templates(self, file_type):
        return list(map(lambda x: os.path.join(self.reader.template_path, x),
                        self.reader.DEFAULT_TEMPLATE_FILES[file_type]))

    def render_templates(self, package_info: QnnPackageInfo,
                         template_file_path,
                         **per_operator_args):
        """
         This method handles the template substitution by calling mako on templates that have been created for the
         package. Mako subtitutes the template fields with the user provided information and
         returns a rendered string

        :return: A string containing the result of mako template substitution
        :raises: An ImportError if mako cannot be found
                 An IOError if template files or the template cannot be created from the template file path
                 Unknown Error otherwise, and a system exit
        """
        try:
            import mako
            from mako.lookup import TemplateLookup
            from mako.template import Template
        except ImportError as e:
            raise Exception("{}: Mako template dependency not found. "
                            "Please ensure mako is installed".format(type(e)))

        mytemplate = ''
        template_dir = self.reader.template_path
        directory_lookup = TemplateLookup(directories=[template_dir])

        log_debug("Rendering template {}", template_file_path)
        try:
            mytemplate = Template(filename=template_file_path, lookup=directory_lookup)
            if not mytemplate:
                raise IOError
        except IOError as e:
            log_error("Unknown Error {}:Could not find auto-generation code dependency: {}. "
                      "Please make sure QNN_SDK_ROOT"
                      " environment variable is set", str(e), template_file_path)
        except Exception as e:
            log_error('UNKNOWN ERROR: {} : {}', str(e), type(e))
            sys.exit(-1)
        log_debug_msg_as_status("Auto-generating code")
        rendered = mytemplate.render(package_info=package_info, **per_operator_args)
        log_debug("Auto-generation complete")
        return rendered.lstrip()
