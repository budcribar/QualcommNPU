# ==============================================================================
#
#  Copyright (c) 2020-2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from __future__ import print_function
import logging
import sys
import os
from time import sleep
import shutil

if os.path.isdir(os.path.join(os.path.abspath(
        os.path.dirname(__file__)), '../lib/benchmarks/')):
    sys.path.insert(
        0,
        os.path.join(
            os.path.abspath(
                os.path.dirname(__file__)),
            '../lib/benchmarks/'))
    sys.path.insert(
        0,
        os.path.join(
            os.path.abspath(
                os.path.dirname(__file__)),
            '../lib'))
else:
    sys.path.insert(
        0,
        os.path.join(
            os.path.abspath(
                os.path.dirname(__file__)),
            'utils'))
    sys.path.insert(
        0,
        os.path.join(
            os.path.abspath(
                os.path.dirname(__file__)),
            'bm_utils'))

from common_utils.constants import LOG_FORMAT
from common_utils import exceptions
from bm_utils.qnn import QNN
from bm_utils.error import Error
from bm_utils import bm_parser, bm_config, bm_bm, bm_md5, bm_writer, bm_device , bm_constants

logger = None


def _find_shell_binary_on_target(device):
    sh_path = '/system/bin/sh'
    if device.adb.check_file_exists(sh_path) is False:
        sh_cmd = 'which sh'
        ret, out, err = device.adb.shell(sh_cmd)
        sh_path = out[0]
        if ret != 0:
            sh_path = ''
        if sh_path == '' or "not found" in sh_path:
            logger.error('Could not find md5 checksum binary on device.')
            sh_path = ''
    return sh_path.rstrip()


def _config_logger(product, debug, device_id=None):
    global logger
    log_prefix = product.BENCH_NAME + ('_' + device_id if device_id else "")
    logger = logging.getLogger(log_prefix)
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format=LOG_FORMAT)


def bench(program_name, args_list, device_msm_os_dict=None):
    try:
        args_parser = bm_parser.ArgsParser(program_name, args_list)
        product_type_obj = QNN.getInstance()
        if args_parser.device_id_override:
            _config_logger(
                product_type_obj,
                args_parser.debug_enabled,
                args_parser.device_id_override[0])
        else:
            _config_logger(product_type_obj, args_parser.debug_enabled)
        logger.info(
            "Running {0} with {1}".format(
                product_type_obj.BENCH_NAME,
                args_parser.args))
        config = bm_config.ConfigFactory.make_config(args_parser.config_file_path,
                                                     args_parser.output_basedir_override,
                                                     args_parser.device_id_override, args_parser.host_name,
                                                     args_parser.device_os_type_override,
                                                     '', args_parser.backend_config, args_parser.perfprofile,
                                                     args_parser.profilinglevel, args_parser.iterations,
                                                     args_parser.runtimes, product_type_obj,
                                                     args_parser.dsp_type, args_parser.htp_serialized,
                                                     args_parser.arm_prepare, args_parser.use_signed_skel,
                                                     args_parser.enable_cache, args_parser.shared_buffer,
                                                     args_parser.discard_output, args_parser.test_duration)
        if config is None:
            sys.exit(Error.ERRNUM_CONFIG_ERROR)
        logger.info(config)
        # Dictionary is
        # {"cpu_arm_all_SNPE_BENCH_NAMEMemory":ZdlSnapDnnCppDroidBenchmark
        # object}
        benchmarks, results_dir = bm_bm.BenchmarkFactory.make_benchmarks(
            config, product_type_obj, args_parser.device_os_type_override,
            args_parser.htp_serialized, args_parser.shared_buffer)
        # Now loop through all the devices and run the benchmarks on them
        for device_id in config.devices:
            device = bm_device.DeviceFactory.make_device(
                device_id, config, product_type_obj)
            if args_parser.device_os_type_override != \
            product_type_obj.CONFIG_DEVICEOSTYPES_WINDOWS_AARCH64 and \
               not device.adb.is_device_online():
                logger.warn("Device not online. Trying to bring it up.")
                device_state = device.adb.recover_device(issue="crash")
                if not device_state:
                    logger.warn(
                        "Couldn't bring up the device. Device dead before benchmark could start")
                    exceptions.AdbShellCmdFailedException(
                        "Device not recovered from the bad state. Exiting Job")
            # don't need to capture retcode/err since error handling is
            # done in the fuction and behaviour depends on 'fatal' argument
            if args_parser.device_os_type_override != \
            product_type_obj.CONFIG_DEVICEOSTYPES_WINDOWS_AARCH64:
                _, device_info, _ = device.adb.get_device_info(
                fatal=((args_parser.device_os_type_override != 'le' and \
                args_parser.device_os_type_override != 'le64'\
                        and args_parser.device_os_type_override != \
                        product_type_obj.CONFIG_DEVICEOSTYPES_QNX_AARCH64)))
            else:
                device_info = []
            sh_path = ''
            if args_parser.device_os_type_override != \
            product_type_obj.CONFIG_DEVICEOSTYPES_WINDOWS_AARCH64:
                logger.debug("Perform md5 checksum on %s" % device_id)
                bm_md5.perform_md5_check(device, \
                [item for sublist in config.artifacts.values() for item in sublist]
                + config.dnn_model.artifacts, product_type_obj)
                logger.debug("Artifacts on %s passed checksum" % device_id)
                sh_path = _find_shell_binary_on_target(device)
            else:
                logger.info("Pushing Artifacts on device: %s" % device_id)
                device.push_win_artifacts([item for sublist in config.artifacts.values() \
                for item in sublist] + config.dnn_model.artifacts)
            benchmarks_ran = []
            # Run each benchmark on device, and pull results
            for bm in benchmarks:
                matches = [value for key, value in product_type_obj.RUNTIMES.items()
                           if bm.runtime_flavor_measure.startswith(key)]
                if matches:
                    logger.info(
                        'Running on {}'.format(
                            bm.runtime_flavor_measure))
                    if args_parser.device_os_type_override != \
                    product_type_obj.CONFIG_DEVICEOSTYPES_WINDOWS_AARCH64: bm.sh_path = sh_path
                    # running iterations of the same runtime.  Two possible failure cases:
                    # 1. Say GPU runtime is not available
                    # 2. Transient failure
                    # For now, for either of those cases, we will mark the whole runtime
                    # as bad, so I break out of for loop as soon as a failure
                    # is detected
                    iterations = config.iterations

                    for i in range(iterations):
                        logger.info("Run " + str(i + 1))
                        bm.run_number = i + 1
                        if args_parser.device_os_type_override != \
                        product_type_obj.CONFIG_DEVICEOSTYPES_WINDOWS_AARCH64:
                            try:
                                device.execute(bm.pre_commands(
                                    args_parser.dsp_type, args_parser.arm_prepare,
                                    args_parser.enable_cache, args_parser.cdsp_id))
                                device.start_measurement(bm)
                                # Sleep to let things cool off
                                if args_parser.sleep != 0:
                                    logger.debug("Sleeping: " +
                                                 str(args_parser.sleep))
                                    sleep(args_parser.sleep)
                                device.execute(bm.commands)

                                device.stop_measurement(bm)
                                device.execute(bm.post_commands)
                            except exceptions.AdbShellCmdFailedException as e:
                                logger.warning(
                                    'Failed to perform benchmark for %s.' %
                                    bm.runtime_flavor_measure)
                                break
                            finally:
                                device.stop_measurement(bm)
                        else:
                            try:
                                device.execute_win(bm.pre_commands(
                                    args_parser.dsp_type, args_parser.arm_prepare,
                                    args_parser.enable_cache, args_parser.cdsp_id))
                                device.start_measurement(bm)
                                if args_parser.sleep != 0:
                                    logger.debug("Sleeping: " +
                                                 str(args_parser.sleep))
                                    sleep(args_parser.sleep)
                                device.execute_win(bm.commands_win)
                                device.stop_measurement(bm)
                                device.execute_win(bm.post_commands_win)
                            except Exception as e:
                                logger.warning(
                                    'Failed to perform benchmark for %s.' %
                                    bm.runtime_flavor_measure)
                                break

                        bm.process_results()

                    else:  # Ran through iterations without failing
                        benchmarks_ran.append((bm.runtime_flavor_measure, bm))

                else:
                    logger.error("The specified runtime with  %s is not a supported runtime,"
                                 " benchmarks will not be running with this runtime" % bm.runtime_flavor_measure)

            if len(benchmarks_ran) == 0:
                logger.error(
                    'None of the selected benchmarks ran, therefore no results reported')
                sys.exit(Error.ERRNUM_NOBENCHMARKRAN_ERROR)
            else:
                os_type, device_meta = device.adb.getmetabuild()
                metabuild_id = ('Meta_Build_ID', device_meta)
                device_info.append(metabuild_id)

                os_type = ('OS_Type', os_type)
                device_info.append(os_type)

                if args_parser.clean_artifacts:
                    databins_path = os.path.join(config.device_path,config._dnnmodel.name)
                    logger.info("Cleaning {}".format(databins_path))
                    device.adb._execute("shell",["rm -rf",databins_path])

                if device_msm_os_dict is not None:
                    chipset = ('Chipset', device_msm_os_dict[device_id][1])
                    if device_msm_os_dict[device_id][2] == '':
                        OS = ('OS', device_msm_os_dict[device_id][3])
                    else:
                        OS = ('OS', device_msm_os_dict[device_id][2])
                    device_info.append(chipset)
                    device_info.append(OS)

                product_version = benchmarks_ran[0][1].get_product_version(
                    config)
                basewriter = bm_writer.Writer(
                    product_version,
                    benchmarks_ran,
                    config,
                    device_info,
                    args_parser.sleep,
                    product_type_obj)

                if args_parser.generate_json:
                    basewriter.writejson(
                        os.path.join(
                            results_dir,
                            "benchmark_stats_{0}.json".format(
                                config.name)))
                basewriter.writecsv(
                    os.path.join(
                        results_dir,
                        "benchmark_stats_{0}.csv".format(
                            config.name)))

        if args_parser.output_basedir_override:
            if os.path.exists(os.path.abspath(os.path.join(args_parser.output_basedir_override,'tmp_work'))):
                shutil.rmtree(os.path.abspath(os.path.join(args_parser.output_basedir_override,'tmp_work')))
    except exceptions.ConfigError as ce:
        logger.error(ce)
        sys.exit(Error.ERRNUM_CONFIG_ERROR)
    except exceptions.AdbShellCmdFailedException as ae:
        logger.error(ae)
        sys.exit(Error.ERRNUM_ADBSHELLCMDEXCEPTION_ERROR)
    except Exception as e:
        logger.error(e)
        sys.exit(Error.ERRNUM_GENERALEXCEPTION_ERROR)


if __name__ == "__main__":
    bench(sys.argv[0], sys.argv[1:])
