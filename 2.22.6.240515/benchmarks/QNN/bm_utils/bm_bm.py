# ==============================================================================
#
#  Copyright (c) 2020-2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import stat
import sys
import statistics

from .bm_config_restrictions import *
from .bm_parser import LogParserFactory
from collections import OrderedDict
import time
import os
import logging

logger = logging.getLogger(__name__)


class BenchmarkStat(object):
    def __init__(self, log_parser, stat_type):
        self._stats = []
        self._stddev = []
        self._resources = OrderedDict()
        self._log_parser = log_parser
        self._type = stat_type

    def __iter__(self):
        return self._stats.__iter__()

    @property
    def stats(self):
        return self._stats

    @property
    def type(self):
        return self._type

    def _process(self, input_dir):
        data_frame, std_dev, resource = self._log_parser.parse(input_dir)
        if len(resource) != 0 and len(self._resources) == 0:
            self._resources = resource
        self._stats.append(data_frame)
        self._stddev.append(std_dev)

    def get_resources(self):
        return self._resources

    @property
    def average(self):
        avg_dict = OrderedDict()
        for stat in self._stats:
            for channel, _sum, _len, _max, _min, _runtime in stat:
                if channel in avg_dict:
                    avg_dict[channel][0] += _sum
                    avg_dict[channel][1] += _len
                else:
                    avg_dict[channel] = [_sum, _len]
        avgs = OrderedDict()
        for channel in avg_dict:
            avgs[channel] = int(avg_dict[channel][0] / avg_dict[channel][1])
        return avgs

    @property
    def max(self):
        max_dict = OrderedDict()
        for stat in self._stats:
            for channel, _sum, _len, _max, _min, _runtime in stat:
                if channel in max_dict:
                    max_dict[channel] = max(max_dict[channel], _max)
                else:
                    max_dict[channel] = _max
        return max_dict

    @property
    def min(self):
        min_dict = OrderedDict()
        for stat in self._stats:
            for channel, _sum, _len, _max, _min, _runtime in stat:
                if channel in min_dict:
                    min_dict[channel] = min(min_dict[channel], _min)
                else:
                    min_dict[channel] = _min
        return min_dict

    @property
    def runtime(self):
        runtime_dict = OrderedDict()
        for stat in self._stats:
            for channel, _sum, _len, _max, _min, _runtime in stat:
                if channel not in runtime_dict:
                    runtime_dict[channel] = _runtime
        return runtime_dict

    @property
    def stddev(self):
        stddev_dict = OrderedDict()
        event_data = {}
        for iter in self._stddev:
            for k in iter.keys():
                if k not in event_data.keys():
                    event_data[k] = []
                event_data[k].extend(iter[k])
        for k in event_data:
            if len(event_data[k]) <= 1:
                stddev_dict[k] = "NA"
            else:
                stddev_dict[k] = statistics.stdev(event_data[k])
        return stddev_dict


class BenchmarkCommand(object):
    def __init__(self, function, params):
        self.function = function
        self.params = params


class BenchmarkFactory(object):
    def __init__(self):
        pass

    @staticmethod
    def make_benchmarks(config, product, device_os_type, htp_serialized, shared_buffer):
        assert config, "config is required"
        assert config.measurement_types_are_valid(), "You asked for %s, but only these types of measurements" \
                                                     " are supported: %s" % (config.measurements,
                                                                             CONFIG_VALID_MEASURMENTS)
        host_result_dirs = {}

        for arch in config.architectures:
            if arch == product.ARCH_AARCH64 or arch == product.ARCH_ARM:
                if 'droid' not in host_result_dirs:
                    host_result_dirs['droid'] = \
                        SnapDnnCppDroidBenchmark.create_host_result_dir(
                            config.host_resultspath, product)

        benchmarks = []
        for runtime, flavor in config.return_valid_run_flavors():
            for measurement in config.measurements:
                dev_bin_path = config.get_device_artifacts_bin(runtime)
                dev_lib_path = config.get_device_artifacts_lib(runtime)
                exe_name = config.get_exe_name()
                parser = LogParserFactory.make_parser(
                    measurement, config, product)
                benchmark = SnapDnnCppDroidBenchmark(
                    dev_bin_path,
                    dev_lib_path,
                    exe_name,
                    config.dnn_model.device_rootdir,
                    os.path.basename(config.dnn_model.model),
                    config.dnn_model.input_list_name,
                    config.userbuffer_mode,
                    config.perfprofile,
                    config.backend_config,
                    config.profilinglevel,
                    config.host_rootpath,
                    product,
                    arch,
                    htp_serialized,
                    shared_buffer,
                    device_os_type,
                    config.discard_output,
                    config.test_duration
                )
                benchmark.measurement = BenchmarkStat(
                    parser, measurement)
                benchmark.runtime = runtime
                benchmark.host_output_dir = host_result_dirs['droid']
                benchmark.name = flavor
                benchmarks.append(benchmark)
        return benchmarks, host_result_dirs['droid']


class SnapDnnCppDroidBenchmark(object):
    @staticmethod
    def create_host_result_dir(host_output_dir, product):
        # Create results output dir, and a "latest_results" that links to it
        _now = time.localtime()[0:6]
        _host_output_datetime_dir = os.path.join(
            host_output_dir,
            product.BENCH_OUTPUT_DIR_DATETIME_FMT %
            _now)
        os.makedirs(_host_output_datetime_dir)
        sim_link_path = os.path.join(
            host_output_dir, product.LATEST_RESULTS_LINK_NAME)
        if os.path.islink(sim_link_path):
            os.remove(sim_link_path)
        if 'linux' in sys.platform:
            os.symlink(
                os.path.relpath(
                    _host_output_datetime_dir,
                    host_output_dir),
                sim_link_path)
        return _host_output_datetime_dir

    def __init__(self, exe_dir, dep_lib_dir, exe_name, model_dir, container_name,
                 input_list_name, userbuffer_mode, perfprofile, backend_config,
                 profilinglevel, host_rootpath, product=None, arch=None, htp_serialized=False,
                 shared_buffer=False, device_os_type=False, discard_output=False, test_duration=False):
        assert model_dir, "model dir is required"
        assert container_name, "container is required"
        assert input_list_name, "input_list is required"
        self.product = product
        self._exe_dir = exe_dir
        self._model_dir = model_dir
        self._dep_lib_dir = dep_lib_dir
        self._exe_name = exe_name
        self._container = container_name
        self._input_list = input_list_name
        self.output_dir = 'output'
        self.host_output_dir = None
        self.host_result_dir = None
        self.runtime = product.RUNTIME_CPU
        self.name = None
        self.run_number = 0
        self.measurement = None
        self.sh_path = '/system/bin/sh'
        self.userbuffer_mode = userbuffer_mode
        self.perfprofile = perfprofile
        self.backend_config = backend_config
        self.profilinglevel = profilinglevel
        self.host_rootpath = host_rootpath
        self.arch = arch
        self.device_os_type= device_os_type
        self.htp_serialized = htp_serialized
        self.shared_buffer = shared_buffer
        self.discard_output = discard_output
        self.test_duration = test_duration

    @property
    def runtime_flavor_measure(self):
        if self.name != '':
            return '{}_{}_{}'.format(
                self.runtime, self.name, self.measurement.type)
        else:
            return '{}_{}'.format(self.runtime, self.measurement.type)

    @property
    def exe_name(self):
        return self.product.BATCHRUN_EXE

    def __create_diagview_script(self):
        cmds = ['cd ' + self._model_dir]
        if self.profilinglevel == "backend" and self.runtime == "DSP":
            run_cmd = self.product.diagview_cmd.format(
                '/'.join([self._exe_dir, self.product.DIAGVIEW_EXE]),
                '/'.join([self.output_dir, self.product.BENCH_DIAG_OUTPUT_FILE]),
                '/'.join([self.output_dir, self.product.BENCH_DIAG_JSON_FILE]),
                '/'.join([self.output_dir, self.product.PARSE_OUT_FILE]))
            run_cmd = run_cmd  + " --reader " + '/'.join([self._dep_lib_dir,self.product.DIAGVIEW_LIB])
        else:
            run_cmd = self.product.diagview_cmd.format(
                '/'.join([self._exe_dir, self.product.DIAGVIEW_EXE]),
                '/'.join([self.output_dir, self.product.BENCH_DIAG_OUTPUT_FILE]),
                '/'.join([self.output_dir, self.product.BENCH_DIAG_CSV_FILE]),
                '/'.join([self.output_dir, self.product.PARSE_OUT_FILE]))
        cmds.append(run_cmd)
        cmd_script_path = ''
        if self.device_os_type != self.product.CONFIG_DEVICEOSTYPES_WINDOWS_AARCH64:
            cmd_script_path = os.path.join(
                self.host_rootpath, self.product.DIAGVIEW_SCRIPT)
        else:
            cmd_script_path = os.path.join(
                self.host_rootpath, self.product.DIAGVIEW_SCRIPT_WIN)

        if os.path.isfile(cmd_script_path):
            os.remove(cmd_script_path)
        with open(cmd_script_path, 'w') as cmd_script:
            for cmd in cmds:
                if self.device_os_type == self.product.CONFIG_DEVICEOSTYPES_WINDOWS_AARCH64:
                    cmd = cmd.replace('/', '\\')
                    cmd_script.write(cmd + '\n')
                else:
                    cmd_script.write(cmd + ';')
        os.chmod(cmd_script_path, stat.S_IRWXU)
        return cmd_script_path

    def __create_script(self, dsp_type, arm_prepare, enable_cache, cdsp_id):
        if self.device_os_type != self.product.CONFIG_DEVICEOSTYPES_WINDOWS_AARCH64:
            cmds = [
            'export LD_LIBRARY_PATH=' + self._dep_lib_dir + ':' + self._model_dir + ':/vendor/dsp'
                                                                                    '/cdsp:' +
                                                                                    '/usr/lib:' +
                                                                                    '/mnt/lib64:' +
            self.product.ARCH_VENDOR_LIB_PATH[self.arch] + ':$LD_LIBRARY_PATH',
            'export ADSP_LIBRARY_PATH=\"' + self._dep_lib_dir +
            '/../../dsp/lib;/system/lib/rfsa/adsp;/usr/lib/rfsa/adsp;/vendor/dsp/cdsp;/system'
            '/vendor/lib/rfsa/adsp;/dsp;/etc/images/dsp;\"',
            'cd ' + self._model_dir,
            'rm -rf ' + self.output_dir,
            'chmod +x {}/*'.format(self._exe_dir)]
        else:
            cmds = ['set PATH=%PATH%;'+ self._dep_lib_dir + ';' + self._model_dir + \
                    ';' + os.path.abspath(self._dep_lib_dir + '\..\..\dsp\lib'),
            'cd ' + self._model_dir, 'rmdir /s /q ' + self.output_dir]
        if enable_cache:
            run_cmd = self.product.cache_cmd_options.format(
                '/'.join([self._exe_dir, self.product.CHACHE_EXE]),
                self._container, "qnngraph.serialized", self._model_dir)
            if self.backend_config:
                run_cmd += " --config_file " + \
                       '/'.join([self._dep_lib_dir, os.path.basename(self.backend_config)])
            run_cmd += self.product.RUNTIMES[self.runtime]
            cmds.append(run_cmd)
            run_cmd = self.product.serialized_run_cmd_options.format(
                'CDSP_ID=' + cdsp_id + ' ' + '/'.join([self._exe_dir, self._exe_name]),
                "qnngraph.serialized.bin", self._input_list,
                self.output_dir)

        elif not self.htp_serialized:
            # Use retrieve context if .bin is provided
            if self._container.endswith('.bin'):
                run_cmd = self.product.serialized_run_cmd_options.format(
                    '/'.join([self._exe_dir, self._exe_name]),
                    self._container, self._input_list, self.output_dir)
            else:
                run_cmd = self.product.run_cmd_options.format(
                    '/'.join([self._exe_dir, self._exe_name]),
                    self._container, self._input_list, self.output_dir)

        else:
            run_cmd = self.product.serialized_run_cmd_options.format(
                'CDSP_ID=' + cdsp_id + ' ' + '/'.join([self._exe_dir, self._exe_name]),
                "qnngraph.serialized.bin", self._input_list,
                self.output_dir)

        if 'DSP' in self.runtime:
            if dsp_type in ['v68', 'v69', 'v69-plus', 'v73', 'v75', 'v79']:
                runtime = 'HTP_' + dsp_type
            else:
                runtime = 'DSP_' + dsp_type
        else:
            runtime = self.runtime

        if self.device_os_type != self.product.CONFIG_DEVICEOSTYPES_WINDOWS_AARCH64:
            run_cmd += self.product.RUNTIMES[runtime]
        else:
            run_cmd += self.product.RUNTIMES_WIN[runtime]
        run_cmd += " --log_level error "

        # add option userbuffer mode
        if self.name in self.product.BUFFER_MODES:
            run_cmd += self.product.BUFFER_MODES[self.name]
        if self.profilinglevel:
            run_cmd += " --profiling_level " + self.profilinglevel
        if self.perfprofile:
            run_cmd += " --perf_profile " + self.perfprofile
        if self.discard_output:
            if self.device_os_type == self.product.CONFIG_DEVICEOSTYPES_WINDOWS_AARCH64:
                run_cmd += " --keep_num_outputs 1"
            else:
                run_cmd += " --keep_num_outputs 0"
        if self.test_duration:
            run_cmd += " --duration " + self.test_duration
        if self.shared_buffer:
            run_cmd += " --shared_buffer "
        if 'DSP' in runtime or 'HTP' in runtime or 'GPU_FP16' in runtime or 'HTA' in runtime or 'GPU' in runtime:
            if self.backend_config:
                 run_cmd += " --config_file " + \
                               '/'.join([self._dep_lib_dir, os.path.basename(self.backend_config)])

        if self.device_os_type == self.product.CONFIG_DEVICEOSTYPES_WINDOWS_AARCH64:
            run_cmd = run_cmd.replace('/', '\\')
        cmds.append(run_cmd)
        logger.info(run_cmd)
        cmd_script_path = ''
        if self.device_os_type != self.product.CONFIG_DEVICEOSTYPES_WINDOWS_AARCH64:
            cmd_script_path = os.path.join(
                self.host_rootpath, self.product.BENCH_SCRIPT)
        else:
            cmd_script_path = os.path.join(
                self.host_rootpath, self.product.BENCH_SCRIPT_WIN)
        if os.path.isfile(cmd_script_path):
            os.remove(cmd_script_path)
        with open(cmd_script_path, 'w') as cmd_script:
            for ln in cmds:
                if self.device_os_type == self.product.CONFIG_DEVICEOSTYPES_WINDOWS_AARCH64:
                    if not 'rmdir' in ln:
                        ln = ln.replace('/', '\\')
                    cmd_script.write(ln + '\n')
                else:
                    cmd_script.write(ln + ';')
        os.chmod(cmd_script_path, stat.S_IRWXU)
        return cmd_script_path

    def pre_commands(self, dsp_type, arm_prepare, enable_cache, cdsp_id):
        self.host_result_dir = os.path.join(
            self.host_output_dir,
            self.measurement.type, '_'.join(
                filter(''.__ne__, (self.runtime, self.name))),
            "Run" + str(self.run_number)
        )
        os.makedirs(self.host_result_dir)
        cmd_script = self.__create_script(dsp_type, arm_prepare, enable_cache, cdsp_id)
        diagview_script = self.__create_diagview_script()
        diag_rm_files = '/'.join([
            self._model_dir, self.output_dir, self.product.BENCH_DIAG_REMOVE])
        if self.device_os_type != self.product.CONFIG_DEVICEOSTYPES_WINDOWS_AARCH64:
            return [BenchmarkCommand('shell', ['rm', ['-f', diag_rm_files]]),
                BenchmarkCommand('push', [cmd_script, self._model_dir]),
                BenchmarkCommand('push', [diagview_script, self._model_dir])]
        else:
            return [['push', cmd_script, self._model_dir],
                ['push', diagview_script, self._model_dir]]

    @property
    def commands(self):
        return [BenchmarkCommand('shell', ['sh', ['/'.join([self._model_dir,
                                                            self.product.BENCH_SCRIPT])]]),
                BenchmarkCommand('shell', ['sh', ['/'.join([self._model_dir,
                                                            self.product.DIAGVIEW_SCRIPT])]])]
    @property
    def commands_win(self):
        return [['shell', '\\'.join([self._model_dir,self.product.BENCH_SCRIPT_WIN])],
                ['shell', '\\'.join([self._model_dir,self.product.DIAGVIEW_SCRIPT_WIN])]]

    @property
    def post_commands(self):
        if self.host_output_dir is None:
            return []
        device_output_dir = '/'.join([self._model_dir, self.output_dir])
        file_format = None
        if self.profilinglevel == "backend" and self.runtime == "DSP":
            file_format =  self.product.BENCH_DIAG_JSON_FILE
        else:
            file_format = self.product.BENCH_DIAG_CSV_FILE
        # now will also pull the script file used to generate the results
        return [BenchmarkCommand('shell', ['chmod', ['777', device_output_dir]]),
                BenchmarkCommand('pull', ['/'.join([self._model_dir,
                                                    self.product.BENCH_SCRIPT]),
                                          self.host_result_dir]),
                BenchmarkCommand('pull', [
                    '/'.join([device_output_dir, self.product.BENCH_DIAG_OUTPUT_FILE]),
                    self.host_result_dir]),
                BenchmarkCommand('pull', [
                    '/'.join([device_output_dir, file_format]),
                    self.host_result_dir]),
                BenchmarkCommand('pull', [
                    '/'.join([device_output_dir, self.product.PARSE_OUT_FILE]),
                    self.host_result_dir])
                ]

    @property
    def post_commands_win(self):
        if self.host_output_dir is None:
            return []
        device_output_dir = '\\'.join([self._model_dir, self.output_dir])
        # now will also pull the script file used to generate the results
        return [['dird', '\\'.join([device_output_dir, 'Result_0'])],
                ['pull', '\\'.join([device_output_dir, self.product.BENCH_DIAG_OUTPUT_FILE]),
                    self.host_result_dir],
                ['pull', '\\'.join([device_output_dir, self.product.BENCH_DIAG_CSV_FILE]),
                    self.host_result_dir],
                ['pull', '\\'.join([device_output_dir, self.product.PARSE_OUT_FILE]),
                    self.host_result_dir]
                ]

    def get_product_version(self, config):
        product_version_parser = LogParserFactory.make_parser(
            self.product.MEASURE_PRODUCT_VERSION, config, self.product)
        return product_version_parser.parse(self.host_result_dir)

    def process_results(self):
        assert os.path.isdir(
            self.host_result_dir), "ERROR: no host result directory"
        self.measurement._process(self.host_result_dir)
