# =============================================================================
#
#  Copyright (c) 2020-2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

"""Provide TShell utilities to communicate with Windows devices."""

# pylint: disable=useless-object-inheritance

import abc
import collections
import enum
import logging
import queue
import re
import selectors
import socket
import subprocess
import sys
import threading

import decorator
import invoke
import os

import functools

_TSHELL_COMMAND = [
    'C:\\Windows\\SysWOW64\\WindowsPowerShell\\v1.0\\powershell.exe',
    '-ExecutionPolicy',
    'Unrestricted']

'''
_TSHELL_COMMAND = [
    'C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe',
    '-ExecutionPolicy',
    'Unrestricted']
'''
def make_task_decorator(target, base_task, *args, **kwargs):
    """Create new task decorator to enhace invoke.task or fabric.task.

    To use:
    def new_task(*args, **args):
        @decorator.decorator  # to preserve signature for help message
        def _task(func, context, *args, **args):
            # do something
            return func(context, *args, **args)
        return utility.task_decorator(_task, invoke.task, *args, **args)
    """
    @decorator.decorator
    def inner(func, context, *args, **kwargs):
        _load_user_configuration(context)
        func(context, *args, **kwargs)

    def outer(func, *args, **kwargs):
        return base_task(inner(target(func)), *args, **kwargs)

    if len(args) == 1 and callable(args[0]):
        # the decorator is used without argument, e.g. @task
        result = outer(args[0])
    else:
        # the decorator is used with arguments, e.g. @task(name='xxx')
        result = functools.partial(outer, *args, **kwargs)
    return result


def _load_user_configuration(context):
    """Load user config file.

    By default, ~/.fabric.xxx has lower priority than the one in project
    To have user configuration take effect, load it here.
    """
    config = context.config
    config.set_runtime_path(config['SNPE']['USER_CONFIG_FILE'])
    config.load_runtime()


class CommandMatcher(abc.ABC):
    """Check input lines are commands or return code results.

    This class is an abstract class as an interface for CommandResultFramer
    to match the command output.
    """
    @abc.abstractmethod
    def match_main_command(self, line):
        """
        Arg: line
        Return: the command if matched, else none
        """

    @abc.abstractmethod
    def match_return_code_command(self, line):
        """
        Arg: line
        Return: the command if matched, else none
        """

    @abc.abstractmethod
    def match_return_code_result(self, line):
        """
        Arg: line
        Return: the return code if matched, else none
        """


class CommandResultFramer(threading.Thread):
    """Split the input stream data to command execution results.

    This class inherits threading.Thread so that it can run in background to
    read and parse the stream without blocking foreground program execution.
    """

    CommandResult = collections.namedtuple(
        'CommandResult', ['command', 'output', 'return_code'])

    class _State(enum.Enum):
        WAITING_COMMAND = 0
        WAITING_EXECUTION = 1
        WAITING_RETURN_CODE = 2

    def __init__(self, in_stream, result_queue, command_matcher):
        """
        Args:
            in_stream: the stream to be parsed
            result_queue: the framed results will be put to this queue
            command_matcher: functions required to framing the output
        """
        super().__init__()
        self._in_stream = in_stream
        self._result_queue = result_queue
        self._command_matcher = command_matcher

        self._echo_stream = None
        self._command = None
        self._output_buffer = []
        self._logger = logging.getLogger('tshell.CommandResultFramer')
        if(os.getenv('TSHELL_LOG_LEVEL')):
            print("log level from environment var TSHELL_LOG_LEVEL: %s" % (os.getenv('TSHELL_LOG_LEVEL')))
            self._logger.setLevel(logging.DEBUG)
        else:
            self._logger.setLevel(logging.INFO)

    def set_echo_stream(self, echo_stream):
        """Set echo stream.

        The input stream will be echoed to echo stream lively. So
        the caller can get the result progressively instead of at the
        end of the command execution.
        """
        self._echo_stream = echo_stream

    def run(self):
        operation_dict = {
            self._State.WAITING_COMMAND: self._waiting_command,
            self._State.WAITING_EXECUTION: self._waiting_execution,
            self._State.WAITING_RETURN_CODE: self._waiting_return_code}

        self._logger.info('Thread Start')
        state = self._State.WAITING_COMMAND
        self._logger.debug('State %s', state)
        while True:
            line = self._in_stream.readline()
            if line:
                line = line.decode('utf-8')
                state = operation_dict[state](line)
                self._logger.debug('Line from in_stream "%s"', line)
                self._logger.debug('State %s', state)
            else:
                break

        self._logger.info('Thread End')

    def _waiting_command(self, line):
        assert not self._command
        assert not self._output_buffer

        match = self._command_matcher.match_main_command(line)
        if match:
            self._command = match
            next_state = self._State.WAITING_EXECUTION
        else:
            next_state = self._State.WAITING_COMMAND
        return next_state

    def _waiting_execution(self, line):
        match = self._command_matcher.match_return_code_command(line)
        if match:
            next_state = self._State.WAITING_RETURN_CODE
        else:
            self._output_buffer.append(line)
            if self._echo_stream:
                self._echo_stream.write(line)
                self._echo_stream.flush()
            next_state = self._State.WAITING_EXECUTION
        return next_state

    def _waiting_return_code(self, line):
        match = self._command_matcher.match_return_code_result(line)
        if match is not None:
            return_code = match
            result = self.CommandResult(
                self._command, ''.join(self._output_buffer), return_code)
            self._result_queue.put(result)

            self._command = None
            self._output_buffer = []
            next_state = self._State.WAITING_COMMAND
        else:
            next_state = self._State.WAITING_RETURN_CODE
        return next_state


class TShellRunnerBase(abc.ABC):
    """Base class for running tshell.

    This class handles the core logic of executing tshell commands. Note the
    tshell process is not launched in this class. It is subclasses'
    responsibility to run the tshell process and provide the input / output
    of the tshell process to this class. With this design, we can here focus
    on executing tshell commands while leaving tshell life cycle management
    to subclasses.
    """

    # Special pattern for matcher to recognize the command from output stream
    _COMMAND_FORMAT = ('if(5487 -eq 5487){{{0}}}\n' +
                       'if(8745 -eq 8745){{echo $?}}\n')

    class _CommandMatcher(CommandMatcher):
        _MAIN_COMMAND_PATTERN = re.compile(r'if\(5487 -eq 5487\){(.+)}')
        _RETURN_CODE_COMMAND_PATTERN = re.compile(r'if\(8745 -eq 8745\){(.+)}')

        def match_main_command(self, line):
            match = self._MAIN_COMMAND_PATTERN.search(line)
            if match:
                command = match.group(1)
            else:
                command = None
            return command

        def match_return_code_command(self, line):
            match = self._RETURN_CODE_COMMAND_PATTERN.search(line)
            if match:
                command = match.group(1)
            else:
                command = None
            return command

        def match_return_code_result(self, line):
            if 'True' in line:
                result = 0
            else:
                assert 'False' in line
                result = 1
            return result

    def __init__(self):
        self._started = False
        self._result_framer = None
        self._result_queue = queue.Queue()
        self._logger = logging.getLogger('tshell.TShellRunnerBase')

    @property
    @abc.abstractmethod
    def in_stream(self):
        """The file-like input stream to tshell."""

    @property
    @abc.abstractmethod
    def out_stream(self):
        """The file-like output stream from tshell."""

    def start(self):
        """Ready the tshell runner."""
        self._logger.debug('Start')
        self._started = True
        self._result_framer = CommandResultFramer(
            self.out_stream, self._result_queue, self._CommandMatcher())
        self._result_framer.setDaemon(True)
        self._result_framer.start()

    def end(self):
        """Stop and clean the tshell runner."""
        self._result_framer.join()
        self._result_framer = None
        self._started = False
        self._logger.info('End')

    def run(self, command, out_stream=None, no_exception=False):
        """Execute tshell command

        Args:
            command: the command to be executed
            out_stream: to which the stdout of the command should go
                        default sys.stdout
                        set to 'hide' for output nothing
            no_exception: do not raise exception when command execution failed
                          default False

        Returns:
            namedtuple contains (command, output, return_code)
                command: the command executed
                output: the stdout and stderr of the command
                return_code: bool indicates command success or not

        Raises:
            RuntimeError: raised when command return code != 0
        """
        if not self._started:
            raise RuntimeError('=== TShellRunner Not Started ===')

        if out_stream is None:
            out_stream = sys.stdout
        elif isinstance(out_stream, str) and out_stream == 'hide':
            out_stream = None

        command = self._COMMAND_FORMAT.format(command)

        try:
            self._result_framer.set_echo_stream(out_stream)

            self.in_stream.write(command.encode('utf-8'))
            self.in_stream.flush()
            self._logger.debug('Wait for result queue')
            result = self._result_queue.get()
            self._logger.debug('Get result queue')
        finally:
            self._result_framer.set_echo_stream(None)

        if not no_exception and result.return_code:
            raise RuntimeError(
                ('=== TShell Command Fail ===:\n' +
                 'command: {0}\n' +
                 'return code: {1}\n' +
                 'output:\n{2}').format(
                     result.command, result.return_code, result.output))
        self._logger.debug(f'Command Output : {result.output}')
        return result

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, _type, _value, _traceback):
        self.end()


class TShellRunner(TShellRunnerBase):
    """Run tshell in background to be able to execute multiple commands without
    loosing the session.

    For example:
        with TShellRunner() as runner:
            runner.run('open-device 00117659B4BC')
            runner.run('cdd workdir')
            runner.run('putd test_data')
            runner.run('execd snpe-net-run.exe')
            runner.run('getd output')
            runner.run('close-device')

    To use tshell to control devices, it is required to open-device before
    issuing commands to the device. However, open-device is time consuming.
    With this class, it can prevent creating new tshell session and call
    open-device for every tshell command.
    """

    def __init__(self):
        super().__init__()
        self._process = None
        self._logger = logging.getLogger('tshell.TShellRunner')

    @property
    def in_stream(self):
        return self._process.stdin

    @property
    def out_stream(self):
        return self._process.stdout

    def start(self):
        self._process = subprocess.Popen(_TSHELL_COMMAND,
                                         stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT)
        super().start()

    def end(self):
        self._process.kill()
        self._process = None
        super().end()


def task(*args, **kwargs):
    """Decorator to replace invoke.task to force using TShellRunner.run."""
    # The formal way of using TShellRunner is to subclass invoke.runner
    # But it is an overkill with our requirement. So here we just replace
    # context.run to TShellRunner.run
    @decorator.decorator
    def _task(func, context, *args, **kwargs):
        """ The meat of the decorator. """
        with TShellRunner() as runner:
            context.run = runner.run  # force replace the run method
            result = func(context, *args, **kwargs)
        return result
    return make_task_decorator(_task, invoke.task, *args, **kwargs)


class TShellServer(threading.Thread):
    """Create a server to run tshell to keep a live session.

    Here we choose socket as the communication interface. It is an overkill for
    local IPC. But it is simple and straight forward with tons of sample code
    available on Internet.

    Also, subclassing threading.Thread to allow executing in background.
    """

    def __init__(self, port):
        super().__init__()
        self._port = port
        self._stop_event = threading.Event()
        self._logger = logging.getLogger('tshell.TShellServer')

    def run(self):
        self._logger.debug('Thread Start')
        process = subprocess.Popen(_TSHELL_COMMAND,
                                   stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT)

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            #server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(('localhost', self._port))
            server.listen(0)

            self._logger.info('Server Started: listening port %d', self._port)

            while not self._stop_event.set():
                self._accept_connection(server, process)

        self._logger.info('Server End')
        process.kill()
        self._logger.info('Thread End')

    def _accept_connection(self, server, process):
        # Note we use selector here instead of creating another thread
        # This is because we need to control not to write the stdout of the
        # process to closed connection
        # Using selectors is much easier to achieve that since all functions
        # are running in a single thread
        select = selectors.DefaultSelector()
        connection, _ = server.accept()
        self._logger.info('Connection Opened %s', connection.getpeername())
        connection_end = False

        def read(conn):
            self._logger.info('Read data from connection')
            nonlocal connection_end
            data = conn.recv(1024)
            if data:
                self._logger.debug('Read data "%s"', data.decode('utf-8'))
                process.stdin.write(data)
                process.stdin.flush()
            else:
                connection_end = True
                connection.close()
                select.unregister(conn)
                select.unregister(process.stdout)
                self._logger.info('Connection Closed')

        def write(out_stream):
            self._logger.info('Write data to connection')
            if not connection_end:
                # Cannot use readline below since it invokes read multiple
                # times and may trap the execution in it
                data = out_stream.read1(1024)
                self._logger.debug('Write data "%s"', data.decode('utf-8'))
                connection.sendall(data)

        select.register(connection, selectors.EVENT_READ, read)
        select.register(process.stdout, selectors.EVENT_READ, write)

        while not connection_end:
            for key, _mask in select.select():
                callback = key.data
                callback(key.fileobj)

        select.close()

    def stop(self):
        """Stop the server gracefully."""
        self._stop_event.set()


class TShellClient(TShellRunnerBase):
    """Client of TShellServer to run tshell command remotely.

    The interface is exactly the same with TShellRunner, except that
    1. You have to run TShellServer in advance
    2. The tshell session persists across client sessions

    For example:
        server = TShellServer(5487)
        server.setDaemone(True)
        server.start()

        with TShellClient(5487) as client:
            client.run('$tempvar=1')

        with TShellClient(5487) as client:
            result = client.run('$tempvar+1')
            assert result.output.strip() == '2'

        server.stop()

    The tshell process is launched in TShellServer side. This class relays the
    socket channels to parent TShellRunnerBase in / out streams.
    """
    def __init__(self, port):
        super().__init__()
        self._port = port
        self._socket = None
        self._socket_in_file = None
        self._socket_out_file = None
        self._logger = logging.getLogger('tshell.TShellClient')

    @property
    def in_stream(self):
        return self._socket_in_file

    @property
    def out_stream(self):
        return self._socket_out_file

    def start(self):
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.connect(('localhost', self._port))
        self._logger.info('Connected %s', self._socket.getpeername())
        self._socket_in_file = self._socket.makefile('wb')
        self._socket_out_file = self._socket.makefile('rb')
        super().start()

    def end(self):
        self._socket.shutdown(socket.SHUT_RDWR)
        self._socket.close()
        self._socket_in_file = None
        self._socket_out_file = None
        self._socket = None
        self._logger.info('Disconnected')
        super().end()


def client_task(*args, **kwargs):
    """Decorator to replace invoke.task to force using TShellClient.run."""
    # The formal way of using TShellClient is to subclass invoke.runner
    # But it is an overkill with our requirement. So here we just replace
    # context.run to TShellRunner.run
    @decorator.decorator
    def _task(func, context, *args, **kwargs):
        """ The meat of the decorator. """
        client = TShellClient(5489)
        try:
            client.start()
        except ConnectionError as exc:
            raise RuntimeError(
                'Failed to connect to TShell Server at port 5487') from exc
        try:
            context.local = context.run
            context.run = client.run  # force replace the run method
            result = func(context, *args, **kwargs)
        finally:
            client.end()
        return result
    return make_task_decorator(_task, invoke.task, *args, **kwargs)


if __name__ == '__main__':
    #_server = TShellServer(5487)
    _server = TShellServer(5489)
    _server.setDaemon(True)
    try:
        print('TShellServer start listening port 5487')
        _server.run()
    finally:
        _server.stop()
        print('TShellServer stopped')
