from qti.aisw.arch_checker.modifier import ModelModifier
import sys
import os
import re
import json
import io
import tarfile
from argparse import Namespace
import qti.aisw.arch_checker.constants as const

try:
    from qti.aisw.converters.backend import qnn_modeltools
    from qti.aisw.converters.qnn_backend import qnn_definitions
    from qti.aisw.converters.common import json_serializer as py_ir_json_serializer
except ImportError as ie:
    print("Failed to find necessary package:")
    print(str(ie))
    print("Please ensure that $QNN_SDK_ROOT/lib/python is in your PYTHONPATH")
    sys.exit(1)

class ModifierConverterBackend():
    def __init__(self):
        self.qnn_binary_tar = None

    def sanitize_name(self, name):
        name = re.sub(r'\W+', "_", name)
        return name if name[0].isalpha() else "_" + name

    def add_tensor_to_qnn_bin(self, tensor_name, tensor):
        # add the actual data to the binary file
        buf = io.BytesIO(tensor.tobytes())
        tensor_tar_info = tarfile.TarInfo(name=tensor_name + ".raw")
        tensor_tar_info.size = len(buf.getbuffer())
        self.qnn_binary_tar.addfile(tarinfo=tensor_tar_info, fileobj=buf)
        buf.close()

class QnnModifier(ModelModifier):
    def __init__(self, c_ir_graph, constraints_json, logger, query, df):
        super(QnnModifier, self).__init__(c_ir_graph, constraints_json, logger, query, df)

    def get_header_name(self):
        node_name = const.O_C_GRAPH_NODENAME
        return node_name

    def save_modified_graph(self, output_json_file):
        if len(self.query) == 0 or self.query == "show":
            # no need to save json in case of show
            return True

        output_base_path = os.path.dirname(os.path.realpath(output_json_file))
        output_base_file = os.path.splitext(output_json_file)[0]
        output_bin_path = output_base_file + ".bin"
        output_json_path = output_base_file + ".json"
        output_cpp_path = output_base_file + ".cpp"
        try:
            backend = ModifierConverterBackend()
            backend.qnn_binary_tar = tarfile.TarFile(output_bin_path, 'w')

            model = qnn_modeltools.QnnModel()
            model_init_res = model.init_model_src_serializer(output_cpp_path, "", "", "", False, backend)
            if not model_init_res:
                self.logger.error("Model init failed when lowering to qnn. Unable to apply modifications to the model.")
                return False

            model_serialize = model.serialize(self.c_ir_graph, backend)
            if not model.save():
                self.logger.error("Model save failed when lowering to qnn. Unable to apply modifications to the model.")
                return False

            self.logger.info("Saved modified cpp at: " + output_cpp_path)
            qnn_raw_files = backend.qnn_binary_tar.getmembers()
            backend.qnn_binary_tar.close()

            if not len(qnn_raw_files):
                self.logger.warning("No raw files found for Model. Saving Model BIN skipped.")
                os.path.exists(output_bin_path) and os.remove(output_bin_path)
            else:
                self.logger.info("Saved modified BIN at: " + output_bin_path)

            # Create json file
            json_serializer = py_ir_json_serializer.IrJsonSerializer()
            init_serialize_res = json_serializer.init_json_serializer(output_base_path, output_bin_path, "", "")
            serialize_res = json_serializer.serialize(self.c_ir_graph)
            ir_json = json_serializer.get_graph_json()
            with open(output_json_path, "w") as json_file:
                json_file.write(ir_json)
            self.logger.info("Saved modified json at: " + output_json_path)
            self.logger.info("Successfully applied all possible modifications!")
            self.logger.warning("Note: The modified model is to help visualize the modifications applied for specific rule and the graph would require retraining.")
        except FileNotFoundError:
            self.logger.error("Invalid file path. Please provide correct output path.")
            return False
        except:
            self.logger.error("Unable to lower the model to qnn. Can not apply modifications to the model.")
            return False

        return True