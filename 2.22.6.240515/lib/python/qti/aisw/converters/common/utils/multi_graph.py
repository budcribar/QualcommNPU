# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from qti.aisw.converters.common.utils.converter_utils import log_error
from collections import defaultdict
import tempfile
import numpy as np
import os


class IrStaticTensorSet:
    def __init__(self):
        self.fp_to_id = dict()
        self.seen_tensors = defaultdict(lambda : defaultdict(lambda : {
            "unique" : set(),
            "shared" : set()}
            ))

    def set_shared_tensor_properties(self, graph):
        id_to_tensor_map = graph.get_tensor_map()
        for tensor in id_to_tensor_map.values():
            if tensor.is_static():
                data = tensor.get_data()
                data_hash = self.__hash_data(data)

                set_of_filepaths = self.seen_tensors[tensor.name()][data_hash]["shared"]

                if len(set_of_filepaths) == 1:
                    fp = list(set_of_filepaths)[0]
                    shared_id = self.fp_to_id[fp]
                    tensor.set_id(shared_id)
                    tensor.set_tensor_type_as_context_static()
                    continue

                for fp in set_of_filepaths:
                    if self.__is_data_same_as_file(data, fp):
                        shared_id = self.fp_to_id[fp]
                        tensor.set_id(shared_id)
                        tensor.set_tensor_type_as_context_static()
                        continue

    def add_graph_tensors(self, graph):
        id_to_tensor_map = graph.get_tensor_map()
        for tensor in id_to_tensor_map.values():
            if tensor.is_static():
                self.__add(tensor)

    def __add(self, tensor):
        data = tensor.get_data()
        data_hash = self.__hash_data(data)

        for fp in self.seen_tensors[tensor.name()][data_hash]["shared"]:
            if self.__is_data_same_as_file(data, fp):
                return

        for fp in self.seen_tensors[tensor.name()][data_hash]["unique"]:
            if self.__is_data_same_as_file(data, fp):
                self.seen_tensors[tensor.name()][data_hash]["unique"].remove(fp)
                self.seen_tensors[tensor.name()][data_hash]["shared"].add(fp)

                if len(self.seen_tensors[tensor.name()][data_hash]["shared"]) > 1:
                    log_error("Error: While converting the model with different \
                               configurations, encountered multiple context static \
                               tensors with the same name which is not yet supported. \
                               Please try using different configurations.")
                return


        with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as file:
            fp = file.name
            np.save(fp, data)
            self.seen_tensors[tensor.name()][data_hash]["unique"].add(fp)
            self.fp_to_id[fp] = tensor.id()

    def __is_data_same_as_file(self, data, fp):
        stored_data = np.load(fp)
        return np.array_equal(data, stored_data)

    def __hash_data(self, data):
        byte_data = data.tobytes()
        shape = data.shape
        hash_input = (shape, byte_data)
        return hash(hash_input)

    def clear_cache(self):
        for tensor_name in self.seen_tensors:
            for data_hash in self.seen_tensors[tensor_name]:
                for fp in self.seen_tensors[tensor_name][data_hash]["shared"]:
                    os.remove(fp)

                for fp in self.seen_tensors[tensor_name][data_hash]["unique"]:
                    os.remove(fp)

