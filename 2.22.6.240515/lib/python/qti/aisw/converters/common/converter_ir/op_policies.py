# ==============================================================================
#
#  Copyright (c) 2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================


class ConversionNamePolicy(object):
    def __init__(self):
        self.type_count = {}

    def get_op_name(self, op):
        count = self.type_count.get(op.type, 0)
        self.type_count[op.type] = count + 1
        if hasattr(op, 'LEGACY_TRANSLATION_KEY'):
            name_prefix_str = str(op.LEGACY_TRANSLATION_KEY)
        else:
            name_prefix_str = str(op.type)
        if op.name:
            return str(op.name)
        else:
            return "%s_%d" % (name_prefix_str, count)

    def get_op_name_by_type(self, op_type, legacy_translation_key, custom_op_type=""):
        count = self.type_count.get(op_type, 0)
        self.type_count[op_type] = count + 1
        if legacy_translation_key:
            name_prefix_str = str(legacy_translation_key)
        else:
            name_prefix_str = str(op_type)
        return "%s_%d" % (name_prefix_str, count)

    def get_input_names(self, op, input_names):
        return list(map(str, input_names))

    def get_output_names(self, op, output_names):
        return list(map(str, output_names))

    def remove_output_name(self, output_name):
        return


class ConversionShapeInferencePolicy(object):

    def infer_shape(self, op, input_shapes):
        raise NotImplementedError("infer_shape for {} not implemented ".format(str(self.__class__.__name__)))
