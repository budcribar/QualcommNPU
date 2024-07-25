# ==============================================================================
#
#  Copyright (c) 2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import tvm.relay.op.op as _op
from tvm import relay

class TfliteDetectionPostprocessRelayOp:
    _op_name = "qti.aisw.detection_postprocess"
    _instance = None

    def __new__(cls):
        # Singleton pattern
        if not cls._instance:
            cls._instance = super().__new__(cls)
            _op.register(cls._op_name)
            custom_op = _op.get(cls._op_name)
            custom_op.set_num_inputs(3)
            custom_op.add_argument("box_prob", "expr", "the input potential coordinates tensor.")
            custom_op.add_argument("class_prob", "expr", "the input class probability tensor.")
            custom_op.add_argument("anchors", "var", "the input pre-defined yxhw anchors tensor.")
            custom_op.set_attrs_type_key("DictAttrs")
            custom_op.add_type_rel(cls._op_name, TfliteDetectionPostprocessRelayOp.relation_func)
            custom_op.set_support_level(1)
            _op.register_pattern(cls._op_name, _op.OpPattern.OPAQUE)
            _op.register_stateful(cls._op_name, False)
        return cls._instance

    @staticmethod
    def relation_func(arg_types, attrs):
        assert len(arg_types) == 3
        batch_num = arg_types[0].shape[0]
        detection_limit = attrs['detection_limit']

        # Return the shape of in sequence:
        # Scores, boxes valid_count and cls_ids
        scores_type = relay.TensorType([batch_num, detection_limit], 'float32')
        boxes_type = relay.TensorType([batch_num, detection_limit, 4], 'float32')
        num_detections_type = relay.TensorType([batch_num], 'int32')
        cls_ids_type = relay.TensorType([batch_num, detection_limit], 'float32')

        return relay.TupleType([scores_type, boxes_type, cls_ids_type, num_detections_type])

TfliteDetectionPostprocessRelayOp()
