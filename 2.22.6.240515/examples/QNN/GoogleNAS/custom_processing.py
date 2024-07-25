# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import numpy as np
import os
import logging

def process_model(model):
    ''' This function is for performing additional model preprocessing prior to
        conversion. In this case Keras saved models are not supported so the model
        is first frozen and saved as a .pb for consumption by QNN
    '''
    logging.info("Running custom model preprocessing")
    frozen_graph_filename = os.path.join(os.path.dirname(model),'frozen_model.pb')
    # Load and convert keras model to frozen graph
    logging.info('Got model %s, freezing to %s, model', model, frozen_graph_filename)
    loaded_model = tf.keras.models.load_model(model)
    full_model = tf.function(lambda x: loaded_model(x))
    full_model = full_model.get_concrete_function(
    tf.TensorSpec(loaded_model.inputs[0].shape, loaded_model.inputs[0].dtype))
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()
    layers = [op.name for op in frozen_func.graph.get_operations()]
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir=os.path.dirname(model),
                      name=frozen_graph_filename,
                      as_text=False)
    logging.info('Conversion complete got %s', frozen_graph_filename)
    return frozen_graph_filename


def process_stats(stats):
    ''' This function is for overriding/modifying/adding stats after execution
    '''
    stats.update({'latency_in_milliseconds': stats['latency_in_us']/(10**3)})
    logging.info('Custom generated stats: %s', stats)
    return stats
