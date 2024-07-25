# =============================================================================
#
#  Copyright (c) 2021,2023,2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import sys
import tensorflow as tf
try:
    tf_compat_v1 = tf.compat.v1
except AttributeError:
    tf_compat_v1 = tf

    # import contrib ops since they are not imported as part of TF by default
    import tensorflow.contrib


# Do some quick validation of python and tensorflow versions
tf_ver = tf.__version__.split('.')

# Only python 3.x and TF 2.x supported
if sys.version_info[0] == 3:
    if int(tf_ver[0]) != 2:
        raise ValueError("Only Tensorflow 2.x supported with python3")
else:
    raise ValueError("Unsupported python {} and tensorflow {} version combination".format(sys.version, tf.__version__))
