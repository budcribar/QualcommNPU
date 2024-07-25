# ==============================================================================
#
#  Copyright (c) 2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
from qti.aisw.op_package_generator.helpers.package_generator_helper import *

# ------------------------------------------------------------------------------
#   Qnn config helpers
# ------------------------------------------------------------------------------

# first step is to create a package manager instance
# the object is a central mechanism for registering and ultimately producing packages
example_package_manager = QnnPackageManager()

# Then create a simple example package
# the package info can be initialized only with a name, and optionally where it should be saved
# since we are working with an htp package, we can set the backend field
# see docs for other optional fields such as version, domain, and for more arguments that can be
# used to create to make a package info
example_package_htp_fp16 = make_package_info(name='ReluHtpFp16', root=os.getcwd(),
                                        backend='HTP_FP16')

# now that we have the package object, we can start creating our operators
# For this example, we will create the Relu operator for HTP
relu_op_def = make_op_def(name="Relu", description="Consumes an input activation and computes"
                                                   " the rectified linear units (Relu) values")

# We would need to define: Relu: inputs and outputs. Relu has no params to define.

# define Relu inputs and outputs
# add inputs to op definition
relu_op_def.add_inputs(make_input_tensor_info(name='data',
                                              description='Data input to a Relu operation',
                                              datatypes=[QNN_DATATYPE_FLOAT_16],
                                              layout=Layout.NHWC,
                                              mandatory=True,
                                              rank=Rank.BATCH_IMAGE))

# add outputs to op def
# since we only have one output, no need to create a variable
relu_op_def.add_outputs(make_output_tensor_info(name='output',
                                                description='The output of the Relu operation',
                                                datatypes=[QNN_DATATYPE_FLOAT_16],
                                                layout=Layout.NHWC,
                                                rank=Rank.BATCH_IMAGE))

# now that the op_def is constructed, we can add it to the package info at once
example_package_htp_fp16.add_op_def(relu_op_def)

# once added we can check to see if the ops are present
if not example_package_htp_fp16.has_operator("Relu"):
    raise LookupError("Relu op was not registered")

# we can also print the package info to verify all our info
# print(example_package_htp_fp16)

# now that the package is created we can add it to the package manager object
# note that if the package name is not unique, we can pass option to mangle the
# name using a incremental combination of package_name, backend, version and domain.
# This turned off by default, meaning a non-unique package name would fail.
example_package_manager.add_package_info(example_package_htp_fp16)

# note we can also defer creation of the package until this point.
# Here we can create and register the package info at once (package name must be unique)
# example_package_manager.create_package_info \
# (name='ReluHtp', root=os.getcwd(), backend='HTP', \
# op_defs=[relu_op_def], register=True)

# when we are done, we can simply generate the packages
# note this will also generate xml files in the config directory
example_package_manager.generate_packages()
