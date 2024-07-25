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
example_package_dsp = make_package_info(name='ExamplePackage', root=os.getcwd(),
                                        backend='HTP')

# now that we have the package object, we can start creating our operators
# For this example, we will create two operators: Softmax and Conv2D
conv2d_op_def = make_op_def(name="Conv2D", description="consumes an input tensor and a filter, "
                                                       "and computes the overlap as the output")

# We would need to define: Conv 2D: inputs, outputs and params
# Softmax inputs, outputs and params

# define conv2d inputs and outputs
conv2d_data_input = make_input_tensor_info(name='data',
                                           description='Data input to a convolution operation',
                                           datatypes=[QNN_DATATYPE_UFIXED_POINT_16,
                                                      QNN_DATATYPE_UFIXED_POINT_8],
                                           layout=Layout.NHWC,
                                           mandatory=True,
                                           rank=Rank.IMAGE)

conv2d_filter_input = make_input_tensor_info(name='filter',
                                             description='The weights for the kernel',
                                             datatypes=[QNN_DATATYPE_UFIXED_POINT_8],
                                             layout=Layout.NHWC,
                                             mandatory=True,
                                             rank=Rank.IMAGE)

conv2d_bias_input = make_input_tensor_info(name='bias',
                                           description='The bias applied to filtered data',
                                           datatypes=[QNN_DATATYPE_UFIXED_POINT_8,
                                                      QNN_DATATYPE_SFIXED_POINT_16],
                                           layout=Layout.NHWC,
                                           mandatory=False,
                                           rank=Rank.VECTOR)
# add inputs to op definition
conv2d_op_def.add_inputs(conv2d_data_input, conv2d_filter_input, conv2d_bias_input)

# add outputs to op def
# since we only have one output, no need to create a variable
conv2d_op_def.add_outputs(make_output_tensor_info(name='output',
                                                  description='The output of the convolution operation',
                                                  datatypes=[QNN_DATATYPE_UFIXED_POINT_8, QNN_DATATYPE_SFIXED_POINT_16],
                                                  layout=Layout.NHWC,
                                                  rank=Rank.IMAGE))

# now define params
stride = make_tensor_param(name="stride",
                           description="The step size between consecutive elements in the data along each spatial axis",
                           mandatory=True,
                           datatypes=[QNN_DATATYPE_UINT_32],
                           rank=Rank.VECTOR)

dilation = make_tensor_param(name="dilation",
                             description="The number of elements inserted between consecutive elements in the kernel",
                             mandatory=False,
                             default=[1, 1],
                             datatypes=[QNN_DATATYPE_UINT_32],
                             rank=Rank.VECTOR)

group = make_scalar_param(name="group",
                          description="The number of divisions between input and output channels",
                          mandatory=False,
                          default=1,
                          datatypes=[QNN_DATATYPE_UINT_32])

pad_amount = make_tensor_param(name="pad_amount",
                               description="The number of pad values to add along each spatial axis",
                               mandatory=True,
                               datatypes=[QNN_DATATYPE_UINT_32],
                               rank=Rank.VECTOR)

# now add params in the same fashion
conv2d_op_def.add_params(stride, pad_amount, group, dilation)

# Similarly we can create the Softmax op
# Since we only have a single input, output and param we can chain the calls
softmax_op_def = make_op_def(name="Softmax", description="Computes the normalized exponential "
                                                         "for each batch of its input data")
softmax_op_def. \
    add_inputs(make_input_tensor_info(name='data',
                                      description='Data input to a softmax operation',
                                      datatypes=[QNN_DATATYPE_FLOAT_32,
                                                 QNN_DATATYPE_UFIXED_POINT_8],
                                      layout=Layout.NHWC,
                                      mandatory=True,
                                      rank=Rank.IMAGE)). \
    add_outputs(make_output_tensor_info(name='output',
                                        description='The output of the softmax operation',
                                        datatypes=[QNN_DATATYPE_FLOAT_32, QNN_DATATYPE_UFIXED_POINT_8],
                                        layout=Layout.NHWC,
                                        rank=Rank.IMAGE)). \
    add_params(make_scalar_param(name="axis",
                                 description="The axis denoting how the input will be coerced to 2D",
                                 mandatory=False,
                                 default=0,
                                 datatypes=[QNN_DATATYPE_UINT_32]))

# now that the op_def is constructed, we can add it to the package info at once
example_package_dsp.add_op_def(conv2d_op_def, softmax_op_def)

# once added we can check to see if the ops are present
if not example_package_dsp.has_operator("Conv2D"):
    raise LookupError("Conv op not registered")
if not example_package_dsp.has_operator("Softmax"):
    raise LookupError("Softmax op not registered")

# we can also print the package info to verify all our info
# print(example_package_dsp)

# now that the package is created we can add it to the package manager object
# note that if the package name is not unique, we can pass option to mangle the
# name using a incremental combination of package_name, backend, version and domain.
# This turned off by default, meaning a non-unique package name would fail.
example_package_manager.add_package_info(example_package_dsp)

# note we can also defer creation of the package until this point.
# Here we can create and register the package info at once (package name must be unique)
# example_package_manager.create_package_info(name='ExamplePackage', root=os.getcwd(), backend='Dsp',
# op_defs=[conv2d_op_def, softmax_op_def], register=True)

# when we are done, we can simply generate the packages
# note this will also generate xml files in the config directory
example_package_manager.generate_packages()
