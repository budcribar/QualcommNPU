#===========================================================================
#
#  Copyright (c) 2022-2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#===========================================================================

apt-get install python3-venv
python3 -m venv pyenv
source pyenv/bin/activate
python -m pip install -U pip
pip install tensorflow
pip install tensorflow_datasets==4.1
pip install tf2onnx
pip install packaging
pip install pandas
${QNN_SDK_ROOT}/bin/check-python-dependency

git clone https://github.com/ARM-software/sesr
cd sesr/
git apply ${QNN_SDK_ROOT}/examples/Models/SESR/sesr.patch
export PYTHONPATH=$PWD:$PYTHONPATH
cd ..


python ${QNN_SDK_ROOT}/examples/Models/SESR/model_gen.py
python ${QNN_SDK_ROOT}/examples/Models/SESR/modify_model.py
python ${QNN_SDK_ROOT}/examples/Models/SESR/extract_weights.py

cp ${QNN_SDK_ROOT}/examples/Models/SESR/rand_input_* .
cp ${QNN_SDK_ROOT}/examples/Models/SESR/PerfSetting.conf .

qnn-onnx-converter \
      --input_network modified_model.onnx \
      --input_list ${QNN_SDK_ROOT}/examples/Models/SESR/input_list_dynamic.txt \
      --input_dim input_1 1,256,256,1 \
      --input_dim sesr/linear_block_c/Conv2D/ReadVariableOp:0_as_input 32,1,5,5 \
      --input_dim sesr/linear_block_c/BiasAdd/ReadVariableOp:0_as_input 32 \
      --input_dim sesr/linear_block_c_1/Conv2D/ReadVariableOp:0_as_input 32,32,3,3 \
      --input_dim sesr/linear_block_c_1/BiasAdd/ReadVariableOp:0_as_input 32 \
      --input_dim sesr/linear_block_c_2/Conv2D/ReadVariableOp:0_as_input 32,32,3,3 \
      --input_dim sesr/linear_block_c_2/BiasAdd/ReadVariableOp:0_as_input 32 \
      --input_dim sesr/linear_block_c_3/Conv2D/ReadVariableOp:0_as_input 32,32,3,3 \
      --input_dim sesr/linear_block_c_3/BiasAdd/ReadVariableOp:0_as_input 32 \
      --input_dim sesr/linear_block_c_4/Conv2D/ReadVariableOp:0_as_input 32,32,3,3 \
      --input_dim sesr/linear_block_c_4/BiasAdd/ReadVariableOp:0_as_input 32 \
      --input_dim sesr/linear_block_c_5/Conv2D/ReadVariableOp:0_as_input 32,32,3,3 \
      --input_dim sesr/linear_block_c_5/BiasAdd/ReadVariableOp:0_as_input 32 \
      --input_dim sesr/linear_block_c_6/Conv2D/ReadVariableOp:0_as_input 32,32,3,3 \
      --input_dim sesr/linear_block_c_6/BiasAdd/ReadVariableOp:0_as_input 32 \
      --input_dim sesr/linear_block_c_7/Conv2D/ReadVariableOp:0_as_input 32,32,3,3 \
      --input_dim sesr/linear_block_c_7/BiasAdd/ReadVariableOp:0_as_input 32 \
      --input_dim sesr/linear_block_c_8/Conv2D/ReadVariableOp:0_as_input 32,32,3,3 \
      --input_dim sesr/linear_block_c_8/BiasAdd/ReadVariableOp:0_as_input 32 \
      --input_dim sesr/linear_block_c_9/Conv2D/ReadVariableOp:0_as_input 32,32,3,3 \
      --input_dim sesr/linear_block_c_9/BiasAdd/ReadVariableOp:0_as_input 32 \
      --input_dim sesr/linear_block_c_10/Conv2D/ReadVariableOp:0_as_input 32,32,3,3 \
      --input_dim sesr/linear_block_c_10/BiasAdd/ReadVariableOp:0_as_input 32 \
      --input_dim sesr/linear_block_c_11/Conv2D/ReadVariableOp:0_as_input 32,32,3,3 \
      --input_dim sesr/linear_block_c_11/BiasAdd/ReadVariableOp:0_as_input 32 \
      --input_dim sesr/linear_block_c_12/Conv2D/ReadVariableOp:0_as_input 4,32,5,5 \
      --input_dim sesr/linear_block_c_12/BiasAdd/ReadVariableOp:0_as_input 4 \
      --output_path quant_modified_model.cpp


python ${QNN_SDK_ROOT}/examples/Models/SESR/quantize_inputs.py --model_json quant_modified_model_net.json

qnn-model-lib-generator -c quant_modified_model.cpp

qnn-net-run \
      --backend ${QNN_SDK_ROOT}/lib/x86_64-linux-clang/libQnnHtp.so \
      --model libs/x86_64-linux-clang/libqnn_model.so --output_dir x86_htp_outputs \
      --input_list ${QNN_SDK_ROOT}/examples/Models/SESR/input_list_dynamic_quant.txt \
      --input_data_type native \
      --config_file ${QNN_SDK_ROOT}/examples/Models/SESR/HtpConfigFile.json