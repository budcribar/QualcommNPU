//=============================================================================
//
//  Copyright (c) 2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

#include "QnnTFLiteDelegate.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"

#define TFLITE_FUNCTION_CHECK(x)                                       \
  if (!(x)) {                                                          \
    fprintf(stderr, "Error in main function at %s:%d\n", __FUNCTION__, \
            __LINE__);                                                 \
    return false;                                                      \
  }

// In HTP backend, Quantized precision mode also support some float type op,
// If Quantized precision mode, these op still should be pass even though it
// is float type
const std::unordered_set<std::int32_t> htp_quantized_support_float_datatype_op{
    kTfLiteBuiltinCast,       kTfLiteBuiltinReshape, kTfLiteBuiltinTranspose,
    kTfLiteBuiltinDequantize, kTfLiteBuiltinPow,     kTfLiteBuiltinQuantize};

bool IsHtpQuantizedSupportFloatDataTypeOp(const std::int32_t& op_type) {
  return htp_quantized_support_float_datatype_op.count(op_type) > 0;
}

class test {
 public:
  test()
      : delegate_(tflite::Interpreter::TfLiteDelegatePtr(nullptr, nullptr)) {}
  bool CreateInterpreterAndDelegate(std::string model_path) {
    model_file_name_ = model_path;
    tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
    std::unique_ptr<tflite::FlatBufferModel> model =
        tflite::FlatBufferModel::BuildFromFile(model_file_name_.c_str());
    tflite::InterpreterBuilder builder(*model, resolver);
    builder(&interpreter_);
    TFLITE_FUNCTION_CHECK(interpreter_ != nullptr);

    TFLITE_FUNCTION_CHECK(TfLiteGetFP32NodeIds());
    // qnn delegate options
    delegate_options_ = TfLiteQnnDelegateOptionsDefault();
    delegate_options_.backend_type = kHtpBackend;
    delegate_options_.htp_options.precision = kHtpQuantized;
    // skip node ids option
    delegate_options_.skip_options.skip_delegate_node_ids =
        fp32_node_ids_vec_.data();
    delegate_options_.skip_options.skip_delegate_node_ids_nr =
        fp32_node_ids_vec_.size();
    TfLiteDelegate* delegate_ptr = TfLiteQnnDelegateCreate(&delegate_options_);
    delegate_ = tflite::Interpreter::TfLiteDelegatePtr(
        delegate_ptr,
        [](TfLiteDelegate* delegate) { TfLiteQnnDelegateDelete(delegate); });
    TfLiteStatus rval = interpreter_->ModifyGraphWithDelegate(delegate_.get());
    TFLITE_FUNCTION_CHECK(rval == kTfLiteOk);
    return true;
  }

  bool TfLiteGetFP32NodeIds() {
    std::unordered_set<TfLiteType> floating_point_types{
        kTfLiteFloat16, kTfLiteFloat32, kTfLiteFloat64};

    const std::vector<int> execution_plan = interpreter_->execution_plan();
    for (const int& node_index : execution_plan) {
      const std::pair<TfLiteNode, TfLiteRegistration>* node_and_reg_pair =
          interpreter_->node_and_registration(node_index);
      size_t floating_point_tensor_number = 0;
      // Float type are support op, skip
      if (IsHtpQuantizedSupportFloatDataTypeOp(
              node_and_reg_pair->second.builtin_code)) {
        continue;
      }

      for (const auto& input_tensor_index :
           tflite::TfLiteIntArrayView(node_and_reg_pair->first.inputs)) {
        const TfLiteTensor* input_tensor =
            interpreter_->tensor(input_tensor_index);
        floating_point_tensor_number +=
            floating_point_types.count(input_tensor->type);
      }
      for (const auto& output_tensor_index :
           tflite::TfLiteIntArrayView(node_and_reg_pair->first.outputs)) {
        const TfLiteTensor* output_tensor =
            interpreter_->tensor(output_tensor_index);

        floating_point_tensor_number +=
            floating_point_types.count(output_tensor->type);
      }
      if (floating_point_tensor_number > 0) {
        fp32_node_ids_vec_.emplace_back(node_index);
      }
    }

    std::cout << "There are " << fp32_node_ids_vec_.size()
              << " floating point input or output in the " << model_file_name_
              << std::endl;
    std::cout << "Skip Node List: ";
    for (int i = 0; i < fp32_node_ids_vec_.size(); i++) {
      std::cout << fp32_node_ids_vec_[i] << ", ";
    }
    std::cout << std::endl;

    return true;
  }

 private:
  std::string model_file_name_{"mix_precision_sample.tflite"};
  std::unique_ptr<tflite::Interpreter> interpreter_;
  tflite::Interpreter::TfLiteDelegatePtr delegate_;
  std::vector<int> fp32_node_ids_vec_;
  TfLiteQnnDelegateOptions delegate_options_;
};

int main(int argc, char* argv[]) {
  if (argc < 2) {
    fprintf(stderr,
            "Compiled " __DATE__ __TIME__
            "\n"
            "Usage!!!: %s <tflite model>"
            "\n",
            argv[0]);
    return 1;
  }
  std::string model_path = argv[1];

  test testClass;
  testClass.CreateInterpreterAndDelegate(model_path);
  return 0;
}
