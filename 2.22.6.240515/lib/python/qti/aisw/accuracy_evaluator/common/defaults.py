##############################################################################
#
# Copyright (c) 2022-2023 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# All data and information contained in or disclosed by this document are
# confidential and proprietary information of Qualcomm Technologies, Inc., and
# all rights therein are expressly reserved. By accepting this material, the
# recipient agrees that this material and the information contained therein
# are held in confidence and in trust and will not be used, copied, reproduced
# in whole or in part, nor its contents revealed in any manner to others
# without the express written permission of Qualcomm Technologies, Inc.
#
##############################################################################
import qti.aisw.accuracy_evaluator.common.exceptions as ce

DEFAULTS_VALUES = {
    "common": {
        "inference_retry_count": "4",
        "inference_timeout": "5000",
    },
    "qacc": {
        "auto_quantization": {
            "algorithms": "default | cle",
            "param_quantizer": "tf | symmetric | enhanced | adjusted",
            "use_per_channel_quantization": "True | False",
            "use_per_row_quantization": "True | False"
        },
        "comparator": {
            "enabled": True,
            "fetch-top": 1,
            "tol": 0.01,
            "type": "abs"
        },
        "default_inference_schemas": {
            # Setting default inference schema to run int8 precision model on AIC device using QNN engine
            "aic_x86": {
                "inference_schema": {
                    "backend": "aic",
                    "backend_extensions": {
                        "compiler_num_of_cores": 8,
                        "compiler_perfWarnings": True
                    },
                    "converter_params": {
                        "algorithms": "default | cle",
                        "param_quantizer": "tf | symmetric | enhanced | adjusted",
                        "use_per_channel_quantization": "True | False",
                        "use_per_row_quantization": "True | False"
                    },
                    "name": "qnn",
                    "precision": "quant",
                    "tag": "qnn_int8",
                    "target_arch": "x86_64-linux-clang"
                }
            },
            # Setting default inference schema to run fp32 precision model on cpu of host machine using QNN engine
            "cpu": {
                "inference_schema": {
                    "backend": "cpu",
                    "name": "qnn",
                    "precision": "fp32",
                    "tag": "qnn_cpu_fp32",
                    "target_arch": "x86_64-linux-clang"
                }
            },
            # Setting default inference schema to run fp32 precision model on cpu of android device using QNN engine
            "cpu_android": {
                "inference_schema": {
                    "backend": "cpu",
                    "name": "qnn",
                    "precision": "fp32",
                    "tag": "qnn_cpu_fp32_android",
                    "target_arch": "aarch64-android"
                }
            },
            # Setting default inference schema to run fp32 precision model on GPU backend in android device (aarch64-android architecture) using QNN engine
            "gpu": {
                "inference_schema": {
                    "backend": "gpu",
                    "name": "qnn",
                    "precision": "fp32",
                    "tag": "qnn_gpu_fp32",
                    "target_arch": "aarch64-android"
                }
            },
            # Setting default inference schema to run int8 precision model on waipio device (aarch64-android architecture) using QNN engine
            "dspv69": {
                "inference_schema": {
                    "backend": "dspv69",
                    "converter_params": {
                        "algorithms": "default | cle",
                        "param_quantizer": "tf | symmetric | enhanced | adjusted",
                        "use_per_channel_quantization": "True | False",
                        "use_per_row_quantization": "True | False"
                    },
                    "name": "qnn",
                    "precision": "quant",
                    "backend_extensions": {
                        "rpc_control_latency": 100,
                        "vtcm_mb": 4
                    },
                    "tag": "qnn_int8",
                    "target_arch": "aarch64-android"
                }
            },
            # Setting default inference schema to run int8 precision model on kailua device (aarch64-android architecture) using QNN engine
            "dspv73": {
                "inference_schema": {
                    "backend": "dspv73",
                    "converter_params": {
                        "algorithms": "default | cle",
                        "param_quantizer": "tf | symmetric | enhanced | adjusted",
                        "use_per_channel_quantization": "True | False",
                        "use_per_row_quantization": "True | False"
                    },
                    "name": "qnn",
                    "precision": "quant",
                    "backend_extensions": {
                        "rpc_control_latency": 100,
                        "vtcm_mb": 4
                    },
                    "tag": "qnn_int8",
                    "target_arch": "aarch64-android"
                }
            },
            # Setting default inference schema to run int8 precision model on lanai device (aarch64-android architecture) using QNN engine
            "dspv75": {
                "inference_schema": {
                    "backend": "dspv75",
                    "converter_params": {
                        "algorithms": "default | cle",
                        "param_quantizer": "tf | symmetric | enhanced | adjusted",
                        "use_per_channel_quantization": "True | False",
                        "use_per_row_quantization": "True | False"
                    },
                    "name": "qnn",
                    "precision": "quant",
                    "backend_extensions": {
                        "rpc_control_latency": 100,
                        "vtcm_mb": 4
                    },
                    "tag": "qnn_int8",
                    "target_arch": "aarch64-android"
                }
            },
            # Setting default inference schema to run int8 precision model with htp simulation (x86_64-linux-clang architecture) using QNN engine
            "htp_x86": {
                "inference_schema": {
                    "backend": "htp",
                    "converter_params": {
                        "algorithms": "default | cle",
                        "param_quantizer": "tf | symmetric | enhanced | adjusted",
                        "use_per_channel_quantization": "True | False",
                        "use_per_row_quantization": "True | False"
                    },
                    "name": "qnn",
                    "precision": "quant",
                    "backend_extensions": {
                        "rpc_control_latency": 100,
                        "vtcm_mb": 4
                    },
                    "tag": "qnn_int8",
                    "target_arch": "x86_64-linux-clang"
                }
            },
            # Setting default inference schema to run int8 precision model with htp_mcp simulation (x86_64-linux-clang architecture) using QNN engine
            "htp_mcp_x86": {
                "inference_schema": {
                    "backend": "htp_mcp",
                    "converter_params": {
                        "algorithms": "default | cle",
                        "param_quantizer": "tf | symmetric | enhanced | adjusted",
                        "use_per_channel_quantization": "True | False",
                        "use_per_row_quantization": "True | False"
                    },
                    "name": "qnn",
                    "precision": "quant",
                    "backend_extensions": {
                        "num_cores": 1,
                        "heap_size": 256,
                        "elf_path": "lib/hexagon-v68/unsigned/libQnnHtpMcpV68.elf",
                        "timeout": 5000
                    },
                    "tag": "qnn_int8",
                    "target_arch": "x86_64-linux-clang"
                }
            }
        },
        "file_type": {
            "infer": "bin, raw",
            "postproc": "txt",
            "preproc": "raw"
        },
        "zero_output_check": False
    }
}


class Defaults:
    """Default configuration class having all the default values supplied by
    the tool or user."""
    __instance = None

    def __init__(self, app=None):
        if Defaults.__instance != None:
            pass
        else:
            self.load_defaults()
            Defaults.__instance = self
        self._app = app

    @classmethod
    def getInstance(cls, app=None):
        if Defaults.__instance == None:
            Defaults(app=app)
        return cls.__instance

    def load_defaults(self):
        """Loads defaults from the defaults Values dict"""
        self.values = DEFAULTS_VALUES

    def get_value(self, key_string):
        """Returns value from nested defaults dictionary.

        Args:
            key_string: nested keys in string format eg key.key.key

        Returns:
            value: value associated to the key, None otherwise

        Raises:
            DefaultsException: if key not found
        """
        keys = key_string.split('.')
        nested = self.values
        for key in keys:
            if key in nested:
                nested = nested[key]
            else:
                raise ce.DefaultsException('key: {} not found in defaults.yaml file'.format(key))
        return nested

    def set_value(self, key_string, value):
        """Updates the value for the key string in nested defaults dictionary.

        Args:
            key_string: nested keys in string format eg key.key.key
            value: Value to be updated for the key_string passed

        Returns:
            value: value associated to the key, None otherwise

        Raises:
            DefaultsException: if key not found
        """
        keys = key_string.split('.')
        nested = self.values
        updated_key = keys[-1]  # key to update with the provided value
        for idx, key in enumerate(keys[:-1]):  # Loop till the last sub dict
            if key in nested:
                nested = nested[key]  # Keep fetching internal dict
                if idx == len(keys) - 2:  # Stop one level above the actual dict to update.
                    # Update the last subdict with passed key:value
                    nested.update({updated_key: value})
            else:
                raise ce.DefaultsException('key: {} not found in defaults.yaml file'.format(key))
