/* =============================================================================
  Copyright (c) 2021 Qualcomm Technologies, Inc.
  All Rights Reserved.
  Confidential and Proprietary - Qualcomm Technologies, Inc.

 =============================================================================
*/

const {PythonShell} = require('python-shell');
const path = require('path');
const fs = require('fs');
const server_file_name = 'pythonserver.py'

/**
 * Flattens json object into a list.
 * @param object json object
 * @returns {null|*[]} list
 */
function flatten(object) {
    if (object === null) {
        return null;
    }
    return Object.keys(object).reduce(function (r, k) {
        return r.concat(k, object[k]);
    }, []);
}

/**
 * Gets file path to entry point of Python backend.
 * @returns {string|boolean} directory of pythonserver.py
 */
function getServerDir() {
    // package location
    if (fs.existsSync(path.join(__dirname, '../../app.asar.unpacked/tools/'))) {
        return path.join(__dirname, '../../app.asar.unpacked/tools/')
    }
    // build or developer location
    else if (fs.existsSync(path.join(__dirname, '../tools/'))) {
        return path.join(__dirname, '../tools/')
    } else {
        try {
            alert("Unable to locate Python backend.")
        } catch {
            throw "Unable to locate Python backend."
        }
        return false;
    }
}

/**
 * Processes user-specified parameters from the GUI and passes them to the Python backend.
 * @param os_args json object of os arguments required for Windows execution (empty string if executing on Linux)
 * @param linux_workspace output directory for all files created by running the tool
 * @param use_case INFERENCE_V_INFERENCE (1), or GOLDEN_V_INFERENCE (2), or OUTPUT_V_OUTPUT (3)
 * @param inf1_args json object of arguments for first inference configuration (null for use case 3)
 * @param inf2_args json object of arguments for second inference configuration (null for use case 2 and 3)
 * @param verif_args json object of arguments for verification
 * @param verif_param_json Path to the verifiers config file
 * @returns {Promise<unknown>} resolve if backend completes without error; reject otherwise
 */
module.exports = function (os_args, linux_workspace, use_case, inf1_args, inf2_args, verif_args, verif_param_json = null) {
    let options = {args: [os_args, linux_workspace, use_case, flatten(inf1_args), flatten(inf2_args), flatten(verif_args),verif_param_json]}
    let server_path = path.join(getServerDir(), server_file_name)

    return new Promise((resolve, reject) => {
        // wraps the spawning of an async child process
        let shell = new PythonShell(server_path, options);

        // collect all print statements from Python backend
        let messages = []
        shell.on('message', function (message) {
            messages.push(message)
        })

        shell.end(function (failure) {
            if (failure) {
                let err = "";
                if (messages.length > 0) {
                    err = messages[messages.length - 1]
                }
                else if (failure.message.length > 0) {
                    err = failure.message;
                } else {
                    err = failure.stack.toString();
                }
                reject(Error(err));
            }
            else resolve();
        });
    })
}
