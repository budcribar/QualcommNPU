/* =============================================================================
  Copyright (c) 2021-2022 Qualcomm Technologies, Inc.
  All Rights Reserved.
  Confidential and Proprietary - Qualcomm Technologies, Inc.

  MIT License

  Copyright (c) Lutz Roeder

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
 =============================================================================
*/

/*
    * OPEN_SOURCE_START
    * The following code is derived from the netron open source project in order to setup UI event listeners and setup
    * netron GUI functionality and button actions
    * Project Link: https://github.com/lutzroeder/netron/
*/
var host = host || {};
const electron = require('electron');
const fs = require('fs');
const http = require('http');
const https = require('https');
const process = require('process');
const path = require('path');
const os = require('os');
const querystring = require('querystring');

host.ElectronHost = class {

    constructor() {
        process.on('uncaughtException', (err) => {
            this.exception(err, true);
        });
        this._document = window.document;
        this._window = window;
        this._window.eval = global.eval = () => {
            throw new Error('window.eval() not supported.');
        };
        this._window.addEventListener('unload', () => {
            if (typeof __coverage__ !== 'undefined') {
                const file = path.join('.nyc_output', path.basename(window.location.pathname, '.html')) + '.json';
                /* eslint-disable no-undef */
                fs.writeFileSync(file, JSON.stringify(__coverage__));
                /* eslint-enable no-undef */
            }
        });
        this._environment = electron.ipcRenderer.sendSync('get-environment', {});
        this._queue = [];
    }

    get window() {
        return this._window;
    }

    get document() {
        return this._document;
    }

    get version() {
        return this._environment.version;
    }

    get type() {
        return 'Diff Loading';
    }

    get browser() {
        return false;
    }

    initialize(view) {
        this._view = view;
        electron.ipcRenderer.on('open', (_, data) => {
            this._openFile(data.file);
        });
        return new Promise((resolve /*, reject */) => {
            resolve()
        });
    }

    start() {
        this._view.show('welcome');
        this.document.getElementById('welcome_spinner').classList.toggle('hide');
        if (this._queue) {
            const queue = this._queue;
            delete this._queue;
            if (queue.length > 0) {
                const file = queue.pop();
                this._openFile(file);
            }
        }

        electron.ipcRenderer.on('export', (_, data) => {
            this._view.export(data.file);
        });
        electron.ipcRenderer.on('cut', () => {
            this._view.cut();
        });
        electron.ipcRenderer.on('copy', () => {
            this._view.copy();
        });
        electron.ipcRenderer.on('paste', () => {
            this._view.paste();
        });
        electron.ipcRenderer.on('selectall', () => {
            this._view.selectAll();
        });
        this.document.addEventListener('dragover', (e) => {
            e.preventDefault();
        });
        this.document.addEventListener('drop', (e) => {
            e.preventDefault();
        });
        this.document.body.addEventListener('drop', (e) => {
            e.preventDefault();
            const files = Array.from(e.dataTransfer.files).map(((file) => file.path));
            if (files.length > 0) {
                electron.ipcRenderer.send('drop-files', { files: files });
            }
            return false;
        });
        electron.ipcRenderer.on('toggle-attributes', () => {
            this._view.toggleAttributes();
            this._update('show-attributes', this._view.showAttributes);
        });
        electron.ipcRenderer.on('toggle-initializers', () => {
            this._view.toggleInitializers();
            this._update('show-initializers', this._view.showInitializers);
        });
        electron.ipcRenderer.on('toggle-names', () => {
            this._view.toggleNames();
            this._update('show-names', this._view.showNames);
        });
        electron.ipcRenderer.on('toggle-direction', () => {
            this._view.toggleDirection();
            this._update('show-horizontal', this._view.showHorizontal);
        });
        electron.ipcRenderer.on('zoom-in', () => {
            this.document.getElementById('zoom-in-button').click();
        });
        electron.ipcRenderer.on('zoom-out', () => {
            this.document.getElementById('zoom-out-button').click();
        });
        electron.ipcRenderer.on('reset-zoom', () => {
            this._view.resetZoom();
        });
        electron.ipcRenderer.on('show-properties', () => {
            this.document.getElementById('menu-button').click();
        });
        electron.ipcRenderer.on('find', () => {
            this._view.find();
        });
        this.document.getElementById('menu-button').addEventListener('click', () => {
            this._view.showModelProperties();
        });

        /* OPEN_SOURCE_END */

        //toggle the different forms for the different use cases
        var jquery;
        jquery = require('jquery');
        jquery(document).ready(function () {
            toggleFields();
            toggleDevFields('#targetDevice_2','#devices_config_path_2','#devices_config_path_2_p',
                '#x86_64-linux-clang_2','#aarch64-android_2','#arch_2',
                '#runtime_2', '#cpu_2', '#gpu_2', '#dspv68_2', '#dspv69_2', '#dspv73_2');
            toggleDevFields('#tarDev_1','#devices_config_path_1','#devices_config_path_1_p',
                '#x86_64-linux-clang_1','#aarch64-android_1','#arch_1',
                '#runtime_1', '#cpu_1', '#gpu_1', '#dspv68_1', '#dspv69_1', '#dspv73_1');

            toggleDevFields('#tarDev_1_2','#devices_config_path_1_2','#devices_config_path_1_2_p',
                '#x86_64-linux-clang_1_2','#aarch64-android_1_2','#arch_1_2',
                '#runtime_1_2', '#cpu_1_2', '#gpu_1_2', '#dspv68_1_2', '#dspv69_1_2', '#dspv73_1_2');
            toggleVerifierFields('#default_ver_1','#verifier_config_text_1','#verifier_config_text_2','#verifier_input_1','#verifier_input_2');


            jquery("#caseType").change(function () {
                toggleFields();
            });
            jquery("#targetDevice_2").change(function () {
                toggleDevFields('#targetDevice_2','#devices_config_path_2','#devices_config_path_2_p',
                    '#x86_64-linux-clang_2','#aarch64-android_2','#arch_2',
                    '#runtime_2', '#cpu_2', '#gpu_2', '#dspv68_2', '#dspv69_2', '#dspv73_2');

            });
            jquery("#tarDev_1").change(function () {
                toggleDevFields('#tarDev_1','#devices_config_path_1','#devices_config_path_1_p',
                    '#x86_64-linux-clang_1','#aarch64-android_1','#arch_1',
                    '#runtime_1', '#cpu_1', '#gpu_1', '#dspv68_1', '#dspv69_1', '#dspv73_1');
                toggleProfFields('#tarDev_1','#tarDev_1_2','#runtime_1','#runtime_1_2',"#prof_level_1");

            });
            jquery("#tarDev_1_2").change(function () {
                toggleDevFields('#tarDev_1_2','#devices_config_path_1_2','#devices_config_path_1_2_p',
                    '#x86_64-linux-clang_1_2','#aarch64-android_1_2','#arch_1_2',
                    '#runtime_1_2', '#cpu_1_2', '#gpu_1_2', '#dspv68_1_2', '#dspv69_1_2', '#dspv73_1_2');
                toggleProfFields('#tarDev_1','#tarDev_1_2','#runtime_1','#runtime_1_2',"#prof_level_1");

            });
            jquery("#runtime_1_2").change(function () {
                toggleProfFields('#tarDev_1','#tarDev_1_2','#runtime_1','#runtime_1_2',"#prof_level_1");

            });
            jquery("#runtime_1").change(function () {
                toggleProfFields('#tarDev_1','#tarDev_1_2','#runtime_1','#runtime_1_2',"#prof_level_1");
            });

            jquery("#prof_level_2").change(function () {
                if (jquery("#prof_level_2").val()=='on'){
                    jquery("#perf_threshold_input_2").show();
                }else{
                    jquery("#perf_threshold_input_2").hide();
                }


            });
            jquery("#prof_level_1").change(function () {
                if (jquery("#prof_level_1").val()=='on'){
                    jquery("#perf_threshold_input_1").show();
                }else{
                    jquery("#perf_threshold_input_1").hide();
                }


            });
            jquery("#default_ver_1").change(function () {
                toggleVerifierFields('#default_ver_1','#verifier_config_text_1','#verifier_config_text_2','#verifier_input_1','#verifier_input_2');
            });
            jquery("#default_ver_2").change(function () {
                toggleVerifierFields('#default_ver_2','#verifier_config_text_2_1','#verifier_config_text_2_2','#verifier_input_2_1','#verifier_input_2_2');
            });
            jquery("#default_ver_3").change(function () {
                toggleVerifierFields('#default_ver_3','#verifier_config_text_3_1','#verifier_config_text_3_2','#verifier_input_3_1','#verifier_input_3_2');
            });

            if (process.env.QNN_SDK_ROOT){
                let setEnginePath = (enginePath) => enginePath.textContent = enginePath.value = process.env.QNN_SDK_ROOT;
                setEnginePath(document.getElementById('dev_engine_path_2'));
                setEnginePath(document.getElementById('engine_path_1'));
                setEnginePath(document.getElementById('engine_path_1_2'));
            }

        });
        function toggleFields() {
            if (jquery("#caseType").val() === "usecase1")
                jquery("#useCase1Form").show();
            else
                jquery("#useCase1Form").hide();
            if (jquery("#caseType").val() === "usecase2")
                jquery("#useCase2Form").show();
            else
                jquery("#useCase2Form").hide();
            if (jquery("#caseType").val() === "usecase3")
                jquery("#useCase3Form").show();
            else
                jquery("#useCase3Form").hide();
        }
        //toggle config value based on if host or on-device is selected
        function toggleDevFields(tarDev, devConfig, devConfig_p, linux, aarch64, arch, runtime, cpu, gpu, dspv68,
                                 dspv69, dspv73) {
            if (jquery(tarDev).val() === "x86"){
                jquery(devConfig).val('host');
                jquery(devConfig_p).hide();
                jquery(arch).val('x86_64-linux-clang');
                jquery(linux).prop('disabled',false);
                jquery(aarch64).prop('disabled',true);
                jquery(runtime).val('cpu');
                jquery(gpu).prop('disabled',true);
                jquery(dspv68).prop('disabled',true);
                jquery(dspv69).prop('disabled',true);
                jquery(dspv73).prop('disabled',true);
            }else{
                jquery(devConfig).val('');
                jquery(devConfig_p).show();
                jquery(arch).val('aarch64-android');
                jquery(linux).prop('disabled',true);
                jquery(aarch64).prop('disabled',false);
                jquery(runtime).val('cpu');
                jquery(gpu).prop('disabled',false);
                jquery(dspv68).prop('disabled',false);
                jquery(dspv69).prop('disabled',false);
                jquery(dspv73).prop('disabled',false);
            }
        }

        //toggle prof level config value based on if host or on-device is selected
        function toggleProfFields(tar_dev_1, tar_dev_2,runtime_1 ,runtime_2,prof) {
            if (jquery(tar_dev_1).val() == jquery(tar_dev_2).val() && jquery(runtime_1).val() == jquery(runtime_2).val() ){
                jquery(prof).prop('disabled',false);
                jquery(prof).val('on');
                jquery("#profile_comparison_1").show();

            }else{
                jquery(prof).prop('disabled',true);
                jquery("#profile_comparison_1").hide();
                jquery(prof).val('off');
            }
        }
        function toggleVerifierFields(verif,text1,text2,input1,input2){
            switch(jquery(verif).val()){
                case "L1Error":
                    jquery(text1).show();
                    jquery(text2).show();
                    jquery(text1).text('multiplier');
                    jquery(text2).text('scale');
                    jquery(text1).css("display",'inline');
                    jquery(text2).css("display",'inline');
                    jquery(input1).show();
                    jquery(input2).show();
                    jquery(input1).prop('value','1.0');
                    jquery(input2).prop('value','1.0');
                    break;
                case "RtolAtol":
                    jquery(text1).show();
                    jquery(text2).show();
                    jquery(text1).css("display",'inline');
                    jquery(text2).css("display",'inline');
                    jquery(text1).text('rtolmargin');
                    jquery(text2).text('atolmargin');
                    jquery(input1).show();
                    jquery(input2).show();
                    jquery(input1).prop('value','0.01');
                    jquery(input2).prop('value','0.01');
                    break;
                case "TopK":
                    jquery(text1).show();
                    jquery(text2).show();
                    jquery(text1).css("display",'inline');
                    jquery(text2).css("display",'inline');
                    jquery(text1).text('k');
                    jquery(text2).text('ordered');
                    jquery(input1).show();
                    jquery(input2).show();
                    jquery(input1).prop('value','1');
                    jquery(input2).prop('value','false');
                    break;
                case "CosineSimilarity":
                    jquery(text1).show();
                    jquery(text2).show();
                    jquery(text1).css("display",'inline');
                    jquery(text2).css("display",'inline');
                    jquery(text1).text('multiplier');
                    jquery(text2).text('scale');
                    jquery(input1).show();
                    jquery(input2).show();
                    jquery(input1).prop('value','1.0');
                    jquery(input2).prop('value','1.0');
                    break;
                case "MeanIOU" :
                    jquery(text1).show();
                    jquery(text2).hide();
                    jquery(text1).css("display",'inline');

                    jquery(text1).text('background_classification');
                    jquery(input1).show();
                    jquery(input2).hide();
                    jquery(input1).prop('value','1.0');
                    jquery(input2).prop('value','1.0');
                    break;
                case "AdjustedRtolAtol":
                    jquery(text1).show();
                    jquery(text2).hide();
                    jquery(text1).css("display",'inline');
                    jquery(text1).text('levels_num');
                    jquery(input1).show();
                    jquery(input2).hide();
                    jquery(input1).prop('value','1');
                    jquery(input2).prop('value','1');
                    break;
                case "MSE":
                case "SQNR":
                case "MAE":
                    jquery(text1).hide();
                    jquery(text2).hide();
                    jquery(input1).hide();
                    jquery(input2).hide();
                    break;
                default:
                    jquery(text1).hide();
                    jquery(text2).hide();
                    jquery(input1).hide();
                    jquery(input2).hide();
            }
        }
        function getVeriferConfig(text1,input1,text2,input2){
            let ret = '';
            if(jquery(text1).css("display") !=  "none"){
                ret +=","
                ret += jquery(text1).text();
                ret +=","
                ret += jquery(input1).val();
            }
            if(jquery(text2).css("display") !=  "none"){
                ret +=","
                ret += jquery(text2).text();
                ret +=","
                ret += jquery(input2).val();
            }
            return ret;
        }


        //detect operating system
        let systemVal = this.document.getElementById("systemVal");
        systemVal.innerHTML="OS: " +os.type;


        //set up variables for use case 3
        let modelJson_3= this.document.getElementById('model_json_3');
        modelJson_3.addEventListener('click', () => {
            modelJson_3.textContent = modelJson_3.value = getFile();
            var modelJson_3_obj = JSON.parse(fs.readFileSync(modelJson_3.value));
            modelBin_3.textContent = modelBin_3.value = modelJson_3_obj['model.bin'];
        });

        let modelBin_3= this.document.getElementById('model_bin_3');
        modelBin_3.addEventListener('click', () => {
            modelBin_3.textContent = modelBin_3.value = getBinFile();
            //check file names
            let splitJson3 = modelJson_3.textContent.split('/');
            let splitBin3 = modelBin_3.textContent.split('/');
            if(splitJson3[splitJson3.length-2]!== splitBin3[splitBin3.length-2] ){
                alert('The Json and Bin file you entered may not be for the same model!')
            }
        });
        let modelRaw_3= this.document.getElementById('path_to_raw_3');
        modelRaw_3.addEventListener('click', () => {
            modelRaw_3.textContent = modelRaw_3.value = getPath();
        });
        let modelGoldens_3= this.document.getElementById('path_to_goldens_3');
        modelGoldens_3.addEventListener('click', () => {
            modelGoldens_3.textContent = modelGoldens_3.value = getPath();
        });
        let default_verifier_3 = '';
        let verifier_acc_thr_3 = 0.01;
        let verifier_perf_thr_3 = 1;
        let workspace_3= this.document.getElementById('workspace_3');
        workspace_3.addEventListener('click', () => {
            workspace_3.textContent = workspace_3.value = getPath();
        });

        //set up variables for use case 2
        let targetDev_2= '';
        let runtime_2= '';
        let architecture_2 = '';
        let default_verifier_2 = '';
        let verifier_acc_thr_2 = 0.01;
        let verifier_perf_thr_2 = 1;


        let modelJson_2= this.document.getElementById('model_json_2');
        modelJson_2.addEventListener('click', () => {
            modelJson_2.textContent = modelJson_2.value = getFile();
            var modelJson_2_obj = JSON.parse(fs.readFileSync(modelJson_2.value));
            modelCpp_2.textContent = modelCpp_2.value = modelJson_2_obj['model.cpp'];
            modelBin_2.textContent = modelBin_2.value = modelJson_2_obj['model.bin'];
        });
        let modelBin_2= this.document.getElementById('model_bin_2');
        modelBin_2.addEventListener('click', () => {
            modelBin_2.textContent = modelBin_2.value = getBinFile();
            let splitJson2 = modelJson_2.textContent.split('/');
            let splitBin2 = modelBin_2.textContent.split('/');
            if(splitJson2[splitJson2.length-2]!== splitBin2[splitBin2.length-2] ){
                alert('The Json and Bin file you entered may not be for the same model!')
            }
        });

        let modelCpp_2= this.document.getElementById('model_cpp_2');
        modelCpp_2.addEventListener('click', () => {
            modelCpp_2.textContent = modelCpp_2.value = getFile();
            let splitJson2_cpp = modelJson_2.textContent.split('/');
            let splitCpp2 = modelCpp_2.textContent.split('/');
            if (splitJson2_cpp[splitJson2_cpp.length - 2] !== splitCpp2[splitCpp2.length - 2]) {
                alert('The Json and Cpp file you entered may not be for the same model!')
            }
        });

        let ndkPath_2= this.document.getElementById('ndk_path_2');
        ndkPath_2.addEventListener('click', () => {
            ndkPath_2.textContent = ndkPath_2.value = getPath();
        });
        let devEnginePath_2= this.document.getElementById('dev_engine_path_2');
        devEnginePath_2.addEventListener('click', () => {
            devEnginePath_2.textContent = devEnginePath_2.value = getPath();
        });
        let inputList_2= this.document.getElementById('input_list_2');
        inputList_2.addEventListener('click', () => {
            inputList_2.textContent = inputList_2.value = getFile();
        });

        let modelGoldens_2= this.document.getElementById('path_to_goldens_2');
        modelGoldens_2.addEventListener('click', () => {
            modelGoldens_2.textContent = modelGoldens_2.value = getPath();
        });

        let workspace_2= this.document.getElementById('workspace_2');
        workspace_2.addEventListener('click', () => {
            workspace_2.textContent = workspace_2.value = getPath();
        });
        let inputDataType_2='';
        let outputDataType_2='';
        let profLevel_2='';
        let perfProf_2='';
        let devConfigPath_2='';

        //set up variables for use case 1
        //inference 1
        let targetDev_1 = '';
        let runtime_1 = '';
        let architecture_1 = '';
        let default_verifier_1 = '';
        let verifier_acc_thr_1 = 0.01;
        let verifier_perf_thr_1 = 1;

        let modelJson_1= this.document.getElementById('model_json_1');
        modelJson_1.addEventListener('click', () => {
            modelJson_1.textContent = modelJson_1.value = getFile();
            if(!modelJson_1_2.value){
                modelJson_1_2.textContent = modelJson_1_2.value = modelJson_1.textContent;
            }
            // Auto add the corresponding cpp and bin files
            var modelJson_1_obj = JSON.parse(fs.readFileSync(modelJson_1.value));
            modelCpp_1.textContent = modelCpp_1.value = modelJson_1_obj['model.cpp'];
            modelBin_1.textContent = modelBin_1.value = modelJson_1_obj['model.bin'];
            if(!modelCpp_1_2.value){
                modelCpp_1_2.textContent = modelCpp_1_2.value = modelJson_1_obj['model.cpp'];
            }
            if(!modelBin_1_2.value) {
                modelBin_1_2.textContent = modelBin_1_2.value = modelJson_1_obj['model.bin'];
            }
        });

        let modelCpp_1= this.document.getElementById('model_cpp_1');
        modelCpp_1.addEventListener('click', () => {
            modelCpp_1.textContent = modelCpp_1.value = getFile();
            if (!modelCpp_1_2.value) {
                modelCpp_1_2.textContent = modelCpp_1_2.value = modelCpp_1.textContent;
            }
            let splitJson1_cpp = modelJson_1.textContent.split('/');
            let splitCpp1 = modelCpp_1.textContent.split('/');
            if (splitJson1_cpp[splitJson1_cpp.length - 2] !== splitCpp1[splitCpp1.length - 2]) {
                alert('The Json and Cpp file you entered may not be for the same model!')
            }
        });

        let modelBin_1= this.document.getElementById('model_bin_1');
        modelBin_1.addEventListener('click', () => {
            modelBin_1.textContent = modelBin_1.value = getBinFile();
            if (!modelBin_1_2.value) {
                modelBin_1_2.textContent = modelBin_1_2.value = modelBin_1.textContent;
            }
            let splitJson1_bin = modelJson_1.textContent.split('/');
            let splitBin1 = modelBin_1.textContent.split('/');
            if (splitJson1_bin[splitJson1_bin.length - 2] !== splitBin1[splitBin1.length - 2]) {
                alert('The Json and Bin file you entered may not be for the same model!')
            }
        });
        let ndkPath_1= this.document.getElementById('ndk_path_1');
        ndkPath_1.addEventListener('click', () => {
            ndkPath_1.textContent = ndkPath_1.value = getPath();
            if (!ndk_path_1_2.value) {
                ndk_path_1_2.textContent = ndk_path_1_2.value = ndkPath_1.textContent;
            }
        });

        let devEnginePath_1= this.document.getElementById('engine_path_1');
        devEnginePath_1.addEventListener('click', () => {
            devEnginePath_1.textContent = devEnginePath_1.value = getPath();
            if (!devEnginePath_1_2.value) {
                devEnginePath_1_2.textContent = devEnginePath_1_2.value = devEnginePath_1.textContent;
            }
        });
        let inputList_1= this.document.getElementById('input_list_1');
        inputList_1.addEventListener('click', () => {
            inputList_1.textContent = inputList_1.value = getFile();
            if (!inputList_1_2.value) {
                inputList_1_2.textContent = inputList_1_2.value = inputList_1.textContent;
            }
        });
        let workspace_1= this.document.getElementById('workspace_1');
        workspace_1.addEventListener('click', () => {
            workspace_1.textContent= workspace_1.value = getPath();
        });

        let inputDataType_1='';
        let outputDataType_1='';
        let profLevel_1='';
        let perfProf_1='';
        let devConfigPath_1='';


        //inference 2
        let targetDev_1_2 = '';
        let runtime_1_2= '';
        let architecture_1_2 = '';
        //let default_verifier_1_2 = '';
        let modelJson_1_2= this.document.getElementById('model_json_1_2');
        modelJson_1_2.addEventListener('click', () => {
            modelJson_1_2.textContent = modelJson_1_2.value = getFile();
            // Auto add the corresponding cpp and bin files
            var modelJson_1_2_obj = JSON.parse(fs.readFileSync(modelJson_1_2.value));
            modelCpp_1_2.textContent = modelCpp_1_2.value = modelJson_1_2_obj['model.cpp'];
            modelBin_1_2.textContent = modelBin_1_2.value = modelJson_1_2_obj['model.bin'];
        });
        let modelCpp_1_2= this.document.getElementById('model_cpp_1_2');
        modelCpp_1_2.addEventListener('click', () => {
            modelCpp_1_2.textContent = modelCpp_1_2.value = getFile();
            let splitJson1_2_cpp = modelJson_1_2.textContent.split('/');
            let splitCpp1_2 = modelCpp_1_2.textContent.split('/');
            if (splitJson1_2_cpp[splitJson1_2_cpp.length - 2] !== splitCpp1_2[splitCpp1_2.length - 2]) {
                alert('The Json and Cpp file you entered may not be for the same model!')
            }
        });

        let modelBin_1_2= this.document.getElementById('model_bin_1_2');
        modelBin_1_2.addEventListener('click', () => {
            modelBin_1_2.textContent = modelBin_1_2.value = getBinFile();
            let splitJson1_2_bin = modelJson_1_2.textContent.split('/');
            let splitBin1_2 = modelBin_1_2.textContent.split('/');
            if (splitJson1_2_bin[splitJson1_2_bin.length - 2] !== splitBin1_2[splitBin1_2.length - 2]) {
                alert('The Json and Bin file you entered may not be for the same model!')
            }
        })
        let ndkPath_1_2= this.document.getElementById('ndk_path_1_2');
        ndkPath_1_2.addEventListener('click', () => {
            ndkPath_1_2.textContent = ndkPath_1_2.value = getPath();
        });

        let devEnginePath_1_2= this.document.getElementById('engine_path_1_2');
        devEnginePath_1_2.addEventListener('click', () => {
            devEnginePath_1_2.textContent = devEnginePath_1_2.value = getPath();
        });
        let inputList_1_2= this.document.getElementById('input_list_1_2');
        inputList_1_2.addEventListener('click', () => {
            inputList_1_2.textContent = inputList_1_2.value = getFile();
        });
        let inputDataType_1_2='';
        let outputDataType_1_2='';
        let profLevel_1_2='';
        let perfProf_1_2='';
        let devConfigPath_1_2='';

        //load saved run
        let savedRun= this.document.getElementById('saved_run_file');
        savedRun.addEventListener('click', () => {
            savedRun.textContent = savedRun.value = getFile()
            var data = JSON.parse(fs.readFileSync(savedRun.value));

            //populating use case 3
            if(data['useCase'] == 'usecase3'){
                modelJson_3.textContent = modelJson_3.value = data['Inference1']['--model'];
                modelBin_3.textContent = modelBin_3.value = data['Inference1']['--qnn_model_bin_path'];
                modelRaw_3.textContent = modelRaw_3.value = data['Verifier']['--inference_results'];
                modelGoldens_3.textContent = modelGoldens_3.value = data['Verifier']['--framework_results'];
                this.document.getElementById('default_ver_3').value = data['Verifier']['--default_verifier'];
                if(typeof data['Verifier']['--verifier_acc_thr'] != "undefined" && data['Verifier']['--verifier_acc_thr'] != null && data['Verifier']['--verifier_acc_thr'] != ""){
                    this.document.getElementById('verifier_threshold_value_3').value = data['Verifier']['--verifier_acc_thr'];
                }
                workspace_3.textContent = workspace_3.value = data['Inference1']['--working_dir'];
                toggleVerifierFields('#default_ver_3_1','#verifier_config_text_3_1','#verifier_config_text_3_2','#verifier_input_3_1','#verifier_input_3_2');

            }

            //populating use case 2
            if(data['useCase'] == 'usecase2'){
                this.document.getElementById('targetDevice_2').value = data['Inference1']['--target_device'];
                this.document.getElementById('runtime_2').value = data['Inference1']['--runtime'];
                this.document.getElementById('arch_2').value = data['Inference1']['--architecture'];
                this.document.getElementById('default_ver_2').value = data['Verifier']['--default_verifier'];
                if(typeof data['Verifier']['--verifier_acc_thr'] != "undefined" && data['Verifier']['--verifier_acc_thr'] != null && data['Verifier']['--verifier_acc_thr'] != ""){
                    this.document.getElementById('verifier_threshold_value_2').value = data['Verifier']['--verifier_acc_thr'];
                }
                modelCpp_2.textContent = modelCpp_2.value = data['Inference1']['--qnn_model_cpp_path'];
                modelJson_2.textContent = modelJson_2.value = data['Inference1']['--model'];
                modelBin_2.textContent = modelBin_2.value = data['Inference1']['--qnn_model_bin_path'];
                ndkPath_2.textContent = ndkPath_2.value = data['Inference1']['--ndk_path'];
                devEnginePath_2.textContent = devEnginePath_2.value = data['Inference1']['--engine_path'];
                inputList_2.textContent = inputList_2.value = data['Inference1']['--input_list'];
                modelGoldens_2.textContent = modelGoldens_2.value = data['Verifier']['--framework_results'];
                workspace_2.textContent = workspace_2.value = data['Inference1']['--working_dir'];
                this.document.getElementById('input_data_type_2').value = data['Inference1']['--input_data_type'];
                this.document.getElementById('output_data_type_2').value = data['Inference1']['--output_data_type'];
                this.document.getElementById('prof_level_2').value = data['Inference1']['--profiling_level'] =="detailed"? "on":"off";
                this.document.getElementById('perf_prof_2').value = data['Inference1']['--perf_profile'];
                this.document.getElementById('devices_config_path_2').value = data['Inference1']['--devices_config_path'];
                if(this.document.getElementById('targetDevice_2').value == 'android'){
                    jquery('#devices_config_path_2_p').show();
                }
                toggleDevFields('#targetDevice_2','#devices_config_path_2','#devices_config_path_2_p',
                '#x86_64-linux-clang_2','#aarch64-android_2','#arch_2',
                '#runtime_2', '#cpu_2', '#gpu_2', '#dspv68_2', '#dspv69_2', '#dspv73_2');
                toggleVerifierFields('#default_ver_2_1','#verifier_config_text_2_1','#verifier_config_text_2_2','#verifier_input_2_1','#verifier_input_2_2');

            }
            //populating use case 1
            if(data['useCase'] == 'usecase1'){
                //inference1
                this.document.getElementById('tarDev_1').value = data['Inference1']['--target_device'];
                toggleDevFields('#tarDev_1','#devices_config_path_1','#devices_config_path_1_p',
                '#x86_64-linux-clang_1','#aarch64-android_1','#arch_1',
                '#runtime_1', '#cpu_1', '#gpu_1', '#dspv68_1', '#dspv69_1', '#dspv73_1');
                this.document.getElementById('runtime_1').value = data['Inference1']['--runtime'];
                this.document.getElementById('arch_1').value = data['Inference1']['--architecture'];
                this.document.getElementById('default_ver_1').value = data['Verifier']['--default_verifier'];
                if(typeof data['Verifier']['--verifier_acc_thr'] != "undefined" && data['Verifier']['--verifier_acc_thr'] != null && data['Verifier']['--verifier_acc_thr'] != ""){
                    this.document.getElementById('verifier_threshold_value_1').value = data['Verifier']['--verifier_acc_thr'];
                }
                modelCpp_1.textContent = modelCpp_1.value = data['Inference1']['--qnn_model_cpp_path'];
                modelJson_1.textContent = modelJson_1.value = data['Inference1']['--model'];
                modelBin_1.textContent = modelBin_1.value = data['Inference1']['--qnn_model_bin_path'];
                ndkPath_1.textContent = ndkPath_1.value = data['Inference1']['--ndk_path'];
                devEnginePath_1.textContent = devEnginePath_1.value = data['Inference1']['--engine_path'];
                inputList_1.textContent = inputList_1.value = data['Inference1']['--input_list'];
                workspace_1.textContent = workspace_1.value = data['Inference1']['--working_dir'];
                this.document.getElementById('input_data_type_1').value = data['Inference1']['--input_data_type'];
                this.document.getElementById('output_data_type_1').value = data['Inference1']['--output_data_type'];
                this.document.getElementById('prof_level_1').value = data['Inference1']['--profiling_level'] =="detailed"? "on":"off";
                this.document.getElementById('perf_prof_1').value = data['Inference1']['--perf_profile'];
                this.document.getElementById('devices_config_path_1').value = data['Inference1']['--devices_config_path'];
                if(this.document.getElementById('tarDev_1').value == 'android'){
                    jquery('#devices_config_path_1_p').show();
                }

                //inference 2
                this.document.getElementById('tarDev_1_2').value = data['Inference2']['--target_device'];
                toggleDevFields('#tarDev_1_2','#devices_config_path_1_2','#devices_config_path_1_2_p',
                '#x86_64-linux-clang_1_2','#aarch64-android_1_2','#arch_1_2',
                '#runtime_1_2', '#cpu_1_2', '#gpu_1_2', '#dspv68_1_2', '#dspv69_1_2', '#dspv73_1_2');

                this.document.getElementById('runtime_1_2').value = data['Inference2']['--runtime'];
                this.document.getElementById('arch_1_2').value = data['Inference2']['--architecture'];
                //this.document.getElementById('default_ver_1_2').textContent = this.document.getElementById('default_ver_1_2').value = data['Verifier']['--default_verifier'];
                modelCpp_1_2.textContent = modelCpp_1_2.value = data['Inference2']['--qnn_model_cpp_path'];
                modelJson_1_2.textContent = modelJson_1_2.value = data['Inference2']['--model'];
                modelBin_1_2.textContent = modelBin_1_2.value = data['Inference2']['--qnn_model_bin_path'];
                ndkPath_1_2.textContent = ndkPath_1_2.value = data['Inference2']['--ndk_path'];
                devEnginePath_1_2.textContent = devEnginePath_1_2.value = data['Inference2']['--engine_path'];
                inputList_1_2.textContent = inputList_1_2.value = data['Inference2']['--input_list'];
                this.document.getElementById('input_data_type_1_2').value = data['Inference2']['--input_data_type'];
                this.document.getElementById('output_data_type_1_2').value = data['Inference2']['--output_data_type'];
                this.document.getElementById('perf_prof_1_2').value = data['Inference2']['--perf_profile'];
                this.document.getElementById('devices_config_path_1_2').value = data['Inference2']['--devices_config_path'];
                if(this.document.getElementById('tarDev_1_2').value == 'android'){
                    jquery('#devices_config_path_1_2_p').show();
                }
                toggleVerifierFields('#default_ver_1','#verifier_config_text_1','#verifier_config_text_2','#verifier_input_1','#verifier_input_2');



            }

        });

        //set path for where the configuration will be saved
        const fileRunPath= this.document.getElementById('save_run_configs');
        fileRunPath.addEventListener('click', () => {
            fileRunPath.textContent = fileRunPath.value = getPath();
        });

        //what happens when the user presses run button
        const openTestButton = this.document.getElementById('run');
        if(openTestButton){
            openTestButton.style.opacity = 1;
            openTestButton.addEventListener('click', () => {

                //get values for usecase 3
                default_verifier_3 = this.document.getElementById('default_ver_3').value;
                verifier_acc_thr_3 = this.document.getElementById('verifier_threshold_value_3').value;
                verifier_perf_thr_3= this.document.getElementById('verifier_threshold_value_3_2').value;

                //get values for usecase 2
                targetDev_2= this.document.getElementById('targetDevice_2').value;
                runtime_2= this.document.getElementById('runtime_2').value;
                architecture_2 = this.document.getElementById('arch_2').value;
                default_verifier_2 = this.document.getElementById('default_ver_2').value;
                verifier_acc_thr_2 = this.document.getElementById('verifier_threshold_value_2').value;;
                verifier_perf_thr_2 = this.document.getElementById('verifier_threshold_value_2_2').value;;

                devConfigPath_2= this.document.getElementById('devices_config_path_2').value
                inputDataType_2=this.document.getElementById('input_data_type_2').value;
                outputDataType_2=this.document.getElementById('output_data_type_2').value;
                profLevel_2=this.document.getElementById('prof_level_2').value;
                perfProf_2=this.document.getElementById('perf_prof_2').value;


                // get values for usecase 1
                //inference 1
                targetDev_1 = this.document.getElementById('tarDev_1').value;
                runtime_1 = this.document.getElementById('runtime_1').value;
                architecture_1 = this.document.getElementById('arch_1').value;
                default_verifier_1 = this.document.getElementById('default_ver_1').value;
                verifier_acc_thr_1 = this.document.getElementById('verifier_threshold_value_1').value;
                verifier_perf_thr_1 = this.document.getElementById('verifier_threshold_value_1_2').value;

                devConfigPath_1= this.document.getElementById('devices_config_path_1').value;
                inputDataType_1=this.document.getElementById('input_data_type_1').value;
                outputDataType_1=this.document.getElementById('output_data_type_1').value;
                profLevel_1=this.document.getElementById('prof_level_1').value;
                perfProf_1=this.document.getElementById('perf_prof_1').value;


                //inference 2
                targetDev_1_2 = this.document.getElementById('tarDev_1_2').value;
                runtime_1_2= this.document.getElementById('runtime_1_2').value;
                architecture_1_2 = this.document.getElementById('arch_1_2').value;
                // default_verifier_1_2 = this.document.getElementById('default_ver_1_2').value;
                inputDataType_1_2=this.document.getElementById('input_data_type_1_2').value;
                outputDataType_1_2=this.document.getElementById('output_data_type_1_2').value;
                //prof level same to inference 1
                profLevel_1_2=this.document.getElementById('prof_level_1').value;
                perfProf_1_2=this.document.getElementById('perf_prof_1_2').value;
                devConfigPath_1_2= this.document.getElementById('devices_config_path_1_2').value


                //json objects needed for golden eye tool
                let jsonObject1={};
                let jsonObject2={};
                let verificationParams={};
                let verifierHyperParameterJson='';
                let useCaseForm = '';
                let graphFile = '';
                let binFile='';
                let verifierThreshold={};
                let useCase= this.document.getElementById("caseType").value;

                //populate json for use case 3
                if(useCase === "usecase3"){
                    useCaseForm = this.document.getElementById('useCase3Form')
                    graphFile= modelJson_3.value;
                    binFile = modelBin_3.value;
                    jsonObject1['--engine']='QNN';
                    jsonObject1['--stage']='converted';
                    jsonObject1["--model"]=modelJson_3.value;
                    jsonObject1["--qnn_model_bin_path"]=modelBin_3.value;
                    verificationParams["--framework_results"]=modelGoldens_3.value;
                    verificationParams["--inference_results"]=modelRaw_3.value;
                    verificationParams["--default_verifier"]=default_verifier_3;
                    verificationParams["--verifier_acc_thr"]=verifier_acc_thr_3;
                    verificationParams["--verifier_perf_thr"]=verifier_perf_thr_3;

                    verificationParams['--default_verifier']=default_verifier_3 + getVeriferConfig('#verifier_config_text_3_1','#verifier_input_3_1','#verifier_config_text_3_2','#verifier_input_3_2');

                    jsonObject1['--working_dir']=workspace_3.value;
                }

                //populate json for use case 2
                if(useCase === "usecase2"){
                    useCaseForm = this.document.getElementById('useCase2Form')
                    graphFile = modelJson_2.value;
                    binFile = modelBin_2.value;
                    jsonObject1['--engine']='QNN';
                    jsonObject1['--stage']='converted';
                    jsonObject1['--working_dir']=workspace_2.value;
                    jsonObject1['--target_device']=targetDev_2;
                    jsonObject1['--runtime']=runtime_2;
                    jsonObject1['--architecture']=architecture_2;
                    verificationParams['--default_verifier']=default_verifier_2;
                    verificationParams['--verifier_acc_thr']=verifier_acc_thr_2;
                    verificationParams['--verifier_perf_thr']=verifier_perf_thr_2;
                    verificationParams['--default_verifier']=default_verifier_2 + getVeriferConfig('#verifier_config_text_2_1','#verifier_input_2_1','#verifier_config_text_2_2','#verifier_input_2_2');

                    jsonObject1['--qnn_model_cpp_path']=modelCpp_2.value;
                    jsonObject1['--model'] =modelJson_2.value;
                    jsonObject1['--qnn_model_bin_path']=modelBin_2.value;
                    jsonObject1['--ndk_path']=ndkPath_2.value;
                    jsonObject1['--engine_path']=devEnginePath_2.value;
                    jsonObject1['--input_list']=inputList_2.value;
                    jsonObject1['--devices_config_path']=devConfigPath_2;
                    verificationParams['--framework_results']=modelGoldens_2.value;
                    jsonObject1['--input_data_type']=inputDataType_2;
                    jsonObject1['--output_data_type']=outputDataType_2;
                    if(profLevel_2=="on"){
                        jsonObject1['--profiling_level']="detailed";
                    }
                    jsonObject1['--perf_profile']=perfProf_2;
                }

                //populate json for use case 1
                if(useCase === "usecase1"){
                    useCaseForm = this.document.getElementById('useCase1Form')
                    //inference 1
                    graphFile = modelJson_1.value;
                    binFile = modelBin_1.value;
                    jsonObject1['--engine']='QNN';
                    jsonObject1['--stage']='converted';
                    jsonObject1['--working_dir']=workspace_1.value;
                    jsonObject1['--target_device']=targetDev_1;
                    jsonObject1['--runtime']=runtime_1;
                    jsonObject1['--architecture']=architecture_1;
                    verificationParams['--default_verifier']=default_verifier_1;
                    verificationParams['--verifier_acc_thr']=verifier_acc_thr_1;
                    verificationParams['--verifier_perf_thr']=verifier_perf_thr_1;
                    verificationParams['--default_verifier']=default_verifier_1 + getVeriferConfig('#verifier_config_text_1','#verifier_input_1','#verifier_config_text_2','#verifier_input_2');
                    jsonObject1['--qnn_model_cpp_path']=modelCpp_1.value;
                    jsonObject1['--qnn_model_bin_path']=modelBin_1.value;
                    jsonObject1['--model']=modelJson_1.value;
                    jsonObject1['--ndk_path']=ndkPath_1.value;
                    jsonObject1['--engine_path']=devEnginePath_1.value;
                    jsonObject1['--input_list']=inputList_1.value;
                    jsonObject1['--devices_config_path']=devConfigPath_1;
                    jsonObject1['--input_data_type']=inputDataType_1;
                    jsonObject1['--output_data_type']=outputDataType_1;
                    if(profLevel_1=="on"){
                        jsonObject1['--profiling_level']="detailed";
                    }
                    jsonObject1['--perf_profile']=perfProf_1;


                    //inference 2
                    jsonObject2['--engine']='QNN';
                    jsonObject2['--stage']='converted';
                    jsonObject2['--working_dir']=workspace_1.value;
                    jsonObject2['--target_device']=targetDev_1_2;
                    jsonObject2['--runtime']=runtime_1_2;
                    jsonObject2['--architecture']=architecture_1_2;
                    jsonObject2['--qnn_model_cpp_path']=modelCpp_1_2.value;
                    jsonObject2['--model']=modelJson_1_2.value;
                    jsonObject2['--qnn_model_bin_path']=modelBin_1_2.value;
                    jsonObject2['--ndk_path']=ndkPath_1_2.value;
                    jsonObject2['--engine_path']=devEnginePath_1_2.value;
                    jsonObject2['--input_list']=inputList_1_2.value;
                    jsonObject2['--devices_config_path']=devConfigPath_1_2;
                    jsonObject2['--input_data_type']=inputDataType_1_2;
                    jsonObject2['--output_data_type']=outputDataType_1_2;
                    //using prof level of inf1
                    if(profLevel_1=="on"){
                        jsonObject2['--profiling_level']="detailed";
                    }

                    jsonObject2['--perf_profile']=perfProf_1_2;

                }
                let json1Verification, json2Verification = true;
                json1Verification = verifyFormfields(jsonObject1);
                if (useCase === "usecase1" && json1Verification ) {
                    json2Verification = verifyFormfields(jsonObject2);
                }

                if (json1Verification && json2Verification) {
                    let workspace = jsonObject1['--working_dir'];
                    let csvPath = path.join(workspace,"csv_outputs"); //workspace +'/output';
                    //clear csv path before run
                    if(fs.existsSync(csvPath)){
                        let fileList=fs.readdirSync(csvPath);
                        fileList.forEach((fileName) => { fs.unlinkSync(path.join(csvPath,fileName));});
                    }
                    //create json to be saved and save file
                    let saveFileJson = {};
                    saveFileJson['useCase'] = useCase;
                    saveFileJson['Inference1'] = jsonObject1;
                    saveFileJson['Inference2'] = jsonObject2;
                    saveFileJson['Verifier'] = verificationParams;

                    let saveFileName= this.document.getElementById('save_run_configs_name').value;
                    if(fileRunPath!=='' && fileRunPath!=='undefined' && saveFileName!=='' && saveFileName!=='File Name' ){
                        makeJSONFile(saveFileJson,fileRunPath,saveFileName);
                    }

                    //delete extra parameters that are not needed by the backend
                    jsonObject1 = deleteExtraParam(jsonObject1);
                    jsonObject2 = deleteExtraParam(jsonObject2);
                    let useCaseForBackend=null;
                    if(useCase=='usecase1'){
                        useCaseForBackend='INFERENCE_V_INFERENCE';
                    }
                    if(useCase=='usecase2'){
                        useCaseForBackend='GOLDEN_V_INFERENCE';
                        jsonObject2=null;
                    }
                    if(useCase=='usecase3'){
                        useCaseForBackend='OUTPUT_V_OUTPUT';
                        jsonObject1=null;
                        jsonObject2=null;
                    }

                    verifierThreshold.acc = verificationParams["--verifier_acc_thr"];
                    verifierThreshold.perf= verificationParams["--verifier_perf_thr"];
                    let backend = require('./backend');

                    // call to backend.js
                    this.document.getElementById('welcome_spinner').classList.toggle('hide');
                    // $("#useCase1Form :input").attr("disabled", true);
                    // $("#tarDev_1").attr('disabled', 'disabled');
                    // $('#useCase1Form').find('input, textarea, button, select').attr('disabled','disabled');
                    backend('', workspace, useCaseForBackend, jsonObject1, jsonObject2, verificationParams,verifierHyperParameterJson)
                        .then(() => {
                            this.document.getElementById('welcome_spinner').classList.toggle('hide');
                            window.close()
                            electron.ipcRenderer.sendSync('open-diff-file',graphFile,csvPath,binFile,verifierThreshold);
                        })
                        .catch((err) =>
                        {
                            let errMsgTitle = "Log Message"
                            if (targetDev_1 !== "x86" || targetDev_1_2 !== "x86" || targetDev_2 !== "x86") {
                                errMsgTitle = "Log Message (Check Adb logcat for more)"
                            }
                            this.document.getElementById('welcome_spinner').classList.toggle('hide');
                            electron.ipcRenderer.send('open-diff-error-dialog', err, errMsgTitle);
                        });
                }
            });
        }
    }

    /*
        * OPEN_SOURCE_START
        * The following code is derived from the netron open source project in order to setup environment, exception
        * throwing, and page view configurations.
        * Project Link: https://github.com/lutzroeder/netron/
        * Note: There are a few minor modifications to accommodate for QNN-Netron use case
    */

    environment(name) {
        return this._environment[name];
    }

    error(message, detail) {
        electron.ipcRenderer.sendSync('show-message-box', {
            type: 'error',
            message: message,
            detail: detail,
        });
    }

    confirm(message, detail) {
        const result = electron.ipcRenderer.sendSync('show-message-box', {
            type: 'question',
            message: message,
            detail: detail,
            buttons: ['Yes', 'No'],
            defaultId: 0,
            cancelId: 1
        });
        return result === 0;
    }

    require(id) {
        try {
            return Promise.resolve(require(id));
        }
        catch (error) {
            return Promise.reject(error);
        }
    }

    save(name, extension, defaultPath, callback) {
        const selectedFile = electron.ipcRenderer.sendSync('show-save-dialog', {
            title: 'Export Tensor',
            defaultPath: defaultPath,
            buttonLabel: 'Export',
            filters: [ { name: name, extensions: [ extension ] } ]
        });
        if (selectedFile) {
            callback(selectedFile);
        }
    }

    export(file, blob) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const data = new Uint8Array(e.target.result);
            fs.writeFile(file, data, null, (err) => {
                if (err) {
                    this.exception(err, false);
                    this.error('Error writing file.', err.message);
                }
            });
        };

        let err = null;
        if (!blob) {
            err = new Error("Export blob is '" + JSON.stringify(blob) + "'.");
        }
        else if (!(blob instanceof Blob)) {
            err = new Error("Export blob type is '" + (typeof blob) + "'.");
        }

        if (err) {
            this.exception(err, false);
            this.error('Error exporting image.', err.message);
        }
        else {
            reader.readAsArrayBuffer(blob);
        }
    }

    request(file, encoding, base) {
        return new Promise((resolve, reject) => {
            const pathname = path.join(base || __dirname, file);
            fs.stat(pathname, (err, stats) => {
                if (err && err.code === 'ENOENT') {
                    reject(new Error("The file '" + file + "' does not exist."));
                }
                else if (err) {
                    reject(err);
                }
                else if (!stats.isFile()) {
                    reject(new Error("The path '" + file + "' is not a file."));
                }
                else if (stats && stats.size < 0x7ffff000) {
                    fs.readFile(pathname, encoding, (err, data) => {
                        if (err) {
                            reject(err);
                        }
                        else {
                            resolve(encoding ? data : new host.ElectronHost.BinaryStream(data));
                        }
                    });
                }
                else if (encoding) {
                    reject(new Error("The file '" + file + "' size (" + stats.size.toString() + ") for encoding '" + encoding + "' is greater than 2 GB."));
                }
                else {
                    resolve(new host.ElectronHost.FileStream(pathname, 0, stats.size, stats.mtimeMs));
                }
            });
        });
    }

    openURL(url) {
        electron.shell.openExternal(url);
    }

    exception(error, fatal) {
        if (error) {
            try {
                const description = [];
                description.push((error && error.name ? (error.name + ': ') : '') + (error && error.message ? error.message : '(null)'));
                if (error.stack) {
                    const match = error.stack.match(/\n {4}at (.*)\((.*)\)/);
                    if (match) {
                        description.push(match[1] + '(' + match[2].split('/').pop().split('\\').pop() + ')');
                    }
                }
            }
            catch (e) {
                // continue regardless of error
            }
        }
    }

    screen(name) {
        //pass
    }


    event(category, action, label, value) {
        //pass
    }

    _openFile(file) {
        if (this._queue) {
            this._queue.push(file);
            return;
        }
        if (file && this._view.accept(file)) {
            this._view.show('welcome spinner');
            const dirname = path.dirname(file);
            console.log(dirname);
            const basename = path.basename(file);
            this.request(basename, null, dirname).then((stream) => {
                const context = new host.ElectronHost.ElectronContext(this, dirname, basename, stream);
                this._view.open(context).then((model) => {
                    this._view.show(null);
                    if (model) {
                        this._update('path', file);
                    }
                    this._update('show-attributes', this._view.showAttributes);
                    this._update('show-initializers', this._view.showInitializers);
                    this._update('show-names', this._view.showNames);
                }).catch((error) => {
                    if (error) {
                        this._view.error(error, null, null);
                        this._update('path', null);
                    }
                    this._update('show-attributes', this._view.showAttributes);
                    this._update('show-initializers', this._view.showInitializers);
                    this._update('show-names', this._view.showNames);
                });
            }).catch((error) => {
                this._view.error(error, 'Error while reading file.', null);
                this._update('path', null);
            });
        }
    }

    _request(url, headers, encoding, timeout) {
        return new Promise((resolve, reject) => {
            const httpModule = url.split(':').shift() === 'https' ? https : http;
            const options = {
                headers: headers
            };
            const request = httpModule.get(url, options, (response) => {
                if (response.statusCode !== 200) {
                    const err = new Error("The web request failed with status code " + response.statusCode + " at '" + url + "'.");
                    err.type = 'error';
                    err.url = url;
                    err.status = response.statusCode;
                    reject(err);
                }
                else {
                    let data = '';
                    response.on('data', (chunk) => {
                        data += chunk;
                    });
                    response.on('err', (err) => {
                        reject(err);
                    });
                    response.on('end', () => {
                        resolve(data);
                    });
                }
            }).on("error", (err) => {
                reject(err);
            });
            if (timeout) {
                request.setTimeout(timeout, () => {
                    request.abort();
                    const err = new Error("The web request timed out at '" + url + "'.");
                    err.type = 'timeout';
                    err.url = url;
                    reject(err);
                });
            }
        });
    }

    _getConfiguration(name) {
        return electron.ipcRenderer.sendSync('get-configuration', { name: name });
    }

    _setConfiguration(name, value) {
        electron.ipcRenderer.sendSync('set-configuration', { name: name, value: value });
    }

    _update(name, value) {
        electron.ipcRenderer.send('update', { name: name, value: value });
    }

    /* OPEN_SOURCE_END */

};

function getFile(){
    let file= electron.ipcRenderer.sendSync('choose-file-dialog', {});
    if (file){
        return file;
    }
    // Continue to return option to original value when no entry
    return this._view.textContents();
}

function getBinFile(){
    let file= electron.ipcRenderer.sendSync('open-binFile-dialog', {});
    if (file){
        return file;
    }
    // Continue to return option to original value when no entry
    return this._view.textContents();
}
function getPath(){
    let path= electron.ipcRenderer.sendSync('choose-path-dialog',{});
    if (path){
        return path;
    }
    // Continue to return option to original when no entry
    return this._view.textContents();
}
function makeJSONFile(saveFileJson,fileRunPath,saveFileName){
    console.log(saveFileJson);
    fileVal = JSON.stringify(saveFileJson);
    fs.writeFileSync(fileRunPath.textContent+'/'+saveFileName+'.json',fileVal,(error) =>{
        if(error) throw error;
    });
}

function deleteExtraParam(json){
    for ( const param in json){
        if(json[param] == '' || json[param] == 'N/A'){
            delete json[param];
        }
        if(param == '--model'){
            delete json[param]
        }
        if(param == '--working_dir'){
            delete json[param]
        }
    }
    return json;
}

function verifyFormfields(json) {
    // field that is allowed to be empty and has no default
    let optionalFields = ["--qnn_model_bin_path"]
    for (const param in json) {
        if(json[param] == '' && !optionalFields.includes(param)){
            alert("Missing form field for: " + param);
            return false;
        }
    }
    return true;
}

/*
    * OPEN_SOURCE_START
    * The following code is derived from the netron open source project in order to setup electron contexts and variety
    * of streams (file or binary)
    * Project Link: https://github.com/lutzroeder/netron/
    * Note: There are a few minor modifications to accommodate for QNN-Netron use case
 */



host.ElectronHost.BinaryStream = class {

    constructor(buffer) {
        this._buffer = buffer;
        this._length = buffer.length;
        this._position = 0;
    }

    get position() {
        return this._position;
    }

    get length() {
        return this._length;
    }

    stream(length) {
        const buffer = this.read(length);
        return new host.ElectronHost.BinaryStream(buffer.slice(0));
    }

    seek(position) {
        this._position = position >= 0 ? position : this._length + position;
        if (this._position > this._buffer.length) {
            throw new Error('Expected ' + (this._position - this._buffer.length) + ' more bytes. The file might be corrupted. Unexpected end of file.');
        }
    }

    skip(offset) {
        this._position += offset;
        if (this._position > this._buffer.length) {
            throw new Error('Expected ' + (this._position - this._buffer.length) + ' more bytes. The file might be corrupted. Unexpected end of file.');
        }
    }

    peek(length) {
        if (this._position === 0 && length === undefined) {
            return this._buffer;
        }
        const position = this._position;
        this.skip(length !== undefined ? length : this._length - this._position);
        const end = this._position;
        this.seek(position);
        return this._buffer.subarray(position, end);
    }

    read(length) {
        if (this._position === 0 && length === undefined) {
            this._position = this._length;
            return this._buffer;
        }
        const position = this._position;
        this.skip(length !== undefined ? length : this._length - this._position);
        return this._buffer.subarray(position, this._position);
    }

    byte() {
        const position = this._position;
        this.skip(1);
        return this._buffer[position];
    }
};

host.ElectronHost.FileStream = class {

    constructor(file, start, length, mtime) {
        this._file = file;
        this._start = start;
        this._length = length;
        this._position = 0;
        this._mtime = mtime;
    }

    get position() {
        return this._position;
    }

    get length() {
        return this._length;
    }

    stream(length) {
        const file = new host.ElectronHost.FileStream(this._file, this._position, length, this._mtime);
        this.skip(length);
        return file;
    }

    seek(position) {
        this._position = position >= 0 ? position : this._length + position;
    }

    skip(offset) {
        this._position += offset;
        if (this._position > this._length) {
            throw new Error('Expected ' + (this._position - this._length) + ' more bytes. The file might be corrupted. Unexpected end of file.');
        }
    }

    peek(length) {
        length = length !== undefined ? length : this._length - this._position;
        if (length < 0x10000000) {
            const position = this._fill(length);
            this._position -= length;
            return this._buffer.subarray(position, position + length);
        }
        const position = this._position;
        this.skip(length);
        this.seek(position);
        const buffer = new Uint8Array(length);
        this._read(buffer, position);
        return buffer;
    }

    read(length) {
        length = length !== undefined ? length : this._length - this._position;
        if (length < 0x10000000) {
            const position = this._fill(length);
            return this._buffer.subarray(position, position + length);
        }
        const position = this._position;
        this.skip(length);
        const buffer = new Uint8Array(length);
        this._read(buffer, position);
        return buffer;
    }

    byte() {
        const position = this._fill(1);
        return this._buffer[position];
    }

    _fill(length) {
        if (this._position + length > this._length) {
            throw new Error('Expected ' + (this._position + length - this._length) + ' more bytes. The file might be corrupted. Unexpected end of file.');
        }
        if (!this._buffer || this._position < this._offset || this._position + length > this._offset + this._buffer.length) {
            this._offset = this._position;
            this._buffer = new Uint8Array(Math.min(0x10000000, this._length - this._offset));
            this._read(this._buffer, this._offset);
        }
        const position = this._position;
        this._position += length;
        return position - this._offset;
    }

    _read(buffer, offset) {
        const descriptor = fs.openSync(this._file, 'r');
        const stat = fs.statSync(this._file);
        if (stat.mtimeMs != this._mtime) {
            throw new Error("File '" + this._file + "' last modified time changed.");
        }
        try {
            fs.readSync(descriptor, buffer, 0, buffer.length, offset + this._start);
        }
        finally {
            fs.closeSync(descriptor);
        }
    }
};

host.ElectronHost.ElectronContext = class {

    constructor(host, folder, identifier, stream) {
        this._host = host;
        this._folder = folder;
        this._identifier = identifier;
        this._stream = stream;
    }

    get identifier() {
        return this._identifier;
    }

    get stream() {
        return this._stream;
    }

    request(file, encoding, base) {
        return this._host.request(file, encoding, base === undefined ? this._folder : base);
    }

    require(id) {
        return this._host.require(id);
    }

    exception(error, fatal) {
        this._host.exception(error, fatal);
    }
};

window.addEventListener('load', () => {
    global.protobuf = require('./protobuf');
    global.flatbuffers = require('./flatbuffers');
    const view = require('./view');
    window.__view__ = new view.View(new host.ElectronHost());
});

/* OPEN_SOURCE_END */
