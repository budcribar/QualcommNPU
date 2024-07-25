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

var qnn = qnn || {};
var json = json || require('./json');
var fs = require('fs');
let path = require('path')
var os = require('os');
const tester = require('electron');

qnn.ModelFactory = class {

    //matching for model type
    match(context) {
        const identifier = context.identifier;
        const extension = identifier.split('.').pop().toLowerCase();
        const obj = context.open('json');
        this._binName = obj['model.bin'];
        console.log(this._binName);
        if (extension === 'json') {
            if(obj['model.bin'] && obj['model.cpp']){
                return true;
            }
        }
        return false;
    }

 //opens the QNN model
    open(context,csv,binFilePath,verifierThreshold){
         console.log("qnnmodelFactoryOpen");
         this._csv = csv || null;
         this._binFile = binFilePath || null;

         if(verifierThreshold&&verifierThreshold.hasOwnProperty("acc") &&verifierThreshold.hasOwnProperty("perf") ){
            this._verifierThreshold = verifierThreshold
         }else{
             this._verifierThreshold = null
         }
         console.log('verifier threshold use here:', this._verifierThreshold);

         let binFile="";
         // check to see if there was a binFile passed in already and if there is not ask the user
         // if they want to upload one
         if((this._binFile == null || this._binFile == 'Add File' || this._binFile == undefined) && this._binName!=='N/A'){
            const result = tester.ipcRenderer.sendSync('show-message-box', {
            type: 'question',
            message: 'Would you like to use this bin file '+ this._binName+' ?',
            buttons: ['Yes', 'No','Change Bin File'],
            defaultId: 0,
            cancelId: 1,
            addFile: 2
        });
         if(result == 2){
         const binNameUser = tester.ipcRenderer.sendSync('open-binFile-dialog', {});
         if(binNameUser !== undefined){
         binFile= binNameUser[0];
         }else{
         alert("No Bin File Found. Maybe look at: "+this._binName);
         binFile ="";
         }
         }else if(result ==0){
         binFile=this._binName;
         }else{
         binFile = "";
         }
         }else {
         binFile = this._binFile;
         }
            return qnn.Metadata.open(context).then((metadata) => {
                //call to create a new model
                return new qnn.Model(metadata,context.open('json'),binFile,this._csv,this._verifierThreshold);
            });
    }
    };


qnn.Model = class{
    constructor(metadata,model,binFile,csv,verifierThreshold){
    this._graphs=[];
    this._model = model;
    //creates graph
    this._graphs.push(new qnn.Graph(metadata, model,binFile,csv,verifierThreshold));
    }

    get converter_command(){
        return this._model.converter_command;
    }

    get copyright(){
        return this._model.copyright_str || "N/A";
    }

    get format(){
        return 'QNN Model';
    }

    get graphs(){
        return this._graphs;
    }

    get model_cpp(){
        return this._model["model.cpp"] || "N/A";
    }

    get model_bin(){
        return this._model["model.bin"] || "N/A";
    }
    get op_types(){
        return this._model.op_types.toString();
    }

    get total_param(){
        return this._model["Total parameters"];
    }

    get total_mac(){
        return this._model["Total MACs per inference"];
    }
};

//enums for the type of argument
const TypeOfArgument = {
    INPUT: 0,
    STATIC: 4,
    OUTPUT: 1,
    NATIVE: 3,
};

//enums for the data Type
const DataType = {
    INT8 : 8,
    INT16 : 22,
    INT32 : 50,
    INT64 : 100,
    UINT8 : 264,
    UINT16 : 278,
    UINT32 : 306,
    UINT64 : 356,
    FLOAT16 : 534,
    FLOAT32 : 562,
    SFIXEDPOINT8 : 776,
    SFIXEDPOINT16 : 790,
    SFIXEDPOINT32 : 818,
    UFIXEDPOINT8 : 1032,
    UFIXEDPOINT16 : 1046,
    UFIXEDPOINT32 : 1074,
};

//returns the data type of the graph
function findDataType(dataType){
    switch(dataType){
                case DataType.INT8:
                    return "INT8" ;
                    break;
                case DataType.INT16:
                    return "INT16";
                    break;
                case DataType.INT32:
                    return "INT32";
                    break;
                case DataType.INT64:
                    return "INT64" ;
                    break;
                case DataType.UINT8:
                    return "UINT8" ;
                    break;
                case DataType.UINT16:
                    return "UINT16" ;
                    break;
                case DataType.UINT32:
                    return "UINT32" ;
                    break;
                case DataType.UINT64:
                    return "UINT64" ;
                    break;
                case DataType.FLOAT16:
                    return "FLOAT16" ;
                    break;
                case DataType.FLOAT32:
                    return "FLOAT32" ;
                    break;
                case DataType.SFIXEDPOINT8:
                    return "SFIXEDPOINT8" ;
                    break;
                case DataType.SFIXEDPOINT16:
                    return "SFIXEDPOINT16" ;
                    break;
                case DataType.SFIXEDPOINT32:
                    return "SFIXEDPOINT32" ;
                    break;
                case DataType.UFIXEDPOINT8:
                    return "UFIXEDPOINT8" ;
                    break;
                case DataType.UFIXEDPOINT16:
                    return "UFIXEDPOINT16" ;
                    break;
                case DataType.UFIXEDPOINT32:
                    return "UFIXEDPOINT32" ;
                    break;
}
}


// creates a map of the tensor names as keys and the corresponding argument as the value
//this method also checks to see what type of argument it is (input,output, or internal) and adds it
//to the corresponding list using IOSetter
qnn.ArgumentMapper = class {

    constructor(iosetter, model,tensors,binFile){

        this._argumentMap = new Map();
        for(let i=0; i< tensors.length; i++){
             const dataType = findDataType(tensors[i][1]["data_type"]);
             const shape = new qnn.TensorShape(tensors[i][1]["current_dims"]);
             let scale_offset='';
             if(tensors[i][1]["quant_params"]["scale_offset"]["axis_scale_offset"]==undefined){
              scale_offset = JSON.stringify(tensors[i][1]["quant_params"]["scale_offset"]);
             }else{
              scale_offset = JSON.stringify(tensors[i][1]["quant_params"]["axis_scale_offset"]["scale_offsets"]);
             }
             const encoding = tensors[i][1]["quant_params"]["encoding"];
             const axis_format = tensors[i][1]["axis_format"];
             const inputType = new qnn.TensorType(dataType +"\n Scale_Offsets:"+scale_offset +"\n Encoding:"+encoding+"\n Axis_Format:" + axis_format +"\n", shape);
             const type = tensors[i][1]["type"];
             let argument = new qnn.Argument(tensors[i][0],inputType);
            switch(type){
                case TypeOfArgument.INPUT:
                    iosetter.addInput(argument);
                    break;
                case TypeOfArgument.STATIC:
                    if(binFile){
                     let tenData = sendToPython(binFile,tensors[i][0],dataType);
                     console.log(tenData);
                     if(tenData == ''){
                     break;
                     }
                     let intial = new qnn.Tensor(inputType,tenData);
                     argument = new qnn.Argument(tensors[i][0],inputType, intial);
                      iosetter.addInput(argument);
                    }

                     break;
                case TypeOfArgument.OUTPUT:
                    iosetter.addOutput(argument);
                    break;
                case TypeOfArgument.NATIVE:
                    break;

            }
            this._argumentMap.set(tensors[i][0],argument);
        }


    }

    get argumentMap() {
    return this._argumentMap;
    }


};

//extracts the data from the .bin file
function sendToPython(binFile, argName, dataType) {
    var spawnSync = require('child_process').spawnSync;
    let tools_dir;
    // package location
    if (fs.existsSync(path.join(__dirname, '../../app.asar.unpacked/tools/'))) {
        tools_dir = path.join(__dirname, '../../app.asar.unpacked/tools/')
    }
    // developer location
    else if (fs.existsSync('tools/')) {
        tools_dir = 'tools'
    } else {
        alert("Unable to load bin file. model.bin reader not found")
        return false;
    }
    var result = spawnSync('python', [path.join(tools_dir, 'qnn.py'), binFile, argName + '.raw', dataType], {
        cwd: process.cwd(),
        env: process.env,
        stdio: 'pipe',
        encoding: 'utf-8'
    });
    var savedOutput = result.stdout;
    var errOutput = result.stderr;
    console.log(savedOutput);
    if(errOutput)
    {
        console.log(errOutput);
        alert(errOutput);
        return false;
    }
    return (savedOutput);
}


qnn.IOSetter = class {

    constructor(){
      this._inputArguments=[];
      this._outputArguments=[];

    }

    addInput(argument){
    this._inputArguments.push(argument);
    }

    addOutput(argument){
    this._outputArguments.push(argument);
    }

    get inputs(){
      return this._inputArguments;
    }

    get outputs(){
   return this._outputArguments;
    }


};

qnn.Attribute = class {

    constructor(name,value) {
        this._name = name;
        this._value = value;
        this._visible =true;
    }

    get name() {
        return this._name;
    }

    get value() {
        return this._value;
    }

    get visible() {
        return this._visible == false ? false : true;
    }
};

//extracts the accuracy data from the csv file
function accuracyInfo(csv) {
    var spawnSync = require('child_process').spawnSync;
    let tools_dir;
    if(csv != null){
        csv=path.join(csv,"summary.csv");
    }
    // package location
    if (fs.existsSync(path.join(__dirname, '../../app.asar.unpacked/tools/'))) {
        tools_dir = path.join(__dirname, '../../app.asar.unpacked/tools/')
    }
    // developer location
    else if (fs.existsSync('tools/')) {
        tools_dir = 'tools'
    } else {
        alert("Unable to parse accuracy CSV. Parser not found.")
        return false;
    }
    var result = spawnSync('python', [path.join(tools_dir, 'AccuracyParser.py'), csv], {
        cwd: process.cwd(),
        env: process.env,
        stdio: 'pipe',
        encoding: 'utf-8'
    });
    var savedOutput = result.stdout;
    var errOutput = result.stderr;
    if(errOutput)
    {
        console.log(errOutput);
        alert(errOutput);
        return false;
    }
    return(savedOutput);
}

function performanceInfo(csv) {
    var spawnSync = require('child_process').spawnSync;
    let tools_dir;
    let csv1;
    let csv2;
    if(csv == null){
        //return null when using visualizer
        return {}
    }
    csv1=path.join(csv,"inf1_profiling.csv");
    csv2=path.join(csv,"inf2_profiling.csv");
    if (!fs.existsSync(csv1) || !fs.existsSync(csv2)) {
        //return null when profiling_level = off
        return {}
    }

    // package location
    if (fs.existsSync(path.join(__dirname, '../../app.asar.unpacked/tools/'))) {
        tools_dir = path.join(__dirname, '../../app.asar.unpacked/tools/')
    }
    // developer location
    else if (fs.existsSync('tools/')) {
        tools_dir = 'tools'
    } else {
        alert("Unable to parse performance CSV. Parser not found.")
        return {};
    }
    var result = spawnSync('python', [path.join(tools_dir, 'RuntimeParser.py'), csv1,csv2], {
        cwd: process.cwd(),
        env: process.env,
        stdio: 'pipe',
        encoding: 'utf-8'
    });
    var savedOutput = result.stdout;
    var obj = JSON.parse(savedOutput);
    return(obj);
}

function getColor(value,thr=1)
{
    if(Math.abs(value) >= thr){
        return '#FFCC99';
    }
    else if(Math.abs(value) > 0 ){
        return '#ADD8E6';
    } else {
        return null;
    }
}
/*
    * OPEN_SOURCE_START
    * The following code is derived from the netron open source project in order to setup the graph object as needed
    * for the QNN Graph use case
    * Note: Minor modifications done to align with QNN requirements
    * Project Link: https://github.com/lutzroeder/netron/
*/
//creates a Graph Object for a QNN Network
qnn.Graph = class {

    constructor(metadata, model,binFile,csv,verifierThreshold) {
        console.log("qnnGraph");
        console.log(csv);
        console.log('verifier threshould value: %f', verifierThreshold)
        this._nodes = [];
        this._inputs = [];
        this._outputs = [];
        this._csvInfo=[];
        const args = {};

       //create argument object for tensors
        const ioSetter = new qnn.IOSetter();
        const graphArgMapper = new qnn.ArgumentMapper(ioSetter,model,Object.entries(model.graph.tensors),binFile);
        const graphArgumentMap =graphArgMapper.argumentMap;

        //get list of the arguments for the absolute inputs an outputs of the graph
        const inputArguments = ioSetter.inputs;
        const outputArguments = ioSetter.outputs;
        this._inputs.push(new qnn.Parameter("input",inputArguments));
        this._outputs.push(new qnn.Parameter("output",outputArguments));
        //create the internal nodes
        const nodeMap = Object.entries(model.graph.nodes);

        let accuracyDiffInfo = []
        let perfDiffInfo = []
        if (csv != null) {
            accuracyDiffInfo = accuracyInfo(csv).split('\n');
            perfDiffInfo = performanceInfo(csv);
        }
         let threshold = {performance:0,accuracy:false};
         console.log(perfDiffInfo);
        if(perfDiffInfo && perfDiffInfo["__root__"]){
            for(var event in perfDiffInfo["__root__"]){
                for( var process in perfDiffInfo["__root__"][event] )
                var csvinfo={value:null,color:null};
                if(event == "root"){
                    csvinfo.value=process.toLowerCase()+": " + perfDiffInfo["__root__"][event][process]["value"]*100.0 +"%";
                    csvinfo.color=getColor(perfDiffInfo["__root__"][event][process]["value"]*100.0,verifierThreshold.perf);
                    this._csvInfo.push(csvinfo);
                }else{
                    csvinfo.value=event +": " + perfDiffInfo["__root__"][event][process]["value"]*100.0 +" %";
                    csvinfo.color=getColor(perfDiffInfo["__root__"][event][process]["value"]*100.0,verifierThreshold.perf);
                    this._csvInfo.push(csvinfo);
                }
            }
        }
       //create all nodes
        for( let k=0; k<nodeMap.length; k++) {
             const node = nodeMap[k];
             const nodeName = node[0];
             const nodeType = node[1].type;
             const inputNames = node[1]['input_names'];
             const outputNames = node[1]['output_names'];
             const tensorParams = node[1]['tensor_params'];
             threshold={performance:"none",accuracy:false};

             //create an array of diff Data to be shown on the graph
             let nodeDiffInfo =[];
             for(let j=0;j<outputNames.length;j++){
             let nodeNameTester = outputNames[j];
             let runNum =1;

            if(perfDiffInfo && perfDiffInfo[nodeName]){
                for(var elem in perfDiffInfo[nodeName]){
                    var csvinfo={value:null,color:null};
                    csvinfo.value=elem.toLowerCase() +" performance: "  +  perfDiffInfo[nodeName][elem]["value"]*100.0 +" %";
                    var perf= Math.abs(perfDiffInfo[nodeName][elem]["value"]*100.0)
                    csvinfo.color=getColor(perf,verifierThreshold.perf);
                    if(perf >= verifierThreshold.perf){
                        threshold.performance ="warn"
                    }else if(perf <verifierThreshold.perf && perf != 0){
                        threshold.performance ="info"
                    }
                    nodeDiffInfo.push(csvinfo);
                }
            }
            for(let i=0; i<accuracyDiffInfo.length;i++){
                if(accuracyDiffInfo[i] !== ''){
                    let nodeDiff = accuracyDiffInfo[i].split(',');
                    if(nodeDiff[0].replaceAll('_', '') == nodeNameTester.replaceAll('_', '')){
                         var csvinfo={value:null,color:null};
                         csvinfo.value=nodeNameTester+" Run "+runNum+":  "+ nodeDiff[1]+"% Error";

                         if(parseFloat(nodeDiff[1])>verifierThreshold.acc){
                            threshold.accuracy = true;
                            csvinfo.color='#ffc3c3';
                         }
                         nodeDiffInfo.push(csvinfo);
                         runNum = runNum +1;
                     }
                 }
            }
            }
            //nodeDiffInfo.push("Performance Summary"); //added
            console.log(nodeDiffInfo);

             //input parameters for the node
             const nodeInputPars=[];
             for(let m=0; m<inputNames.length; m++){
                 const nodeInputArgs=[];
                 nodeInputArgs.push(graphArgumentMap.get(inputNames[m]));
                 nodeInputPars.push(new qnn.Parameter(inputNames[m],nodeInputArgs));
             }



             //output parameters for the node
             const nodeOutputPars=[];
             for(let l=0; l<outputNames.length; l++){
                  const nodeOutputArgs=[];
                  nodeOutputArgs.push(graphArgumentMap.get(outputNames[l]));
                  nodeOutputPars.push(new qnn.Parameter(outputNames[l],nodeOutputArgs));

             }

             //put in attributes if there are any for the node
             const tensorParamsMap = Object.entries(tensorParams);
             const attributeList =[];
             for(let j=0; j<tensorParamsMap.length; j++){
               console.log("attributes");
               const param = tensorParamsMap[j][0];
               const attributes = tensorParamsMap[j][1];
               const attributeArgumentMap = Object.entries(attributes);
               const attributeName = attributeArgumentMap[0][0];
               const attributenum = attributeArgumentMap[0][1]['type'];
               const scale = "Scale: " + attributeArgumentMap[0][1]['quant_params']['scale_offset']['scale'];
               const offset = "Offset: "+ attributeArgumentMap[0][1]['quant_params']['scale_offset']['offset'];
               console.log(attributeArgumentMap[0][1]);
               let attributeType ='';
               switch(attributenum){
                case TypeOfArgument.INPUT:
                     attributeType = "Type: INPUT ";
                    break;
                case TypeOfArgument.STATIC:
                     attributeType = "Type: STATIC";
                    break;
                case TypeOfArgument.OUTPUT:
                     attributeType = "Type: OUTPUT";
                case TypeOfArgument.NATIVE:
                     attributeType = "Type: NATIVE";
                    break;

              }

               const attributeDim = "Dimension: "+ attributeArgumentMap[0][1]['current_dims'];
               const attributeData = "Data: "+attributeArgumentMap[0][1]['data'];

               const attParam = [];
               attParam.push(attributeType);
               attParam.push(attributeDim);
               attParam.push(attributeData);
               attParam.push(scale);
               attParam.push(offset);


               attributeList.push(new qnn.Attribute(param,attParam));

             }

             this._nodes.push(new qnn.Node(nodeInputPars,metadata,nodeName,nodeOutputPars,nodeType,attributeList,nodeDiffInfo,threshold));

        }

    }

    get name() {
        return this._name;
    }

    get type() {
     return '';
    }

    get inputs() {
        return this._inputs;
    }

    get outputs() {
        return this._outputs;
    }

    get nodes() {
        return this._nodes;
    }
    get csv() {
        return this._csvInfo;
    }

};

/* OPEN_SOURCE_END */

/*
    * OPEN_SOURCE_START
    * The following code is derived from the netron open source project in order to keep track of node information
    * in QNN Graph similiar to how netron does it
    * Project Link: https://github.com/lutzroeder/netron/
*/

qnn.Node = class {

    constructor(inputs,metadata,name,outputs,type,params,csv,match) {
        this._operator = '';
        this._name = '';
        this._type='';
        this._outputs = [''];
        this._chain=[];
        this._inputs = [''];
        this._category = '';
        this._match = match;

        this._inputs=inputs;
        this._metadata = metadata;
        this._name=name;
        this._outputs=outputs;
        this._type=type;
        this._attributes = params || null;
        this._csvInfo =csv;


    }

    get operator() {
        return this._operator;
    }

    get name() {
        return this._name;
    }

    get domain() {
        return null;
    }

    get documentation() {
        return '';
    }

    get category() {
        return this._category;
    }

    get inputs() {
        return this._inputs;
    }

    get outputs() {
        return this._outputs;
    }

    get attributes() {
        return this._attributes;
    }

     get type() {
        return this._type;
    }

    get nodes() {
        return this._nodes;
    }
   get chain() {
        return this._chain;
    }

    get metadata() {
        return this._metadata.type(this._type);
    }



    get csv(){
      return this._csvInfo;
    }
    get match(){
      return this._match;
    }


};

/* OPEN_SOURCE_END */

qnn.Attribute = class {

    constructor(name, value) {
        this._name = name;
        this._value = value;
        console.log("qnnAttribute");

    }

    get name() {
        return this._name;
    }

    get value() {
        return this._value;
    }

    get visible() {
        return this._visible == false ? false : true;
    }
};

/*
    * OPEN_SOURCE_START
    * The following code is derived from the netron open source project in order to keep track of parameter information
    * as well as argument information in QNN Graph similar to how netron does it
    * Project Link: https://github.com/lutzroeder/netron/
*/

qnn.Parameter = class {

    constructor(name, args) {
        this._name = name;
        this._arguments = args;
    }

    get name() {
        return this._name;
    }

    get visible() {
        return true;
    }

    get arguments() {
        return this._arguments;
    }
};


qnn.Argument = class {

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new caffe.Error("Invalid argument identifier '" + JSON.stringify(name) + "'.");
        }
        this._name = name;
        this._type = type || null;
        this._initializer = initializer || null;
    }

    get name() {
        return this._name;
    }

    get type() {
        return this._type;
    }

    get initializer() {
        return this._initializer;
    }
};

/* OPEN_SOURCE_END */

/*
    * OPEN_SOURCE_START
    * The following code is derived from the netron open source project in order to keep track of tensor information
    * in QNN Graph similiar to how netron does it
    * Project Link: https://github.com/lutzroeder/netron/
    * Note: There are a few minor modifications to accommodate for QNN-Netron use case
*/

qnn.Tensor = class {

    constructor(tensorInfo, data) {
        this._name = '';
        this._type = tensorInfo;
        this._kind = '';
        this._binVals=data;
        this._data = null;
    }

    get name() {
        return this._name;
    }

    get kind() {
        return this._kind;
    }

    get type() {
        return this._type;
    }

    get state() {
        return this._context().state;
    }

    get value() {
        let context = this._context();
        if (context.state) {
            return null;
        }
        context.limit = Number.MAX_SAFE_INTEGER;
        return this._decode(context, 0);
    }

    toString() {
        let context = this._context();
        if (context.state) {
            return '';
        }
        context.limit = 10000;
        let value = this._decode(context, 0);
        return JSON.stringify(value, null, 4);
    }

    _context() {
        let context = {};
        context.state = null;
        context.index = 0;
        context.count = 0;

        if (this._data == null) {
            context.state = this._binVals;
            return context;
        }

        context.dataType = this._type.dataType;
        context.shape = this._type.shape.dimensions;
        context.dimensions = this.type.shape.dimensions;
        return context;
    }

    _decode(context, dimension) {
        let shape = context.shape;
        if (shape.length == 0) {
            shape = [ 1 ];
        }
        let size = shape[dimension];
        let results = [];
        if (dimension == shape.length - 1) {
            for (let i = 0; i < size; i++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                switch (context.dataType) {
                    case 'float16':
                        results.push(context.data.getFloat16(context.index, true));
                        context.index += 2;
                        context.count++;
                        break;
                    case 'float32':
                        results.push(context.data.getFloat32(context.index, true));
                        context.index += 4;
                        context.count++;
                        break;
                    case 'quint8':
                        results.push(context.data.getUint8(context.index));
                        context.index += 1;
                        context.count++;
                        break;
                    case 'qint16':
                        results.push(context.data.getInt16(context.index, true));
                        context.index += 2;
                        context.count++;
                        break;
                    case 'int32':
                        results.push(context.data.getInt32(context.index, true));
                        context.index += 4;
                        context.count++;
                        break;
                    case 'boolean':
                        results.push(context.data.getInt8(context.index));
                        context.index += 1;
                        context.count++;
                        break;
                    default:
                        break;
                }
            }
        }
        else {
            for (let j = 0; j < size; j++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                results.push(this._decode(context, dimension + 1));
            }
        }
        if (context.shape.length == 0) {
            return results[0];
        }
        return results;
    }

    get kind() {
        return null;
    }

    /* OPEN_SOURCE_END */

};

/*
    * OPEN_SOURCE_START
    * The following code is derived from the netron open source project in order to keep track of tensor type, shape,
    * model meta-data, and error throwing for loading a QNN model
    * Project Link: https://github.com/lutzroeder/netron/
    * Note: There are a few minor modifications to accommodate for QNN-Netron use case
*/

qnn.TensorType = class {

    constructor(dataType, shape) {
        this._dataType = dataType;
        this._shape = shape;
    }

    get dataType() {
        return this._dataType;
    }

    get shape() {
        return this._shape;
    }

    toString() {
        return (this.dataType || '?') + this._shape.toString();
    }
};


qnn.TensorShape = class {

    constructor(dimensions) {
        this._dimensions = dimensions;
    }

    get dimensions() {
        return this._dimensions;
    }

    toString() {
        if (!this._dimensions || this._dimensions.length == 0) {
            return '';
        }
        return '[' + this._dimensions.map((dimension) => dimension.toString()).join(',') + ']';
    }
};

qnn.Metadata = class {

    static open(context) {
        if (qnn.Metadata._metadata) {
            return Promise.resolve(qnn.Metadata._metadata);
        }
        return context.request('qnn-metadata.json', 'utf-8', null).then((data) => {
            qnn.Metadata._metadata = new qnn.Metadata(data);
            return qnn.Metadata._metadata;
        }).catch(() => {
            qnn.Metadata._metadata = new qnn.Metadata(null);
            return qnn.Metadata._metadata;
        });
    }

  constructor(data) {
        this._map = new Map();
        this._attributeCache = {};
        if (data) {
            const metadata = JSON.parse(data);
            this._map = new Map(metadata.map((item) => [ item.name, item ]));
        }
    }

    type(name) {
        return this._map.get(name);
    }

    attribute(type, name) {
        let map = this._attributeCache[type];
        if (!map) {
            map = {};
            const schema = this.type(type);
            if (schema && schema.attributes && schema.attributes.length > 0) {
                for (const attribute of schema.attributes) {
                    map[attribute.name] = attribute;
                }
            }
            this._attributeCache[type] = map;
        }
        return map[name] || null;
    }
};



qnn.Error = class extends Error {
    constructor(message) {
        super(message);
        this.name = 'Error loading QNN model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = qnn.ModelFactory;
}

/* OPEN_SOURCE_END */
