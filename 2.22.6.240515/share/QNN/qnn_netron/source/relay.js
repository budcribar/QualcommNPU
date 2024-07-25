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

/* jshint esversion: 6 */

var relay = relay || {};
var json = json || require('./json');

const EXPR_TYPE = 'expr';
class Value {
    constructor(dtype, value) {
        this._dtype = dtype;
        this._val = value;
    }
    get value() {
        return this._val;
    }
    get dtype() {
        return this._dtype;
    }
}

function GetValue(val_in, type_meta, nodes) {
    const val = parseInt(val_in);
    let ret_type = undefined;
    let ret_val = undefined;

    const convertExpr = (expr_node) => {
        let expr_ret = undefined;
        if(expr_node.type_key === 'IntImm') {
            expr_ret = parseInt(expr_node.attrs.value);
            ret_type = expr_node.attrs.dtype;
        }
        else if(expr_node.type_key === 'runtime.String') {
            expr_ret = expr_node.repr_str;
            ret_type = 'string';
            if(!expr_ret) {
                expr_ret = '';
            }
        }
        else if(expr_node.type_key === 'FloatImm') {
            expr_ret = parseFloat(expr_node.attrs.value);
            ret_type = expr_node.attrs.dtype;
        }
        else if(expr_node.type_key === '') {
            expr_ret = '';
            ret_type = '';
        }
        else {
            console.error(`Unknown expr_node.type_key ${expr_node.type_key}. Index ${val_in}. Object: `, expr_node);
        }
        return expr_ret;
    };

    if(Array.isArray(type_meta)) {
        const node = nodes[val];
        ret_val = [];

        // Array must hold the same type. So only need type_meta[0].
        if(node.type_key !== 'Array') {
            console.debug(`Expect Array for node index ${val_in}, but got ${node.type_key}. Treat it as expr.`);
            // bet that this node is actually expr. This happen to SplitAttr.
            ret_val = convertExpr(node);
        }
        else if(type_meta[0] !== EXPR_TYPE) {
            for(let i = 0; i < node.data.length; ++i) {
                ret_val.push(node.data[i]);
            }
        }
        else {
            if(Array.isArray(node.data)) {
                for(let i = 0; i < node.data.length; ++i) {
                    const v = GetValue(node.data[i], type_meta[0], nodes);
                    ret_type = v.dtype + '[]';
                    ret_val.push(v.value);
                }
            }
            else {
                console.debug(`Node index ${val} is Array but data is empty or undefined.`);
                ret_type = '[]';
                ret_val = [];
            }
        }
    }
    else if(type_meta === EXPR_TYPE) {
        const node = nodes[val];
        ret_val = convertExpr(node);
    }
    else if(type_meta !== EXPR_TYPE) {
        ret_type = type_meta;
        ret_val = val_in;
    }
    return new Value(ret_type, ret_val);
}

// return type as string.
// Empty string if the node is not valid.
function ParseTensorType(node_idx, nodes) {
    if(nodes[node_idx].type_key === 'relay.TensorType') {
        const node_obj = nodes[node_idx];
        const dtype = node_obj.attrs.dtype;
        const shape = GetValue(node_obj.attrs.shape, [EXPR_TYPE], nodes);
        const tensor_type = new relay.TensorType(dtype, new relay.TensorShape(shape.value));
        return tensor_type.toString();
    }
    else if(nodes[node_idx].type_key === 'TupleType') {
        const fields_idx = parseInt(nodes[node_idx].attrs.fields);
        const tensor_type_idx_array = nodes[fields_idx];
        if(tensor_type_idx_array.type_key !== 'Array') {
            console.error('Unknown tensor type representation.');
            return 'TupleType_unknown';
        }
        const ret = [];
        for(let i = 0; i < tensor_type_idx_array.data.length; ++i) {
            ret.push(ParseTensorType(tensor_type_idx_array.data[i], nodes));
        }
        return '(' + ret.join(', ') + ')';
    }
    return '';
}

class NdArray {

    static fromB64Str(blob, index) {
        return new NdArray(blob, index, 'base64');
    }

    constructor(blob, index, encoding) {
        this._buf = Buffer.from(blob, encoding);
        this._idx = index;
        this._offset = 0;
        // followings are TVM ND Array attributes
        this._header = undefined;
        this._reserved = undefined;
        this._dev = undefined;
        this._ndims = undefined;
        this._dtype = undefined;
        this._shape = undefined;
        this._data_byte_size = undefined;
        this._data = undefined;
        // followings are for netron
        this._tensor_type = undefined;
        this._tensor_shape = undefined;
        this._tensor = undefined;
        this._state = '';

        // begin to decode
        this._decodeHeader();
        this._decodeReserved();
        this._decodeDev();
        this._decodeNdim();
        this._decodeDType();
        this._decodeShape();
        this._decodeDataAndTensor();

    }

    get tensor() {
        return this._tensor;
    }

    _advanceOffset(num_bits) { this._offset += num_bits / 8; }

    _codeToDType(code) {
        // refer to DLDataTypeCode
        const map = {
            0: "Int",
            1: "UInt",
            2: "Float",
            3: "OpaqueHandle",
            4: "Bfloat",
            5: "Complex",
        };
        return map[code];
    }

    _decodeHeader() {
        const header = this._buf.readBigUInt64LE(this._offset);
        // kTVMNDArrayMagic
        if(header !== BigInt("15951258332257624383")) {
            console.error(`NDArray in index ${this._idx} has wrong header ${header}`);
            this._state += `NDArray in index ${this._idx} has wrong header ${header}\n`;
        }
        else {
            this._header = header;
        }
        this._advanceOffset(64);
    }

    _decodeReserved() {
        this._reserved = this._buf.readBigUInt64LE(this._offset);
        this._advanceOffset(64);
    }

    _decodeDev() {
        // hope TVM don't change struct Device frequently.
        const device_type = this._buf.readInt32LE(this._offset);
        this._advanceOffset(32);
        if(device_type !== 1) {
            console.error(`NdArray index ${this._idx} with device_type ${device_type} !== 1(CPU)`);
            this._state += `NdArray index ${this._idx} with device_type ${device_type} !== 1(CPU)\n`;
        }

        const device_id = this._buf.readInt32LE(this._offset);
        this._advanceOffset(32);

        this._dev = {
            "device_type": device_type,
            "device_id": device_id,
        };
    }

    _decodeNdim() {
        const ndim = this._buf.readInt32LE(this._offset);
        this._advanceOffset(32);
        this._ndims = ndim;
    }

    _decodeDType() {
        const code = this._buf.readUInt8(this._offset);
        this._advanceOffset(8);

        const bits = this._buf.readUInt8(this._offset);
        this._advanceOffset(8);

        const lanes = this._buf.readUInt16LE(this._offset);
        this._advanceOffset(16);

        this._dtype = {
            "code": code,
            "bits": bits,
            "lanes": lanes,
        };
    }

    _decodeShape() {
        if(this._ndims === undefined) {
            console.error(`NDArray index ${this._idx} has undefined ndims.`);
            this._state += `NDArray index ${this._idx} has undefined ndims.\n`;
            return;
        }
        const shape = [];
        for(let i = 0; i < this._ndims; ++i) {
            shape.push(this._buf.readBigInt64LE(this._offset));
            this._advanceOffset(64);
        }
        this._shape = shape;
    }

    _decodeDataAndTensor() {
        if(this._dtype === undefined) {
            console.error(`NDArray index ${this._idx} has undefined dtype.`);
            this._state += `NDArray index ${this._idx} has undefined dtype.\n`;
            return;
        }
        if(this._shape === undefined) {
            console.error(`NDArray index ${this._idx} has undefined shape.`);
            this._state += `NDArray index ${this._idx} has undefined shape.\n`;
            return;
        }

        const elem_bytes = Math.floor((this._dtype.bits + 7) / 8);
        const dims = this._shape;
        let num_elems = BigInt("1");
        for(let i = 0; i < dims.length; ++i) {
            num_elems *= dims[i];
        }
        const data_byte_size = this._buf.readBigInt64LE(this._offset);
        this._advanceOffset(64);

        if(data_byte_size !== num_elems * BigInt(elem_bytes)) {
            console.error(`data_byte_size ${data_byte_size} !== ${num_elems}(num_elems) * ${elem_bytes}(elem_bytes)`);
            this._state.concat(`data_byte_size ${data_byte_size} !== ${num_elems}(num_elems) * ${elem_bytes}(elem_bytes)\n`);
            return;
        }
        this._data_byte_size = data_byte_size;

        // for netron
        this._tensor_shape = new relay.TensorShape(dims);
        this._tensor_type = new relay.TensorType(`${this._codeToDType(this._dtype.code)}${this._dtype.bits}`, this._tensor_shape);
        const dtype_lower = this._tensor_type.dataType.toLowerCase();
        const data = [];
        switch(dtype_lower) {
            case 'uint8':
                for(let i = 0; i < num_elems; ++i) {
                    data.push(this._buf.readUInt8(this._offset));
                    this._advanceOffset(8);
                }
                break;
            case 'int8':
                for(let i = 0; i < num_elems; ++i) {
                    data.push(this._buf.readInt8(this._offset));
                    this._advanceOffset(8);
                }
                break;
            case 'uint16':
                for(let i = 0; i < num_elems; ++i) {
                    data.push(this._buf.readUInt16LE(this._offset));
                    this._advanceOffset(16);
                }
                break;
            case 'int16':
                for(let i = 0; i < num_elems; ++i) {
                    data.push(this._buf.readInt16LE(this._offset));
                    this._advanceOffset(16);
                }
                break;
            case 'uint32':
                for(let i = 0; i < num_elems; ++i) {
                    data.push(this._buf.readUInt32LE(this._offset));
                    this._advanceOffset(32);
                }
                break;
            case 'int32':
                for(let i = 0; i < num_elems; ++i) {
                    data.push(this._buf.readInt32LE(this._offset));
                    this._advanceOffset(32);
                }
                break;
            case 'float32':
                for(let i = 0; i < num_elems; ++i) {
                    data.push(this._buf.readFloatLE(this._offset));
                    this._advanceOffset(32);
                }
                break;
            default:
                console.error(`NDArray index ${this._idx} with unknown dtype ${dtype_lower}`);
                this._state += `NDArray index ${this._idx} with unknown dtype ${dtype_lower}\n`;
                break;
        }

        this._data = data;
        this._tensor = new relay.Tensor(this._tensor_type, data, this._state);

        if(this._offset !== this._buf.byteLength) {
            console.error(`NDArray index ${this._idx} read ${this._offset} bytes, which don't match the buffer size ${this._buf.byteLength}`);
            this._state += `NDArray index ${this._idx} read ${this._offset} bytes, which don't match the buffer size ${this._buf.byteLength}\n`;
        }
    }
}

class ModelReader {
    constructor(json_model, metadata) {
        this._metadata = metadata;
        this._root = json_model.root;
        this._nodes = json_model.nodes;
        this._b64ndarrays = json_model.b64ndarrays;
        this._attrs = json_model.attrs;
        // index(in this._nodes) to outputs
        this._idx_to_output_arg = new Map();
        this._model_inputs = [];
        this._model_nodes = [];

        this._consumed_arg_idx_set = new Set();

        // first, add all arguments.
        this._addArgs();
        // then, add nodes.
        this._addNodes();
    }

    get model_inputs() {
        return this._model_inputs;
    }

    get model_nodes() {
        return this._model_nodes;
    }

    get model_outputs() {
        const all_args_idx = Array.from(this._idx_to_output_arg.keys(), x => x);
        const unconsumed_args_idx = all_args_idx.filter(x => !this._consumed_arg_idx_set.has(x));
        const args = Array.from(unconsumed_args_idx, x => this._idx_to_output_arg.get(x));
        return [new relay.Parameter('output', args)];
    }

    _addArgs() {

        // Relay Node don't have name for outputs.
        // Here we try to assign one to them.
        for(let idx = 0; idx < this._nodes.length; ++idx) {

            if(this._nodes[idx].type_key === 'relay.Var') {
                console.debug(`_addArgs(), relay.Var, idx=${idx}`);
                const vid_idx = parseInt(this._nodes[idx].attrs.vid);
                const name_idx = parseInt(this._nodes[vid_idx].attrs.name_hint);
                const name = this._nodes[name_idx].repr_str;
                // try to parse type
                const tensor_type_idx = parseInt(this._nodes[idx].attrs.type_annotation);
                const arg_type = ParseTensorType(tensor_type_idx, this._nodes);
                const arg = new relay.Argument(name, arg_type, null);
                this._idx_to_output_arg.set(idx, arg);
                // Also they are possibly graph inputs.
                this._model_inputs.push(new relay.Parameter(name, [ arg ]));
            }

            else if(this._nodes[idx].type_key === 'relay.Call') {
                console.debug(`_addArgs(), relay.Call, idx=${idx}`);
                const op_idx = parseInt(this._nodes[idx].attrs.op);
                const op_name = this._nodes[op_idx].repr_str;
                const op_output_name = op_name + '_' + idx.toString() + '_out';
                const checked_type_idx = parseInt(this._nodes[idx].attrs._checked_type_);
                const arg_type = ParseTensorType(checked_type_idx, this._nodes);
                this._idx_to_output_arg.set(idx, new relay.Argument(op_output_name, arg_type, null));
            }

            else if(this._nodes[idx].type_key === 'relay.TupleGetItem') {
                console.debug(`_addArgs(), relay.TupleGetItem, idx=${idx}`);
                const item_index_str = this._nodes[idx].attrs.index;
                const output_name = 'TupleGetItem' + idx + '_index_' + item_index_str;
                const checked_type_idx = parseInt(this._nodes[idx].attrs._checked_type_);
                const arg_type = ParseTensorType(checked_type_idx, this._nodes);
                this._idx_to_output_arg.set(idx, new relay.Argument(output_name, arg_type, null));
            }

            else if(this._nodes[idx].type_key === 'relay.Constant') {
                const blob_idx = parseInt(this._nodes[idx].attrs.data);
                const tensor = NdArray.fromB64Str(this._b64ndarrays[blob_idx], idx).tensor;
                const output_name = 'Constant_' + idx;
                if(tensor) {
                    this._idx_to_output_arg.set(idx, new relay.Argument(output_name, tensor.type.toString(), tensor));
                }
                else {
                    console.error(`Failed to parse b64ndarrays[${blob_idx}]`);
                    this._idx_to_output_arg.set(idx, new relay.Argument(output_name, 'error parsed NdArray', null));
                }
            }

            else if(this._nodes[idx].type_key === 'relay.Tuple') {
                const output_name = 'Tuple_' + idx;
                const checked_type_idx = parseInt(this._nodes[idx].attrs._checked_type_);
                const arg_type = ParseTensorType(checked_type_idx, this._nodes);
                this._idx_to_output_arg.set(idx, new relay.Argument(output_name, arg_type, null));
            }
        }
    }

    _addNodes() {

        for(let idx = 0; idx < this._nodes.length; ++idx ) {
            if(this._nodes[idx].type_key === 'relay.Call') {
                this._call(idx);
            }
            else if(this._nodes[idx].type_key === 'relay.TupleGetItem') {
                this._tupleGetItem(idx);
            }
            else if(this._nodes[idx].type_key === 'relay.Tuple') {
                this._tuple(idx);
            }
        }
    }

    _tuple(node_idx) {
        const op_def = this._metadata.type('relay.Tuple');
        const param_name = op_def.inputs[0].name;

        const fields_idx = parseInt(this._nodes[node_idx].attrs.fields);
        if(this._nodes[fields_idx].type_key !== 'Array') {
            console.error(`Tuple node ${node_idx}: fields ${fields_idx} is ${this._nodes[fields_idx].type_key}, not Array.`);
            return;
        }
        const args_idx_array = this._nodes[fields_idx].data;

        const input_args = [];
        for(let i = 0; i < args_idx_array.length; ++i) {
            input_args.push(this._idx_to_output_arg.get(args_idx_array[i]));
            this._consumed_arg_idx_set.add(args_idx_array[i]);
        }
        const inputs = [ new relay.Parameter(param_name, input_args) ];
        const outputs = [ new relay.Parameter('output', [ this._idx_to_output_arg.get(node_idx) ]) ];
        this._model_nodes.push(
            new relay.Node('Tuple_' + node_idx.toString(),
                'relay.Tuple',
                inputs,
                [],
                outputs,
                this._metadata
            )
        );
    }

    _tupleGetItem(node_idx) {
        const node_obj = this._nodes[node_idx];
        const arg_idx = parseInt(node_obj.attrs.tuple_value);
        const op_def = this._metadata.type('relay.TupleGetItem');
        const param_name = op_def.inputs[0].name;

        const inputs = [new relay.Parameter(param_name, [ this._idx_to_output_arg.get(arg_idx) ])];
        this._consumed_arg_idx_set.add(arg_idx);

        const item_index = GetValue(node_obj.attrs.index, op_def.attrs.index.type, this._nodes);
        const attrs = [new relay.Attribute('index', item_index.dtype, item_index.value)];

        const outputs = [ new relay.Parameter('output', [ this._idx_to_output_arg.get(node_idx) ]) ];
        this._model_nodes.push(
            new relay.Node('TupleGetItem_' + node_idx.toString(),
                'relay.TupleGetItem',
                inputs,
                attrs,
                outputs,
                this._metadata));
    }

    _call(node_idx) {
        const node_obj = this._nodes[node_idx];
        const args_container_idx = parseInt(node_obj.attrs.args);
        const args_idx_array = this._nodes[args_container_idx].data;
        const op_idx = parseInt(node_obj.attrs.op);
        const op_name = this._nodes[op_idx].repr_str;
        const op_def = this._metadata.type(op_name);

        if(!op_def) {
            console.error(`${op_name} not found in relay-metadata.json.`);
        }

        const inputs = [];
        for(let idx = 0; idx < args_idx_array.length; ++idx) {
            let param_name = 'in' + idx.toString();
            if(op_def) {
                param_name = op_def.inputs[idx].name;
            }
            const arg_idx = args_idx_array[idx];
            this._consumed_arg_idx_set.add(arg_idx);
            inputs.push(new relay.Parameter(param_name, [ this._idx_to_output_arg.get(arg_idx) ]));
        }

        const attrs = [];
        const attrs_idx = parseInt(node_obj.attrs.attrs);
        const attrs_node = this._nodes[attrs_idx];
        if(attrs_node['attrs']) {
            if(op_def) {
                const attrs_meta = op_def.attrs;
                for(const key in attrs_meta) {
                    if(attrs_node.attrs[key]) {
                        const val = GetValue(attrs_node.attrs[key], attrs_meta[key].type, this._nodes);
                        attrs.push(new relay.Attribute(key, val.dtype, val.value));
                    }
                    else {
                        attrs.push(new relay.Attribute(key, attrs_meta[key].type.toString(), attrs_node.attrs[key]));
                    }
                }
            }
            else {
                for(const key in attrs_node.attrs) {
                    attrs.push(new relay.Attribute(key, "error", "op not in relay-metadata.json"));
                }
            }
        }

        const outputs = [ new relay.Parameter('output', [ this._idx_to_output_arg.get(node_idx) ]) ];
        this._model_nodes.push(
            new relay.Node(op_name + '_' + node_idx.toString(),
                op_name,
                inputs,
                attrs,
                outputs,
                this._metadata));
    }

}

relay.ModelFactory = class {

    match(context) {
        const identifier = context.identifier;
        const extension = identifier.split('.').pop().toLowerCase();
        if (extension === 'json') {
            const obj = context.open('json');
            if (obj['attrs']) {
                return !!obj['attrs']['tvm_version'];
            }
        }
        return false;
    }

    open(context) {
        return relay.Metadata.open(context).then((metadata) => {
            return new relay.Model(metadata, context.open('json'));
        });
    }
};

relay.Model = class {
    constructor(metadata, model) {
        this._model = model;
        this._graphs = [];
        this._graphs.push(new relay.Graph(metadata, this._model));
    }

    get format() {
        return 'TVM Relay Model';
    }

    get graphs() {
        return this._graphs;
    }
};

/*
    * OPEN_SOURCE_START
    * The following code is derived from the netron open source project in order to keep track of relay graph
    * similar to how netron does it
    * Note: Minor changes for accommodating relay graph format
    * Project Link: https://github.com/lutzroeder/netron/
*/

relay.Graph = class {

    constructor(metadata, model) {

        this._name = 'TVM Graph';

        const reader = new ModelReader(model, metadata);
        this._inputs = reader.model_inputs;
        this._nodes = reader.model_nodes;
        this._outputs = reader.model_outputs;
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
};

/* OPEN_SOURCE_END */


/*
    * OPEN_SOURCE_START
    * The following code is derived from the netron open source project in order to keep track of relay tensors,
    * parameters, attributes, nodes, arguments, tensor shapes, tensor types, and error throwing for opening relay models
    * similar to how netron does it
    * Project Link: https://github.com/lutzroeder/netron/
    * Note: There are a few minor modifications to accommodate for relay QNN-Netron use case
*/

relay.Parameter = class {

    constructor(name, args) {
        this._name = name;
        const strip_undef = (x, idx) => {
            if(x !== undefined) {
                return x;
            }
            console.error(`Parameter ${this._name} has undefined argument with index ${idx}`);
            return new relay.Argument(`undefined ${idx}`, 'error', null);
        };
        this._args = args.map(strip_undef);
    }

    get name() {
        return this._name;
    }

    get visible() {
        return true;
    }

    get arguments() {
        return this._args;
    }
};

relay.Argument = class {

    constructor(name, type, initializer) {
        this._name = name;
        this._type = type;
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


relay.Node = class {

    constructor(name, type, inputs, attrs, outputs, metadata) {
        this._name = name;
        this._type_name = type;
        this._attrs = attrs;
        this._inputs = inputs;
        this._outputs = outputs;
        this._metadata = metadata;
    }

    get name() {
        return this._name;
    }

    get type() {
        return this._type_name;
    }

    get metadata() {
        return this._metadata.type(this._type_name);
    }

    get attributes() {
        return this._attrs;
    }

    get inputs() {
        return this._inputs;
    }

    get outputs() {
        return this._outputs;
    }

};

relay.Attribute = class {

    constructor(name, type, value) {
        this._name = name;
        this._type = type;
        this._value = value;
    }

    get name() {
        return this._name;
    }

    get type() {
        return this._type;
    }

    get value() {
        if(Array.isArray(this._value)) {
            return JSON.stringify(this._value, null, 4);
        }
        return this._value;
    }

    get visible() {
        const ret = (!!this._value) || (!!this._type);
        return ret;
    }
};

relay.Tensor = class {

    constructor(tensor_type, data, state) {
        this._type = tensor_type;
        this._data_array = data;
        this._state = state;
    }

    get type() {
        return this._type;
    }

    get state() {
        return this._state || null;
    }

    get value() {
        const context = this._context();
        if(context.state) {
            console.error(context.state);
            return '';
        }
        return this._decode(context, 0);
    }

    toString() {
        const context = this._context();
        context.limit = 1000;
        const value = this._decode(context, 0);
        if(context.state) {
            console.error('Error tensor context: ', context);
            return 'parsing error';
        }

        if(Array.isArray(value)) {
            return JSON.stringify(value, null, 4);
        }

        return value;
    }

    _context() {
        const context = {};
        context.state = this._state;
        context.index = 0;
        context.count = 0;
        context.limit = Number.MAX_SAFE_INTEGER;
        if(!this._data_array) {
            context.state += 'Tensor data is empty.\n';
            return context;
        }
        context.data_array = this._data_array;
        context.dims = this._type.shape.dimensions;
        context.isScalar = false;
        if(!context.dims || context.dims.length === 0) {
            context.isScalar = true;
        }
        return context;
    }

    _decode(context, dim) {
        if(context.isScalar) {
            if(context.data_array.length == 1){
                return context.data_array[0];
            }
            context.state += `Data is a scalar but data_array.length = ${context.data_array.length}`;
            return [];
        }
        const results = [];
        const size = context.dims[dim];
        if(dim === context.dims.length - 1) {
            for(let i = 0; i < size; ++i) {
                if(context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                results.push(context.data_array[context.index]);
                context.index++;
                context.count++;
            }
        }
        else {
            for(let i = 0; i < size; ++i) {
                if(context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                results.push(this._decode(context, dim + 1));
            }
        }
        return results;
    }
};

relay.TensorType = class {

    constructor(data_type, shape) {
        this._data_type = data_type;
        this._shape = shape;
    }

    get dataType() {
        return this._data_type;
    }

    get shape() {
        return this._shape;
    }

    toString() {
        return (this.dataType || '?') + ' ' + this._shape.toString();
    }
};

relay.TensorShape = class {

    constructor(dims) {
        this._dims = dims;
    }

    get dimensions() {
        return this._dims;
    }

    toString() {
        if (!this._dims || this._dims.length === 0) {
            return '';
        }
        return '[' + this._dims.map((dim) => dim.toString()).join(',') + ']';
    }
};

relay.Metadata = class {

    static open(context) {
        if (relay.Metadata._metadata) {
            return Promise.resolve(relay.Metadata._metadata);
        }
        return context.request('relay-metadata.json', 'utf-8', null).then((data) => {
            relay.Metadata._metadata = new relay.Metadata(data);
            return relay.Metadata._metadata;
        }).catch(() => {
            relay.Metadata._metadata = new relay.Metadata(null);
            return relay.Metadata._metadata;
        });
    }

    constructor(data) {
        this._map = new Map();
        if (data) {
            const metadata = JSON.parse(data);
            this._map = new Map(metadata.map((obj) => [ obj.type, obj ]));
        }
    }

    type(type_name) {
        return this._map.get(type_name);
    }

};

relay.Error = class extends Error {
    constructor(message) {
        super(message);
        this.name = 'Error loading Relay model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = relay.ModelFactory;
}

/* OPEN_SOURCE_END */