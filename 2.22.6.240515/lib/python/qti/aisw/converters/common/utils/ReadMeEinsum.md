# Guide on how to extend Einsum Optimizer to support newer equations.

#### Einsum Optimizer uses einsum_equ.py file which has mapping of all the supported einsum equation to its respective mathematically equivalent sub-graph.

#### In einsum_equ.py file, the sub-graph is presented as below.
- All the nodes in sub-graph can be represented with the help of Node class which is defined in the same file.
```python
Node(<node_name>, <node_op_type>, <list_of_input_node_names>, <dict_for_attributes>)

Node("unsqueeze_3", "Unsqueeze", inputs=["ip_1"], attrs={"axis": [0]})
```
-  Each sub-graph starts with "Input" op_type nodes, which can be 1 or more and ends with single "Output" op_type node.
- The edges of sub-graphs are defined by inputs parameter of each node. The inputs parameter shall include the name of the node with which the given node is connected. "Input" op_type nodes will not have any inputs since they are input nodes.
- Node names used in the einsum_equ.py file are for representation purpose only. After einsum optimization, model will not have these names.
- The sub-graph described using Node class shall be a valid sub-graph with no dangling nodes. In case of invalid sub-graph, the einsum optimizer will fail to generate the correct model.
- The name of the "Input" op_type nodes shall be with pattern "input_*". The name of the "Output" op_type shall be "output".
- The einsum equation can be provided using any character. e.g. "abc,cd->abd" and "xyz,zp->xyp" both will represent same functionality. Hence, only mapping for one can be provided in the einsum_equ.py file.

#### Example for "abc,cd->abd" einsum equation
For "abc,cd->abd" equation, the inner dimensions shall be same in order to properly matmul these array. This einsum equation is equivalent to a single matmul operation. For this equation, the sub-graph can be defined as follows.

```python
{
    "abc,cd->abd": [
        Node("input_1", "Input", inputs=[]),        # First nodes are always input nodes.
        Node("input_2", "Input", inputs=[]),        # First nodes are always input nodes.
        Node("matmul_3", "Matmul", inputs=["input_1", "input_2"]),        # This will create a matmul node with "input_1" and "input_2" as inputs. Mind the ordering of inputs.
        Node("output", "Output", inputs=["matmul_3"]),  # Last node will be an output node with input as "matmul_3" node.
    ]
}
```

Add the above sub-graph representation into einsum mapping file appropriately to provide the support for the same in the converter stack.