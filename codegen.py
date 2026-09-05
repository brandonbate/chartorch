from dataset_store import dataloader_snippet

MODULES = {
    "Linear": [
        {"name": "in_features", "type": "int"},
        {"name": "out_features", "type": "int"},
        {"name": "bias", "type": "bool"},
    ],
    "Conv2d": [
        {"name": "in_channels", "type": "int"},
        {"name": "out_channels", "type": "int"},
        {"name": "kernel_size", "type": "dim"},
        {"name": "stride", "type": "dim"},
        {"name": "padding", "type": "dim"},
        {"name": "dilation", "type": "dim"},
        {"name": "groups", "type": "int"},
        {"name": "bias", "type": "bool"},
        {"name": "padding_mode", "type": "enum"},
    ],
    "MaxPool2d": [
        {"name": "kernel_size", "type": "dim"},
        {"name": "stride", "type": "intOrNone"},
        {"name": "padding", "type": "dim"},
        {"name": "dilation", "type": "dim"},
        {"name": "return_indices", "type": "bool"},
        {"name": "ceil_mode", "type": "bool"},
    ],
    "Flatten": [
        {"name": "start_dim", "type": "int"},
        {"name": "end_dim", "type": "int"},
    ],
    "ReLU": [{"name": "inplace", "type": "bool"}],
    "Sigmoid": [],
    "Tanh": [],
    "Dropout": [
        {"name": "p", "type": "float"},
        {"name": "inplace", "type": "bool"},
    ],
    "BatchNorm2d": [
        {"name": "num_features", "type": "int"},
        {"name": "eps", "type": "float"},
        {"name": "momentum", "type": "float"},
        {"name": "affine", "type": "bool"},
        {"name": "track_running_stats", "type": "bool"},
    ],
    "LayerNorm": [
        {"name": "normalized_shape", "type": "dim"},
        {"name": "eps", "type": "float"},
        {"name": "elementwise_affine", "type": "bool"},
        {"name": "bias", "type": "bool"},
    ],
}

OPTIMIZERS = {
    "SGD": {
        "className": "SGD",
        "args": [
            {"name": "lr", "type": "float"},
            {"name": "momentum", "type": "float"},
            {"name": "dampening", "type": "float"},
            {"name": "weight_decay", "type": "float"},
            {"name": "nesterov", "type": "bool"},
            {"name": "maximize", "type": "bool"},
            {"name": "foreach", "type": "boolOrNone"},
            {"name": "differentiable", "type": "bool"},
            {"name": "fused", "type": "boolOrNone"},
        ],
    },
    "Adam": {
        "className": "Adam",
        "args": [
            {"name": "lr", "type": "float"},
            {"name": "betas", "type": "floatTuple"},
            {"name": "eps", "type": "float"},
            {"name": "weight_decay", "type": "float"},
            {"name": "amsgrad", "type": "bool"},
            {"name": "foreach", "type": "boolOrNone"},
            {"name": "maximize", "type": "bool"},
            {"name": "capturable", "type": "bool"},
            {"name": "differentiable", "type": "bool"},
            {"name": "fused", "type": "boolOrNone"},
        ],
    },
    "AdamW": {
        "className": "AdamW",
        "args": [
            {"name": "lr", "type": "float"},
            {"name": "betas", "type": "floatTuple"},
            {"name": "eps", "type": "float"},
            {"name": "weight_decay", "type": "float"},
            {"name": "amsgrad", "type": "bool"},
            {"name": "maximize", "type": "bool"},
            {"name": "foreach", "type": "boolOrNone"},
            {"name": "capturable", "type": "bool"},
            {"name": "differentiable", "type": "bool"},
            {"name": "fused", "type": "boolOrNone"},
        ],
    },
}

LOSSES = {
    "MSE": {
        "className": "MSELoss",
        "args": [{"name": "reduction", "type": "enum"}],
    },
    "L1": {
        "className": "L1Loss",
        "args": [{"name": "reduction", "type": "enum"}],
    },
    "Huber": {
        "className": "HuberLoss",
        "args": [
            {"name": "reduction", "type": "enum"},
            {"name": "delta", "type": "float"},
        ],
    },
    "CrossEntropy": {
        "className": "CrossEntropyLoss",
        "args": [
            {"name": "weight", "type": "noneOrExpr"},
            {"name": "ignore_index", "type": "int"},
            {"name": "reduction", "type": "enum"},
            {"name": "label_smoothing", "type": "float"},
        ],
    },
    "BCE": {
        "className": "BCELoss",
        "args": [
            {"name": "weight", "type": "noneOrExpr"},
            {"name": "reduction", "type": "enum"},
        ],
    },
    "BCEWithLogits": {
        "className": "BCEWithLogitsLoss",
        "args": [
            {"name": "weight", "type": "noneOrExpr"},
            {"name": "reduction", "type": "enum"},
            {"name": "pos_weight", "type": "noneOrExpr"},
        ],
    },
}


def python_value(field, value):
    if value is True:
        return "True"
    if value is False:
        return "False"
    if value is None:
        return "None"
    if field["type"] == "enum":
        return f"'{value}'"
    return str(value)


def _outgoing(edges):
    return {edge["from"]: edge for edge in edges.values()}


def _incoming_ids(edges):
    return {edge["to"] for edge in edges.values()}


def linear_path(nodes, edges):
    n = len(nodes)
    if n == 0:
        return None
    if len(edges) != n - 1:
        return None
    incoming = _incoming_ids(edges)
    outgoing = _outgoing(edges)
    sources = [node for node in nodes.values() if node["id"] not in incoming]
    if len(sources) != 1:
        return None
    path = []
    seen = set()
    current = sources[0]
    while current:
        node_id = current["id"]
        if node_id in seen:
            return None
        seen.add(node_id)
        path.append(current)
        edge = outgoing.get(node_id)
        current = nodes.get(edge["to"]) if edge else None
    if len(path) != n:
        return None
    return path


def to_python(payload):
    nodes = payload.get("nodes") or {}
    edges = payload.get("edges") or {}
    training = payload.get("training") or {}
    snippet = dataloader_snippet(payload.get("dataset"))
    if snippet is None:
        return None
    path = linear_path(nodes, edges)
    if path is None:
        return None

    opt = OPTIMIZERS.get(training.get("optimizer"))
    loss = LOSSES.get(training.get("loss"))
    if opt is None or loss is None:
        return None
    opt_args_vals = training.get("optimizerArgs") or {}
    loss_args_vals = training.get("lossArgs") or {}
    epochs = training.get("epochs")
    if not isinstance(epochs, int):
        return None

    layers = []
    for node in path:
        fields = MODULES.get(node.get("type"))
        if fields is None:
            return None
        args = node.get("args") or {}
        if not fields:
            layers.append(f"    nn.{node['type']}()")
        else:
            joined = ", ".join(
                f"{field['name']}={python_value(field, args.get(field['name']))}"
                for field in fields
            )
            layers.append(f"    nn.{node['type']}({joined})")

    opt_args = ",\n".join(
        f"    {field['name']}={python_value(field, opt_args_vals.get(field['name']))}"
        for field in opt["args"]
    )
    loss_args = ", ".join(
        f"{field['name']}={python_value(field, loss_args_vals.get(field['name']))}"
        for field in loss["args"]
    )
    layer_block = ",\n".join(layers)

    return f"""import torch
import torch.nn as nn

{snippet}

model = nn.Sequential(
{layer_block},
)

criterion = nn.{loss['className']}({loss_args})
optimizer = torch.optim.{opt['className']}(
    model.parameters(),
{opt_args},
)

epochs = {epochs}
model.train()
for epoch in range(epochs):
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
"""
