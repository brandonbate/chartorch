from dataset_store import dataloader_snippet, test_dataloader_snippet

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

CLASSIFICATION_LOSSES = frozenset({"CrossEntropy", "BCE", "BCEWithLogits"})

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


def _parts(payload):
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
    loss_name = training.get("loss")
    loss = LOSSES.get(loss_name)
    if opt is None or loss is None:
        return None
    opt_args_vals = training.get("optimizerArgs") or {}
    loss_args_vals = training.get("lossArgs") or {}
    epochs = training.get("epochs")
    if not isinstance(epochs, int):
        return None
    batch_size = training.get("batchSize")
    if not isinstance(batch_size, int) or batch_size < 1:
        return None
    shuffle = training.get("shuffle")
    if not isinstance(shuffle, bool):
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
    return {
        "snippet": snippet,
        "opt": opt,
        "loss": loss,
        "loss_name": loss_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "opt_args": opt_args,
        "loss_args": loss_args,
        "layer_block": ",\n".join(layers),
        "dataset": payload.get("dataset"),
    }


def to_python(payload):
    parts = _parts(payload)
    if parts is None:
        return None

    return f"""import json
import torch
import torch.nn as nn

batch_size = {parts["batch_size"]}
shuffle = {python_value({"type": "bool"}, parts["shuffle"])}

{parts["snippet"]}

model = nn.Sequential(
{parts["layer_block"]},
)

criterion = nn.{parts["loss"]["className"]}({parts["loss_args"]})
optimizer = torch.optim.{parts["opt"]["className"]}(
    model.parameters(),
{parts["opt_args"]},
)

epochs = {parts["epochs"]}
model.train()
for epoch in range(epochs):
    running = 0.0
    n = 0
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running += loss.item()
        n += 1
    if n:
        _loss = running / n
        print(f"epoch {{epoch + 1}}/{{epochs}} loss {{_loss}}", flush=True)
        with open("metrics.jsonl", "a", encoding="utf-8") as _f:
            _f.write(json.dumps({{"epoch": epoch + 1, "loss": _loss}}) + "\\n")
            _f.flush()

torch.save(model.state_dict(), "model.pt")
torch.set_printoptions(threshold=float("inf"), linewidth=120)
with open("params.txt", "w", encoding="utf-8") as _f:
    for _name, _param in model.named_parameters():
        _f.write(f"{{_name}} {{tuple(_param.shape)}}\\n{{_param.detach().cpu()}}\\n\\n")
"""


def to_test_python(payload):
    parts = _parts(payload)
    if parts is None:
        return None
    test_snippet = test_dataloader_snippet(parts["dataset"])
    if test_snippet is None:
        return None
    classify = parts["loss_name"] in CLASSIFICATION_LOSSES
    if parts["loss_name"] == "CrossEntropy":
        pred_lines = "        preds = outputs.argmax(dim=1)\n        gold = targets"
    elif parts["loss_name"] == "BCEWithLogits":
        pred_lines = "        preds = outputs >= 0\n        gold = targets >= 0.5"
    else:
        pred_lines = "        preds = outputs >= 0.5\n        gold = targets >= 0.5"
    if classify:
        eval_block = f"""    failures = []
    idx = 0
    for inputs, targets in test_dataloader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        bs = inputs.size(0)
        if criterion.reduction == "mean":
            running += loss.item() * bs
        elif criterion.reduction == "sum":
            running += loss.item()
        else:
            running += loss.sum().item()
        n += bs
{pred_lines}
        mismatch = preds != gold
        if mismatch.ndim > 1:
            bad = mismatch.reshape(mismatch.size(0), -1).any(dim=1)
        else:
            bad = mismatch
        for i in range(bs):
            if bool(bad[i]):
                failures.append({{
                    "index": idx + i,
                    "label": _fmt(gold[i]),
                    "pred": _fmt(preds[i]),
                }})
        idx += bs
    _loss = running / n if n else 0.0
    n_failures = len(failures)
    accuracy = (n - n_failures) / n if n else 0.0
    result = {{
        "loss": _loss,
        "n": n,
        "accuracy": accuracy,
        "n_failures": n_failures,
        "failures": failures,
    }}"""
    else:
        eval_block = """    for inputs, targets in test_dataloader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        bs = inputs.size(0)
        if criterion.reduction == "mean":
            running += loss.item() * bs
        elif criterion.reduction == "sum":
            running += loss.item()
        else:
            running += loss.sum().item()
        n += bs
    _loss = running / n if n else 0.0
    result = {
        "loss": _loss,
        "n": n,
        "accuracy": None,
        "n_failures": None,
        "failures": [],
    }"""

    return f"""import json
import torch
import torch.nn as nn

batch_size = {parts["batch_size"]}

{test_snippet}

def _fmt(t):
    t = t.detach().cpu()
    if t.ndim == 0:
        v = t.item()
        if isinstance(v, float) and v == int(v):
            return str(int(v))
        return str(v)
    return str(t.tolist())

model = nn.Sequential(
{parts["layer_block"]},
)
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
model.eval()
criterion = nn.{parts["loss"]["className"]}({parts["loss_args"]})

running = 0.0
n = 0
with torch.no_grad():
{eval_block}

print(f"test loss {{result['loss']}} accuracy {{result['accuracy']}} failures {{result['n_failures']}}", flush=True)
with open("test_results.json", "w", encoding="utf-8") as _f:
    json.dump(result, _f)
    _f.flush()
"""
