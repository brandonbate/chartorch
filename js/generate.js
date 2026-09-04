import { incoming, outgoing, state } from "./model.js";
import { MODULES, pythonValue } from "./modules.js";
import { closeEditor } from "./render.js";
import { LOSSES, OPTIMIZERS, training } from "./training.js";

const INVALID_MSG = "the network architecture is invalid";

export function linearPath() {
  const nodes = Object.values(state.nodes);
  const n = nodes.length;
  if (n === 0) return null;
  if (Object.keys(state.edges).length !== n - 1) return null;
  const sources = nodes.filter((node) => !incoming(node.id));
  if (sources.length !== 1) return null;
  const path = [];
  const seen = new Set();
  let current = sources[0];
  while (current) {
    if (seen.has(current.id)) return null;
    seen.add(current.id);
    path.push(current);
    const edge = outgoing(current.id);
    current = edge ? state.nodes[edge.to] : null;
  }
  if (path.length !== n) return null;
  return path;
}

export function toPython(path) {
  const layers = path.map((node) => {
    const fields = MODULES[node.type].args;
    if (fields.length === 0) return `    nn.${node.type}()`;
    const args = fields
      .map((field) => `${field.name}=${pythonValue(field, node.args[field.name])}`)
      .join(", ");
    return `    nn.${node.type}(${args})`;
  });

  const opt = OPTIMIZERS[training.optimizer];
  const optArgs = opt.args
    .map((field) => `    ${field.name}=${pythonValue(field, training.optimizerArgs[field.name])}`)
    .join(",\n");

  const loss = LOSSES[training.loss];
  const lossArgs = loss.args
    .map((field) => `${field.name}=${pythonValue(field, training.lossArgs[field.name])}`)
    .join(", ");

  return `import torch
import torch.nn as nn

model = nn.Sequential(
${layers.join(",\n")},
)

criterion = nn.${loss.className}(${lossArgs})
optimizer = torch.optim.${opt.className}(
    model.parameters(),
${optArgs},
)

# TODO: provide a DataLoader named dataloader
# dataloader = ...

epochs = ${training.epochs}
model.train()
for epoch in range(epochs):
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
`;
}

export function initGenerate() {
  const panel = document.getElementById("generate-panel");
  const textarea = document.getElementById("generated-code");

  document.getElementById("btn-generate").addEventListener("click", () => {
    closeEditor(true);
    const path = linearPath();
    if (!path) {
      window.alert(INVALID_MSG);
      return;
    }
    textarea.value = toPython(path);
    panel.hidden = false;
  });

  document.getElementById("panel-close").addEventListener("click", () => {
    panel.hidden = true;
  });

  document.getElementById("btn-copy").addEventListener("click", async () => {
    const text = textarea.value;
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      textarea.select();
      document.execCommand("copy");
    }
  });
}
