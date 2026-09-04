import { incoming, outgoing, state } from "./model.js";
import { MODULES } from "./modules.js";
import { closeEditor } from "./render.js";

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
  return `import torch.nn as nn\n\nmodel = nn.Sequential(\n${layers.join(",\n")},\n)\n`;
}

function pythonValue(field, value) {
  if (value === true) return "True";
  if (value === false) return "False";
  if (value === null) return "None";
  if (field.type === "enum") return `'${value}'`;
  return String(value);
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
