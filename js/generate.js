import { incoming, outgoing, state } from "./model.js";
import { closeEditor } from "./render.js";
import { training } from "./training.js";
import { dataState } from "./data.js";

const INVALID_MSG = "the network architecture is invalid";
const GENERATE_FAIL_MSG = "failed to generate code";

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
    fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        nodes: state.nodes,
        edges: state.edges,
        training: {
          optimizer: training.optimizer,
          optimizerArgs: training.optimizerArgs,
          loss: training.loss,
          lossArgs: training.lossArgs,
          epochs: training.epochs,
        },
        dataset: dataState.dataset,
      }),
    })
      .then(async (r) => {
        if (!r.ok) {
          if (r.status === 400) {
            window.alert(INVALID_MSG);
            return;
          }
          throw new Error();
        }
        const payload = await r.json();
        if (!payload || typeof payload.code !== "string") throw new Error();
        textarea.value = payload.code;
        panel.hidden = false;
      })
      .catch(() => {
        window.alert(GENERATE_FAIL_MSG);
      });
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
