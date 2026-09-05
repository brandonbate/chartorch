import { incoming, outgoing, state } from "./model.js";
import { closeEditor } from "./render.js";
import { training } from "./training.js";
import { dataState } from "./data.js";

const INVALID_MSG = "the network architecture is invalid";
const GENERATE_FAIL_MSG = "failed to generate code";
const TRAIN_FAIL_MSG = "failed to start training";

let generatedCode = "";
let trainBusy = false;
let pollTimer = null;
let lossChart = null;

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
  const btnTrain = document.getElementById("btn-train");

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
        generatedCode = payload.code;
        setCode(generatedCode);
        showTraceback("");
        setRightTab("code");
        panel.hidden = false;
        setTrainEnabled();
      })
      .catch(() => {
        window.alert(GENERATE_FAIL_MSG);
      });
  });

  document.getElementById("tab-code").addEventListener("click", () => setRightTab("code"));
  document.getElementById("tab-train").addEventListener("click", () => setRightTab("train"));

  btnTrain.addEventListener("click", () => {
    if (trainBusy || !generatedCode) return;
    trainBusy = true;
    setTrainEnabled();
    showTraceback("");
    fetch("/api/train", { method: "POST" })
      .then(async (r) => {
        if (!r.ok) throw new Error();
        const payload = await r.json();
        const epochs = payload.epochs;
        if (!Number.isInteger(epochs) || epochs < 1) throw new Error();
        initChart(epochs);
        startPoll();
      })
      .catch(() => {
        trainBusy = false;
        setTrainEnabled();
        window.alert(TRAIN_FAIL_MSG);
      });
  });

  document.getElementById("panel-close").addEventListener("click", () => {
    panel.hidden = true;
  });

  document.getElementById("btn-copy").addEventListener("click", async () => {
    const text = generatedCode;
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      const ta = document.createElement("textarea");
      ta.value = text;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      ta.remove();
    }
  });
}

function setRightTab(name) {
  document.getElementById("tab-code").classList.toggle("active", name === "code");
  document.getElementById("tab-train").classList.toggle("active", name === "train");
  document.getElementById("code-tab").hidden = name !== "code";
  document.getElementById("train-tab").hidden = name !== "train";
  if (name === "train" && lossChart) lossChart.resize();
}

function setTrainEnabled() {
  document.getElementById("btn-train").disabled = trainBusy || !generatedCode;
}

function setCode(script) {
  const pre = document.getElementById("generated-code");
  const code = pre.querySelector("code");
  code.textContent = script;
  pre.removeAttribute("data-line");
  if (window.Prism) window.Prism.highlightElement(code);
}

function parseTrainLines(text) {
  const lines = [];
  const re = /File "([^"]+)", line (\d+)/g;
  let m;
  while ((m = re.exec(text))) {
    const base = m[1].replaceAll("\\", "/").split("/").pop();
    if (base === "train.py") lines.push(m[2]);
  }
  return [...new Set(lines)];
}

function showTraceback(text) {
  const box = document.getElementById("traceback");
  const pre = document.getElementById("generated-code");
  const code = pre.querySelector("code");
  if (!text) {
    box.hidden = true;
    box.value = "";
    pre.removeAttribute("data-line");
  } else {
    box.hidden = false;
    box.value = text;
    const lines = parseTrainLines(text);
    if (lines.length) pre.setAttribute("data-line", lines.join(","));
    else pre.removeAttribute("data-line");
  }
  if (window.Prism) window.Prism.highlightElement(code);
}

function initChart(epochs) {
  const wrap = document.getElementById("loss-chart-wrap");
  const canvas = document.getElementById("loss-chart");
  wrap.hidden = false;
  if (lossChart) lossChart.destroy();
  const labels = Array.from({ length: epochs }, (_, i) => i + 1);
  lossChart = new window.Chart(canvas, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          data: Array(epochs).fill(null),
          spanGaps: true,
          showLine: true,
          pointRadius: 3,
          borderColor: "#333",
          backgroundColor: "#333",
        },
      ],
    },
    options: {
      animation: false,
      maintainAspectRatio: false,
      scales: {
        x: { ticks: { stepSize: 1, autoSkip: false } },
        y: {},
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          displayColors: false,
          callbacks: {
            title: () => "",
            label: (ctx) => `(${ctx.label}, ${ctx.parsed.y})`,
          },
        },
      },
    },
  });
}

function applyPoints(points) {
  if (!lossChart) return;
  const data = lossChart.data.datasets[0].data;
  let min = Infinity;
  let max = -Infinity;
  for (const p of points) {
    const i = p.epoch - 1;
    if (i >= 0 && i < data.length) data[i] = p.loss;
    if (typeof p.loss === "number") {
      min = Math.min(min, p.loss);
      max = Math.max(max, p.loss);
    }
  }
  if (min !== Infinity) {
    const pad = min === max ? Math.abs(min) * 0.1 || 0.1 : (max - min) * 0.1;
    lossChart.options.scales.y.min = min - pad;
    lossChart.options.scales.y.max = max + pad;
  }
  lossChart.update();
}

function startPoll() {
  stopPoll();
  const tick = () => {
    fetch("/api/train/status")
      .then((r) => {
        if (!r.ok) throw new Error();
        return r.json();
      })
      .then((s) => {
        applyPoints(s.points || []);
        if (s.done) {
          stopPoll();
          trainBusy = false;
          setTrainEnabled();
          if (s.exit_code) {
            showTraceback(s.traceback || "training failed");
            setRightTab("code");
          } else {
            showTraceback("");
          }
        }
      })
      .catch(() => {});
  };
  tick();
  pollTimer = setInterval(tick, 400);
}

function stopPoll() {
  if (pollTimer !== null) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}
