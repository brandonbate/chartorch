import { incoming, outgoing, state } from "./model.js";
import { closeEditor } from "./render.js";
import { training } from "./training.js";
import { chartRange, dataState, renderMedia, scatterDatasets, scatterOptions } from "./data.js";

const INVALID_MSG = "the network architecture is invalid";
const GENERATE_FAIL_MSG = "failed to generate code";
const TRAIN_FAIL_MSG = "failed to start training";
const TEST_FAIL_MSG = "failed to start testing";

let generatedCode = "";
let generatedDataset = "";
let trainBusy = false;
let testReady = false;
let testBusy = false;
let pollTimer = null;
let lossChart = null;
let testPlotChart = null;
let failures = [];
let failIndex = 0;
let failLoading = false;
let failSeq = 0;

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
          batchSize: training.batchSize,
          shuffle: training.shuffle,
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
        generatedDataset = dataState.dataset;
        setCode(generatedCode);
        showTraceback("");
        setRightTab("code");
        panel.hidden = false;
        stopPoll();
        trainBusy = false;
        testBusy = false;
        testReady = false;
        resetTestUi();
        setButtons();
      })
      .catch(() => {
        window.alert(GENERATE_FAIL_MSG);
      });
  });

  document.getElementById("tab-code").addEventListener("click", () => setRightTab("code"));
  document.getElementById("tab-train").addEventListener("click", () => setRightTab("train"));

  document.getElementById("btn-stop").addEventListener("click", () => {
    if (!trainBusy) return;
    fetch("/api/train/stop", { method: "POST" }).catch(() => {});
  });

  btnTrain.addEventListener("click", () => {
    if (trainBusy || testBusy || !generatedCode) return;
    trainBusy = true;
    testReady = false;
    resetTestUi();
    setButtons();
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
        setButtons();
        window.alert(TRAIN_FAIL_MSG);
      });
  });

  document.getElementById("btn-test").addEventListener("click", () => {
    if (trainBusy || testBusy || !testReady) return;
    testBusy = true;
    setButtons();
    showTestError("");
    document.getElementById("test-stats").hidden = true;
    document.getElementById("fail-sample").hidden = true;
    hideTestPlot();
    fetch("/api/test", { method: "POST" })
      .then((r) => {
        if (!r.ok) throw new Error();
        startTestPoll();
      })
      .catch(() => {
        testBusy = false;
        setButtons();
        window.alert(TEST_FAIL_MSG);
      });
  });

  const failPrev = document.getElementById("fail-prev");
  const failNext = document.getElementById("fail-next");
  const failIndexInput = document.getElementById("fail-index");
  failPrev.addEventListener("click", () => {
    if (failLoading || failIndex <= 0) return;
    failIndex -= 1;
    loadFailPreview();
  });
  failNext.addEventListener("click", () => {
    if (failLoading || failIndex >= failures.length - 1) return;
    failIndex += 1;
    loadFailPreview();
  });
  const commitFailIndex = () => {
    const parsed = Number.parseInt(failIndexInput.value, 10);
    if (Number.isInteger(parsed) && failures.length) {
      const next = Math.min(failures.length - 1, Math.max(0, parsed));
      if (next !== failIndex) {
        failIndex = next;
        loadFailPreview();
      }
    }
    failIndexInput.value = String(failIndex);
  };
  failIndexInput.addEventListener("change", commitFailIndex);
  failIndexInput.addEventListener("blur", commitFailIndex);
  failIndexInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      failIndexInput.blur();
    }
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
  if (name === "train" && testPlotChart) testPlotChart.resize();
}

function setButtons() {
  document.getElementById("btn-train").disabled = trainBusy || testBusy || !generatedCode;
  const btnStop = document.getElementById("btn-stop");
  btnStop.hidden = !trainBusy;
  const btnTest = document.getElementById("btn-test");
  btnTest.hidden = !testReady;
  btnTest.disabled = trainBusy || testBusy || !testReady;
}

function resetTestUi() {
  failures = [];
  failIndex = 0;
  document.getElementById("btn-test").hidden = true;
  document.getElementById("test-stats").hidden = true;
  document.getElementById("fail-sample").hidden = true;
  hideTestPlot();
  showTestError("");
}

function showTestError(text) {
  const el = document.getElementById("test-error");
  if (!text) {
    el.hidden = true;
    el.textContent = "";
    return;
  }
  el.hidden = false;
  el.textContent = text;
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
        x: { ticks: { autoSkip: true, maxRotation: 0, minRotation: 0 } },
        y: { ticks: { autoSkip: true } },
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
          if (s.cancelled) {
            testReady = false;
            setButtons();
            showTraceback("");
          } else if (s.exit_code) {
            testReady = false;
            setButtons();
            showTraceback(s.traceback || "training failed");
            setRightTab("code");
          } else {
            testReady = true;
            setButtons();
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

function startTestPoll() {
  stopPoll();
  const tick = () => {
    fetch("/api/test/status")
      .then((r) => {
        if (!r.ok) throw new Error();
        return r.json();
      })
      .then((s) => {
        if (!s.done) return;
        stopPoll();
        testBusy = false;
        setButtons();
        if (s.exit_code) {
          showTestError(s.traceback || "testing failed");
          return;
        }
        if (!s.result) {
          showTestError("testing failed");
          return;
        }
        applyTestResult(s.result);
      })
      .catch(() => {});
  };
  tick();
  pollTimer = setInterval(tick, 400);
}

function hideTestPlot() {
  const wrap = document.getElementById("test-plot-wrap");
  wrap.hidden = true;
  if (testPlotChart) {
    testPlotChart.destroy();
    testPlotChart = null;
  }
}

function mixWithWhite(hex) {
  const raw = String(hex || "").replace(/^#/, "");
  let r;
  let g;
  let b;
  if (raw.length === 3) {
    r = parseInt(raw[0] + raw[0], 16);
    g = parseInt(raw[1] + raw[1], 16);
    b = parseInt(raw[2] + raw[2], 16);
  } else if (raw.length === 6) {
    r = parseInt(raw.slice(0, 2), 16);
    g = parseInt(raw.slice(2, 4), 16);
    b = parseInt(raw.slice(4, 6), 16);
  } else {
    return "#eeeeee";
  }
  if (![r, g, b].every((v) => Number.isFinite(v))) return "#eeeeee";
  const toHex = (v) => Math.round((v + 255) / 2).toString(16).padStart(2, "0");
  return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
}

function regionPlugin(plot) {
  return {
    id: "regions2d",
    beforeDatasetsDraw(chart) {
      const nx = plot.nx;
      const ny = plot.ny;
      const pred = plot.pred;
      if (!nx || !ny || !Array.isArray(pred) || pred.length < nx * ny) return;
      const xScale = chart.scales.x;
      const yScale = chart.scales.y;
      const ctx = chart.ctx;
      const area = chart.chartArea;
      const dx = (plot.x_max - plot.x_min) / nx;
      const dy = (plot.y_max - plot.y_min) / ny;
      const colors = plot.class_colors || {};
      const fills = {};
      for (const key of Object.keys(colors)) fills[key] = mixWithWhite(colors[key]);
      ctx.save();
      ctx.beginPath();
      ctx.rect(area.left, area.top, area.right - area.left, area.bottom - area.top);
      ctx.clip();
      for (let iy = 0; iy < ny; iy++) {
        for (let ix = 0; ix < nx; ix++) {
          const cls = pred[iy * nx + ix];
          ctx.fillStyle = fills[String(cls)] || fills[cls] || "#eeeeee";
          const x0 = plot.x_min + ix * dx;
          const x1 = x0 + dx;
          const y0 = plot.y_min + iy * dy;
          const y1 = y0 + dy;
          const left = xScale.getPixelForValue(x0);
          const right = xScale.getPixelForValue(x1);
          const top = yScale.getPixelForValue(y1);
          const bottom = yScale.getPixelForValue(y0);
          ctx.fillRect(left, top, right - left + 1, bottom - top + 1);
        }
      }
      ctx.restore();
    },
  };
}

function renderTestPlot(plot) {
  const wrap = document.getElementById("test-plot-wrap");
  const canvas = document.getElementById("test-plot");
  const points = Array.isArray(plot.points) ? plot.points : [];
  const curve = Array.isArray(plot.curve) ? plot.curve : [];
  const regions = plot.kind === "regions2d" && Array.isArray(plot.pred) && plot.pred.length;
  if (!points.length && !curve.length && !regions) {
    hideTestPlot();
    return;
  }
  wrap.hidden = false;
  if (testPlotChart) testPlotChart.destroy();
  if (plot.kind === "regions2d") {
    const xr =
      typeof plot.x_min === "number" && typeof plot.x_max === "number"
        ? [plot.x_min, plot.x_max]
        : chartRange(points.map((p) => p.x));
    const yr =
      typeof plot.y_min === "number" && typeof plot.y_max === "number"
        ? [plot.y_min, plot.y_max]
        : chartRange(points.map((p) => p.y));
    testPlotChart = new window.Chart(canvas, {
      type: "scatter",
      data: { datasets: scatterDatasets(points) },
      plugins: [regionPlugin(plot)],
      options: {
        ...scatterOptions(),
        scales: {
          x: xr ? { min: xr[0], max: xr[1] } : {},
          y: yr ? { min: yr[0], max: yr[1] } : {},
        },
      },
    });
    return;
  }
  const xs = points.map((p) => p.x).concat(curve.map((p) => p.x));
  const ys = points.map((p) => p.y).concat(curve.map((p) => p.y));
  const xr = chartRange(xs);
  const yr = chartRange(ys);
  testPlotChart = new window.Chart(canvas, {
    type: "scatter",
    data: {
      datasets: [
        ...scatterDatasets(points),
        {
          data: curve,
          showLine: true,
          pointRadius: 0,
          borderColor: "#1a73e8",
          backgroundColor: "#1a73e8",
        },
      ],
    },
    options: {
      ...scatterOptions(),
      scales: {
        x: xr ? { min: xr[0], max: xr[1] } : {},
        y: yr ? { min: yr[0], max: yr[1] } : {},
      },
    },
  });
}

function applyTestResult(result) {
  showTestError("");
  const stats = document.getElementById("test-stats");
  stats.hidden = false;
  document.getElementById("test-loss").textContent = `test loss = ${result.loss}`;
  const accEl = document.getElementById("test-accuracy");
  const failEl = document.getElementById("test-failures");
  if (result.plot) {
    failEl.hidden = true;
    document.getElementById("fail-sample").hidden = true;
    failures = [];
    if (result.accuracy == null) {
      accEl.hidden = true;
    } else {
      accEl.hidden = false;
      accEl.textContent = `test accuracy = ${result.accuracy}`;
    }
    renderTestPlot(result.plot);
    return;
  }
  hideTestPlot();
  if (result.accuracy == null || result.n_failures == null) {
    accEl.hidden = true;
    failEl.hidden = true;
    document.getElementById("fail-sample").hidden = true;
    failures = [];
    return;
  }
  accEl.hidden = false;
  failEl.hidden = false;
  accEl.textContent = `test accuracy = ${result.accuracy}`;
  failEl.textContent = `test failures = ${result.n_failures}`;
  failures = Array.isArray(result.failures) ? result.failures : [];
  failIndex = 0;
  if (!failures.length) {
    document.getElementById("fail-sample").hidden = true;
    return;
  }
  document.getElementById("fail-sample").hidden = false;
  loadFailPreview();
}

function setFailNav() {
  const prev = document.getElementById("fail-prev");
  const next = document.getElementById("fail-next");
  const indexInput = document.getElementById("fail-index");
  const n = failures.length;
  const busy = failLoading || n <= 0;
  prev.disabled = busy || failIndex <= 0;
  next.disabled = busy || failIndex >= n - 1;
  indexInput.disabled = busy;
}

function loadFailPreview() {
  const item = failures[failIndex];
  const sample = document.getElementById("fail-sample");
  if (!item) {
    sample.hidden = true;
    return;
  }
  const seq = ++failSeq;
  failLoading = true;
  setFailNav();
  const testIndex = item.index;
  fetch(
    `/api/preview/${encodeURIComponent(generatedDataset)}?index=${encodeURIComponent(testIndex)}&split=test`,
  )
    .then((r) => {
      if (!r.ok) throw new Error();
      return r.json();
    })
    .then((payload) => {
      if (seq !== failSeq) return;
      failLoading = false;
      renderMedia(payload, document.getElementById("fail-media"));
      document.getElementById("fail-label").textContent = `label = ${item.label}`;
      document.getElementById("fail-pred").textContent = `pred = ${item.pred}`;
      document.getElementById("fail-test-index").textContent = `test_index = ${testIndex}`;
      document.getElementById("fail-index").value = String(failIndex);
      document.getElementById("fail-size").textContent = `size = ${failures.length}`;
      setFailNav();
    })
    .catch(() => {
      if (seq !== failSeq) return;
      failLoading = false;
      document.getElementById("fail-media").replaceChildren();
      document.getElementById("fail-label").textContent = "Failed to load sample.";
      document.getElementById("fail-pred").textContent = "";
      document.getElementById("fail-test-index").textContent = "";
      document.getElementById("fail-size").textContent = "";
      setFailNav();
    });
}
