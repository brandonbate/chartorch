export const dataState = {
  dataset: "MNIST",
  index: 0,
  size: null,
};

let requestSeq = 0;
let loading = false;

export function initData() {
  const sel = document.getElementById("dataset-type");
  const dataPanel = document.getElementById("data-panel");
  const trainingPanel = document.getElementById("training-panel");
  const tabData = document.getElementById("tab-data");
  const tabTraining = document.getElementById("tab-training");
  const btnPrev = document.getElementById("dataset-prev");
  const btnNext = document.getElementById("dataset-next");
  const indexInput = document.getElementById("dataset-index");

  const setPanelMode = (mode) => {
    document.body.dataset.panelMode = mode;
    tabData.classList.toggle("active", mode === "data");
    tabTraining.classList.toggle("active", mode === "training");
    dataPanel.hidden = mode !== "data";
    trainingPanel.hidden = mode !== "training";
  };

  tabData.addEventListener("click", () => setPanelMode("data"));
  tabTraining.addEventListener("click", () => setPanelMode("training"));

  sel.addEventListener("change", () => {
    dataState.dataset = sel.value;
    dataState.index = 0;
    dataState.size = null;
    indexInput.value = "0";
    loadPreview(dataState.dataset, 0, true);
  });

  btnPrev.addEventListener("click", () => {
    if (loading || dataState.size == null || dataState.index <= 0) return;
    dataState.index -= 1;
    loadPreview(dataState.dataset, dataState.index, false);
  });

  btnNext.addEventListener("click", () => {
    if (loading || dataState.size == null || dataState.index >= dataState.size - 1) return;
    dataState.index += 1;
    loadPreview(dataState.dataset, dataState.index, false);
  });

  const commitIndex = () => {
    const parsed = Number.parseInt(indexInput.value, 10);
    if (Number.isInteger(parsed) && dataState.size != null) {
      const next = Math.min(dataState.size - 1, Math.max(0, parsed));
      if (next !== dataState.index) {
        dataState.index = next;
        loadPreview(dataState.dataset, next, false);
      }
    }
    indexInput.value = String(dataState.index);
  };
  indexInput.addEventListener("change", commitIndex);
  indexInput.addEventListener("blur", commitIndex);
  indexInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      indexInput.blur();
    }
  });

  setNavEnabled();

  fetch("/api/datasets")
    .then((r) => {
      if (!r.ok) throw new Error();
      return r.json();
    })
    .then((payload) => {
      const ids = payload.datasets || [];
      if (ids.length && !ids.includes(dataState.dataset)) {
        dataState.dataset = ids[0];
      }
      fillSelect(sel, ids.length ? ids : ["MNIST"], dataState.dataset);
      loadPreview(dataState.dataset, dataState.index, true);
    })
    .catch(() => {
      fillSelect(sel, ["MNIST"], dataState.dataset);
      loadPreview(dataState.dataset, dataState.index, true);
    });
}

function fillSelect(sel, names, current) {
  sel.replaceChildren();
  for (const name of names) {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    if (name === current) opt.selected = true;
    sel.appendChild(opt);
  }
}

function setNavEnabled() {
  const prev = document.getElementById("dataset-prev");
  const next = document.getElementById("dataset-next");
  const indexInput = document.getElementById("dataset-index");
  const size = dataState.size;
  const busy = loading || size == null || size <= 0;
  prev.disabled = busy || dataState.index <= 0;
  next.disabled = busy || dataState.index >= size - 1;
  indexInput.disabled = busy;
}

function loadPreview(id, index, clear) {
  const seq = ++requestSeq;
  loading = true;
  setNavEnabled();
  if (clear) showSampleStatus("Loading sample…");
  fetch(`/api/preview/${encodeURIComponent(id)}?index=${encodeURIComponent(index)}`)
    .then((r) => {
      if (!r.ok) throw new Error();
      return r.json();
    })
    .then((payload) => {
      if (seq !== requestSeq) return;
      loading = false;
      if (typeof payload.index === "number") dataState.index = payload.index;
      if (typeof payload.size === "number") dataState.size = payload.size;
      renderPreview(payload);
      setNavEnabled();
    })
    .catch(() => {
      if (seq !== requestSeq) return;
      loading = false;
      setNavEnabled();
      if (clear) showSampleStatus("Failed to load sample.");
    });
}

function showSampleStatus(text) {
  document.getElementById("dataset-media").replaceChildren();
  document.getElementById("dataset-label").textContent = text;
  document.getElementById("dataset-size").textContent = "";
  document.getElementById("dataset-input-shape").textContent = "";
  document.getElementById("dataset-label-shape").textContent = "";
}

function renderPreview(payload) {
  renderMedia(payload, document.getElementById("dataset-media"));
  document.getElementById("dataset-label").textContent = `label = ${payload.label}`;
  document.getElementById("dataset-index").value = String(dataState.index);
  document.getElementById("dataset-size").textContent = `size = ${payload.size}`;
  document.getElementById("dataset-input-shape").textContent =
    `input_shape = ${formatShape(displayInputShape(payload.input_shape))}`;
  document.getElementById("dataset-label-shape").textContent =
    `label_shape = ${formatShape(payload.label_shape)}`;
}

export function renderMedia(payload, el) {
  el.replaceChildren();
  if (payload.kind === "image") {
    const img = document.createElement("img");
    img.alt = String(payload.label ?? "");
    img.src = `data:${payload.mime};base64,${payload.data}`;
    el.appendChild(img);
    return;
  }
  if (payload.kind === "audio") {
    const audio = document.createElement("audio");
    audio.controls = true;
    audio.src = `data:${payload.mime};base64,${payload.data}`;
    el.appendChild(audio);
  }
}

export function displayInputShape(shape) {
  if (!Array.isArray(shape)) return [];
  if (shape.length && shape[0] === 1) return shape.slice(1);
  return shape;
}

export function formatShape(shape) {
  if (!Array.isArray(shape)) return "[]";
  return `[${shape.join(", ")}]`;
}
