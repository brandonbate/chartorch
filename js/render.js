import { state } from "./model.js";
import { MODULES, cardHTML, defaultArgs, formatValue, getField, parseValue } from "./modules.js";
import {
  boundsOf,
  edgePoints,
  nodePortRight,
  pointsAttr,
  routeOrthogonal,
} from "./edges.js";

const body = document.body;
const workspace = document.getElementById("workspace");
const nodesEl = document.getElementById("nodes");
const edgesSvg = document.getElementById("edges");
const previewSvg = document.getElementById("preview");
const ghostEl = document.getElementById("ghost");
const btnMove = document.getElementById("btn-move");
const btnConnect = document.getElementById("btn-connect");

export function render() {
  renderToolbar();
  renderNodes();
  measureNodes();
  sizeWorkspace();
  renderEdges();
  renderPreview();
  renderGhost();
}

export function renderToolbar() {
  body.dataset.mode = state.mode;
  if (state.placingType) body.dataset.placing = "true";
  else delete body.dataset.placing;
  btnMove.classList.toggle("active", state.mode === "move");
  btnConnect.classList.toggle("active", state.mode === "connect");
  for (const btn of document.querySelectorAll(".module-btn")) {
    btn.classList.toggle("active", btn.dataset.type === state.placingType);
  }
}

export function renderNodes() {
  const keep = new Set();
  for (const node of Object.values(state.nodes)) {
    keep.add(node.id);
    let el = document.getElementById(`node-${node.id}`);
    if (!el) {
      el = document.createElement("div");
      el.className = "node";
      el.id = `node-${node.id}`;
      el.dataset.id = node.id;
      el.innerHTML = cardHTML(node.type, node.args);
      nodesEl.appendChild(el);
    }
    el.style.left = `${node.x}px`;
    el.style.top = `${node.y}px`;
    el.classList.toggle(
      "selected",
      !!(state.selected && state.selected.kind === "node" && state.selected.id === node.id)
    );
    if (state.editing && state.editing.nodeId === node.id) continue;
    for (const field of MODULES[node.type].args) {
      const valueEl = el.querySelector(`.arg-value[data-name="${field.name}"]`);
      if (!valueEl || valueEl.querySelector("input, select")) continue;
      valueEl.textContent = formatValue(field, node.args[field.name]);
    }
  }
  for (const el of [...nodesEl.children]) {
    if (!keep.has(el.dataset.id)) el.remove();
  }
}

export function measureNodes() {
  for (const node of Object.values(state.nodes)) {
    const el = document.getElementById(`node-${node.id}`);
    if (!el) continue;
    node.w = el.offsetWidth;
    node.h = el.offsetHeight;
  }
}

export function sizeWorkspace() {
  const canvas = document.getElementById("canvas");
  let w = canvas.clientWidth;
  let h = canvas.clientHeight;
  for (const node of Object.values(state.nodes)) {
    w = Math.max(w, node.x + node.w + 80);
    h = Math.max(h, node.y + node.h + 80);
  }
  workspace.style.width = `${w}px`;
  workspace.style.height = `${h}px`;
}

export function renderEdges() {
  const defs = edgesSvg.querySelector("defs");
  const keep = new Set();
  for (const edge of Object.values(state.edges)) {
    keep.add(edge.id);
    const pts = edgePoints(edge);
    const selected =
      state.selected && state.selected.kind === "edge" && state.selected.id === edge.id;
    let g = edgesSvg.querySelector(`[data-id="${edge.id}"]`);
    if (!g) {
      g = document.createElementNS("http://www.w3.org/2000/svg", "g");
      g.classList.add("edge");
      g.dataset.id = edge.id;
      const hit = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
      hit.classList.add("edge-hit");
      const line = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
      line.classList.add("edge-line");
      const marquee = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      marquee.classList.add("edge-marquee");
      marquee.setAttribute("hidden", "");
      g.append(hit, line, marquee);
      edgesSvg.appendChild(g);
    }
    const attr = pointsAttr(pts);
    g.querySelector(".edge-hit").setAttribute("points", attr);
    const line = g.querySelector(".edge-line");
    line.setAttribute("points", attr);
    line.setAttribute("marker-end", selected ? "url(#arrow-selected)" : "url(#arrow)");
    g.classList.toggle("selected", selected);
    const marquee = g.querySelector(".edge-marquee");
    if (selected) {
      const box = boundsOf(pts, 8);
      marquee.removeAttribute("hidden");
      marquee.setAttribute("x", box.x);
      marquee.setAttribute("y", box.y);
      marquee.setAttribute("width", box.w);
      marquee.setAttribute("height", box.h);
    } else {
      marquee.setAttribute("hidden", "");
    }
  }
  for (const g of [...edgesSvg.querySelectorAll("g.edge")]) {
    if (!keep.has(g.dataset.id)) g.remove();
  }
  if (defs) edgesSvg.appendChild(defs);
}

let lastPreviewCursor = null;

export function renderPreview(cursor) {
  if (cursor) lastPreviewCursor = cursor;
  if (!state.connectingFrom) {
    lastPreviewCursor = null;
    previewSvg.replaceChildren();
    return;
  }
  previewSvg.replaceChildren();
  const from = state.nodes[state.connectingFrom];
  if (!from) return;
  const start = nodePortRight(from);
  const c = cursor || lastPreviewCursor;
  const tx = c ? c.x : start.x + 40;
  const ty = c ? c.y : start.y;
  const pts = routeOrthogonal(start.x, start.y, tx, ty);
  const line = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
  line.id = "preview-line";
  line.setAttribute("points", pointsAttr(pts));
  line.setAttribute("marker-end", "url(#arrow-selected)");
  previewSvg.appendChild(line);
}

export function renderGhost(pos) {
  if (!state.placingType) {
    ghostEl.hidden = true;
    ghostEl.dataset.type = "";
    return;
  }
  if (ghostEl.dataset.type !== state.placingType) {
    ghostEl.dataset.type = state.placingType;
    ghostEl.innerHTML = cardHTML(state.placingType, defaultArgs(state.placingType));
  }
  if (pos) {
    ghostEl.hidden = false;
    ghostEl.style.left = `${pos.x}px`;
    ghostEl.style.top = `${pos.y}px`;
  }
}

export function updateNodePosition(id) {
  const node = state.nodes[id];
  const el = document.getElementById(`node-${id}`);
  if (el && node) {
    el.style.left = `${node.x}px`;
    el.style.top = `${node.y}px`;
  }
  sizeWorkspace();
  renderEdges();
  renderPreview();
}

export function openEditor(nodeId, argName) {
  closeEditor(true);
  const node = state.nodes[nodeId];
  if (!node) return;
  const field = getField(node.type, argName);
  if (!field) return;
  const el = document.getElementById(`node-${nodeId}`);
  const valueEl = el.querySelector(`.arg-value[data-name="${argName}"]`);
  if (!valueEl) return;
  state.editing = { nodeId, argName };
  const current = formatValue(field, node.args[argName]);
  let control;
  if (field.type === "bool") {
    control = document.createElement("select");
    for (const opt of ["True", "False"]) {
      const option = document.createElement("option");
      option.value = opt;
      option.textContent = opt;
      if (opt === current) option.selected = true;
      control.appendChild(option);
    }
  } else if (field.type === "enum") {
    control = document.createElement("select");
    for (const opt of field.options) {
      const option = document.createElement("option");
      option.value = opt;
      option.textContent = opt;
      if (opt === current) option.selected = true;
      control.appendChild(option);
    }
  } else {
    control = document.createElement("input");
    control.type = "text";
    control.value = current;
  }
  valueEl.textContent = "";
  valueEl.appendChild(control);
  control.focus();
  if (control.select) control.select();

  const commit = () => closeEditor(true);
  const cancel = () => closeEditor(false);
  control.addEventListener("blur", commit);
  if (control.tagName === "SELECT") control.addEventListener("change", commit);
  control.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      commit();
    } else if (e.key === "Escape") {
      e.preventDefault();
      e.stopPropagation();
      cancel();
    }
  });
  control.addEventListener("mousedown", (e) => e.stopPropagation());
  control.addEventListener("click", (e) => e.stopPropagation());
  control.addEventListener("dblclick", (e) => e.stopPropagation());
}

export function closeEditor(commit) {
  const editing = state.editing;
  if (!editing) return;
  const node = state.nodes[editing.nodeId];
  const el = document.getElementById(`node-${editing.nodeId}`);
  state.editing = null;
  if (!node || !el) return;
  const field = getField(node.type, editing.argName);
  const valueEl = el.querySelector(`.arg-value[data-name="${editing.argName}"]`);
  if (!valueEl) return;
  const control = valueEl.querySelector("input, select");
  if (commit && control && field) {
    const parsed = parseValue(field, control.value);
    if (parsed !== undefined) node.args[editing.argName] = parsed;
  }
  valueEl.replaceChildren();
  valueEl.textContent = formatValue(field, node.args[editing.argName]);
}

export function workspacePoint(event) {
  const rect = workspace.getBoundingClientRect();
  return {
    x: event.clientX - rect.left,
    y: event.clientY - rect.top,
  };
}

export { workspace, ghostEl };
