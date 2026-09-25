import {
  addEdge,
  addNode,
  canConnect,
  clearPlacing,
  removeEdge,
  removeNode,
  selectedEdge,
  selectedNode,
  state,
} from "./model.js";
import {
  closeEditor,
  ghostEl,
  openEditor,
  render,
  renderGhost,
  renderPreview,
  updateNodePosition,
  workspace,
  workspacePoint,
} from "./render.js";

const DRAG_THRESHOLD = 5;

let drag = null;
let suppressClick = false;
let cursor = null;

export function initCanvas() {
  const canvas = document.getElementById("canvas");

  canvas.addEventListener("mousemove", (e) => {
    cursor = workspacePoint(e);
    if (state.placingType) renderGhost(cursor);
    if (state.connectingFrom) renderPreview(cursor);
  });

  canvas.addEventListener("mouseleave", () => {
    if (state.placingType) {
      ghostEl.hidden = true;
    }
  });

  canvas.addEventListener("mouseenter", (e) => {
    cursor = workspacePoint(e);
    if (state.placingType) renderGhost(cursor);
  });

  workspace.addEventListener("mousedown", onPointerDown);
  window.addEventListener("mousemove", onWindowMove);
  window.addEventListener("mouseup", onWindowUp);
  workspace.addEventListener("click", onClick);
  workspace.addEventListener("dblclick", onDblClick);
  window.addEventListener("keydown", onKeyDown);
}

function onPointerDown(e) {
  if (e.button !== 0) return;
  if (e.target.closest("input, select")) return;
  const nodeEl = e.target.closest(".node");
  if (!nodeEl || state.mode !== "move" || state.placingType) return;
  const id = nodeEl.dataset.id;
  const node = state.nodes[id];
  if (!node) return;
  const pt = workspacePoint(e);
  drag = {
    id,
    offsetX: pt.x - node.x,
    offsetY: pt.y - node.y,
    originX: pt.x,
    originY: pt.y,
    moved: false,
  };
  e.preventDefault();
}

function onWindowMove(e) {
  if (!drag) return;
  const pt = workspacePoint(e);
  const dx = pt.x - drag.originX;
  const dy = pt.y - drag.originY;
  if (!drag.moved && Math.hypot(dx, dy) < DRAG_THRESHOLD) return;
  drag.moved = true;
  closeEditor(true);
  const node = state.nodes[drag.id];
  node.x = Math.max(0, pt.x - drag.offsetX);
  node.y = Math.max(0, pt.y - drag.offsetY);
  const el = document.getElementById(`node-${drag.id}`);
  if (el) el.classList.add("dragging");
  updateNodePosition(drag.id);
}

function onWindowUp() {
  if (!drag) return;
  const el = document.getElementById(`node-${drag.id}`);
  if (el) el.classList.remove("dragging");
  if (!drag.moved) {
    closeEditor(true);
    state.selected = { kind: "node", id: drag.id };
    render();
  } else {
    suppressClick = true;
    sizeAfterDrag();
  }
  drag = null;
}

function sizeAfterDrag() {
  render();
}

function onClick(e) {
  if (suppressClick) {
    suppressClick = false;
    return;
  }
  if (e.target.closest("input, select")) return;

  const nodeEl = e.target.closest(".node");
  const edgeEl = e.target.closest(".edge");
  const pt = workspacePoint(e);

  if (state.placingType) {
    if (!nodeEl && !edgeEl) {
      closeEditor(true);
      addNode(state.placingType, pt.x, pt.y);
      clearPlacing();
      render();
    }
    return;
  }

  if (state.mode === "connect") {
    if (nodeEl) {
      if (e.target.closest(".arg-value")) return;
      const id = nodeEl.dataset.id;
      if (!state.connectingFrom) {
        closeEditor(true);
        state.connectingFrom = id;
        state.selected = { kind: "node", id };
        render();
        renderPreview(pt);
        return;
      }
      if (canConnect(state.connectingFrom, id)) {
        closeEditor(true);
        addEdge(state.connectingFrom, id);
        state.connectingFrom = null;
        render();
      }
      return;
    }
    closeEditor(true);
    if (edgeEl) {
      state.connectingFrom = null;
      state.selected = { kind: "edge", id: edgeEl.dataset.id };
      render();
      return;
    }
    state.connectingFrom = null;
    state.selected = null;
    render();
    return;
  }

  if (nodeEl) return;

  closeEditor(true);
  if (edgeEl) {
    state.selected = { kind: "edge", id: edgeEl.dataset.id };
  } else {
    state.selected = null;
  }
  render();
}

function onDblClick(e) {
  const valueEl = e.target.closest(".arg-value");
  if (!valueEl || valueEl.querySelector("input, select")) return;
  const nodeEl = valueEl.closest(".node");
  if (!nodeEl) return;
  e.preventDefault();
  e.stopPropagation();
  state.connectingFrom = null;
  openEditor(nodeEl.dataset.id, valueEl.dataset.name);
}

function onKeyDown(e) {
  const typing = e.target.closest("input, select, textarea");
  if (e.key === "Escape") {
    if (state.editing) {
      closeEditor(false);
      return;
    }
    if (state.connectingFrom || state.placingType) {
      state.connectingFrom = null;
      clearPlacing();
      render();
    }
    return;
  }
  if (typing) return;
  if (e.key !== "Delete" && e.key !== "Backspace") return;
  const node = selectedNode();
  const edge = selectedEdge();
  if (node) {
    e.preventDefault();
    removeNode(node.id);
    render();
  } else if (edge) {
    e.preventDefault();
    removeEdge(edge.id);
    render();
  }
}
