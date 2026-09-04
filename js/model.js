import { defaultArgs, MODULES } from "./modules.js";

let nextNodeId = 1;
let nextEdgeId = 1;

export const state = {
  mode: "move",
  placingType: null,
  selected: null,
  connectingFrom: null,
  editing: null,
  nodes: {},
  edges: {},
};

export function addNode(type, x, y) {
  const id = `n${nextNodeId++}`;
  const argCount = MODULES[type].args.length;
  state.nodes[id] = {
    id,
    type,
    args: defaultArgs(type),
    x,
    y,
    w: 220,
    h: 36 + argCount * 22 + 12,
  };
  state.selected = { kind: "node", id };
  return id;
}

export function removeNode(id) {
  for (const edge of Object.values(state.edges)) {
    if (edge.from === id || edge.to === id) {
      delete state.edges[edge.id];
    }
  }
  delete state.nodes[id];
  if (state.selected && state.selected.kind === "node" && state.selected.id === id) {
    state.selected = null;
  }
  if (state.connectingFrom === id) state.connectingFrom = null;
  if (state.editing && state.editing.nodeId === id) state.editing = null;
}

export function addEdge(from, to) {
  if (!canConnect(from, to)) return null;
  const id = `e${nextEdgeId++}`;
  state.edges[id] = { id, from, to };
  state.selected = { kind: "edge", id };
  return id;
}

export function removeEdge(id) {
  delete state.edges[id];
  if (state.selected && state.selected.kind === "edge" && state.selected.id === id) {
    state.selected = null;
  }
}

export function outgoing(nodeId) {
  return Object.values(state.edges).find((edge) => edge.from === nodeId) || null;
}

export function incoming(nodeId) {
  return Object.values(state.edges).find((edge) => edge.to === nodeId) || null;
}

export function reaches(fromId, toId) {
  let current = fromId;
  const seen = new Set();
  while (current && !seen.has(current)) {
    if (current === toId) return true;
    seen.add(current);
    const edge = outgoing(current);
    current = edge ? edge.to : null;
  }
  return false;
}

export function canConnect(from, to) {
  if (!from || !to || from === to) return false;
  if (!state.nodes[from] || !state.nodes[to]) return false;
  if (outgoing(from) || incoming(to)) return false;
  if (reaches(to, from)) return false;
  return true;
}

export function setMode(mode) {
  state.mode = mode;
  state.placingType = null;
  state.connectingFrom = null;
  state.editing = null;
}

export function startPlacing(type) {
  state.mode = "move";
  state.placingType = type;
  state.connectingFrom = null;
  state.editing = null;
}

export function clearPlacing() {
  state.placingType = null;
}

export function selectedNode() {
  if (state.selected && state.selected.kind === "node") {
    return state.nodes[state.selected.id] || null;
  }
  return null;
}

export function selectedEdge() {
  if (state.selected && state.selected.kind === "edge") {
    return state.edges[state.selected.id] || null;
  }
  return null;
}
