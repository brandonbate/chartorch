import { state } from "./model.js";

export function nodePortRight(node) {
  return { x: node.x + node.w, y: node.y + node.h / 2 };
}

export function nodePortLeft(node) {
  return { x: node.x, y: node.y + node.h / 2 };
}

export function routeOrthogonal(sx, sy, tx, ty) {
  const stub = 24;
  if (Math.abs(sy - ty) < 0.5 && tx >= sx) {
    return [
      [sx, sy],
      [tx, ty],
    ];
  }
  if (tx - sx >= 40) {
    const midX = (sx + tx) / 2;
    return [
      [sx, sy],
      [midX, sy],
      [midX, ty],
      [tx, ty],
    ];
  }
  const outX = sx + stub;
  const inX = tx - stub;
  const midY = (sy + ty) / 2;
  return [
    [sx, sy],
    [outX, sy],
    [outX, midY],
    [inX, midY],
    [inX, ty],
    [tx, ty],
  ];
}

export function pointsAttr(pts) {
  return pts.map(([x, y]) => `${x},${y}`).join(" ");
}

export function boundsOf(pts, pad = 6) {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const [x, y] of pts) {
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
  }
  return {
    x: minX - pad,
    y: minY - pad,
    w: Math.max(maxX - minX, 1) + pad * 2,
    h: Math.max(maxY - minY, 1) + pad * 2,
  };
}

export function edgePoints(edge) {
  const from = state.nodes[edge.from];
  const to = state.nodes[edge.to];
  const start = nodePortRight(from);
  const end = nodePortLeft(to);
  return routeOrthogonal(start.x, start.y, end.x, end.y);
}
