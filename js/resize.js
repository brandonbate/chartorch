export function initPanelResize() {
  const stage = document.getElementById("stage");
  bindResize(document.getElementById("settings-panel"), document.getElementById("settings-resize"), {
    edge: "right",
    min: 220,
    maxFraction: 0.5,
    stage,
  });
  bindResize(document.getElementById("generate-panel"), document.getElementById("generate-resize"), {
    edge: "left",
    min: 220,
    maxFraction: 0.5,
    stage,
  });
}

function bindResize(panel, handle, { edge, min, maxFraction, stage }) {
  handle.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    e.preventDefault();
    handle.setPointerCapture(e.pointerId);
    const startX = e.clientX;
    const startW = panel.getBoundingClientRect().width;

    const onMove = (ev) => {
      const dx = ev.clientX - startX;
      const max = stage.getBoundingClientRect().width * maxFraction;
      let w = edge === "right" ? startW + dx : startW - dx;
      w = Math.min(max, Math.max(min, w));
      panel.style.width = `${w}px`;
    };

    const onUp = () => {
      handle.removeEventListener("pointermove", onMove);
      handle.removeEventListener("pointerup", onUp);
    };

    handle.addEventListener("pointermove", onMove);
    handle.addEventListener("pointerup", onUp);
  });
}
