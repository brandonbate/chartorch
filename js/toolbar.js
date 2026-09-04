import { MODULE_ORDER } from "./modules.js";
import { setMode, startPlacing } from "./model.js";
import { closeEditor, render } from "./render.js";

export function initToolbar() {
  const host = document.getElementById("module-buttons");
  for (const type of MODULE_ORDER) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "module-btn";
    btn.dataset.type = type;
    btn.textContent = type;
    btn.addEventListener("click", () => {
      closeEditor(true);
      startPlacing(type);
      render();
    });
    host.appendChild(btn);
  }

  document.getElementById("btn-move").addEventListener("click", () => {
    closeEditor(true);
    setMode("move");
    render();
  });

  document.getElementById("btn-connect").addEventListener("click", () => {
    closeEditor(true);
    setMode("connect");
    render();
  });
}
