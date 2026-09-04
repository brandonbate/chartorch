import { initToolbar } from "./toolbar.js";
import { initCanvas } from "./canvas.js";
import { initGenerate } from "./generate.js";
import { initSettings } from "./training.js";
import { initPanelResize } from "./resize.js";
import { render } from "./render.js";

initToolbar();
initCanvas();
initSettings();
initGenerate();
initPanelResize();
render();
