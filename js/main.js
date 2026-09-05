import { initToolbar } from "./toolbar.js";
import { initCanvas } from "./canvas.js";
import { initGenerate } from "./generate.js";
import { initSettings } from "./training.js";
import { initData } from "./data.js";
import { initPanelResize } from "./resize.js";
import { render } from "./render.js";

initToolbar();
initCanvas();
initSettings();
initData();
initGenerate();
initPanelResize();
render();
