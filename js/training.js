import { defaultsFrom, formatValue, parseValue } from "./modules.js";

const REDUCTION = ["none", "mean", "sum"];

export const OPTIMIZERS = {
  SGD: {
    className: "SGD",
    args: [
      { name: "lr", type: "float", default: 0.001 },
      { name: "momentum", type: "float", default: 0 },
      { name: "dampening", type: "float", default: 0 },
      { name: "weight_decay", type: "float", default: 0 },
      { name: "nesterov", type: "bool", default: false },
      { name: "maximize", type: "bool", default: false },
      { name: "foreach", type: "boolOrNone", default: null },
      { name: "differentiable", type: "bool", default: false },
      { name: "fused", type: "boolOrNone", default: null },
    ],
  },
  Adam: {
    className: "Adam",
    args: [
      { name: "lr", type: "float", default: 0.001 },
      { name: "betas", type: "floatTuple", default: "(0.9, 0.999)" },
      { name: "eps", type: "float", default: 1e-8 },
      { name: "weight_decay", type: "float", default: 0 },
      { name: "amsgrad", type: "bool", default: false },
      { name: "foreach", type: "boolOrNone", default: null },
      { name: "maximize", type: "bool", default: false },
      { name: "capturable", type: "bool", default: false },
      { name: "differentiable", type: "bool", default: false },
      { name: "fused", type: "boolOrNone", default: null },
    ],
  },
  AdamW: {
    className: "AdamW",
    args: [
      { name: "lr", type: "float", default: 0.001 },
      { name: "betas", type: "floatTuple", default: "(0.9, 0.999)" },
      { name: "eps", type: "float", default: 1e-8 },
      { name: "weight_decay", type: "float", default: 0.01 },
      { name: "amsgrad", type: "bool", default: false },
      { name: "maximize", type: "bool", default: false },
      { name: "foreach", type: "boolOrNone", default: null },
      { name: "capturable", type: "bool", default: false },
      { name: "differentiable", type: "bool", default: false },
      { name: "fused", type: "boolOrNone", default: null },
    ],
  },
};

export const LOSSES = {
  MSE: {
    className: "MSELoss",
    args: [{ name: "reduction", type: "enum", default: "mean", options: REDUCTION }],
  },
  L1: {
    className: "L1Loss",
    args: [{ name: "reduction", type: "enum", default: "mean", options: REDUCTION }],
  },
  Huber: {
    className: "HuberLoss",
    args: [
      { name: "reduction", type: "enum", default: "mean", options: REDUCTION },
      { name: "delta", type: "float", default: 1.0 },
    ],
  },
  CrossEntropy: {
    className: "CrossEntropyLoss",
    args: [
      { name: "weight", type: "noneOrExpr", default: null },
      { name: "ignore_index", type: "int", default: -100 },
      { name: "reduction", type: "enum", default: "mean", options: REDUCTION },
      { name: "label_smoothing", type: "float", default: 0.0 },
    ],
  },
  BCE: {
    className: "BCELoss",
    args: [
      { name: "weight", type: "noneOrExpr", default: null },
      { name: "reduction", type: "enum", default: "mean", options: REDUCTION },
    ],
  },
  BCEWithLogits: {
    className: "BCEWithLogitsLoss",
    args: [
      { name: "weight", type: "noneOrExpr", default: null },
      { name: "reduction", type: "enum", default: "mean", options: REDUCTION },
      { name: "pos_weight", type: "noneOrExpr", default: null },
    ],
  },
};

export const OPTIMIZER_ORDER = ["SGD", "Adam", "AdamW"];
export const LOSS_ORDER = ["MSE", "L1", "Huber", "CrossEntropy", "BCE", "BCEWithLogits"];

export const training = {
  optimizer: "SGD",
  optimizerArgs: defaultsFrom(OPTIMIZERS.SGD.args),
  loss: "MSE",
  lossArgs: defaultsFrom(LOSSES.MSE.args),
  epochs: 10,
};

export function initSettings() {
  const optSel = document.getElementById("optimizer-type");
  const lossSel = document.getElementById("loss-type");
  const optArgs = document.getElementById("optimizer-args");
  const lossArgs = document.getElementById("loss-args");
  const epochsInput = document.getElementById("epochs-input");

  fillSelect(optSel, OPTIMIZER_ORDER, training.optimizer);
  fillSelect(lossSel, LOSS_ORDER, training.loss);
  epochsInput.value = String(training.epochs);

  renderArgFields(optArgs, OPTIMIZERS[training.optimizer].args, training.optimizerArgs);
  renderArgFields(lossArgs, LOSSES[training.loss].args, training.lossArgs);

  optSel.addEventListener("change", () => {
    training.optimizer = optSel.value;
    training.optimizerArgs = defaultsFrom(OPTIMIZERS[training.optimizer].args);
    renderArgFields(optArgs, OPTIMIZERS[training.optimizer].args, training.optimizerArgs);
  });

  lossSel.addEventListener("change", () => {
    training.loss = lossSel.value;
    training.lossArgs = defaultsFrom(LOSSES[training.loss].args);
    renderArgFields(lossArgs, LOSSES[training.loss].args, training.lossArgs);
  });

  const commitEpochs = () => {
    const parsed = parseValue({ type: "int" }, epochsInput.value);
    if (parsed !== undefined && parsed >= 1) {
      training.epochs = parsed;
    }
    epochsInput.value = String(training.epochs);
  };
  epochsInput.addEventListener("change", commitEpochs);
  epochsInput.addEventListener("blur", commitEpochs);
}

function fillSelect(sel, names, current) {
  for (const name of names) {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    if (name === current) opt.selected = true;
    sel.appendChild(opt);
  }
}

function renderArgFields(host, fields, args) {
  host.replaceChildren();
  for (const field of fields) {
    const row = document.createElement("div");
    row.className = "setting-arg";
    const label = document.createElement("span");
    label.className = "arg-name";
    label.textContent = field.name;
    row.append(label, document.createTextNode(" = "), makeControl(field, args));
    host.appendChild(row);
  }
}

function makeControl(field, args) {
  if (field.type === "bool" || field.type === "enum" || field.type === "boolOrNone") {
    const sel = document.createElement("select");
    const options =
      field.type === "bool"
        ? ["True", "False"]
        : field.type === "boolOrNone"
          ? ["None", "True", "False"]
          : field.options;
    const current = formatValue(field, args[field.name]);
    for (const name of options) {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      if (name === current) opt.selected = true;
      sel.appendChild(opt);
    }
    sel.addEventListener("change", () => {
      const parsed = parseValue(field, sel.value);
      if (parsed !== undefined) args[field.name] = parsed;
    });
    return sel;
  }
  const input = document.createElement("input");
  input.type = "text";
  input.value = formatValue(field, args[field.name]);
  const commit = () => {
    const parsed = parseValue(field, input.value);
    if (parsed !== undefined) args[field.name] = parsed;
    input.value = formatValue(field, args[field.name]);
  };
  input.addEventListener("change", commit);
  input.addEventListener("blur", commit);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      input.blur();
    }
  });
  return input;
}
