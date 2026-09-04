export const MODULE_ORDER = [
  "Linear",
  "Conv2d",
  "MaxPool2d",
  "Flatten",
  "ReLU",
  "Sigmoid",
  "Tanh",
  "Dropout",
  "BatchNorm2d",
  "LayerNorm",
];

export const MODULES = {
  Linear: {
    args: [
      { name: "in_features", type: "int", default: 64 },
      { name: "out_features", type: "int", default: 64 },
      { name: "bias", type: "bool", default: true },
    ],
  },
  Conv2d: {
    args: [
      { name: "in_channels", type: "int", default: 3 },
      { name: "out_channels", type: "int", default: 16 },
      { name: "kernel_size", type: "dim", default: 3 },
      { name: "stride", type: "dim", default: 1 },
      { name: "padding", type: "dim", default: 0 },
      { name: "dilation", type: "dim", default: 1 },
      { name: "groups", type: "int", default: 1 },
      { name: "bias", type: "bool", default: true },
      {
        name: "padding_mode",
        type: "enum",
        default: "zeros",
        options: ["zeros", "reflect", "replicate", "circular"],
      },
    ],
  },
  MaxPool2d: {
    args: [
      { name: "kernel_size", type: "dim", default: 2 },
      { name: "stride", type: "intOrNone", default: null },
      { name: "padding", type: "dim", default: 0 },
      { name: "dilation", type: "dim", default: 1 },
      { name: "return_indices", type: "bool", default: false },
      { name: "ceil_mode", type: "bool", default: false },
    ],
  },
  Flatten: {
    args: [
      { name: "start_dim", type: "int", default: 1 },
      { name: "end_dim", type: "int", default: -1 },
    ],
  },
  ReLU: {
    args: [{ name: "inplace", type: "bool", default: false }],
  },
  Sigmoid: {
    args: [],
  },
  Tanh: {
    args: [],
  },
  Dropout: {
    args: [
      { name: "p", type: "float", default: 0.5 },
      { name: "inplace", type: "bool", default: false },
    ],
  },
  BatchNorm2d: {
    args: [
      { name: "num_features", type: "int", default: 16 },
      { name: "eps", type: "float", default: 1e-5 },
      { name: "momentum", type: "float", default: 0.1 },
      { name: "affine", type: "bool", default: true },
      { name: "track_running_stats", type: "bool", default: true },
    ],
  },
  LayerNorm: {
    args: [
      { name: "normalized_shape", type: "dim", default: 64 },
      { name: "eps", type: "float", default: 1e-5 },
      { name: "elementwise_affine", type: "bool", default: true },
      { name: "bias", type: "bool", default: true },
    ],
  },
};

export function defaultArgs(type) {
  const args = {};
  for (const field of MODULES[type].args) {
    args[field.name] = field.default;
  }
  return args;
}

export function getField(type, name) {
  return MODULES[type].args.find((field) => field.name === name);
}

export function formatValue(field, value) {
  if (value === true) return "True";
  if (value === false) return "False";
  if (value === null) return "None";
  return String(value);
}

export function parseValue(field, text) {
  const t = text.trim();
  switch (field.type) {
    case "bool":
      if (t === "True" || t === "true") return true;
      if (t === "False" || t === "false") return false;
      return undefined;
    case "enum":
      return field.options.includes(t) ? t : undefined;
    case "int": {
      if (!/^-?\d+$/.test(t)) return undefined;
      return parseInt(t, 10);
    }
    case "float": {
      const n = Number(t);
      return Number.isFinite(n) ? n : undefined;
    }
    case "intOrNone": {
      if (t === "None" || t === "none") return null;
      if (!/^-?\d+$/.test(t)) return undefined;
      return parseInt(t, 10);
    }
    case "dim": {
      if (/^-?\d+$/.test(t)) return parseInt(t, 10);
      if (/^\(\s*-?\d+\s*(,\s*-?\d+\s*)*,?\s*\)$/.test(t)) return t;
      return undefined;
    }
    default:
      return undefined;
  }
}

export function cardHTML(type, args) {
  const rows = MODULES[type].args
    .map((field) => {
      const value = formatValue(field, args[field.name]);
      return `<div class="arg"><span class="arg-name">${field.name}</span> = <span class="arg-value" data-name="${field.name}">${value}</span></div>`;
    })
    .join("");
  return `<div class="node-title">${type}</div><div class="node-args">${rows}</div>`;
}
