import { plot, project, trajectory, titles, unproject } from "./plots.js";
import { strengthAt, strengthPosition } from "./strength.js";

let instance = 0;
const methods = {
  coordinate: "Coordinate descent",
  proximal: "Proximal gradient",
  irls: "CD (IRLS)",
  hybrid: "Hybrid",
};
const defaults = {
  loss: "gaussian",
  penalty: "l1",
  correlation: 0.7,
  signal: [2, -1],
  strength: 0.05,
  mixing: 0.5,
  rank_ratio: 0.5,
};
const number = (value) => Number(value.toPrecision(4)).toString();
const penaltyFormula = (c) =>
  ({
    l1: "λ(|β₁| + |β₂|)",
    ridge: "λ(β₁² + β₂²)/2",
    "elastic-net": "λ[α(|β₁| + |β₂|) + (1 − α)(β₁² + β₂²)/2]",
    slope: "λ[max(|β₁|, |β₂|) + r min(|β₁|, |β₂|)]",
    "clustered-lasso": "λ[α(|β₁| + |β₂|) + (1 − α)|β₁ − β₂|]",
  })[c.penalty];

export async function mount(root, { signal }) {
  const id = `optimization-${++instance}`;
  root.className = "optimization-explorer";
  const select = (name, label, options) =>
    `<label>${label}<select name="${name}">${options.map(([value, text]) => `<option value="${value}">${text}</option>`).join("")}</select></label>`;
  const range = (name, label, min, max, step, value) =>
    `<label>${label}<output data-value="${name}">${value}</output><input name="${name}" type="range" min="${min}" max="${max}" step="${step}" value="${value}"/></label>`;
  root.innerHTML = `<div class="explorer-intro"><p class="explorer-eyebrow">Build a model, watch it fit</p><p>Choose a loss and penalty. Then follow a solver across the objective it minimizes.</p></div>
<form class="explorer-controls" aria-label="Optimization problem">
<fieldset class="explorer-data"><legend>Data</legend><div class="explorer-group-fields">
${select("loss", "Datafit", [
  ["gaussian", "Gaussian"],
  ["binomial", "Binomial"],
])}
${range("correlation", "Correlation", -0.9, 0.9, 0.05, defaults.correlation)}
${range("signal1", "Signal β₁", -3, 3, 0.1, defaults.signal[0])}
${range("signal2", "Signal β₂", -3, 3, 0.1, defaults.signal[1])}
</div></fieldset>
<fieldset><legend>Penalty</legend><div class="explorer-group-fields">
${select("penalty", "Penalty", [
  ["l1", "Lasso (L1)"],
  ["ridge", "Ridge"],
  ["elastic-net", "Elastic net"],
  ["slope", "SLOPE"],
  ["clustered-lasso", "Clustered lasso"],
])}
${range("strength", "Penalty strength λ", 0, 1, "any", strengthPosition(defaults.strength))}
<div data-shape="mixing" hidden>${range("mixing", "Sparsity mixing", 0, 1, 0.05, 0.5)}</div>
<div data-shape="rank" hidden>${range("rank_ratio", "Second rank weight / first", 0, 1, 0.05, 0.5)}</div>
</div></fieldset>
<fieldset><legend>Solver</legend><div class="explorer-group-fields">
${select("method", "Solver", Object.entries(methods))}
</div></fieldset>
</form>
<div class="explorer-plots">${Object.entries(titles)
    .map(
      ([field, title]) =>
        `<figure data-panel="${field}"><figcaption>${title}</figcaption><div data-plot="${field}"></div><p data-caption="${field}"></p>${field === "loss" ? '<p data-caption="penalty"></p>' : `<p class="explorer-hint" id="${id}-drag-hint">Drag the hand to choose a starting point.</p>`}</figure>`,
    )
    .join("")}</div>
<p class="explorer-status" role="status">Loading the explorer…</p>`;
  const form = root.querySelector("form");
  form.addEventListener("submit", (event) => event.preventDefault(), {
    signal,
  });
  const input = (name) => root.querySelector(`[name="${name}"]`);
  const status = root.querySelector('[role="status"]');
  const solver = input("method");
  const objective = root.querySelector('[data-plot="objective"]');
  solver.disabled = true;
  const worker = new Worker(new URL("./worker.js", import.meta.url), {
    type: "module",
  });
  let scene,
    trace,
    displayedConfig,
    start = [0, 0],
    serial = 0,
    busy = false,
    pending,
    debounce;
  let resolveInitial, rejectInitial;
  const initialized = new Promise((resolve, reject) => {
    resolveInitial = resolve;
    rejectInitial = reject;
  });
  const timeout = setTimeout(
    () => rejectInitial(new Error("The Rust worker did not initialize.")),
    30000,
  );
  const cleanup = () => {
    clearTimeout(debounce);
    clearTimeout(timeout);
    worker.terminate();
    rejectInitial(new DOMException("Component unmounted", "AbortError"));
  };
  signal.addEventListener("abort", cleanup, { once: true });
  if (signal.aborted) {
    cleanup();
    return;
  }

  const config = () => ({
    loss: input("loss").value,
    penalty: input("penalty").value,
    strength: strengthAt(+input("strength").value),
    correlation: +input("correlation").value,
    signal: [+input("signal1").value, +input("signal2").value],
    mixing: +input("mixing").value,
    rank_ratio: +input("rank_ratio").value,
  });
  const updateFields = () => {
    form.querySelectorAll("output").forEach((output) => {
      const position = +input(output.dataset.value).value;
      output.value = number(
        output.dataset.value === "strength" ? strengthAt(position) : position,
      );
    });
    input("strength").setAttribute(
      "aria-valuetext",
      `λ = ${number(strengthAt(+input("strength").value))}`,
    );
    const penalty = input("penalty").value;
    root.querySelector('[data-shape="mixing"]').hidden = ![
      "elastic-net",
      "clustered-lasso",
    ].includes(penalty);
    root.querySelector('[data-shape="rank"]').hidden = penalty !== "slope";
    const mixingLabel = input("mixing").parentElement.firstChild;
    mixingLabel.textContent =
      penalty === "elastic-net" ? "L1 mixing α" : "Sparsity mixing α";
  };
  const showTrace = () => {
    const path = trajectory(scene, trace.points, trace.points.length - 1);
    objective.querySelector("[data-trajectory]").innerHTML = path;
    updateHandle();
  };
  const updateHandle = () => {
    let handle = objective.querySelector("[data-start-handle]");
    if (!handle) {
      objective.insertAdjacentHTML(
        "beforeend",
        `<button type="button" class="explorer-start-handle" data-start-handle aria-label="Move starting point" aria-describedby="${id}-drag-hint" title="Drag to move the starting point. Arrow keys also move it."><svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M8 21h8l4-7V9a1.5 1.5 0 0 0-3 0v3V5a1.5 1.5 0 0 0-3 0v6V3a1.5 1.5 0 0 0-3 0v8V5a1.5 1.5 0 0 0-3 0v10l-2-2a1.5 1.5 0 0 0-2 2l4 6Z" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg></button>`,
      );
      handle = objective.querySelector("[data-start-handle]");
    }
    const [x, y] = project(scene, start);
    handle.style.left = `${(x / 310) * 100}%`;
    handle.style.top = `${(y / 310) * 100}%`;
    handle.setAttribute(
      "aria-description",
      `β₁ = ${number(start[0])}, β₂ = ${number(start[1])}. Use arrow keys to move.`,
    );
  };
  const flush = () => {
    if (busy || !pending || signal.aborted) return;
    busy = true;
    const request = pending;
    pending = undefined;
    worker.postMessage(request);
  };
  const request = (immediate = false, trajectoryOnly = false) => {
    const current = config();
    trajectoryOnly =
      trajectoryOnly && JSON.stringify(current) === displayedConfig;
    solver.disabled = true;
    root.setAttribute("aria-busy", "true");
    status.hidden = trajectoryOnly;
    status.textContent = trajectoryOnly
      ? "Computing trajectory…"
      : "Computing contours…";
    serial += 1;
    pending = {
      id: serial,
      config: current,
      method: solver.value,
      start,
      trajectoryOnly,
    };
    clearTimeout(debounce);
    if (immediate) flush();
    else debounce = setTimeout(flush, 150);
  };
  worker.onmessage = ({ data }) => {
    busy = false;
    if (signal.aborted) return;
    if (data.id !== serial) {
      flush();
      return;
    }
    root.setAttribute("aria-busy", "false");
    solver.disabled = false;
    if (data.error) {
      clearTimeout(timeout);
      rejectInitial(new Error(data.error));
      status.hidden = false;
      status.textContent = `Calculation failed: ${data.error}`;
      return;
    }
    trace = data.trace;
    if (data.scene) {
      scene = data.scene;
      displayedConfig = JSON.stringify(config());
      const current = config();
      for (const field of Object.keys(titles))
        root.querySelector(`[data-plot="${field}"]`).innerHTML = plot(
          scene,
          field,
          `${id}-${field}`,
        );
      root.querySelector('[data-caption="loss"]').textContent =
        (current.loss === "gaussian"
          ? "Averaged squared error. "
          : "Averaged binomial negative log likelihood. Responses are proportions. ") +
        (scene.reference.converged
          ? "Orange dot: penalized optimum."
          : "Reference optimum unavailable.");
      const zero = current.strength === 0;
      const fusion = current.penalty === "clustered-lasso";
      root.querySelector('[data-caption="penalty"]').textContent =
        `P(β) = ${penaltyFormula(current)}. ` +
        (!scene.reference.converged
          ? "The constraint region is unavailable until the reference fit converges."
          : `Matching constraint: P(β) ≤ P(β̂) = ${number(scene.penalty_budget)}. ` +
            (zero
              ? "Zero penalty: the entire coefficient plane is feasible."
              : scene.penalty_budget === 0
                ? scene.penalty_zero_set === "diagonal"
                  ? "The region collapses to the line β₁ = β₂."
                  : "The region collapses to the origin."
                : `${scene.penalty_clipped ? "The region continues beyond this view. " : "Shading includes the boundary. "}${fusion && current.mixing === 0 ? "Pure fusion gives an unbounded strip." : ""}`));
      root.querySelector('[data-caption="objective"]').textContent = scene
        .reference.converged
        ? "Loss + penalty, evaluated at every grid point. Orange dot: penalized optimum. Black line: accepted steps."
        : "Loss + penalty. Reference optimum unavailable.";
      solver.innerHTML = scene.compatible_methods
        .map(
          (method) => `<option value="${method}">${methods[method]}</option>`,
        )
        .join("");
    }
    solver.value = data.method;
    showTrace();
    status.hidden = trace.fit.converged;
    status.textContent = trace.fit.converged
      ? ""
      : "The solver stopped before reaching the optimum.";
    clearTimeout(timeout);
    resolveInitial();
    root.dataset.ready = "true";
    flush();
  };
  worker.onerror = (event) => {
    busy = false;
    root.setAttribute("aria-busy", "false");
    solver.disabled = false;
    clearTimeout(timeout);
    rejectInitial(new Error(event.message));
    status.hidden = false;
    status.textContent = "The Rust worker failed. Reload this page to retry.";
  };
  form.addEventListener(
    "input",
    (event) => {
      updateFields();
      const onlyMethod = event.target.name === "method";
      request(onlyMethod, onlyMethod);
    },
    { signal },
  );
  let drag;
  const pointerPosition = (event) => {
    const matrix = objective.querySelector("svg").getScreenCTM();
    if (!matrix) return null;
    const point = new DOMPoint(event.clientX, event.clientY).matrixTransform(
      matrix.inverse(),
    );
    return [point.x, point.y];
  };
  const moveStart = (chosen) => {
    if (chosen[0] === start[0] && chosen[1] === start[1]) return;
    start = chosen;
    updateHandle();
    request(true, true);
  };
  objective.addEventListener(
    "pointerdown",
    (event) => {
      const handle = event.target.closest("[data-start-handle]");
      if (!scene || !handle || event.button !== 0 || !event.isPrimary) return;
      const point = pointerPosition(event);
      if (!point) return;
      const origin = project(scene, start);
      drag = {
        id: event.pointerId,
        offset: point.map((value, i) => value - origin[i]),
      };
      event.preventDefault();
      handle.focus({ preventScroll: true });
      objective.setPointerCapture(event.pointerId);
      objective.dataset.dragging = "true";
    },
    { signal },
  );
  const movePointer = (event) => {
    if (!drag || drag.id !== event.pointerId) return;
    const point = pointerPosition(event);
    if (point)
      moveStart(
        unproject(
          scene,
          point.map((value, i) => value - drag.offset[i]),
          true,
        ),
      );
  };
  objective.addEventListener("pointermove", movePointer, { signal });
  const finishDrag = (event) => {
    if (!drag || drag.id !== event.pointerId) return;
    if (event.type === "pointerup") movePointer(event);
    drag = undefined;
    delete objective.dataset.dragging;
    if (objective.hasPointerCapture(event.pointerId))
      objective.releasePointerCapture(event.pointerId);
  };
  for (const type of ["pointerup", "pointercancel", "lostpointercapture"])
    objective.addEventListener(type, finishDrag, { signal });
  objective.addEventListener(
    "keydown",
    (event) => {
      if (!scene || !event.target.closest("[data-start-handle]")) return;
      const direction = {
        ArrowLeft: [-1, 0],
        ArrowRight: [1, 0],
        ArrowUp: [0, 1],
        ArrowDown: [0, -1],
      }[event.key];
      if (!direction) return;
      event.preventDefault();
      const step = scene.radius * (event.shiftKey ? 0.2 : 0.05);
      moveStart(
        start.map((value, i) =>
          Math.max(
            -scene.radius,
            Math.min(scene.radius, value + step * direction[i]),
          ),
        ),
      );
    },
    { signal },
  );
  input("strength").title =
    "Logarithmic spacing with zero included. The displayed value is λ.";
  updateFields();
  request(true);
  await initialized;
  return cleanup;
}
