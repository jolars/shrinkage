import { contours } from "d3-contour";

const escape = (value) =>
  String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll('"', "&quot;");
const colors = {
  penalty: "#4682b4",
  optimum: "#cc6600",
  trace: "#000000",
};
// Keep the minimum slightly darker, matching Basin's contour palette.
const shades = Array.from({ length: 9 }, (_, i) => {
  const t = 1 - i / 8;
  const light = Math.round(195 + 50 * t);
  const dark = Math.round(45 + 45 * t);
  return `light-dark(rgb(${light} ${light} ${light}), rgb(${dark} ${dark} ${dark}))`;
});
export const titles = {
  loss: "Datafit + penalty region",
  objective: "Penalized objective",
};
const left = 44,
  top = 22,
  size = 240;
export const project = (scene, b) => [
  left + ((b[0] + scene.radius) / (2 * scene.radius)) * size,
  top + ((scene.radius - b[1]) / (2 * scene.radius)) * size,
];

export const unproject = (scene, [x, y], clamp = false) => {
  if (clamp) {
    x = Math.max(left, Math.min(left + size, x));
    y = Math.max(top, Math.min(top + size, y));
  }
  if (x < left || x > left + size || y < top || y > top + size) return null;
  return [
    (((x - left) / size) * 2 - 1) * scene.radius,
    (1 - ((y - top) / size) * 2) * scene.radius,
  ];
};

function polygonPath(polygons, scene) {
  return polygons.coordinates
    .map((polygon) =>
      polygon
        .map(
          (ring) =>
            ring
              .map(([x, y], i) => {
                const p = project(scene, [
                  scene.grid_low + x * scene.grid_step,
                  scene.grid_low + y * scene.grid_step,
                ]);
                return `${i ? "L" : "M"}${p[0].toFixed(3)},${p[1].toFixed(3)}`;
              })
              .join("") + "Z",
        )
        .join(""),
    )
    .join("");
}

function contourPaths(scene, field) {
  if (field === "penalty") {
    if (!scene.reference.converged) return { fills: "", lines: "" };
    if (scene.penalty_budget === 0) {
      const [x, y] = project(scene, [0, 0]);
      const color = colors.penalty;
      const shape =
        scene.penalty_zero_set === "plane"
          ? `<path d="M${left},${top}h${size}v${size}h-${size}Z" fill="${color}" fill-opacity=".13"/>`
          : scene.penalty_zero_set === "diagonal"
            ? `<path d="M${left},${top + size}L${left + size},${top}" fill="none" stroke="${color}" stroke-width="1.8"/>`
            : `<circle cx="${x}" cy="${y}" r="6" fill="${color}" fill-opacity=".13" stroke="${color}" stroke-width="1.8"/>`;
      return {
        fills: `<g data-field="penalty" data-budget="0" data-region="${scene.penalty_zero_set}">${shape}</g>`,
        lines: "",
      };
    }
  }
  const values = scene[field];
  const color = field === "penalty" ? colors.penalty : "#ffffff";
  const minimum =
    field === "loss" ? scene.loss_minimum : scene.reference.objective;
  let maximum = minimum;
  for (const value of values) maximum = Math.max(maximum, value);
  const levels =
    field === "penalty"
      ? [scene.penalty_budget]
      : [0.003, 0.01, 0.03, 0.07, 0.15, 0.3, 0.55, 0.85].map(
          (f) => minimum + f * (maximum - minimum),
        );
  // Negation yields sublevel polygons, including the complete penalty interior.
  const generator = contours()
    .size([scene.grid_size, scene.grid_size])
    .thresholds(levels.map((v) => -v).sort((a, b) => a - b));
  const polygons = generator(Array.from(values, (v) => -v)).map((p) =>
    polygonPath(p, scene),
  );
  if (field === "penalty") {
    return {
      fills: `<g data-field="penalty" data-budget="${scene.penalty_budget}" data-region="level-set">${polygons.map((path) => `<path d="${path}" fill="${color}" fill-opacity=".13" stroke="${color}" stroke-width="1.8"/>`).join("")}</g>`,
      lines: "",
    };
  }
  const fills = `<g data-field="${field}" data-contour-fills><path d="M${left},${top}h${size}v${size}h-${size}Z" fill="${shades[0]}"/>${polygons.map((path, i) => `<path d="${path}" fill="${shades[i + 1]}"/>`).join("")}</g>`;
  const lines = polygons
    .map(
      (path) =>
        `<path d="${path}" fill="none" stroke="${color}" stroke-width="1.1"/>`,
    )
    .join("");
  const contact =
    field === "loss" &&
    scene.reference.converged &&
    scene.loss_at_optimum > scene.loss_minimum
      ? generator
          .thresholds([-scene.loss_at_optimum])(Array.from(values, (v) => -v))
          .map(
            (p) =>
              `<path data-contact-contour d="${polygonPath(p, scene)}" fill="none" stroke="${color}" stroke-width="1.8"/>`,
          )
          .join("")
      : "";
  return { fills, lines: `<g data-contour-lines>${lines}${contact}</g>` };
}

export function plot(scene, field, id) {
  const contours = contourPaths(scene, field);
  const penalty = field === "loss" ? contourPaths(scene, "penalty").fills : "";
  const muted = "var(--muted, #536777)";
  const ticks = [
    -scene.radius,
    -scene.radius / 2,
    0,
    scene.radius / 2,
    scene.radius,
  ];
  const [zeroX, zeroY] = project(scene, [0, 0]);
  const axes = ticks
    .map((t) => {
      const [x, y] = project(scene, [t, t]);
      return `<path d="M${x},${top}V${top + size}M${left},${y}H${left + size}" stroke="#ffffff" stroke-opacity=".12" stroke-width=".6"/><text x="${x}" y="${top + size + 18}" text-anchor="middle">${Number(t.toFixed(1))}</text><text x="${left - 9}" y="${y + 4}" text-anchor="end">${Number(t.toFixed(1))}</text>`;
    })
    .join("");
  const minimumPoint = scene.reference.converged
    ? scene.reference.coefficients
    : null;
  const marker = minimumPoint
    ? (() => {
        const [x, y] = project(scene, minimumPoint);
        return `<circle data-optimum cx="${x}" cy="${y}" r="4" fill="${colors.optimum}" stroke="none"><title>Penalized optimum</title></circle>`;
      })()
    : "";
  return `<svg width="310" height="310" viewBox="0 0 310 310" role="img" aria-label="${escape(titles[field])}: beta one on the horizontal axis and beta two on the vertical axis" xmlns="http://www.w3.org/2000/svg">
<defs><clipPath id="${id}"><path d="M${left},${top}h${size}v${size}h-${size}Z"/></clipPath></defs>
<g clip-path="url(#${id})">${contours.fills}</g>
<g fill="${muted}" font-family="system-ui, sans-serif" font-size="11">${axes}<text x="${left + size / 2}" y="303" text-anchor="middle">β₁</text><text x="13" y="${top + size / 2}" text-anchor="middle">β₂</text></g>
<path d="M${zeroX},${top}V${top + size}M${left},${zeroY}H${left + size}" stroke="#ffffff" stroke-opacity=".2"/>
<g clip-path="url(#${id})">${contours.lines}${penalty}${field === "objective" ? "<g data-trajectory></g>" : ""}${marker}</g></svg>`;
}

export function trajectory(scene, points, index) {
  const visible = points.slice(0, index + 1);
  const path = visible
    .map((point, i) => {
      const [x, y] = project(scene, point.coefficients);
      return `${i ? "L" : "M"}${x},${y}`;
    })
    .join("");
  const dots = visible
    .slice(1)
    .filter((point) => point.kind !== "Final")
    .map((point) => {
      const [x, y] = project(scene, point.coefficients);
      const label = point.inner
        ? `Outer iteration ${point.iteration}, inner iteration ${point.inner_iteration}`
        : `Iteration ${point.iteration}`;
      return `<circle data-iterate="${point.iteration}"${point.inner ? ' data-inner-iterate=""' : ""} cx="${x}" cy="${y}" r="2.6" fill="${point.inner ? "#ffffff" : colors.trace}" stroke="${point.inner ? colors.trace : "none"}" stroke-width=".7"><title>${label}: ${escape(point.kind)}</title></circle>`;
    })
    .join("");
  return `<path d="${path}" fill="none" stroke="${colors.trace}" stroke-width="1.7" stroke-linejoin="round"/>${dots}`;
}
