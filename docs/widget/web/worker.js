import init, { evaluateScene, fitTrace } from "../pkg/shrinkage_widget.js";
const ready = init();
self.onmessage = async ({
  data: { id, config, method, start, trajectoryOnly },
}) => {
  try {
    await ready;
    let scene = trajectoryOnly ? null : evaluateScene(config, 0);
    const startRadius = Math.max(...start.map(Math.abs));
    if (scene && startRadius >= scene.radius)
      scene = evaluateScene(config, startRadius);
    let selectedMethod =
      !scene || scene.compatible_methods.includes(method)
        ? method
        : scene.compatible_methods[0];
    let trace;
    try {
      trace = fitTrace(config, selectedMethod, start);
    } catch (error) {
      if (selectedMethod === "proximal") throw error;
      selectedMethod = "proximal";
      trace = fitTrace(config, selectedMethod, start);
    }
    const radius = trace.points.reduce(
      (r, point) => Math.max(r, ...point.coefficients.map(Math.abs)),
      0,
    );
    if (scene && radius >= scene.radius) scene = evaluateScene(config, radius);
    const transfer = [];
    for (const field of scene ? ["loss", "penalty", "objective"] : []) {
      scene[field] = Float64Array.from(scene[field]);
      transfer.push(scene[field].buffer);
    }
    self.postMessage({ id, scene, trace, method: selectedMethod }, transfer);
  } catch (error) {
    self.postMessage({ id, error: String(error) });
  }
};
