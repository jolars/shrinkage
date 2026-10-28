import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import init, { evaluateScene, fitTrace } from "../pkg/shrinkage_widget.js";
import { plot, project, trajectory, unproject } from "../web/plots.js";

await init({
  module_or_path: readFileSync(
    new URL("../pkg/shrinkage_widget_bg.wasm", import.meta.url),
  ),
});
const base = {
  loss: "gaussian",
  penalty: "l1",
  correlation: 0.7,
  signal: [2, -1],
  strength: 0.05,
  mixing: 0.5,
  rank_ratio: 0.5,
};
const close = (a, b, tolerance = 1e-10) =>
  assert.ok(Math.abs(a - b) <= tolerance, `${a} differs from ${b}`);

for (const loss of ["gaussian", "binomial"])
  for (const penalty of [
    "l1",
    "ridge",
    "elastic-net",
    "slope",
    "clustered-lasso",
  ]) {
    test(`native and WebAssembly agree for ${loss} ${penalty}`, () => {
      const config = { ...base, loss, penalty };
      const native = spawnSync(
        fileURLToPath(new URL("../target/debug/fixture", import.meta.url)),
        [JSON.stringify(config)],
        { encoding: "utf8", maxBuffer: 16 * 1024 * 1024 },
      );
      assert.equal(native.status, 0, native.stderr);
      const expected = JSON.parse(native.stdout);
      const actual = evaluateScene(config, 0);
      close(actual.reference.objective, expected.scene.reference.objective);
      close(actual.penalty_budget, expected.scene.penalty_budget);
      close(actual.loss_at_optimum, expected.scene.loss_at_optimum);
      for (let j = 0; j < 2; j++)
        close(
          actual.reference.coefficients[j],
          expected.scene.reference.coefficients[j],
          1e-8,
        );
      for (const field of ["loss", "penalty", "objective"])
        for (let i = 0; i < actual[field].length; i += 137)
          close(actual[field][i], expected.scene[field][i]);
      const trace = fitTrace(config, "proximal", [0, 0]);
      close(trace.fit.objective, expected.trace.fit.objective);
      assert.equal(trace.fit.converged, expected.trace.fit.converged);
      for (let j = 0; j < 2; j++)
        close(
          trace.fit.coefficients[j],
          expected.trace.fit.coefficients[j],
          1e-8,
        );
      for (const method of actual.compatible_methods) {
        const fit = fitTrace(config, method, [0, 0]);
        assert.equal(fit.fit.converged, true);
        close(fit.fit.objective, actual.reference.objective, 1e-8);
        assert.deepEqual(fit.points[0].coefficients, [0, 0]);
        assert.deepEqual(fit.points.at(-1).coefficients, fit.fit.coefficients);
        const initialized = fitTrace(config, method, [-2, 1.5]);
        assert.deepEqual(initialized.points[0].coefficients, [-2, 1.5]);
        assert.equal(initialized.fit.converged, true);
        close(initialized.fit.objective, actual.reference.objective, 1e-8);
      }
    });
  }

test("SVG contours and trajectories share coefficient coordinates", () => {
  const scene = evaluateScene(base, 0);
  assert.deepEqual(project(scene, [0, 0]), [164, 142]);
  for (const b of [
    [0, 0],
    [-2, 1.5],
    [scene.radius, -scene.radius],
  ]) {
    const actual = unproject(scene, project(scene, b));
    actual.forEach((v, i) => close(v, b[i]));
  }
  assert.equal(unproject(scene, [0, 0]), null);
  const svg = plot(scene, "objective", "test-clip");
  assert.ok(svg.includes("data-trajectory"));
  assert.ok(svg.includes('clip-path="url(#test-clip)"'));
  assert.ok(!svg.includes("NaN"));
  const trace = fitTrace(base, "coordinate", [0, 0]);
  const path = trajectory(scene, trace.points, trace.points.length - 1);
  const updates = trace.points.filter(
    (point) => !["Initial", "Final"].includes(point.kind),
  );
  assert.equal([...path.matchAll(/data-iterate=/g)].length, updates.length);
  for (const point of updates) {
    const [x, y] = project(scene, point.coefficients);
    assert.ok(path.includes(`cx="${x}" cy="${y}"`));
  }
  assert.ok(path.includes('stroke="#000000"'));
  assert.ok(!path.includes("NaN"));
});

test("zero penalty, pure fusion, and invalid requests retain clear semantics", () => {
  const zero = evaluateScene({ ...base, strength: 0 }, 0);
  assert.ok(zero.penalty.every((v) => v === 0));
  assert.equal(zero.penalty_clipped, true);
  const fusion = evaluateScene(
    { ...base, penalty: "clustered-lasso", mixing: 0 },
    0,
  );
  assert.equal(fusion.penalty_clipped, true);
  assert.throws(() =>
    fitTrace({ ...base, penalty: "ridge" }, "coordinate", [0, 0]),
  );
  assert.throws(() => fitTrace(base, "proximal", [NaN, 0]));
  assert.throws(() => evaluateScene({ ...base, correlation: 1 }, 0));
});

test("the constraint boundary passes through the penalized optimum", () => {
  const scene = evaluateScene({ ...base, strength: 0.35 }, 0);
  const svg = plot(scene, "loss", "boundary-regression");
  const region = svg.match(/<g data-field="penalty"[^>]*>(.*?)<\/g>/s)[1];
  const vertices = [...region.matchAll(/[ML](-?[\d.]+),(-?[\d.]+)/g)].map(
    (m) => [+m[1], +m[2]],
  );
  const point = project(scene, scene.reference.coefficients);
  const distance = Math.min(
    ...vertices.slice(1).map((b, i) => {
      const a = vertices[i];
      const d = [b[0] - a[0], b[1] - a[1]];
      const t = Math.max(
        0,
        Math.min(
          1,
          ((point[0] - a[0]) * d[0] + (point[1] - a[1]) * d[1]) /
            (d[0] ** 2 + d[1] ** 2 || 1),
        ),
      );
      return Math.hypot(point[0] - a[0] - t * d[0], point[1] - a[1] - t * d[1]);
    }),
  );
  assert.ok(
    distance < 1,
    `The optimum is ${distance} SVG pixels from the boundary.`,
  );
});

test("zero-budget regions render as a point, line, or plane", () => {
  for (const [config, shape] of [
    [{ ...base, correlation: 0, strength: 2 }, "origin"],
    [
      {
        ...base,
        correlation: 0,
        strength: 2,
        penalty: "clustered-lasso",
        mixing: 0,
      },
      "diagonal",
    ],
    [{ ...base, strength: 0 }, "plane"],
  ]) {
    const scene = evaluateScene(config, 0);
    assert.equal(scene.penalty_budget, 0);
    const svg = plot(scene, "loss", "zero-set");
    assert.ok(svg.includes(`data-region="${shape}"`));
    assert.ok(!svg.includes("NaN"));
    assert.ok(svg.includes("data-optimum"));
  }
});
