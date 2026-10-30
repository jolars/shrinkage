import { createHash } from "node:crypto";
import { test, expect } from "@playwright/test";
import { strengthPosition } from "../web/strength.js";

const explorer = (page) => page.locator(".optimization-explorer");
const openWidget = async (page) => {
  await page.goto("/");
  const component = page.locator(".diplodocus-component");
  if (await component.count()) await component.first().scrollIntoViewIfNeeded();
};
const ready = async (page) => {
  await openWidget(page);
  await expect(explorer(page)).toHaveAttribute("data-ready", "true");
};
const changeRange = async (page, name, value) => {
  await explorer(page)
    .locator(`[name="${name}"]`)
    .evaluate(
      (element, value) => {
        element.value = String(value);
        element.dispatchEvent(new Event("input", { bubbles: true }));
      },
      name === "strength" ? strengthPosition(value) : value,
    );
};

test("shows the full trajectory from zero without playback controls", async ({
  page,
}) => {
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await ready(page);
  const root = explorer(page);
  await expect(root.locator("svg[role=img]")).toHaveCount(2);
  await expect(
    root.locator("[data-plot=loss] [data-field=penalty] path"),
  ).toHaveCount(1);
  await expect(root.locator("[data-panel=penalty]")).toHaveCount(0);
  await expect
    .poll(async () =>
      Number(
        await root.locator("[data-field=penalty]").getAttribute("data-budget"),
      ),
    )
    .toBeCloseTo(0.1333333333333333, 8);
  await expect(
    root.locator("[data-plot=loss] [data-contact-contour]"),
  ).toHaveCount(1);
  const optima = root.locator("[data-optimum]");
  await expect(optima).toHaveCount(2);
  for (const attribute of ["cx", "cy"])
    await expect(optima.nth(0)).toHaveAttribute(
      attribute,
      await optima.nth(1).getAttribute(attribute),
    );
  const solver = root
    .locator("form")
    .getByRole("combobox", { name: "Solver", exact: true });
  await expect(solver).toHaveValue("coordinate");
  await expect(solver.locator("option")).toHaveText([
    "Coordinate descent",
    "Proximal gradient",
    "CD (IRLS)",
  ]);
  await expect(
    root.locator(
      "[name=preset], .explorer-solvers, .explorer-start, .explorer-compatibility",
    ),
  ).toHaveCount(0);
  await expect(root.locator(".explorer-hint")).toContainText(
    "choose a starting point",
  );
  await expect(root).toHaveCSS("border-top-width", "0px");
  await expect(root).not.toContainText("Available here:");
  await expect(root).not.toContainText("Starts at");
  await changeRange(page, "strength", 0.05);
  await expect(root).toHaveAttribute("aria-busy", "false");
  await expect(root).toHaveAttribute("aria-busy", "false");
  await expect
    .poll(() => root.locator("[data-plot=objective] [data-iterate]").count())
    .toBeGreaterThan(1);
  await expect(
    root.locator("[data-plot=loss] [data-trajectory] circle"),
  ).toHaveCount(0);
  await expect(
    root.locator(
      ".explorer-playback, [name=timeline], [name=speed], [data-action]",
    ),
  ).toHaveCount(0);
  await expect(
    root.getByRole("button", { name: "Move starting point" }),
  ).toBeVisible();
  await expect(
    root.locator("[data-plot=objective] [data-trajectory] path"),
  ).toHaveAttribute("d", /^M164,142L/);
  await expect(
    root.locator(".explorer-diagnostics, .explorer-conventions"),
  ).toHaveCount(0);
  await expect(root.locator("[role=status]")).toBeHidden();
  await expect(root.locator("figcaption")).toHaveText([
    "Datafit + penalty region",
    "Penalized objective",
  ]);
  await expect(root.locator("figcaption").first()).toHaveCSS(
    "text-align",
    "center",
  );
  await expect(root.locator('[data-caption="penalty"]')).toContainText("R(β)");
  await expect(root.locator('[data-caption="penalty"]')).not.toContainText("P(β)");
  await expect(root.locator('[data-caption="objective"]')).toContainText(
    "P(0, β) = F(0, β) + R(β)",
  );
  await expect(root).not.toContainText("θ");
  await expect(root.locator("[name=signal1]").locator("..")).toContainText(
    "β₁",
  );
  await expect(root.locator("[name=penalty] option")).toHaveCount(5);
  await expect(
    root.locator(
      '[name=penalty] option[value="group-lasso"], [name=penalty] option[value="chain-fused-lasso"]',
    ),
  ).toHaveCount(0);
  expect(errors).toEqual([]);
});

test("problem changes select a compatible solver and preserve supported selections", async ({
  page,
}) => {
  await ready(page);
  const root = explorer(page);
  const solver = root.getByRole("combobox", { name: "Solver", exact: true });
  await root.locator("[name=loss]").selectOption("binomial");
  await expect(root.locator("[data-caption=loss]")).toContainText(
    "proportions",
  );
  await expect(solver).toHaveValue("proximal");
  await expect(solver.locator("option")).toHaveText([
    "Proximal gradient",
    "CD (IRLS)",
  ]);
  await solver.selectOption("irls");
  await expect(root).toHaveAttribute("aria-busy", "false");
  await expect(root).toHaveAttribute("aria-busy", "false");
  await root.locator("[name=penalty]").selectOption("slope");
  await expect(root.locator("[name=rank_ratio]")).toBeVisible();
  await expect(solver).toHaveValue("proximal");
  await expect(solver.locator("option")).toHaveText([
    "Proximal gradient",
    "Hybrid",
  ]);
  await solver.selectOption("hybrid");
  await expect(root).toHaveAttribute("aria-busy", "false");
  await root.locator("[name=penalty]").selectOption("clustered-lasso");
  await expect(root).toHaveAttribute("aria-busy", "false");
  await expect(solver).toHaveValue("hybrid");
  await expect(root).toHaveAttribute("aria-busy", "false");
});

test("latest slider input wins and pure fusion and zero strength remain visible", async ({
  page,
}) => {
  await ready(page);
  const root = explorer(page);
  await changeRange(page, "correlation", -0.9);
  await changeRange(page, "correlation", 0.9);
  await changeRange(page, "correlation", 0.2);
  await expect(root.locator("[data-value=correlation]")).toHaveText("0.2");
  await root.locator("[name=penalty]").selectOption("clustered-lasso");
  await changeRange(page, "mixing", 0);
  await expect(root.locator("[data-caption=penalty]")).toContainText(
    "unbounded strip",
  );
  await changeRange(page, "strength", 2);
  await expect(root.locator("[data-field=penalty]")).toHaveAttribute(
    "data-region",
    "diagonal",
  );
  await expect(root.locator("[data-caption=penalty]")).toContainText(
    "line β₁ = β₂",
  );
  await root.locator("[name=penalty]").selectOption("l1");
  await expect(root.locator("[data-field=penalty]")).toHaveAttribute(
    "data-region",
    "origin",
  );
  await changeRange(page, "strength", 0);
  await expect(root.locator("[data-field=penalty]")).toHaveAttribute(
    "data-region",
    "plane",
  );
  await expect(root.locator("[data-caption=penalty]")).toContainText(
    "entire coefficient plane",
  );
});

const clickPlot = async (page, x, y, field = "objective") => {
  await explorer(page)
    .locator(`[data-plot=${field}] > svg`)
    .scrollIntoViewIfNeeded();
  const position = await explorer(page)
    .locator(`[data-plot=${field}] > svg`)
    .evaluate(
      (svg, [x, y]) => {
        const point = new DOMPoint(x, y).matrixTransform(svg.getScreenCTM());
        return { x: point.x, y: point.y };
      },
      [x, y],
    );
  await page.mouse.click(position.x, position.y);
};

const dragHandle = async (page, x, y) => {
  const root = explorer(page);
  await root.locator("[data-plot=objective] > svg").scrollIntoViewIfNeeded();
  const handle = root.getByRole("button", { name: "Move starting point" });
  const box = await handle.boundingBox();
  const point = await root.locator("[data-plot=objective] > svg").evaluate(
    (svg, [x, y]) => {
      const point = new DOMPoint(x, y).matrixTransform(svg.getScreenCTM());
      return { x: point.x, y: point.y };
    },
    [x, y],
  );
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(point.x, point.y, { steps: 6 });
  await page.mouse.up();
  await expect(root).toHaveAttribute("aria-busy", "false");
};

const expectStart = async (page, x, y) => {
  const path = explorer(page).locator(
    "[data-plot=objective] [data-trajectory] path",
  );
  await expect
    .poll(async () => {
      const [_, sx, sy] = (await path.getAttribute("d")).match(
        /^M([^,]+),([^L]+)/,
      );
      return Math.abs(Number(sx) - x) < 2 && Math.abs(Number(sy) - y) < 2;
    })
    .toBe(true);
};

test("dragging the hand refits the objective trajectory", async ({ page }) => {
  await ready(page);
  const root = explorer(page);
  const handle = root.getByRole("button", { name: "Move starting point" });
  await expect(handle).toBeVisible();
  await expect(handle).toHaveCSS("cursor", "grab");
  await dragHandle(page, 104, 82);
  await expectStart(page, 104, 82);
  await root
    .getByRole("combobox", { name: "Solver", exact: true })
    .selectOption("proximal");
  await expect(root).toHaveAttribute("aria-busy", "false");
  await expectStart(page, 104, 82);
  await dragHandle(page, 224, 202);
  await expectStart(page, 224, 202);
  const path = root.locator("[data-plot=objective] [data-trajectory] path");
  const before = await path.getAttribute("d");
  await clickPlot(page, 164, 142);
  await clickPlot(page, 104, 82, "loss");
  await expect(path).toHaveAttribute("d", before);
  await expect(root.locator("[data-plot=loss] [data-trajectory]")).toHaveCount(
    0,
  );
  await dragHandle(page, 164, 142);
  await expectStart(page, 164, 142);
  await handle.focus();
  await page.keyboard.press("ArrowRight");
  await expect(root).toHaveAttribute("aria-busy", "false");
  await expectStart(page, 170, 142);
  await expect(handle).toBeFocused();
});

test("keyboard problem controls show the complete path regardless of motion preference", async ({
  page,
}) => {
  await page.emulateMedia({ reducedMotion: "no-preference" });
  await ready(page);
  const root = explorer(page);
  await root.locator("[name=correlation]").focus();
  await page.keyboard.press("ArrowLeft");
  await expect(root.locator("[data-value=correlation]")).toHaveText("0.65");
  await root
    .getByRole("combobox", { name: "Solver", exact: true })
    .selectOption("proximal");
  await expect(root).toHaveAttribute("aria-busy", "false");
  const path = root.locator("[data-plot=objective] [data-trajectory] path");
  const before = await path.getAttribute("d");
  await page.emulateMedia({ reducedMotion: "reduce" });
  await expect(path).toHaveAttribute("d", before);
});

test("capture explorer in both themes without horizontal overflow", async ({
  page,
}, info) => {
  await ready(page);
  const root = explorer(page);
  await changeRange(page, "strength", 0.05);
  await expect(root).toHaveAttribute("aria-busy", "false");
  await expect(root).toHaveAttribute("aria-busy", "false");
  for (const colorScheme of ["light", "dark"]) {
    await page.emulateMedia({ colorScheme });
    await expect(root.locator("svg[role=img]")).toHaveCount(2);
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    ).toBe(true);
    await root.screenshot({
      path: `artifacts/${info.project.name}-${colorScheme}.png`,
    });
  }
});

test("keeps the static explanation when WebAssembly cannot load", async ({
  page,
  context,
}) => {
  await context.route("**/*.wasm", (route) => route.abort("failed"));
  await openWidget(page);
  if (process.env.SHRINKAGE_DOCS_SITE) {
    await expect(page.locator("[data-component-fallback]")).toBeVisible();
    await expect(
      page.locator(".diplodocus-component").getByRole("status"),
    ).toContainText("could not load");
    await expect(explorer(page)).toHaveCount(0);
  } else {
    await expect(explorer(page)).toContainText(/failed|fetch/i);
    await expect(explorer(page)).not.toHaveAttribute("data-ready", "true");
  }
});

test("reserves the explorer's space while its scripts and worker load", async ({
  page,
}, info) => {
  test.skip(
    !process.env.SHRINKAGE_DOCS_SITE,
    "Requires the generated landing page.",
  );
  let releaseScript, releaseWasm;
  const scriptGate = new Promise((resolve) => {
    releaseScript = resolve;
  });
  const wasmGate = new Promise((resolve) => {
    releaseWasm = resolve;
  });
  await page.route("**/assets/components.js", async (route) => {
    await scriptGate;
    await route.continue();
  });
  await page.route("**/*.wasm", async (route) => {
    await wasmGate;
    await route.continue();
  });
  try {
    await page.goto("/", { waitUntil: "commit" });
    const component = page.locator(".diplodocus-component");
    const loading = component.locator("[data-explorer-loading]");
    await expect(loading).toBeVisible();
    await expect(loading).toHaveAttribute("role", "status");
    await expect(loading).toContainText("Loading the optimization explorer");
    await expect(component.locator("[data-component-fallback] img")).toBeHidden();
    await component.scrollIntoViewIfNeeded();
    const before = await component.boundingBox();
    releaseScript();
    await expect(explorer(page)).toBeAttached();
    await expect(loading).toBeVisible();
    await expect(explorer(page)).toBeHidden();
    const indicator = loading.locator(".explorer-loading-indicator");
    await page.emulateMedia({ reducedMotion: "reduce" });
    await expect(indicator).toHaveCSS("animation-name", "none");
    await page.emulateMedia({ reducedMotion: "no-preference" });
    await expect(indicator).toHaveCSS("animation-name", "explorer-loading");
    await page.emulateMedia({ reducedMotion: "reduce" });
    await component.screenshot({
      path: `artifacts/${info.project.name}-loading.png`,
    });
    releaseWasm();
    await expect(explorer(page)).toHaveAttribute("data-ready", "true");
    await expect(loading).toBeHidden();
    const after = await component.boundingBox();
    expect(Math.abs(after.height - before.height)).toBeLessThan(2);
  } finally {
    releaseScript();
    releaseWasm();
  }
});

test("shows the static explanation when JavaScript is disabled", async ({
  browser,
}, info) => {
  test.skip(
    !process.env.SHRINKAGE_DOCS_SITE,
    "Requires the generated landing page.",
  );
  const context = await browser.newContext({
    javaScriptEnabled: false,
    viewport: info.project.use.viewport,
  });
  try {
    const page = await context.newPage();
    await page.goto("http://127.0.0.1:8792/");
    await expect(page.locator("[data-component-fallback] img")).toBeVisible();
    await expect(page.locator("[data-explorer-loading]")).toBeHidden();
  } finally {
    await context.close();
  }
});

test("terminates its worker when the component unmounts", async ({ page }) => {
  await page.addInitScript(() => {
    const NativeWorker = window.Worker;
    window.activeExplorerWorkers = 0;
    window.Worker = class extends NativeWorker {
      constructor(...args) {
        super(...args);
        window.activeExplorerWorkers += 1;
      }
      terminate() {
        if (!this.closed) {
          this.closed = true;
          window.activeExplorerWorkers -= 1;
        }
        super.terminate();
      }
    };
  });
  await ready(page);
  expect(await page.evaluate(() => window.activeExplorerWorkers)).toBe(1);
  if (process.env.SHRINKAGE_DOCS_SITE) {
    await page.getByRole("link", { name: "Get started", exact: true }).click();
    await expect(page).toHaveURL(/installation/);
  } else await page.evaluate(() => window.disposeExplorer());
  await expect
    .poll(() => page.evaluate(() => window.activeExplorerWorkers))
    .toBe(0);
});

test("automatically fits with a compatible solver on load and after problem changes", async ({
  page,
}) => {
  await ready(page);
  const root = explorer(page);
  const path = root.locator("[data-plot=objective] [data-trajectory] path");
  await expect(path).toHaveAttribute("d", /^M164,142L/);
  const solver = root.getByRole("combobox", { name: "Solver", exact: true });
  await expect(solver).toHaveValue("coordinate");
  await root.locator("[name=penalty]").selectOption("slope");
  await expect(root).toHaveAttribute("aria-busy", "false");
  await expect(solver).toHaveValue("proximal");
  await expect(path).toHaveAttribute("d", /^M164,142L/);
  await expect(solver.locator("option")).toHaveText([
    "Proximal gradient",
    "Hybrid",
  ]);
});

test("dragging a new start preserves both plots and their limits", async ({
  page,
}) => {
  await ready(page);
  const root = explorer(page);
  const snapshot = () =>
    root.locator("[data-plot] > svg").evaluateAll((svgs) =>
      svgs.map((svg) => {
        const clone = svg.cloneNode(true);
        clone.querySelector("[data-trajectory]")?.replaceChildren();
        return clone.outerHTML;
      }),
    );
  const before = await snapshot();
  const hashes = (items) =>
    items.map((value) => createHash("sha256").update(value).digest("hex"));
  await dragHandle(page, 64, 261);
  expect(hashes(await snapshot())).toEqual(hashes(before));
});

test("only Gaussian trajectories include inner iterations", async ({
  page,
}) => {
  await ready(page);
  const root = explorer(page);
  const inner = root.locator("[data-plot=objective] [data-inner-iterate]");
  await changeRange(page, "strength", 0.05);
  await expect(root).toHaveAttribute("aria-busy", "false");
  for (const loss of ["gaussian", "binomial"]) {
    await root.locator("[name=loss]").selectOption(loss);
    await expect(root).toHaveAttribute("aria-busy", "false");
    for (const [penalty, method] of [
      ["l1", "irls"],
      ["slope", "hybrid"],
      ["clustered-lasso", "hybrid"],
    ]) {
      await root.locator("[name=penalty]").selectOption(penalty);
      await expect(root).toHaveAttribute("aria-busy", "false");
      await root.locator("[name=method]").selectOption(method);
      await expect(root).toHaveAttribute("aria-busy", "false");
      if (loss === "binomial") {
        await expect(inner).toHaveCount(0);
      } else {
        await expect.poll(() => inner.count()).toBeGreaterThan(0);
        await expect(inner.first()).toHaveAttribute("fill", "#ffffff");
        await expect(inner.first()).toHaveCSS("fill", "rgb(255, 255, 255)");
        await expect(inner.first()).toHaveCSS("stroke", "rgb(0, 0, 0)");
        await expect(inner.first().locator("title")).toContainText(
          "inner iteration",
        );
      }
      await expect(root.locator("[role=status]")).toBeHidden();
    }
  }
});

test("large binomial quadratic proposals preserve a useful viewport", async ({
  page,
}) => {
  await ready(page);
  const result = await page.evaluate(async () => {
    const module = document.querySelector("[data-component-module]")
      ?.dataset.componentModule;
    const entry = module ? new URL(module, location.href) : location.href;
    const worker = new Worker(new URL("worker.js", entry), { type: "module" });
    try {
      return await new Promise((resolve, reject) => {
        worker.onmessage = ({ data }) => resolve(data);
        worker.onerror = (event) => reject(new Error(event.message));
        worker.postMessage({
          id: 1,
          config: {
            loss: "binomial",
            penalty: "clustered-lasso",
            correlation: 0.7,
            signal: [2, -1],
            strength: 0.00788,
            mixing: 0.5,
            rank_ratio: 0.5,
          },
          method: "hybrid",
          start: [38.5, -1],
          trajectoryOnly: false,
        });
      });
    } finally {
      worker.terminate();
    }
  });
  expect(result.error).toBeUndefined();
  expect(result.method).toBe("hybrid");
  expect(result.trace.fit.converged).toBe(true);
  const accepted = result.trace.points.filter((point) => !point.inner);
  expect(accepted).toHaveLength(result.trace.points.length);
  const extent = Math.max(
    ...accepted.flatMap((point) => point.coefficients.map(Math.abs)),
  );
  expect(result.scene.radius).toBeGreaterThan(extent);
  expect(result.scene.radius).toBeLessThan(2 * extent);
});

test("penalty slider resolves weak binomial penalties and includes zero", async ({
  page,
}) => {
  await ready(page);
  const root = explorer(page);
  await root.locator("[name=loss]").selectOption("binomial");
  await expect(root).toHaveAttribute("aria-busy", "false");
  const strength = root.locator("[name=strength]");
  const value = root.locator("[data-value=strength]");
  await strength.focus();
  await page.keyboard.press("Home");
  await expect(value).toHaveText("0");
  await expect(root).toHaveAttribute("aria-busy", "false");
  await expect(root.locator("[data-field=penalty]")).toHaveAttribute(
    "data-region",
    "plane",
  );
  await page.keyboard.press("ArrowRight");
  await expect
    .poll(async () => Number(await value.textContent()))
    .toBeGreaterThan(0);
  expect(Number(await value.textContent())).toBeLessThan(0.001);
  await expect(root).toHaveAttribute("aria-busy", "false");
  await strength.evaluate((element) => {
    element.value = String((Number(element.min) + Number(element.max)) / 2);
    element.dispatchEvent(new Event("input", { bubbles: true }));
  });
  await expect
    .poll(async () => Number(await value.textContent()))
    .toBeLessThan(0.1);
  expect(Number(await value.textContent())).toBeGreaterThan(0.01);
  await expect(root).toHaveAttribute("aria-busy", "false");
  await expect(
    root.locator("[data-plot=loss] [data-field=penalty] path"),
  ).toHaveCount(1);
  await page.keyboard.press("End");
  await expect(value).toHaveText("2");
  await expect(root).toHaveAttribute("aria-busy", "false");
  await expect(root.locator("[data-field=penalty]")).toHaveAttribute(
    "data-region",
    "origin",
  );
});

test("touch dragging the hand preserves page position and releases capture", async ({
  page,
  isMobile,
}) => {
  test.skip(!isMobile, "Touch interaction uses the mobile viewport.");
  await ready(page);
  const root = explorer(page);
  const objective = root.locator("[data-plot=objective]");
  await objective.locator(":scope > svg").scrollIntoViewIfNeeded();
  const session = await page.context().newCDPSession(page);
  const scroll = await page.evaluate(() => scrollY);
  const touch = async (x, y, finish) => {
    const box = await root
      .getByRole("button", { name: "Move starting point" })
      .boundingBox();
    const point = await objective.locator(":scope > svg").evaluate(
      (svg, [x, y]) => {
        const point = new DOMPoint(x, y).matrixTransform(svg.getScreenCTM());
        return { x: point.x, y: point.y };
      },
      [x, y],
    );
    await session.send("Input.dispatchTouchEvent", {
      type: "touchStart",
      touchPoints: [{ x: box.x + box.width / 2, y: box.y + box.height / 2 }],
    });
    await session.send("Input.dispatchTouchEvent", {
      type: "touchMove",
      touchPoints: [point],
    });
    await session.send("Input.dispatchTouchEvent", {
      type: finish,
      touchPoints: [],
    });
    await expect(root).toHaveAttribute("aria-busy", "false");
    await expect(objective).not.toHaveAttribute("data-dragging", "true");
    await expectStart(page, x, y);
  };
  await touch(104, 82, "touchEnd");
  expect(await page.evaluate(() => scrollY)).toBe(scroll);
  await touch(224, 202, "touchCancel");
  await touch(164, 142, "touchEnd");
  await session.detach();
});
