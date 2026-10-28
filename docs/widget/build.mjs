import { build } from "esbuild";
import { mkdir, copyFile } from "node:fs/promises";
await mkdir("dist", { recursive: true });
await build({
  entryPoints: ["web/index.js", "web/worker.js"],
  outdir: "dist",
  bundle: true,
  format: "esm",
  target: "es2022",
  minify: true,
});
await Promise.all([
  copyFile("pkg/shrinkage_widget_bg.wasm", "dist/shrinkage_widget_bg.wasm"),
  copyFile("web/style.css", "dist/style.css"),
]);
