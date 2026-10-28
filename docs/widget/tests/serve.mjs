import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { resolve, extname, sep } from "node:path";
const site = process.env.SHRINKAGE_DOCS_SITE;
const root = resolve(site || "dist");
const page = `<!doctype html><html><head><meta name="viewport" content="width=device-width, initial-scale=1"><link rel="stylesheet" href="style.css"><style>
:root{color-scheme:light dark;--ink:light-dark(#202c38,#e6edf3);--muted:light-dark(#4d606b,#a5b3bd);--border:light-dark(#d6dedc,#34434c);--background:light-dark(white,#11191f);--surface:light-dark(#fafbf9,#202b33);--selected:light-dark(#e8f0ee,#293e43);--accent:light-dark(#30776b,#90cfbe);--accent-background:light-dark(#2d7064,#90cfbe);--on-accent:light-dark(white,#10251e);--focus-ring:light-dark(#a04605,#f5b36b);--warning-border:light-dark(#a04605,#f5b36b)}*{box-sizing:border-box}body{font:16px/1.5 system-ui;background:var(--background);color:var(--ink);margin:0;padding:24px}main{max-width:1100px;margin:auto}h1{font-weight:550}button,input,select{font:inherit}
</style></head><body><main><h1>Shrinkage</h1><div id="widget">Static optimization explanation.</div></main><script type="module">import {mount} from './index.js'; const controller=new AbortController();window.disposeExplorer=()=>controller.abort();mount(document.querySelector('#widget'),{signal:controller.signal}).catch(error=>document.querySelector('#widget').textContent=String(error));</script></body></html>`;
createServer(async (request, response) => {
  try {
    let pathname = decodeURIComponent(
      new URL(request.url, "http://localhost").pathname,
    );
    if (!site && pathname === "/") {
      response.setHeader("Content-Type", "text/html");
      response.end(page);
      return;
    }
    if (pathname.endsWith("/")) pathname += "index.html";
    if (!extname(pathname)) pathname += ".html";
    const file = resolve(root, "." + pathname);
    if (!file.startsWith(root + sep)) throw new Error("Invalid path");
    const bytes = await readFile(file);
    response.setHeader(
      "Content-Type",
      {
        ".html": "text/html",
        ".js": "text/javascript",
        ".css": "text/css",
        ".json": "application/json",
        ".wasm": "application/wasm",
        ".svg": "image/svg+xml",
        ".png": "image/png",
        ".woff2": "font/woff2",
        ".woff": "font/woff",
        ".ttf": "font/ttf",
      }[extname(file)] || "application/octet-stream",
    );
    response.end(bytes);
  } catch {
    response.writeHead(404).end();
  }
}).listen(8792, "127.0.0.1");
