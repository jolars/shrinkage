import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createReleasePlan } from "../docs/node_modules/versionary/dist/release/plan.js";
import { prepareSeparateReleasePrs } from "../docs/node_modules/versionary/dist/release/pr.js";

const repository = fileURLToPath(new URL("..", import.meta.url));
const config = JSON.parse(
  fs.readFileSync(path.join(repository, "versionary.jsonc"), "utf8"),
);

for (const scenario of ["rust", "python", "both", "documentation"]) {
  const directory = fs.mkdtempSync(
    path.join(os.tmpdir(), "shrinkage-release-"),
  );
  const git = (...args) =>
    execFileSync("git", args, {
      cwd: directory,
      encoding: "utf8",
      stdio: ["ignore", "pipe", "pipe"],
    }).trim();
  const write = (name, body) => {
    const target = path.join(directory, name);
    fs.mkdirSync(path.dirname(target), { recursive: true });
    fs.writeFileSync(target, body);
  };
  const commit = (message) => {
    git("add", ".");
    git("commit", "-m", message);
  };
  try {
    git("init", "-b", "main");
    git("config", "user.name", "Release fixture");
    git("config", "user.email", "fixture@example.invalid");
    git("config", "commit.gpgsign", "false");
    git("config", "tag.gpgsign", "false");
    write("versionary.jsonc", JSON.stringify(config));
    write(
      "Cargo.toml",
      '[package]\nname="shrinkage"\nversion="1.0.0"\nedition="2024"\n',
    );
    write("src/lib.rs", "pub const VALUE: i32 = 1;\n");
    write(
      "packages/python/Cargo.toml",
      '[package]\nname="shrinkage-python"\nversion="0.3.0"\nedition="2024"\npublish=false\n[dependencies]\nshrinkage={path="../.."}\n[workspace]\n',
    );
    write("packages/python/src/lib.rs", "pub use shrinkage::VALUE;\n");
    write(
      "packages/python/pyproject.toml",
      '[project]\nname="shrinkage"\nversion="0.3.0"\nrequires-python=">=3.11"\n',
    );
    write(
      "packages/python/src/shrinkage/__init__.py",
      '__version__ = "0.3.0"\n',
    );
    for (const cwd of [directory, path.join(directory, "packages/python")]) {
      execFileSync("cargo", ["generate-lockfile", "--offline"], {
        cwd,
        stdio: "pipe",
      });
    }
    execFileSync("uv", ["lock", "--offline"], {
      cwd: path.join(directory, "packages/python"),
      stdio: "pipe",
    });
    commit("chore: initialize release fixture");
    git("tag", "v1.0.0");
    git("tag", "shrinkage-python-v0.3.0");
    if (scenario === "rust" || scenario === "both") {
      write("src/lib.rs", "pub const VALUE: i32 = 2;\n");
      commit("feat: improve Rust core");
    }
    if (scenario === "python" || scenario === "both") {
      write(
        "packages/python/src/shrinkage/_api.py",
        'MESSAGE = "new binding"\n',
      );
      commit("fix: repair Python binding");
    }
    if (scenario === "documentation") {
      write("docs/pages/index.md", "# Documentation\n");
      commit("feat: improve documentation");
    }
    const expected = {
      ...(scenario === "rust" || scenario === "both" ? { ".": "1.1.0" } : {}),
      ...(scenario === "python" || scenario === "both"
        ? { "packages/python": "0.3.1" }
        : {}),
    };
    const plan = createReleasePlan(directory);
    assert.deepEqual(
      Object.fromEntries(
        plan.packages
          .filter((pkg) => pkg.nextVersion)
          .map((pkg) => [pkg.path, pkg.nextVersion]),
      ),
      expected,
    );
    if (scenario !== "documentation") {
      const head = git("rev-parse", "HEAD");
      const releases = prepareSeparateReleasePrs(directory);
      assert.equal(releases.length, 1);
      assert.deepEqual(
        [...releases[0].packagePaths].sort(),
        Object.keys(expected).sort(),
      );
      const branch = releases[0].branch;
      if (expected["packages/python"]) {
        for (const name of [
          "Cargo.toml",
          "Cargo.lock",
          "pyproject.toml",
          "uv.lock",
          "src/shrinkage/__init__.py",
        ]) {
          assert.match(
            git("show", `${branch}:packages/python/${name}`),
            /0\.3\.1/,
          );
        }
        assert.equal(
          git("show", `${branch}:Cargo.toml`).includes('version="1.1.0"'),
          scenario === "both",
        );
      }
      assert.equal(git("rev-parse", "HEAD"), head);
      assert.equal(git("status", "--porcelain"), "");
    }
    process.stdout.write(`Release scenario passed: ${scenario}\n`);
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
}
