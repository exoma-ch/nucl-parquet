// Give CommonJS consumers CommonJS type declarations.
//
// `tsc --emitDeclarationOnly` writes one set of .d.ts files into dist/. This
// package is `"type": "module"`, so TypeScript reads those as ESM — correct for
// `import`, wrong for `require`, which gets dist/index.cjs at runtime. That
// mismatch is arethetypeswrong's "Masquerading as ESM" under node16, and 0.17.1
// shipped it: tsup emitted an index.d.cts that `exports` never pointed at.
//
// The declarations are pure type re-exports, so the same files are valid CJS
// types. Copy them into dist/cjs/, next to a package.json that marks the
// directory CommonJS; `exports["."].require.types` points there. `attw --pack`
// in scripts/ci.sh checks every resolution mode.
import { copyFileSync, mkdirSync, readdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const dist = "dist";
const cjs = join(dist, "cjs");
mkdirSync(cjs, { recursive: true });
for (const file of readdirSync(dist)) {
  if (file.endsWith(".d.ts")) copyFileSync(join(dist, file), join(cjs, file));
}
writeFileSync(join(cjs, "package.json"), JSON.stringify({ type: "commonjs" }) + "\n");
