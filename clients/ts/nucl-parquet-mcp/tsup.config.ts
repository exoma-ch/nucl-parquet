import { defineConfig } from "tsup";

export default defineConfig({
  entry: ["src/index.ts"],
  format: ["esm"],
  // Declarations come from `tsc --emitDeclarationOnly` (the build script): tsup's
  // `dts` build drives the TypeScript compiler API, which TypeScript 7 does not ship.
  dts: false,
  clean: true,
  sourcemap: true,
  banner: { js: "#!/usr/bin/env node" },
});
