import { defineConfig } from "tsup";

export default defineConfig({
  entry: ["src/index.ts"],
  format: ["esm", "cjs"],
  // Declarations come from `tsc --emitDeclarationOnly` (the build script), not
  // tsup: its `dts` build drives the TypeScript compiler API, which TypeScript 7
  // (the native port) does not ship. tsup bundles the JS; tsc owns the types.
  dts: false,
  clean: true,
  sourcemap: true,
});
