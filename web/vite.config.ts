import { defineConfig, Plugin } from "vite";
import { resolve } from "node:path";

function isolateHeaders(): Plugin {
  return {
    name: "cross-origin-isolation",
    configureServer(server) {
      server.middlewares.use((_req, res, next) => {
        res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
        res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
        next();
      });
    },
    configurePreviewServer(server) {
      server.middlewares.use((_req, res, next) => {
        res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
        res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
        next();
      });
    },
  };
}

export default defineConfig({
  base: "./",
  plugins: [isolateHeaders()],
  worker: {
    format: "es",
  },
  build: {
    target: "es2020",
    rollupOptions: {
      input: {
        index: resolve("index.html"),
        bench: resolve("bench.html"),
      },
    },
  },
  optimizeDeps: {
    exclude: ["onnxruntime-web"],
  },
});