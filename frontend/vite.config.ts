import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "path";

export default defineConfig({
    plugins: [
        react(),
        tailwindcss(),
    ],
    resolve: {
        alias: {
            "@": path.resolve(__dirname, "src"),
            "@assets": path.resolve(__dirname, "public/assets"),
        },
    },
    root: ".",
    build: {
        outDir: path.resolve(__dirname, "../static"),
        emptyOutDir: false, // Don't empty since we have plots/ folder
        rollupOptions: {
            output: {
                manualChunks: undefined,
            },
        },
    },
    server: {
        port: 5173,
        host: "0.0.0.0",
    },
});
