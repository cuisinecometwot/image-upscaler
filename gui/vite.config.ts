import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Fixed port so tauri.conf.json devUrl matches.
export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  server: { port: 1420, strictPort: true, watch: { ignored: ["**/src-tauri/**", "**/core/**"] } },
});
