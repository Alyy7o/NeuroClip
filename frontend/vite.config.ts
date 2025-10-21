import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";
import { componentTagger } from "lovable-tagger";
import { VitePWA } from "vite-plugin-pwa";

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
  server: {
    host: "::",
    port: 8080,
  },
  plugins: [
    react(),
    mode === "development" && componentTagger(),
    VitePWA({
      registerType: "autoUpdate",
      injectRegister: "auto",
      includeAssets: ["favicon.ico", "robots.txt", "logo-2-n-192.png", "logo-2-n-512.png"],
      manifest: {
        name: "NeuroClip",
        short_name: "NeuroClip",
        description: "NeuroClip - AI-powered video editing suite",
        theme_color: "#1e40af",
        background_color: "#ffffff",
        display: "standalone",
        start_url: "/",
        icons: [
          { src: "/logo-2-n-192.png", sizes: "192x192", type: "image/png", purpose: "any" },
          { src: "/logo-2-n-512.png", sizes: "512x512", type: "image/png", purpose: "any" },
          { src: "/logo-2-n-512.png", sizes: "512x512", type: "image/png", purpose: "maskable" }
        ]
      }
    })
  ].filter(Boolean),
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
}));
