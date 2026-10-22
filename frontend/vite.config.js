import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig(({ command }) => ({
  // GitHub Pages serves the site from /ai-fake-news-detector/, so production builds need that base path
  base: command === 'build' ? '/ai-fake-news-detector/' : '/',
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      }
    }
  }
}))
