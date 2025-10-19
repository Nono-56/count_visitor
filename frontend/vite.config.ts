import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5423,
    strictPort: true,
    host: '0.0.0.0',
  },
  preview: {
    port: 5423,
    host: '0.0.0.0',
  },
})

