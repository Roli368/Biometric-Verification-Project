/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}", // Added ts/tsx in case you switch to TypeScript later
  ],
  theme: {
    extend: {
      colors: {
        // Deep background and Slate tones for the "Vault" look
        slate: {
          950: '#020617',
          900: '#0f172a',
          800: '#1e293b',
        },
        // Primary Action and Brand Colors
        brand: {
          blue: '#3b82f6',
          cyan: '#06b6d4',
          success: '#10b981',
          danger: '#ef4444',
        }
      },
      animation: {
        // Smooth scanning animation for the biometric oval/box
        'scan-slow': 'scan 3s ease-in-out infinite',
        'pulse-fast': 'pulse 1.5s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
      keyframes: {
        scan: {
          '0%, 100%': { top: '5%' },
          '50%': { top: '90%' },
        }
      },
      backgroundImage: {
        // For the subtle grainy/tech texture in the background
        'tech-grid': "url('https://www.transparenttextures.com/patterns/carbon-fibre.png')",
      }
    },
  },
  plugins: [],
};