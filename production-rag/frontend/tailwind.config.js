/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        ground: '#0F1620',
        panel: { DEFAULT: '#161F2B', raised: '#1C2733', sunken: '#111A24' },
        line: { DEFAULT: '#283644', soft: '#1E2A36' },
        ink: { DEFAULT: '#E6EDF3', muted: '#8B9AAD', faint: '#5C6B7E' },
        signal: '#3FB6C9',
        dense: '#5B9BD5',
        sparse: '#A78BFA',
        pass: '#3FB37F',
        warn: '#E0A33E',
        danger: '#E5705A',
      },
      fontFamily: {
        display: ['"Space Grotesk"', 'system-ui', 'sans-serif'],
        mono: ['"IBM Plex Mono"', 'ui-monospace', 'monospace'],
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      boxShadow: {
        panel:
          '0 1px 0 0 rgba(255,255,255,0.02) inset, 0 8px 24px -12px rgba(0,0,0,0.6)',
      },
    },
  },
  plugins: [],
};
