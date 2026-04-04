import type { Config } from 'tailwindcss'

const config: Config = {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        bg: '#0B0D12',
        surface: '#12151F',
        surface2: '#191D2A',
        surface3: '#21263A',
        border: '#252B3B',
        border2: '#2E3650',
        text1: '#EAE6DC',
        text2: '#7A8499',
        text3: '#4E566A',
        teal: '#0DC5A0',
        amber: '#E8935A',
        danger: '#E05252',
        positive: '#3DBE8C',
      },
      fontFamily: {
        sans: ['Geist', 'system-ui', 'sans-serif'],
        mono: ['Geist Mono', 'monospace'],
      },
      borderRadius: {
        DEFAULT: '2px',
        sm: '2px',
        md: '2px',
        lg: '2px',
        xl: '2px',
        '2xl': '2px',
        full: '9999px',
      },
    },
  },
  plugins: [],
}

export default config
