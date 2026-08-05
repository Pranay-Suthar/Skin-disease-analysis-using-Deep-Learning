/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Brand
        primary:   { DEFAULT: '#7c3aed', light: '#ede9fe', dark: '#5b21b6' },
        accent:    { DEFAULT: '#06b6d4', light: '#ecfeff' },
        // Neutrals
        background: '#ffffff',
        surface:    '#f9fafb',
        muted:      '#f3f4f6',
        border:     '#e5e7eb',
        // Status
        success: '#10b981',
        danger:  '#ef4444',
        warning: '#f59e0b',
      },
      fontFamily: {
        sans:    ['Inter', 'system-ui', 'sans-serif'],
        display: ['"Plus Jakarta Sans"', 'Inter', 'sans-serif'],
      },
      borderRadius: {
        xl:  '12px',
        '2xl': '16px',
        '3xl': '24px',
      },
      boxShadow: {
        sm:   '0 1px 3px rgba(0,0,0,.06), 0 1px 2px rgba(0,0,0,.04)',
        md:   '0 4px 12px rgba(0,0,0,.08)',
        lg:   '0 10px 30px rgba(0,0,0,.10)',
        glow: '0 0 24px rgba(124,58,237,.25)',
        'glow-lg': '0 0 40px rgba(124,58,237,.35)',
      },
      animation: {
        fadeIn:    'fadeIn .5s ease-out',
        slideUp:   'slideUp .5s ease-out',
        slideDown: 'slideDown .35s ease-out',
        shimmer:   'shimmer 2s infinite',
      },
      keyframes: {
        fadeIn:    { from:{ opacity:'0' },                                to:{ opacity:'1' } },
        slideUp:   { from:{ transform:'translateY(20px)', opacity:'0' },  to:{ transform:'translateY(0)', opacity:'1' } },
        slideDown: { from:{ transform:'translateY(-10px)', opacity:'0' }, to:{ transform:'translateY(0)', opacity:'1' } },
        shimmer:   { from:{ backgroundPosition:'-800px 0' },              to:{ backgroundPosition:'800px 0' } },
      },
    },
  },
  plugins: [],
}
