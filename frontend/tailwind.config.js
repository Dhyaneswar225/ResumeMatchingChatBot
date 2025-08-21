// tailwind.config.js
module.exports = {
  content: [
    './public/**/*.{html,js}', // scan HTML + JS
 './public/script.js',  
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          light: '#a7f3d0',
          DEFAULT: '#10b981',
          dark: '#047857',
        },
        sunset: {
          light: '#fbbf24',
          DEFAULT: '#f59e0b',
          dark: '#b45309',
        },
      },
    },
  },
  plugins: [],
};
