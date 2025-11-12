# Frontend Documentation

## Overview
React-based web application for diabetic retinopathy screening with a modern, responsive UI built with Tailwind CSS.

## Technology Stack
- React 18
- Tailwind CSS
- JavaScript (ES6+)

## Project Structure
```
frontend/
├── src/
│   ├── components/
│   │   ├── Header.js          # App header component
│   │   ├── ImageUpload.js     # Image upload and preview
│   │   └── ResultDisplay.js   # Prediction results display
│   ├── App.js                 # Main app component
│   ├── App.css                # App styles
│   └── index.css              # Global styles with Tailwind
├── public/                    # Static files
├── package.json               # Dependencies
└── tailwind.config.js        # Tailwind configuration
```

## Components

### Header
Displays the app title and branding.

### ImageUpload
- File selection with drag-and-drop area
- Image preview before upload
- File validation (type and size)
- Loading state during prediction

### ResultDisplay
- Shows prediction result with severity color coding
- Displays confidence percentage
- Shows all class probabilities
- Provides recommendations based on severity

## Running the Application

### Development
```bash
cd frontend
npm install
npm start
```

The app will open at `http://localhost:3000`

### Production Build
```bash
npm run build
```

### Docker
```bash
docker build -f frontend/Dockerfile -t dr-a-eye-frontend .
docker run -p 3000:3000 dr-a-eye-frontend
```

## API Integration

The frontend communicates with the Flask API at `http://localhost:5000`.

To change the API URL, update the fetch URL in `src/App.js`:
```javascript
const response = await fetch('http://localhost:5000/predict', {
  // ...
});
```

## Styling

The app uses Tailwind CSS for styling. Custom colors and themes can be configured in `tailwind.config.js`.

### Color Scheme
- Primary: Blue shades
- Success (No_DR): Green
- Warning (Mild/Moderate): Yellow/Orange
- Danger (Severe/Proliferate_DR): Red

## Features

1. **Image Upload**: Drag-and-drop or click to select
2. **Real-time Preview**: See selected image before analysis
3. **Loading States**: Visual feedback during prediction
4. **Error Handling**: User-friendly error messages
5. **Responsive Design**: Works on desktop and mobile
6. **Result Visualization**: Color-coded severity indicators
7. **Probability Display**: See confidence for all classes

## Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

