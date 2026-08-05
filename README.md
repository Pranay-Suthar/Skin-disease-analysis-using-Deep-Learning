# AI Skin Disease Detection App

## Quick Start
1. Double-click `run.bat` - that's it!

## What You Need
- Python 3.12.10 (already installed)
- Node.js 18+ with npm (for frontend)

## Setup (One Time Only)

### Step 1: Install Dependencies
Open Command Prompt and run:
```bash
cd d:\Code-Editors\VS-Code\Python\Skin_App
venv\Scripts\activate
pip install -r backend/requirements.txt
cd frontend
npm install
```

### Step 2: Create .env File
In project root (`d:\Code-Editors\VS-Code\Python\Skin_App`), create `.env`:
```
GROQ_API_KEY=your_key_from_https://console.groq.com/
FLASK_ENV=development
```

### Step 3: Run Everything
Double-click `run.bat` or open two terminals:

**Terminal 1 (Backend):**
```bash
cd backend
..\venv\Scripts\activate
python app.py
```
Backend: http://127.0.0.1:5000

**Terminal 2 (Frontend):**
```bash
cd frontend
npm run dev
```
Frontend: http://localhost:5173

## How to Use
1. Upload a skin image (PNG, JPG)
2. Get instant AI prediction with disease name, confidence %, treatments
3. Find nearby hospitals/clinics by entering a city
4. Chat with SkinBot about your condition

## Features
✅ AI-powered skin disease detection (8 conditions)
✅ Hospital/clinic finder with directions
✅ SkinBot AI chatbot (powered by Groq)
✅ Real-time image analysis
✅ Beautiful dark theme UI
✅ CORS enabled for API access

## Diseases Detected
- Melanoma (Critical)
- Basal Cell Carcinoma (High)
- Squamous Cell Carcinoma (High)
- Actinic Keratosis (Moderate)
- Dermatofibroma (Low)
- Benign Keratosis (Low)
- Melanocytic Nevus (Low)
- Vascular Lesion (Low)

## Project Structure
```
Skin_App/
├── backend/
│   ├── app.py (Flask API)
│   ├── requirements.txt
│   ├── uploads/ (saved images)
│   └── config/ (Django config - not used)
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── ImageUploader.jsx
│   │   │   ├── ResultsDisplay.jsx
│   │   │   ├── HospitalLocator.jsx
│   │   │   └── SkinBotChat.jsx
│   │   └── index.css
│   └── package.json
├── models/
│   └── final_model_optimized/ (ML model files)
├── venv/ (Python virtual environment)
├── run.bat (Start both servers)
└── README.md (this file)
```

## API Endpoints
- `POST /api/predict/` - Upload image for analysis
- `POST /api/hospitals/` - Find hospitals by location
- `POST /api/chat/` - Chat with SkinBot
- `GET /api/health/` - Health check

## Troubleshooting

**Backend not responding:**
- Check Flask is running on http://127.0.0.1:5000
- Check terminal for error messages

**Image upload fails:**
- Ensure Flask backend is running
- Try uploading a different image format (PNG, JPG)
- Check browser console (F12) for errors

**Hospital search returns no results:**
- Check internet connection (uses OpenStreetMap)
- Try a different city name
- Use format like "New York, NY" or "London, UK"

**Chat not working:**
- Verify `.env` file has correct GROQ_API_KEY
- Check internet connection

**Frontend not loading:**
- Clear browser cache (Ctrl+Shift+Delete)
- Hard refresh (Ctrl+Shift+R)
- Check npm is running on port 5173

## Tech Stack
**Backend:**
- Flask 3.0.0
- PyTorch 2.13.0
- Transformers (HuggingFace)
- Groq API (LLM)

**Frontend:**
- React 19.2.8
- Vite 8.2.0
- Tailwind CSS 3.4.0
- Lucide React (icons)

## Medical Disclaimer
⚠️ This app is for **educational purposes only**. Always consult a qualified healthcare professional for medical advice.

## License
MIT - Free to use and modify

---

**Questions?** Check the code or error messages in the console!
