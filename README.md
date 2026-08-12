# DermaAI: AI-Powered Skin Disease Detection

![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![React](https://img.shields.io/badge/React-19.2.8-61DAFB?logo=react&logoColor=black)
![Tailwind CSS](https://img.shields.io/badge/Tailwind-3.4.0-38B2AC?logo=tailwind-css&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0.0-000000?logo=flask&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.13.0-EE4C2C?logo=pytorch&logoColor=white)

DermaAI is a comprehensive, state-of-the-art web application designed to help users identify potential skin conditions using advanced Deep Learning computer vision models. It provides instant analysis, curated educational resources, and helps users locate nearby dermatology clinics.

---

## Key Features

- **Instant AI Analysis (Live Model & Webcam)**: Upload a skin image or take a live photo using the native in-browser webcam integration. Receive an immediate prediction across 8 different skin conditions from a real PyTorch backend model, complete with confidence scores and medical context.
- **Downloadable PDF Reports**: Instantly generate and download a comprehensive, dynamically styled A4 PDF report containing the AI diagnosis, uploaded image, recommended home remedies, relevant YouTube videos, and nearby hospitals with clickable Google Maps links.
- **Conversational AI Assistant**: An integrated chat interface (powered by Groq LLM) allows users to ask follow-up questions about their skin conditions and get human-like, conversational guidance.
- **Smart Hospital Locator**: Automatically detects your location (or accepts manual search) and displays nearby dermatology clinics in a beautiful, premium grid layout with Google Maps routing.
- **Rich Educational Resources**: A curated library of skin-health articles and auto-playing YouTube video tutorials, complete with a "Home Remedies & Daily Care" section.
- **Premium Coral UI/UX**: Built with React, TailwindCSS, and Framer Motion, featuring a warm, vibrant Coral & Amber-based color palette, glassmorphism, dynamic gradients, micro-interactions, and a fully responsive design.

---

## Diseases Detected

The deep learning model is trained to identify the following 8 conditions:
- **Melanoma** (Critical)
- **Basal Cell Carcinoma** (High Risk)
- **Squamous Cell Carcinoma** (High Risk)
- **Actinic Keratosis** (Moderate Risk)
- **Dermatofibroma** (Low Risk / Benign)
- **Benign Keratosis** (Low Risk / Benign)
- **Melanocytic Nevus** (Low Risk / Benign)
- **Vascular Lesion** (Low Risk / Benign)

---

## Quick Start (Windows)

1. Double-click the `run_project.bat` file in the root directory.
2. The script will automatically start both the backend API and the frontend server.
3. Open your browser and navigate to `http://localhost:5173`.

---

## Manual Setup (One Time Only)

### Prerequisites
- Python 3.12+ 
- Node.js 18+ with npm

### 1. Install Dependencies
Open Command Prompt in the project root:
```bash
# Setup Python Virtual Environment and install backend packages
python -m venv venv
venv\Scripts\activate
pip install -r backend/requirements.txt

# Install frontend packages
cd frontend
npm install
```

### 2. Environment Variables
In the project root, create a `.env` file:
```env
GROQ_API_KEY=your_key_from_https://console.groq.com/
FLASK_ENV=development
```

### 3. Run the Servers

**Terminal 1 (Backend):**
```bash
cd backend
..\venv\Scripts\activate
python app.py
```
*(Runs on http://127.0.0.1:5000)*

**Terminal 2 (Frontend):**
```bash
cd frontend
npm run dev
```
*(Runs on http://localhost:5173)*

---

## Project Structure

```text
Skin_App/
├── backend/                  # Flask API Server
│   ├── app.py                # Main API routing and logic
│   ├── requirements.txt      # Python dependencies
│   └── uploads/              # Temporary image storage
├── frontend/                 # React UI Application
│   ├── src/
│   │   ├── pages/            # Main application views
│   │   │   ├── Home.tsx      # Landing page
│   │   │   ├── Analyze.tsx   # Image upload & Chat interface
│   │   │   ├── Resources.tsx # Articles, Videos & Remedies
│   │   │   ├── Hospitals.tsx # Clinic locator
│   │   │   └── About.tsx     # Mission & Terms
│   │   ├── components/       # Reusable UI components
│   │   └── utils/            # API hooks and geolocation
│   └── package.json
├── models/                   # PyTorch/HuggingFace model weights
└── README.md
```

---

## API Endpoints

- `POST /api/predict/` - Upload an image for deep-learning analysis.
- `POST /api/chat/` - Send a message to the Groq-powered AI skin assistant.
- `POST /api/hospitals/` - Fetch nearby hospitals using geographic coordinates or city names.
- `POST /api/login/` & `/api/signup/` - User authentication (mock/local).

---

## Tech Stack

Here are the core technologies driving DermaAI:

<div align="center">
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/react/react-original.svg" height="40" alt="react logo"  />
  <img width="12" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/tailwindcss/tailwindcss-original-wordmark.svg" height="40" alt="tailwindcss logo"  />
  <img width="12" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" height="40" alt="python logo"  />
  <img width="12" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/flask/flask-original.svg" height="40" alt="flask logo"  />
  <img width="12" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pytorch/pytorch-original.svg" height="40" alt="pytorch logo"  />
  <img width="12" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/nodejs/nodejs-original.svg" height="40" alt="nodejs logo"  />
</div>

**Frontend:**
- **Framework**: React (v19) + Vite
- **Styling**: Tailwind CSS (v3.4)
- **Icons**: Lucide React & Iconify
- **Animations**: Framer Motion
- **PDF Generation**: html2pdf.js

**Backend:**
- **Framework**: Flask (v3.0)
- **Machine Learning**: PyTorch (v2.13), Transformers (HuggingFace)
- **LLM Integration**: Groq API

---

## Medical Disclaimer

**This application is designed for educational and informational purposes only.** It is not a substitute for professional medical advice, diagnosis, or treatment. The AI models can make mistakes. Always consult a qualified healthcare provider with any questions you may have regarding a medical condition.

---

## License

MIT License - Free to use and modify.
