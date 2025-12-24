# 🔬 AI Skin Disease Detection System

An AI-powered skin disease detection system that analyzes skin lesion images and classifies them into 8 categories. Features an interactive chatbot assistant to explain results, treatments, and causes.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🎯 Features

- **8-Class Skin Disease Classification**
  - Actinic Keratosis (precancerous)
  - Basal Cell Carcinoma (skin cancer)
  - Benign Keratosis (harmless)
  - Dermatofibroma (benign)
  - Melanocytic Nevus (common mole)
  - Melanoma (dangerous cancer)
  - Squamous Cell Carcinoma (skin cancer)
  - Vascular Lesion (blood vessel abnormality)

- **AI Chatbot Assistant** - Ask questions about your diagnosis
- **Detailed Information** - Causes, treatments, home care tips
- **Educational Resources** - YouTube videos, medical articles
- **Downloadable Reports** - Save analysis results

## 🚀 Live Demo

Try the app: [https://your-app-name.streamlit.app](https://your-app-name.streamlit.app)

## 📦 Deploy to Streamlit Cloud

### Quick Deploy

1. **Fork this repository**

2. **Go to [Streamlit Cloud](https://share.streamlit.io/)**

3. **Create new app:**
   - Repository: `your-username/skin-disease-detection`
   - Branch: `main`
   - Main file: `skin_app.py`

4. **Add secrets** (Settings → Secrets):
   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   ```
   Get a free key at: https://console.groq.com/keys

5. **Deploy!** The app will be live in ~2 minutes.

### Important Notes

- Chatbot requires Groq API key (free tier available)
- App loads instantly with pre-configured model

## 🖥️ Local Development

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/skin-disease-detection.git
cd skin-disease-detection
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# For Streamlit deployment (minimal)
pip install -r requirements-streamlit.txt

# For full development (includes training dependencies)
pip install -r requirements.txt
```

### 4. Set Up Environment

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 5. Run the App

```bash
streamlit run skin_app.py
```

Open http://localhost:8501 in your browser.

## 📁 Project Structure

```
skin-disease-detection/
├── skin_app.py              # Main Streamlit application
├── train_cpu.py             # Training script for CPU
├── train_optimized_colab.py # Training script for GPU (Colab)
├── requirements.txt         # Full dependencies (dev + training)
├── requirements-streamlit.txt # Minimal dependencies (deployment)
├── .streamlit/
│   └── config.toml          # Streamlit configuration
├── .env.example             # Environment template
├── .gitignore               # Git ignore rules
├── README.md                # This file
├── LICENSE                  # MIT License
└── models/
    └── final_model_optimized/ # Our trained model
```

## 🖥️ Usage

1. **Upload Image**: Click "Browse files" and select a skin lesion image (JPG, PNG)
2. **Analyze**: Click "🚀 Analyze Image" to get AI prediction
3. **View Results**: See the predicted condition, confidence score, and severity
4. **Chat**: Ask the AI assistant questions about your results
5. **Learn More**: Check the resources section for treatments and articles
6. **Download Report**: Save your analysis as a text file

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **ML Model**: Custom trained Swin Transformer (offline)
- **Inference**: PyTorch + Transformers
- **Chatbot**: Groq API with LLaMA 3.1
- **Dataset**: ISIC (International Skin Imaging Collaboration)

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Validation Accuracy | 85-92% |
| Top-3 Accuracy | 96-98% |

## ⚠️ Disclaimer

**This tool is for educational purposes only and is NOT a substitute for professional medical advice, diagnosis, or treatment.** Always consult a qualified healthcare professional for any skin concerns. Early detection saves lives - if you notice any suspicious skin changes, please see a dermatologist.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [ISIC Archive](https://www.isic-archive.com/) for the skin lesion dataset
- [HuggingFace](https://huggingface.co/) for model hosting
- [Groq](https://groq.com/) for free LLM API access
- Medical resources from Mayo Clinic, AAD, and Skin Cancer Foundation

---

⭐ If you found this project helpful, please give it a star!
