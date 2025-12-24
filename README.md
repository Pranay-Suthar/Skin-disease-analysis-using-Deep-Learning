# 🔬 AI Skin Disease Detection System

An AI-powered skin disease detection system that analyzes skin lesion images and classifies them into 8 categories. Features an interactive chatbot assistant to explain results, treatments, and causes.

## 📁 Model Files

**Important:** The trained model files are too large (>100MB) for GitHub, so they are hosted on Google Drive:

**📥 Download Models:** [Google Drive Link](https://drive.google.com/drive/folders/1lAyxcm0465c5LmyNsOG6i96aUx2LbkaZ?usp=sharing)

Please download the model files and place them in the `models/final_model_optimized/` folder before running the application.

## 👥 Project Team

- **Paresh Suva** - Team Leader
- **Pranay Suthar** - Developer
- **Krish Zalavadiya** - Developer  
- **Samarth Patel** - Developer

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

Run the application locally using the instructions below.

## �️ Local Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Pranay-Suthar/Skin-disease-analysis-using-Deep-Learning.git
cd Skin-disease-analysis-using-Deep-Learning
```

### 2. Download Model Files

Download the trained model from [Google Drive](https://drive.google.com/drive/folders/1lAyxcm0465c5LmyNsOG6i96aUx2LbkaZ?usp=sharing) and extract to:
```
models/final_model_optimized/
├── config.json
├── model.safetensors
└── preprocessor_config.json
```

### 3. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 4. Install Dependencies

```bash
# For running the app
pip install -r requirements-streamlit.txt

# For full development (includes training dependencies)
pip install -r requirements.txt
```

### 5. Set Up Environment (Optional - for chatbot)

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 6. Run the App

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

## 🔄 Using Your Own Model

If you have trained your own skin disease classification model, you can easily integrate it:

### Option 1: Replace Existing Model
1. Place your model files in `models/final_model_optimized/`
2. Ensure files are named: `config.json`, `model.safetensors`, `preprocessor_config.json`
3. Update the `LABEL_MAP` in `skin_app.py` if your classes are different

### Option 2: Add New Model Path
1. Update `LOCAL_MODEL_PATH` in `skin_app.py`:
   ```python
   LOCAL_MODEL_PATH = "models/your_model_folder"
   ```
2. Place your model files in the new folder
3. Restart the application

### Supported Model Formats
- HuggingFace Transformers format (recommended)
- PyTorch models with AutoImageProcessor
- Models with 8-class skin disease classification

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
