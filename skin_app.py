#!/usr/bin/env python3
"""
🔬 AI Skin Disease Checker - Advanced Deep Learning Model
"""

import streamlit as st
import numpy as np
from PIL import Image
from datetime import datetime
import os
from groq import Groq
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Page config
st.set_page_config(
    page_title="🔬 AI Skin Checker",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize states
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []
if 'results' not in st.session_state:
    st.session_state.results = None

# Load Groq API key (supports Streamlit Cloud secrets and .env file)
GROQ_API_KEY = ""

try:
    # Try Streamlit secrets first (for Streamlit Cloud deployment)
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
except:
    pass

if not GROQ_API_KEY:
    # Fallback to .env file (for local development)
    try:
        with open(".env", "r") as f:
            for line in f:
                if line.startswith("GROQ_API_KEY="):
                    GROQ_API_KEY = line.strip().split("=", 1)[1]
    except:
        pass

# Local model path - offline model only
LOCAL_MODEL_PATH = "models/final_model_optimized"

# Map model abbreviations to full names
LABEL_MAP = {
    'AK': 'Actinic Keratosis',
    'BCC': 'Basal Cell Carcinoma', 
    'BKL': 'Benign Keratosis',
    'DF': 'Dermatofibroma',
    'MEL': 'Melanoma',
    'NV': 'Melanocytic Nevus',
    'SCC': 'Squamous Cell Carcinoma',
    'VASC': 'Vascular Lesion'
}

# Disease info (8 classes from model)
DISEASE_INFO = {
    'Actinic Keratosis': {
        'name': 'Actinic Keratosis', 'emoji': '☀️', 'severity': 'Moderate',
        'description': 'A rough, scaly patch on the skin caused by years of sun exposure. It is considered precancerous and can develop into squamous cell carcinoma if left untreated. Most commonly appears on face, lips, ears, back of hands, forearms, scalp, and neck.',
        'causes': '☀️ Cumulative UV radiation damage from sun exposure or tanning beds over many years. Risk factors include: fair skin, history of sunburns, age over 40, living in sunny climates, weakened immune system, and outdoor occupations.',
        'treatments': 'Cryotherapy (freezing), 5-Fluorouracil Cream, Imiquimod, Photodynamic Therapy, Chemical Peels',
        'home_care': 'Apply SPF 50+ sunscreen daily, wear protective clothing and hats, avoid sun during peak hours (10am-4pm), perform regular skin self-exams',
        'youtube': 'https://www.youtube.com/results?search_query=actinic+keratosis+treatment',
        'articles': [
            {'title': 'Mayo Clinic Guide', 'url': 'https://www.mayoclinic.org/diseases-conditions/actinic-keratosis/symptoms-causes/syc-20354969'},
            {'title': 'Skin Cancer Foundation', 'url': 'https://www.skincancer.org/skin-cancer-information/actinic-keratosis/'},
        ]
    },
    'Basal Cell Carcinoma': {
        'name': 'Basal Cell Carcinoma', 'emoji': '⚠️', 'severity': 'High',
        'description': 'The most common type of skin cancer. It begins in the basal cells which produce new skin cells. Usually appears as a slightly transparent bump on sun-exposed skin. While rarely fatal, it can be highly disfiguring if not treated promptly.',
        'causes': '☀️ Primarily caused by long-term exposure to UV radiation from sunlight or tanning beds. DNA mutations in basal cells lead to uncontrolled growth. Risk factors: fair skin, chronic sun exposure, radiation therapy, immunosuppression, arsenic exposure, and genetic syndromes like Gorlin syndrome.',
        'treatments': 'Mohs Micrographic Surgery (most effective), Excisional Surgery, Curettage & Electrodesiccation, Radiation Therapy, Topical Medications',
        'home_care': 'See a dermatologist as soon as possible, document any changes with photos, protect the area from sun, avoid picking or scratching the lesion',
        'youtube': 'https://www.youtube.com/results?search_query=basal+cell+carcinoma+treatment',
        'articles': [
            {'title': 'Mayo Clinic Guide', 'url': 'https://www.mayoclinic.org/diseases-conditions/basal-cell-carcinoma/symptoms-causes/syc-20354187'},
            {'title': 'American Cancer Society', 'url': 'https://www.cancer.org/cancer/types/basal-and-squamous-cell-skin-cancer.html'},
        ]
    },
    'Benign Keratosis': {
        'name': 'Benign Keratosis (Seborrheic Keratosis)', 'emoji': '✅', 'severity': 'Low',
        'description': 'A common non-cancerous skin growth that appears as a waxy, wart-like spot. Often looks like it\'s "stuck on" the skin. These growths are harmless and don\'t become cancerous. They typically appear in middle age and increase with time.',
        'causes': '🧬 The exact cause is unknown, but they tend to run in families (genetic predisposition). They are NOT caused by sun exposure or viral infections. Risk factors: age (more common after 50), family history. They occur when skin cells called keratinocytes multiply excessively.',
        'treatments': 'Usually no treatment needed. If desired for cosmetic reasons: Cryotherapy, Curettage, Electrosurgery, Laser Treatment',
        'home_care': 'No special care required. Keep the area moisturized. Monitor for any sudden changes in appearance. Avoid irritating or scratching the growth.',
        'youtube': 'https://www.youtube.com/results?search_query=seborrheic+keratosis',
        'articles': [
            {'title': 'Mayo Clinic Guide', 'url': 'https://www.mayoclinic.org/diseases-conditions/seborrheic-keratosis/symptoms-causes/syc-20353878'},
            {'title': 'AAD Overview', 'url': 'https://www.aad.org/public/diseases/a-z/seborrheic-keratoses-overview'},
        ]
    },
    'Dermatofibroma': {
        'name': 'Dermatofibroma', 'emoji': '🔵', 'severity': 'Low',
        'description': 'A common benign skin growth that feels like a hard bump under the skin. Usually brownish to red-purple in color. Often develops after minor injuries like insect bites or small cuts. Completely harmless and does not turn into cancer.',
        'causes': '🦟 Often triggered by minor skin injuries such as insect bites, splinters, or small cuts. The body\'s healing response causes an overgrowth of fibrous tissue. More common in women and adults aged 20-50. Not caused by infection or sun exposure.',
        'treatments': 'Usually no treatment needed as they are harmless. If bothersome: Surgical excision, Cryotherapy, Laser removal (may leave a scar)',
        'home_care': 'Protect from repeated trauma or injury. Monitor for any significant changes in size or color. No special care required.',
        'youtube': 'https://www.youtube.com/results?search_query=dermatofibroma',
        'articles': [
            {'title': 'DermNet NZ', 'url': 'https://dermnetnz.org/topics/dermatofibroma'},
            {'title': 'Cleveland Clinic', 'url': 'https://my.clevelandclinic.org/health/diseases/24856-dermatofibroma'},
        ]
    },
    'Melanoma': {
        'name': 'Melanoma', 'emoji': '🚨', 'severity': 'Critical',
        'description': '⚠️ SERIOUS SKIN CANCER - The most dangerous form of skin cancer. Develops from melanocytes (pigment-producing cells). Can spread to other organs if not caught early. Look for the ABCDEs: Asymmetry, Border irregularity, Color variation, Diameter >6mm, Evolving.',
        'causes': '☀️🧬 Caused by DNA damage to melanocytes, primarily from UV radiation (sun/tanning beds). Risk factors: intense sun exposure & sunburns (especially in childhood), many moles (50+), atypical moles, fair skin, family history of melanoma, weakened immune system, and certain genetic mutations (CDKN2A, BRAF).',
        'treatments': 'Wide Excision Surgery, Sentinel Lymph Node Biopsy, Immunotherapy (Keytruda, Opdivo), Targeted Therapy, Radiation Therapy, Chemotherapy',
        'home_care': '🚨 SEEK IMMEDIATE MEDICAL ATTENTION! Do not delay - early detection saves lives. Document the lesion with photos. Avoid sun exposure. Do not attempt to remove or treat at home.',
        'youtube': 'https://www.youtube.com/results?search_query=melanoma+warning+signs+ABCDE',
        'articles': [
            {'title': 'National Cancer Institute', 'url': 'https://www.cancer.gov/types/skin/patient/melanoma-treatment-pdq'},
            {'title': 'American Cancer Society', 'url': 'https://www.cancer.org/cancer/types/melanoma-skin-cancer.html'},
            {'title': 'Melanoma Research Foundation', 'url': 'https://melanoma.org/'},
        ]
    },
    'Melanocytic Nevus': {
        'name': 'Melanocytic Nevus (Common Mole)', 'emoji': '🟤', 'severity': 'Low',
        'description': 'A common benign growth on the skin, commonly known as a mole. Formed by clusters of melanocytes (pigment cells). Most people have 10-40 moles. They can be flat or raised, and range from pink to dark brown. Most moles are harmless.',
        'causes': '🧬☀️ Moles form when melanocytes grow in clusters instead of spreading throughout the skin. Caused by a combination of genetics and sun exposure. Most develop during childhood and adolescence. New moles can appear with sun exposure or hormonal changes (pregnancy, puberty).',
        'treatments': 'Usually no treatment needed. Removal options if desired: Surgical excision, Shave removal. Remove if showing suspicious changes (ABCDE criteria).',
        'home_care': 'Perform monthly self-exams using the ABCDE rule. Protect moles from sun exposure. Take photos to track any changes over time. See a dermatologist annually.',
        'youtube': 'https://www.youtube.com/results?search_query=mole+skin+check+ABCDE',
        'articles': [
            {'title': 'Mayo Clinic Guide', 'url': 'https://www.mayoclinic.org/diseases-conditions/moles/symptoms-causes/syc-20375200'},
            {'title': 'AAD Mole Guide', 'url': 'https://www.aad.org/public/diseases/a-z/moles-overview'},
        ]
    },
    'Squamous Cell Carcinoma': {
        'name': 'Squamous Cell Carcinoma', 'emoji': '⚠️', 'severity': 'High',
        'description': 'The second most common form of skin cancer. Develops in the squamous cells that make up the middle and outer layers of skin. Usually caused by cumulative UV exposure. Can spread to other parts of the body if not treated, though this is uncommon.',
        'causes': '☀️ Primarily caused by cumulative, long-term UV exposure from sun or tanning beds. Can also develop from actinic keratoses (precancerous lesions). Risk factors: fair skin, history of sunburns, chronic sun exposure, weakened immune system, HPV infection, exposure to chemicals (arsenic), and previous radiation therapy.',
        'treatments': 'Mohs Micrographic Surgery, Excisional Surgery, Curettage & Electrodesiccation, Radiation Therapy, Topical Chemotherapy (5-FU), Photodynamic Therapy',
        'home_care': 'See a dermatologist urgently for evaluation. Protect the area from sun. Document changes with photos. Avoid picking or irritating the lesion.',
        'youtube': 'https://www.youtube.com/results?search_query=squamous+cell+carcinoma+skin',
        'articles': [
            {'title': 'Mayo Clinic Guide', 'url': 'https://www.mayoclinic.org/diseases-conditions/squamous-cell-carcinoma/symptoms-causes/syc-20352480'},
            {'title': 'Skin Cancer Foundation', 'url': 'https://www.skincancer.org/skin-cancer-information/squamous-cell-carcinoma/'},
        ]
    },
    'Vascular Lesion': {
        'name': 'Vascular Lesion', 'emoji': '❤️', 'severity': 'Low',
        'description': 'An abnormality of blood vessels in or under the skin. Includes cherry angiomas (small red dots), spider veins, and hemangiomas. These are almost always benign and primarily a cosmetic concern. Very common, especially with aging.',
        'causes': '🧬 Most vascular lesions develop due to aging, genetics, or hormonal changes. Cherry angiomas increase with age (common after 30). Spider veins can be caused by sun exposure, hormonal changes, or prolonged standing. Some are present at birth (hemangiomas). NOT caused by injury or infection.',
        'treatments': 'Usually no treatment needed. Cosmetic options: Pulsed Dye Laser, IPL (Intense Pulsed Light), Sclerotherapy, Electrocautery',
        'home_care': 'Protect from injury to prevent bleeding. No special care required. Cosmetic treatment is optional. Monitor for any rapid growth or changes.',
        'youtube': 'https://www.youtube.com/results?search_query=cherry+angioma+vascular+lesion',
        'articles': [
            {'title': 'DermNet NZ', 'url': 'https://dermnetnz.org/topics/vascular-lesions'},
            {'title': 'Cleveland Clinic', 'url': 'https://my.clevelandclinic.org/health/diseases/17893-cherry-angiomas'},
        ]
    },
}

# CSS Styles
st.markdown("""
<style>
    .main-title {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2rem;
        font-weight: 700;
        text-align: center;
    }
    .result-danger { background: linear-gradient(135deg, #dc2626, #b91c1c); color: white; padding: 1rem; border-radius: 12px; }
    .result-success { background: linear-gradient(135deg, #059669, #047857); color: white; padding: 1rem; border-radius: 12px; }
    .result-warning { background: linear-gradient(135deg, #d97706, #b45309); color: white; padding: 1rem; border-radius: 12px; }
    .resource-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .resource-card a { color: #6366f1; text-decoration: none; font-weight: 600; }
    .resource-card a:hover { text-decoration: underline; }
</style>
""", unsafe_allow_html=True)


# ===== MODEL LOADING =====
@st.cache_resource
def load_model():
    """Load our trained skin cancer classifier model (offline only)"""
    try:
        # Load our trained model from local files only
        if not os.path.exists(LOCAL_MODEL_PATH):
            st.error("❌ Model files not found. Please ensure the trained model is available in the models folder.")
            st.stop()
        
        processor = AutoImageProcessor.from_pretrained(LOCAL_MODEL_PATH, local_files_only=True)
        model = AutoModelForImageClassification.from_pretrained(LOCAL_MODEL_PATH, local_files_only=True)
        model.eval()
        return processor, model
        
    except Exception as e:
        st.error(f"❌ Failed to load the trained model: {str(e)}")
        st.error("Please check that all model files are present and valid.")
        st.stop()


def predict_skin_condition(image, processor, model):
    """Run prediction using the AI model"""
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)[0]
    
    predicted_idx = logits.argmax(-1).item()
    predicted_abbr = model.config.id2label[predicted_idx]
    predicted_label = LABEL_MAP.get(predicted_abbr, predicted_abbr)  # Convert to full name
    confidence = probs[predicted_idx].item()
    
    # Get top 3 predictions with full names
    top_indices = torch.argsort(probs, descending=True)[:3]
    top_3 = []
    for idx in top_indices:
        abbr = model.config.id2label[idx.item()]
        full_name = LABEL_MAP.get(abbr, abbr)
        top_3.append((full_name, probs[idx].item()))
    
    return {
        'disease': predicted_label,
        'confidence': confidence,
        'top_3': top_3
    }


def get_bot_response(message, disease_info):
    """Get chatbot response from Groq"""
    if not GROQ_API_KEY:
        return "⚠️ Add your Groq API key to .env file. Get free key at: console.groq.com/keys"
    
    try:
        client = Groq(api_key=GROQ_API_KEY)
        system = f"""You are SkinBot, a friendly dermatology assistant. 🤖

Current diagnosis:
- Condition: {disease_info.get('name', 'Unknown')}
- Severity: {disease_info.get('severity', 'Unknown')}
- Description: {disease_info.get('description', '')}
- Causes: {disease_info.get('causes', '')}
- Treatments: {disease_info.get('treatments', '')}
- Home Care: {disease_info.get('home_care', '')}

Rules:
- Be friendly, use emojis 😊
- Keep responses concise (2-4 sentences)
- Always recommend seeing a dermatologist
- For serious conditions, strongly urge professional consultation"""
        
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": message}
            ],
            max_tokens=300,
            temperature=0.7
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"❌ Error: {str(e)[:100]}"


def get_disease_info(disease_label):
    """Get disease info from label"""
    # Direct match first
    if disease_label in DISEASE_INFO:
        return DISEASE_INFO[disease_label]
    
    # Try case-insensitive match
    for key in DISEASE_INFO:
        if key.lower() == disease_label.lower():
            return DISEASE_INFO[key]
    
    # Fallback
    return {
        'name': disease_label, 'emoji': '🔍', 'severity': 'Unknown',
        'description': 'Skin condition detected. Please consult a dermatologist for proper evaluation.',
        'treatments': 'Consult a dermatologist',
        'home_care': 'Monitor for changes',
        'youtube': f'https://www.youtube.com/results?search_query={disease_label.replace(" ", "+")}',
        'articles': []
    }


# ===== MAIN APP =====
def main():
    # ===== SIDEBAR - CHATBOT =====
    with st.sidebar:
        st.markdown("## 🤖 SkinBot Assistant")
        
        if st.session_state.results:
            disease = st.session_state.results['disease']
            info = get_disease_info(disease)
            st.caption(f"💬 Discussing: **{info.get('name', disease)}**")
        else:
            info = {}
            st.caption("💬 Upload an image to start chatting")
        
        st.markdown("---")
        
        chat_container = st.container(height=300)
        with chat_container:
            if not st.session_state.chat_messages:
                st.markdown("👋 **SkinBot:** Hi! I'm your AI dermatology assistant. Upload a skin image and I'll help explain the results!")
            else:
                for msg in st.session_state.chat_messages:
                    if msg["role"] == "bot":
                        st.markdown(f"🤖 **SkinBot:** {msg['content']}")
                    else:
                        st.markdown(f"👤 **You:** {msg['content']}")
        
        with st.form(key="chat_form", clear_on_submit=True):
            user_msg = st.text_input("Type your question...", placeholder="e.g., What treatments are available?")
            col1, col2 = st.columns([4, 1])
            with col1:
                send = st.form_submit_button("Send 📤", use_container_width=True)
            with col2:
                clear = st.form_submit_button("🗑️")
        
        if send and user_msg:
            if st.session_state.results:
                st.session_state.chat_messages.append({"role": "user", "content": user_msg})
                response = get_bot_response(user_msg, info)
                st.session_state.chat_messages.append({"role": "bot", "content": response})
                st.rerun()
            else:
                st.warning("Please upload and analyze an image first!")
        
        if clear:
            st.session_state.chat_messages = []
            st.rerun()
        
        st.markdown("---")
        st.caption("⚠️ I'm AI, not a doctor. Always consult professionals.")

    # ===== MAIN CONTENT =====
    st.markdown('<h1 class="main-title">🔬 AI Skin Disease Checker</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center;color:#64748b;">Upload a skin image • Get AI analysis • Chat with SkinBot</p>', unsafe_allow_html=True)
    
    with st.spinner("Loading AI model..."):
        processor, model = load_model()
    
    if processor is None or model is None:
        st.error("❌ Failed to load model. Check your internet connection for first run.")
        return
    
    # Upload and Analysis
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Image")
        uploaded = st.file_uploader("Choose skin image", type=['png', 'jpg', 'jpeg'], label_visibility="collapsed")
        
        if uploaded:
            image = Image.open(uploaded).convert('RGB')
            st.image(image, use_container_width=True)
    
    with col2:
        st.subheader("🔍 Analysis Results")
        
        if uploaded:
            if st.button("🚀 Analyze Image", use_container_width=True, type="primary"):
                with st.spinner("🔬 Analyzing..."):
                    st.session_state.results = predict_skin_condition(image, processor, model)
                    st.session_state.chat_messages = []
                st.rerun()
        
        if st.session_state.results:
            results = st.session_state.results
            disease = results['disease']
            confidence = results['confidence']
            info = get_disease_info(disease)
            
            severity = info.get('severity', 'Low')
            emoji = info.get('emoji', '🔍')
            
            if severity == 'Critical':
                st.markdown(f'''<div class="result-danger">
                    <h3>{emoji} {info.get('name', disease)}</h3>
                    <p>{info.get('description', '')}</p>
                    <p><b>Confidence: {confidence:.1%}</b></p>
                </div>''', unsafe_allow_html=True)
            elif severity == 'High':
                st.markdown(f'''<div class="result-warning">
                    <h3>{emoji} {info.get('name', disease)}</h3>
                    <p>{info.get('description', '')}</p>
                    <p><b>Confidence: {confidence:.1%}</b></p>
                </div>''', unsafe_allow_html=True)
            else:
                st.markdown(f'''<div class="result-success">
                    <h3>{emoji} {info.get('name', disease)}</h3>
                    <p>{info.get('description', '')}</p>
                    <p><b>Confidence: {confidence:.1%}</b></p>
                </div>''', unsafe_allow_html=True)
            
            st.markdown("**Top 3 Predictions:**")
            for name, prob in results['top_3']:
                st.progress(prob, text=f"{name}: {prob:.1%}")


    # ===== RESOURCES SECTION =====
    if st.session_state.results:
        results = st.session_state.results
        disease = results['disease']
        info = get_disease_info(disease)
        
        st.markdown("---")
        st.subheader(f"📚 Resources for {info.get('name', disease)}")
        
        # Causes section - NEW
        st.markdown("### ❓ Why Does This Occur?")
        st.warning(info.get('causes', 'Causes vary. Please consult a dermatologist for more information.'))
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 💊 Treatments")
            st.info(info.get('treatments', 'Consult a dermatologist'))
            
            st.markdown("### 🏠 Home Care")
            st.success(info.get('home_care', 'Monitor for changes'))
        
        with col2:
            st.markdown("### 📺 Video Resources")
            youtube_url = info.get('youtube', f"https://www.youtube.com/results?search_query={disease.replace(' ', '+')}")
            st.markdown(f'''
            <div class="resource-card">
                <p>🎥 <b>Educational Videos</b></p>
                <a href="{youtube_url}" target="_blank">▶️ Watch on YouTube →</a>
            </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            st.markdown("### 📖 Articles")
            articles = info.get('articles', [])
            if articles:
                for article in articles:
                    st.markdown(f'''
                    <div class="resource-card">
                        <a href="{article['url']}" target="_blank">📄 {article['title']} →</a>
                    </div>
                    ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                <div class="resource-card">
                    <a href="https://www.google.com/search?q={disease.replace(' ', '+')}+skin" target="_blank">🔍 Search for more →</a>
                </div>
                ''', unsafe_allow_html=True)
        
        # Download Report
        st.markdown("---")
        report = f"""SKIN ANALYSIS REPORT
{'='*50}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}

RESULTS
-------
Condition: {info.get('name', disease)}
Confidence: {results['confidence']:.1%}
Severity: {info.get('severity', 'Unknown')}

Top 3 Predictions:
{chr(10).join([f"  {i+1}. {name}: {prob:.1%}" for i, (name, prob) in enumerate(results['top_3'])])}

INFORMATION
-----------
Description: {info.get('description', '')}

Why Does This Occur?
{info.get('causes', 'Consult a dermatologist for more information.')}

Treatments: {info.get('treatments', '')}
Home Care: {info.get('home_care', '')}

⚠️ DISCLAIMER
This AI analysis is for educational purposes only.
Always consult a qualified healthcare professional.
"""
        st.download_button("📥 Download Report", report, f"skin_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt", use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.warning("⚠️ **Medical Disclaimer:** This AI tool is for educational purposes only. Always consult a qualified healthcare professional for diagnosis and treatment.")


if __name__ == "__main__":
    main()
