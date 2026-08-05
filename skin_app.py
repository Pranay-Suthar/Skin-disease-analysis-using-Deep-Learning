#!/usr/bin/env python3
"""
🔬 AI Skin Disease Checker - Advanced Deep Learning Model
"""

import streamlit as st
import numpy as np
from PIL import Image
from datetime import datetime
import os
import math
import requests
import io
import re
from groq import Groq
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

st.set_page_config(
    page_title="AI Skin Checker",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []
if 'results' not in st.session_state:
    st.session_state.results = None
if 'user_location' not in st.session_state:
    st.session_state.user_location = None
if 'hospitals' not in st.session_state:
    st.session_state.hospitals = None
if 'location_query' not in st.session_state:
    st.session_state.location_query = ""

GROQ_API_KEY = ""
try:
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
except:
    pass
if not GROQ_API_KEY:
    try:
        with open(".env", "r") as f:
            for line in f:
                if line.startswith("GROQ_API_KEY="):
                    GROQ_API_KEY = line.strip().split("=", 1)[1]
    except:
        pass

LOCAL_MODEL_PATH = "models/final_model_optimized"
LABEL_MAP = {
    'AK': 'Actinic Keratosis', 'BCC': 'Basal Cell Carcinoma',
    'BKL': 'Benign Keratosis', 'DF': 'Dermatofibroma', 'MEL': 'Melanoma',
    'NV': 'Melanocytic Nevus', 'SCC': 'Squamous Cell Carcinoma', 'VASC': 'Vascular Lesion'
}

DISEASE_INFO = {
    'Actinic Keratosis': {
        'name': 'Actinic Keratosis', 'icon': 'fa-solid fa-sun', 'severity': 'Moderate',
        'description': 'A rough, scaly patch on the skin caused by years of sun exposure. It is considered precancerous and can develop into squamous cell carcinoma if left untreated.',
        'causes': '<i class="fa-solid fa-sun icon-inline"></i> Cumulative UV radiation damage from sun exposure or tanning beds over many years. Risk factors include: fair skin, history of sunburns, age over 40, living in sunny climates.',
        'treatments': 'Cryotherapy (freezing), 5-Fluorouracil Cream, Imiquimod, Photodynamic Therapy, Chemical Peels',
        'home_care': 'Apply SPF 50+ sunscreen daily, wear protective clothing and hats, avoid sun during peak hours (10am-4pm), perform regular skin self-exams',
        'youtube': 'https://www.youtube.com/results?search_query=actinic+keratosis+treatment',
        'articles': [
            {'title': 'Mayo Clinic Guide', 'url': 'https://www.mayoclinic.org/diseases-conditions/actinic-keratosis/symptoms-causes/syc-20354969'},
            {'title': 'Skin Cancer Foundation', 'url': 'https://www.skincancer.org/skin-cancer-information/actinic-keratosis/'},
        ]
    },
    'Basal Cell Carcinoma': {
        'name': 'Basal Cell Carcinoma', 'icon': 'fa-solid fa-triangle-exclamation', 'severity': 'High',
        'description': 'The most common type of skin cancer. It begins in the basal cells which produce new skin cells. Usually appears as a slightly transparent bump on sun-exposed skin.',
        'causes': '<i class="fa-solid fa-sun icon-inline"></i> Primarily caused by long-term UV radiation exposure. Risk factors: fair skin, chronic sun exposure, radiation therapy, immunosuppression, arsenic exposure.',
        'treatments': 'Mohs Micrographic Surgery (most effective), Excisional Surgery, Curettage & Electrodesiccation, Radiation Therapy, Topical Medications',
        'home_care': 'See a dermatologist as soon as possible, document any changes with photos, protect the area from sun, avoid picking or scratching the lesion',
        'youtube': 'https://www.youtube.com/results?search_query=basal+cell+carcinoma+treatment',
        'articles': [
            {'title': 'Mayo Clinic Guide', 'url': 'https://www.mayoclinic.org/diseases-conditions/basal-cell-carcinoma/symptoms-causes/syc-20354187'},
            {'title': 'American Cancer Society', 'url': 'https://www.cancer.org/cancer/types/basal-and-squamous-cell-skin-cancer.html'},
        ]
    },
    'Benign Keratosis': {
        'name': 'Benign Keratosis (Seborrheic Keratosis)', 'icon': 'fa-solid fa-circle-check', 'severity': 'Low',
        'description': "A common non-cancerous skin growth that appears as a waxy, wart-like spot. Often looks like it's stuck on the skin. These growths are harmless and don't become cancerous.",
        'causes': '<i class="fa-solid fa-dna icon-inline"></i> The exact cause is unknown, but they tend to run in families. NOT caused by sun exposure or viral infections. More common after age 50.',
        'treatments': 'Usually no treatment needed. If desired: Cryotherapy, Curettage, Electrosurgery, Laser Treatment',
        'home_care': 'No special care required. Keep moisturized. Monitor for sudden changes. Avoid irritating or scratching the growth.',
        'youtube': 'https://www.youtube.com/results?search_query=seborrheic+keratosis',
        'articles': [
            {'title': 'Mayo Clinic Guide', 'url': 'https://www.mayoclinic.org/diseases-conditions/seborrheic-keratosis/symptoms-causes/syc-20353878'},
            {'title': 'AAD Overview', 'url': 'https://www.aad.org/public/diseases/a-z/seborrheic-keratoses-overview'},
        ]
    },
    'Dermatofibroma': {
        'name': 'Dermatofibroma', 'icon': 'fa-solid fa-circle-info', 'severity': 'Low',
        'description': 'A common benign skin growth that feels like a hard bump under the skin. Usually brownish to red-purple. Often develops after minor injuries. Completely harmless.',
        'causes': '<i class="fa-solid fa-bug icon-inline"></i> Often triggered by minor skin injuries such as insect bites, splinters, or small cuts. More common in women and adults aged 20-50.',
        'treatments': 'Usually no treatment needed. If bothersome: Surgical excision, Cryotherapy, Laser removal',
        'home_care': 'Protect from repeated trauma. Monitor for significant changes in size or color.',
        'youtube': 'https://www.youtube.com/results?search_query=dermatofibroma',
        'articles': [
            {'title': 'DermNet NZ', 'url': 'https://dermnetnz.org/topics/dermatofibroma'},
            {'title': 'Cleveland Clinic', 'url': 'https://my.clevelandclinic.org/health/diseases/24856-dermatofibroma'},
        ]
    },
    'Melanoma': {
        'name': 'Melanoma', 'icon': 'fa-solid fa-circle-exclamation', 'severity': 'Critical',
        'description': 'SERIOUS SKIN CANCER - The most dangerous form of skin cancer. Develops from melanocytes. Can spread to other organs if not caught early. Look for ABCDEs: Asymmetry, Border, Color, Diameter >6mm, Evolving.',
        'causes': '<i class="fa-solid fa-sun icon-inline"></i><i class="fa-solid fa-dna icon-inline"></i> Caused by DNA damage to melanocytes from UV radiation. Risk factors: intense sun exposure, many moles, fair skin, family history, weakened immune system.',
        'treatments': 'Wide Excision Surgery, Sentinel Lymph Node Biopsy, Immunotherapy (Keytruda, Opdivo), Targeted Therapy, Radiation, Chemotherapy',
        'home_care': 'SEEK IMMEDIATE MEDICAL ATTENTION! Do not delay - early detection saves lives. Document with photos. Avoid sun exposure.',
        'youtube': 'https://www.youtube.com/results?search_query=melanoma+warning+signs+ABCDE',
        'articles': [
            {'title': 'National Cancer Institute', 'url': 'https://www.cancer.gov/types/skin/patient/melanoma-treatment-pdq'},
            {'title': 'American Cancer Society', 'url': 'https://www.cancer.org/cancer/types/melanoma-skin-cancer.html'},
            {'title': 'Melanoma Research Foundation', 'url': 'https://melanoma.org/'},
        ]
    },
    'Melanocytic Nevus': {
        'name': 'Melanocytic Nevus (Common Mole)', 'icon': 'fa-solid fa-droplet', 'severity': 'Low',
        'description': 'A common benign growth on the skin, known as a mole. Formed by clusters of melanocytes. Most people have 10-40 moles. Can be flat or raised, pink to dark brown.',
        'causes': '<i class="fa-solid fa-dna icon-inline"></i> Moles form when melanocytes grow in clusters. Caused by genetics and sun exposure. Most develop during childhood and adolescence.',
        'treatments': 'Usually no treatment needed. Removal: Surgical excision, Shave removal. Remove if suspicious (ABCDE criteria).',
        'home_care': 'Monthly self-exams using ABCDE rule. Protect from sun. Take photos to track changes. See a dermatologist annually.',
        'youtube': 'https://www.youtube.com/results?search_query=mole+skin+check+ABCDE',
        'articles': [
            {'title': 'Mayo Clinic Guide', 'url': 'https://www.mayoclinic.org/diseases-conditions/moles/symptoms-causes/syc-20375200'},
            {'title': 'AAD Mole Guide', 'url': 'https://www.aad.org/public/diseases/a-z/moles-overview'},
        ]
    },
    'Squamous Cell Carcinoma': {
        'name': 'Squamous Cell Carcinoma', 'icon': 'fa-solid fa-triangle-exclamation', 'severity': 'High',
        'description': 'The second most common skin cancer. Develops in squamous cells. Usually caused by UV exposure. Can spread if not treated.',
        'causes': '<i class="fa-solid fa-sun icon-inline"></i> Cumulative UV exposure from sun or tanning beds. Risk factors: fair skin, chronic sun exposure, weakened immune system, HPV infection.',
        'treatments': 'Mohs Surgery, Excisional Surgery, Curettage & Electrodesiccation, Radiation Therapy, Topical Chemotherapy (5-FU)',
        'home_care': 'See a dermatologist urgently. Protect from sun. Document changes with photos.',
        'youtube': 'https://www.youtube.com/results?search_query=squamous+cell+carcinoma+skin',
        'articles': [
            {'title': 'Mayo Clinic Guide', 'url': 'https://www.mayoclinic.org/diseases-conditions/squamous-cell-carcinoma/symptoms-causes/syc-20352480'},
            {'title': 'Skin Cancer Foundation', 'url': 'https://www.skincancer.org/skin-cancer-information/squamous-cell-carcinoma/'},
        ]
    },
    'Vascular Lesion': {
        'name': 'Vascular Lesion', 'icon': 'fa-solid fa-heart-pulse', 'severity': 'Low',
        'description': 'An abnormality of blood vessels in or under the skin. Includes cherry angiomas, spider veins, hemangiomas. Almost always benign and primarily cosmetic.',
        'causes': '<i class="fa-solid fa-dna icon-inline"></i> Develop due to aging, genetics, or hormonal changes. Cherry angiomas increase with age. Spider veins caused by sun, hormones, or prolonged standing.',
        'treatments': 'Usually no treatment needed. Cosmetic: Pulsed Dye Laser, IPL, Sclerotherapy, Electrocautery',
        'home_care': 'Protect from injury. No special care required. Monitor for rapid growth or changes.',
        'youtube': 'https://www.youtube.com/results?search_query=cherry+angioma+vascular+lesion',
        'articles': [
            {'title': 'DermNet NZ', 'url': 'https://dermnetnz.org/topics/vascular-lesions'},
            {'title': 'Cleveland Clinic', 'url': 'https://my.clevelandclinic.org/health/diseases/17893-cherry-angiomas'},
        ]
    },
}

st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
<style>
    .top-nav-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 0.75rem 1.25rem;
        border-radius: 12px;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.06);
    }
    .nav-logo {
        font-size: 1.2rem;
        font-weight: 700;
        color: #ffffff;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .nav-logo i {
        color: #818cf8;
    }
    .github-contact-btn {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        background: rgba(255, 255, 255, 0.12);
        color: #ffffff !important;
        padding: 0.4rem 0.95rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.85rem;
        text-decoration: none !important;
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.2s ease-in-out;
    }
    .github-contact-btn:hover {
        background: rgba(255, 255, 255, 0.22);
        border-color: rgba(255, 255, 255, 0.4);
        color: #ffffff !important;
    }
    .main-title {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-size: 2.2rem; font-weight: 800; text-align: center; margin-bottom: 0.2rem;
    }
    .main-title i {
        -webkit-text-fill-color: #6366f1;
        margin-right: 0.6rem;
    }
    .sub-title-header {
        font-size: 1.25rem; font-weight: 700; color: #1e293b; margin: 0.8rem 0 0.5rem 0;
        display: flex; align-items: center; gap: 0.5rem;
    }
    .sub-title-header i {
        color: #6366f1;
    }
    .icon-inline { margin-right: 0.4rem; color: #6366f1; }
    .result-danger { background: linear-gradient(135deg, #dc2626, #b91c1c); color: white; padding: 1.2rem; border-radius: 12px; }
    .result-success { background: linear-gradient(135deg, #059669, #047857); color: white; padding: 1.2rem; border-radius: 12px; }
    .result-warning { background: linear-gradient(135deg, #d97706, #b45309); color: white; padding: 1.2rem; border-radius: 12px; }
    .resource-card { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1rem; margin: 0.5rem 0; }
    .resource-card a { color: #6366f1; text-decoration: none; font-weight: 600; }
    .resource-card a:hover { text-decoration: underline; }
    .badge-rank {
        display: inline-flex; align-items: center; justify-content: center;
        background: #6366f1; color: white; border-radius: 50%; font-weight: 700;
        width: 26px; height: 26px; font-size: 0.85rem; margin-right: 0.4rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    try:
        if not os.path.exists(LOCAL_MODEL_PATH):
            st.error("Model files not found.")
            st.stop()
        processor = AutoImageProcessor.from_pretrained(LOCAL_MODEL_PATH, local_files_only=True)
        model = AutoModelForImageClassification.from_pretrained(LOCAL_MODEL_PATH, local_files_only=True)
        model.eval()
        return processor, model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.stop()


def predict_skin_condition(image, processor, model):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)[0]
    predicted_idx = logits.argmax(-1).item()
    predicted_abbr = model.config.id2label[predicted_idx]
    predicted_label = LABEL_MAP.get(predicted_abbr, predicted_abbr)
    confidence = probs[predicted_idx].item()
    top_3 = []
    for idx in torch.argsort(probs, descending=True)[:3]:
        abbr = model.config.id2label[idx.item()]
        top_3.append((LABEL_MAP.get(abbr, abbr), probs[idx].item()))
    return {'disease': predicted_label, 'confidence': confidence, 'top_3': top_3}


def get_bot_response(message, disease_info):
    if not GROQ_API_KEY:
        return "Add your Groq API key to .env file. Get a key at console.groq.com/keys"
    try:
        client = Groq(api_key=GROQ_API_KEY)
        system = f"""You are SkinBot, a friendly dermatology assistant.
Condition: {disease_info.get('name','Unknown')}, Severity: {disease_info.get('severity','Unknown')}.
Description: {disease_info.get('description','')}. Treatments: {disease_info.get('treatments','')}.
Rules: Keep responses professional, clear, 2-4 sentences, and always recommend seeing a dermatologist."""
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "system", "content": system}, {"role": "user", "content": message}],
            max_tokens=300, temperature=0.7
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)[:100]}"


def get_disease_info(disease_label):
    if disease_label in DISEASE_INFO:
        return DISEASE_INFO[disease_label]
    for key in DISEASE_INFO:
        if key.lower() == disease_label.lower():
            return DISEASE_INFO[key]
    return {
        'name': disease_label, 'icon': 'fa-solid fa-magnifying-glass', 'severity': 'Unknown',
        'description': 'Skin condition detected. Please consult a dermatologist.',
        'treatments': 'Consult a dermatologist', 'home_care': 'Monitor for changes',
        'youtube': f'https://www.youtube.com/results?search_query={disease_label.replace(" ", "+")}',
        'articles': []
    }


def clean_html_tags(text):
    if not text:
        return ""
    return re.sub(r'<[^>]+>', '', str(text)).strip()


def generate_pdf_report(results, info, hospitals=None, user_location=""):
    if not REPORTLAB_AVAILABLE:
        buffer = io.BytesIO()
        text_content = f"SKIN ANALYSIS REPORT\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M')}\nCondition: {info.get('name', 'Unknown')}\nConfidence: {results.get('confidence', 0):.1%}\n"
        buffer.write(text_content.encode('utf-8'))
        buffer.seek(0)
        return buffer.getvalue()

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#1e293b')
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#64748b')
    )
    section_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#6366f1'),
        spaceBefore=8,
        spaceAfter=4
    )
    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#334155')
    )
    bold_body = ParagraphStyle(
        'BoldBodyCustom',
        parent=body_style,
        fontName='Helvetica-Bold'
    )
    disclaimer_style = ParagraphStyle(
        'DisclaimerText',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#92400e')
    )

    story = []

    # Header Banner
    story.append(Paragraph("AI SKIN DISEASE CHECKER", title_style))
    story.append(Paragraph(f"Comprehensive Analysis & Clinical Summary Report &bull; Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", subtitle_style))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#6366f1'), spaceAfter=8))

    # Primary Diagnosis Summary
    disease_name = info.get('name', results.get('disease', 'Unknown'))
    confidence_str = f"{results.get('confidence', 0):.1%}"
    severity_str = info.get('severity', 'Low')

    sev_hex = '#059669'
    if severity_str == 'Critical':
        sev_hex = '#dc2626'
    elif severity_str == 'High':
        sev_hex = '#d97706'

    diag_data = [
        [Paragraph("<b>Primary Condition Detected</b>", bold_body), Paragraph(f"<b>{disease_name}</b>", bold_body)],
        [Paragraph("<b>Diagnostic Confidence</b>", body_style), Paragraph(confidence_str, body_style)],
        [Paragraph("<b>Severity Classification</b>", body_style), Paragraph(f"<font color='{sev_hex}'><b>{severity_str}</b></font>", body_style)],
    ]
    diag_table = Table(diag_data, colWidths=[180, 360])
    diag_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#e2e8f0')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('PADDING', (0,0), (-1,-1), 5),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(diag_table)
    story.append(Spacer(1, 8))

    # Top 3 Differential Diagnoses
    story.append(Paragraph("Top Differential Diagnoses", section_style))
    top3_data = [[Paragraph("<b>Rank</b>", bold_body), Paragraph("<b>Condition Name</b>", bold_body), Paragraph("<b>Probability</b>", bold_body)]]
    for idx, (name, prob) in enumerate(results.get('top_3', [])):
        top3_data.append([
            Paragraph(f"#{idx+1}", body_style),
            Paragraph(name, body_style),
            Paragraph(f"{prob:.1%}", body_style)
        ])
    top3_table = Table(top3_data, colWidths=[50, 340, 150])
    top3_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#6366f1')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('PADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(top3_table)
    story.append(Spacer(1, 8))

    # Clinical Information
    story.append(Paragraph("Clinical Overview & Causes", section_style))
    story.append(Paragraph(f"<b>Description:</b> {clean_html_tags(info.get('description', ''))}", body_style))
    story.append(Spacer(1, 3))
    story.append(Paragraph(f"<b>Causes & Risk Factors:</b> {clean_html_tags(info.get('causes', ''))}", body_style))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Treatment & Self Care Guidelines", section_style))
    story.append(Paragraph(f"<b>Medical Treatments:</b> {clean_html_tags(info.get('treatments', ''))}", body_style))
    story.append(Spacer(1, 3))
    story.append(Paragraph(f"<b>Home Care:</b> {clean_html_tags(info.get('home_care', ''))}", body_style))
    story.append(Spacer(1, 10))

    # Nearest Hospitals Section (Up to 6)
    story.append(Paragraph("Nearest Hospitals & Dermatology Facilities", section_style))
    if user_location:
        story.append(Paragraph(f"<i>Showing facilities near location: <b>{user_location}</b></i>", subtitle_style))
        story.append(Spacer(1, 3))

    if hospitals:
        hosp_table_data = [[
            Paragraph("<b>#</b>", bold_body),
            Paragraph("<b>Facility Name</b>", bold_body),
            Paragraph("<b>Type</b>", bold_body),
            Paragraph("<b>Dist.</b>", bold_body),
            Paragraph("<b>Address & Details</b>", bold_body)
        ]]
        for idx, hosp in enumerate(hospitals[:6]):
            phone_txt = f"<br/>Phone: {hosp['phone']}" if hosp.get('phone') else ""
            gmaps_url = f"https://www.google.com/maps/search/?api=1&query={hosp['lat']},{hosp['lon']}"
            gmaps_link = f'<br/><font color="#6366f1"><u><a href="{gmaps_url}">Google Maps Location &rarr;</a></u></font>'
            addr_txt = f"{hosp.get('address', '')}{phone_txt}{gmaps_link}"
            hosp_table_data.append([
                Paragraph(f"#{idx+1}", body_style),
                Paragraph(f"<b>{hosp.get('name', '')}</b>", body_style),
                Paragraph(hosp.get('type', 'Facility'), body_style),
                Paragraph(f"{hosp.get('distance_km', 0)} km", body_style),
                Paragraph(addr_txt, body_style)
            ])
        hosp_table = Table(hosp_table_data, colWidths=[25, 160, 75, 55, 225])
        hosp_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0f172a')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e1')),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
            ('PADDING', (0,0), (-1,-1), 4),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ]))
        story.append(hosp_table)
    else:
        story.append(Paragraph("<i>No hospital location search performed prior to generating this report. Enter your city or address in the app to view and include nearby facilities.</i>", body_style))

    story.append(Spacer(1, 10))

    # Disclaimer Box
    disc_data = [[Paragraph("<b>MEDICAL DISCLAIMER:</b> This report is generated by an artificial intelligence model for educational and informational reference only. It does NOT constitute medical diagnosis or advice. Always consult a board-certified dermatologist or healthcare professional for clinical evaluations.", disclaimer_style)]]
    disc_table = Table(disc_data, colWidths=[540])
    disc_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#fef3c7')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#f59e0b')),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(disc_table)

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ===== HOSPITAL FINDER =====

def geocode_location(query):
    try:
        r = requests.get("https://nominatim.openstreetmap.org/search",
                         params={"q": query, "format": "json", "limit": 1},
                         headers={"User-Agent": "SkinDiseaseChecker/1.0"}, timeout=8)
        data = r.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"]), data[0].get("display_name", query)
    except Exception:
        pass
    return None, None, None


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def fetch_nearby_hospitals(lat, lon):
    headers = {"User-Agent": "SkinDiseaseChecker/1.0"}
    found = []
    seen = set()

    for delta in [0.2, 0.5, 1.0, 2.0]:
        viewbox = f"{lon-delta},{lat+delta},{lon+delta},{lat-delta}"
        for amenity in ["hospital", "clinic"]:
            try:
                resp = requests.get(
                    "https://nominatim.openstreetmap.org/search",
                    params={
                        "q": amenity,
                        "format": "json",
                        "limit": 15,
                        "viewbox": viewbox,
                        "bounded": 1,
                        "addressdetails": 1,
                    },
                    headers=headers,
                    timeout=10,
                )
                for item in resp.json():
                    name = item.get("display_name", "").split(",")[0].strip()
                    if not name or name in seen:
                        continue
                    seen.add(name)
                    h_lat = float(item["lat"])
                    h_lon = float(item["lon"])
                    addr_parts = item.get("address", {})
                    addr = ", ".join(filter(None, [
                        addr_parts.get("road", ""),
                        addr_parts.get("suburb", ""),
                        addr_parts.get("city", "") or addr_parts.get("town", ""),
                        addr_parts.get("state", ""),
                    ])) or item.get("display_name", "").split(",", 1)[-1].strip()
                    found.append({
                        "name": name,
                        "lat": h_lat,
                        "lon": h_lon,
                        "address": addr,
                        "type": amenity.title(),
                        "distance_km": round(haversine_km(lat, lon, h_lat, h_lon), 2),
                        "phone": "",
                        "website": "",
                    })
            except Exception:
                continue

        found.sort(key=lambda x: x["distance_km"])
        deduped = []
        for h in found:
            if not any(haversine_km(h["lat"], h["lon"], d["lat"], d["lon"]) < 0.1 for d in deduped):
                deduped.append(h)
        found = deduped

        if len(found) >= 6:
            break

    return found[:6]


def make_hospital_map(hospital, user_lat, user_lon):
    m = folium.Map(location=[hospital["lat"], hospital["lon"]], zoom_start=14, tiles="OpenStreetMap")
    folium.Marker(
        location=[hospital["lat"], hospital["lon"]],
        popup=folium.Popup(f"<b>{hospital['name']}</b><br>{hospital['address']}<br>{hospital['distance_km']} km away", max_width=220),
        tooltip=hospital["name"],
        icon=folium.Icon(color="red", icon="plus-sign", prefix="glyphicon"),
    ).add_to(m)
    folium.Marker(
        location=[user_lat, user_lon], popup="Your Location", tooltip="Your Location",
        icon=folium.Icon(color="blue", icon="home", prefix="glyphicon"),
    ).add_to(m)
    folium.PolyLine([[user_lat, user_lon], [hospital["lat"], hospital["lon"]]],
                    color="#6366f1", weight=2, dash_array="6", opacity=0.7).add_to(m)
    return m


def show_hospitals_section(disease_name):
    st.markdown("---")
    st.markdown("<div class='sub-title-header'><i class='fa-solid fa-hospital'></i> Find Nearest Hospitals / Dermatology Clinics</div>", unsafe_allow_html=True)
    st.caption("Locate nearby medical facilities to diagnose and treat your skin condition.")

    if not FOLIUM_AVAILABLE:
        st.warning("Map libraries not installed. Run: pip install folium streamlit-folium, then restart the app.")
        return

    col_in, col_btn = st.columns([4, 1])
    with col_in:
        location_text = st.text_input(
            "loc_input", value=st.session_state.location_query,
            placeholder="Enter your city or address  e.g. New York, NY",
            label_visibility="collapsed",
        )
    with col_btn:
        search_clicked = st.button("Search Hospitals", type="primary", use_container_width=True)

    if search_clicked and location_text.strip():
        with st.spinner("Searching for nearby medical facilities…"):
            lat, lon, display = geocode_location(location_text.strip())
            if lat is None:
                st.error("Could not find that location. Try a more specific city or address.")
                return
            st.session_state.user_location = (lat, lon, display)
            st.session_state.location_query = location_text.strip()
            st.session_state.hospitals = fetch_nearby_hospitals(lat, lon)

    if st.session_state.user_location and st.session_state.hospitals is not None:
        lat, lon, display = st.session_state.user_location
        hospitals = st.session_state.hospitals
        st.markdown(f"<div style='color:#059669;font-weight:600;margin-bottom:0.8rem;'><i class='fa-solid fa-location-dot'></i> Showing results near: <b>{display}</b></div>", unsafe_allow_html=True)
        if not hospitals:
            st.warning("No hospitals found within search radius. Try a different location.")
            return

        st.markdown(f"#### Nearest Facilities for **{disease_name}** Treatment (Showing {len(hospitals)})")
        TYPE_COLOR = {"Hospital": "#dc2626", "Clinic": "#059669", "Dermatologist": "#6366f1", "Skin Care": "#a855f7"}

        for chunk_idx in range(0, len(hospitals), 3):
            chunk = hospitals[chunk_idx:chunk_idx+3]
            map_cols = st.columns(len(chunk))
            for i, (col, hosp) in enumerate(zip(map_cols, chunk)):
                hosp_num = chunk_idx + i + 1
                badge_color = TYPE_COLOR.get(hosp["type"], "#6366f1")
                phone_html = f'<div style="font-size:0.75rem;margin-top:0.3rem;"><i class="fa-solid fa-phone" style="color:#64748b;"></i> {hosp["phone"]}</div>' if hosp.get("phone") else ""
                web_html = f'<div style="font-size:0.75rem;"><a href="{hosp["website"]}" target="_blank"><i class="fa-solid fa-globe"></i> Website</a></div>' if hosp.get("website") else ""
                with col:
                    st.markdown(f"""<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:0.8rem;margin-bottom:0.5rem;">
                        <span class="badge-rank">{hosp_num}</span>
                        <span style="font-weight:700;font-size:0.95rem;color:#1e293b;">{hosp['name']}</span>
                        <div style="margin:0.3rem 0;"><span style="background:{badge_color};color:white;font-size:0.7rem;padding:2px 8px;border-radius:999px;">{hosp['type']}</span></div>
                        <div style="font-size:0.8rem;color:#475569;"><i class="fa-solid fa-route" style="color:#6366f1;"></i> <b>{hosp['distance_km']} km</b> away</div>
                        <div style="font-size:0.75rem;color:#94a3b8;margin-top:0.2rem;">{hosp['address']}</div>
                        {phone_html}{web_html}</div>""", unsafe_allow_html=True)
                    m = make_hospital_map(hosp, lat, lon)
                    st_folium(m, width=None, height=240, returned_objects=[], key=f"hosp_map_{chunk_idx}_{i}")

        st.markdown("<div style='font-weight:700;margin:1rem 0 0.5rem 0;'><i class='fa-solid fa-map-location-dot' style='color:#6366f1;'></i> Get Directions (opens Google Maps)</div>", unsafe_allow_html=True)
        for chunk_idx in range(0, len(hospitals), 3):
            chunk = hospitals[chunk_idx:chunk_idx+3]
            dir_cols = st.columns(len(chunk))
            for i, (col, hosp) in enumerate(zip(dir_cols, chunk)):
                hosp_num = chunk_idx + i + 1
                gmaps = f"https://www.google.com/maps/dir/?api=1&origin={lat},{lon}&destination={hosp['lat']},{hosp['lon']}&travelmode=driving"
                short = hosp["name"][:22] + ("…" if len(hosp["name"]) > 22 else "")
                with col:
                    st.markdown(f'<a href="{gmaps}" target="_blank" style="display:block;text-align:center;background:#6366f1;color:white;padding:0.5rem;border-radius:8px;text-decoration:none;font-weight:600;font-size:0.85rem;margin-bottom:0.5rem;"><i class="fa-solid fa-diamond-turn-right"></i> #{hosp_num} {short}</a>', unsafe_allow_html=True)
    else:
        st.info("Enter your location above and click Search Hospitals to view nearby facilities on interactive maps.")


# ===== MAIN APP =====
def main():
    with st.sidebar:
        st.markdown("<div style='font-size:1.4rem;font-weight:700;color:#1e293b;'><i class='fa-solid fa-robot' style='color:#6366f1;'></i> SkinBot Assistant</div>", unsafe_allow_html=True)
        if st.session_state.results:
            disease = st.session_state.results['disease']
            info = get_disease_info(disease)
            st.markdown(f"<div style='font-size:0.8rem;color:#64748b;margin-bottom:0.5rem;'><i class='fa-solid fa-comments' style='color:#6366f1;'></i> Discussing: <b>{info.get('name', disease)}</b></div>", unsafe_allow_html=True)
        else:
            info = {}
            st.markdown("<div style='font-size:0.8rem;color:#64748b;margin-bottom:0.5rem;'><i class='fa-solid fa-circle-info'></i> Upload an image to start chatting</div>", unsafe_allow_html=True)
        st.markdown("---")
        chat_container = st.container(height=300)
        with chat_container:
            if not st.session_state.chat_messages:
                st.markdown("<div style='font-size:0.9rem;'><i class='fa-solid fa-user-doctor' style='color:#6366f1;'></i> <b>SkinBot:</b> Hi! Upload a skin image and I will help explain the results.</div>", unsafe_allow_html=True)
            else:
                for msg in st.session_state.chat_messages:
                    if msg["role"] == "bot":
                        st.markdown(f"<div style='font-size:0.9rem;margin-bottom:0.4rem;'><i class='fa-solid fa-user-doctor' style='color:#6366f1;'></i> <b>SkinBot:</b> {msg['content']}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='font-size:0.9rem;margin-bottom:0.4rem;'><i class='fa-solid fa-user' style='color:#059669;'></i> <b>You:</b> {msg['content']}</div>", unsafe_allow_html=True)
        with st.form(key="chat_form", clear_on_submit=True):
            user_msg = st.text_input("Type your question...", placeholder="e.g., What treatments are available?")
            col1, col2 = st.columns([4, 1])
            with col1:
                send = st.form_submit_button("Send", use_container_width=True)
            with col2:
                clear = st.form_submit_button("Clear")
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
        st.markdown("<div style='font-size:0.75rem;color:#64748b;'><i class='fa-solid fa-shield-halved' style='color:#6366f1;'></i> AI assistant for guidance. Always consult medical professionals.</div>", unsafe_allow_html=True)

    st.markdown("""
<div class="top-nav-bar">
    <div class="nav-logo">
        <i class="fa-solid fa-microscope"></i> AI Skin Checker
    </div>
    <div>
        <a href="https://github.com/Pranay-Suthar" target="_blank" class="github-contact-btn">
            <i class="fa-brands fa-github"></i> Contact Developer
        </a>
    </div>
</div>
""", unsafe_allow_html=True)

    st.markdown('<h1 class="main-title"><i class="fa-solid fa-microscope"></i> AI Skin Disease Checker</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center;color:#64748b;font-weight:500;">Upload skin image &bull; Get AI analysis &bull; Chat with SkinBot</p>', unsafe_allow_html=True)

    with st.spinner("Loading AI model..."):
        processor, model = load_model()
    if processor is None or model is None:
        st.error("Failed to load model.")
        return

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("<div class='sub-title-header'><i class='fa-solid fa-cloud-arrow-up'></i> Upload Image</div>", unsafe_allow_html=True)
        uploaded = st.file_uploader("Choose skin image", type=['png', 'jpg', 'jpeg'], label_visibility="collapsed")
        if uploaded:
            image = Image.open(uploaded).convert('RGB')
            st.image(image, use_container_width=True)
    with col2:
        st.markdown("<div class='sub-title-header'><i class='fa-solid fa-chart-pie'></i> Analysis Results</div>", unsafe_allow_html=True)
        if uploaded:
            if st.button("Analyze Image", use_container_width=True, type="primary"):
                with st.spinner("Analyzing..."):
                    st.session_state.results = predict_skin_condition(image, processor, model)
                    st.session_state.chat_messages = []
                    st.session_state.user_location = None
                    st.session_state.hospitals = None
                    st.session_state.location_query = ""
                st.rerun()
        if st.session_state.results:
            results = st.session_state.results
            disease = results['disease']
            confidence = results['confidence']
            info = get_disease_info(disease)
            severity = info.get('severity', 'Low')
            icon_cls = info.get('icon', 'fa-solid fa-notes-medical')
            if severity == 'Critical':
                st.markdown(f'<div class="result-danger"><h3><i class="{icon_cls}"></i> {info.get("name", disease)}</h3><p>{info.get("description", "")}</p><p><b>Confidence: {confidence:.1%}</b></p></div>', unsafe_allow_html=True)
            elif severity == 'High':
                st.markdown(f'<div class="result-warning"><h3><i class="{icon_cls}"></i> {info.get("name", disease)}</h3><p>{info.get("description", "")}</p><p><b>Confidence: {confidence:.1%}</b></p></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-success"><h3><i class="{icon_cls}"></i> {info.get("name", disease)}</h3><p>{info.get("description", "")}</p><p><b>Confidence: {confidence:.1%}</b></p></div>', unsafe_allow_html=True)
            st.markdown("**Top 3 Predictions:**")
            for name, prob in results['top_3']:
                st.progress(prob, text=f"{name}: {prob:.1%}")

    # ===== RESOURCES SECTION =====
    if st.session_state.results:
        results = st.session_state.results
        disease = results['disease']
        info = get_disease_info(disease)

        st.markdown("---")
        st.markdown(f"<div class='sub-title-header'><i class='fa-solid fa-book-medical'></i> Resources for {info.get('name', disease)}</div>", unsafe_allow_html=True)
        st.markdown("#### <i class='fa-solid fa-circle-question' style='color:#6366f1;'></i> Why Does This Occur?", unsafe_allow_html=True)
        st.warning(info.get('causes', 'Causes vary. Please consult a dermatologist.'))

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("#### <i class='fa-solid fa-pills' style='color:#6366f1;'></i> Treatments", unsafe_allow_html=True)
            st.info(info.get('treatments', 'Consult a dermatologist'))
            st.markdown("#### <i class='fa-solid fa-house-medical' style='color:#059669;'></i> Home Care", unsafe_allow_html=True)
            st.success(info.get('home_care', 'Monitor for changes'))
        with col2:
            st.markdown("#### <i class='fa-solid fa-video' style='color:#dc2626;'></i> Video Resources", unsafe_allow_html=True)
            youtube_url = info.get('youtube', f"https://www.youtube.com/results?search_query={disease.replace(' ', '+')}")
            st.markdown(f'<div class="resource-card"><p><i class="fa-solid fa-circle-play" style="color:#dc2626;"></i> <b>Educational Videos</b></p><a href="{youtube_url}" target="_blank"><i class="fa-brands fa-youtube" style="color:#ff0000;margin-right:0.3rem;"></i> Watch on YouTube &rarr;</a></div>', unsafe_allow_html=True)
        with col3:
            st.markdown("#### <i class='fa-solid fa-newspaper' style='color:#6366f1;'></i> Articles & Guides", unsafe_allow_html=True)
            articles = info.get('articles', [])
            if articles:
                for article in articles:
                    st.markdown(f'<div class="resource-card"><a href="{article["url"]}" target="_blank"><i class="fa-solid fa-file-lines" style="margin-right:0.3rem;"></i> {article["title"]} &rarr;</a></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="resource-card"><a href="https://www.google.com/search?q={disease.replace(" ", "+")}+skin" target="_blank"><i class="fa-solid fa-magnifying-glass" style="margin-right:0.3rem;"></i> Search for more &rarr;</a></div>', unsafe_allow_html=True)

        # ===== NEAREST HOSPITALS =====
        show_hospitals_section(info.get('name', disease))

        # ===== DOWNLOAD REPORT (PDF) =====
        st.markdown("---")
        hospitals_list = st.session_state.hospitals or []
        user_loc_name = st.session_state.user_location[2] if st.session_state.user_location else ""
        pdf_bytes = generate_pdf_report(results, info, hospitals_list, user_loc_name)
        
        st.download_button(
            label="Download Diagnostic Report (PDF)",
            data=pdf_bytes,
            file_name=f"skin_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
            use_container_width=True,
            type="primary"
        )

    st.markdown("---")
    st.markdown("<div style='background:#fef3c7;border:1px solid #f59e0b;color:#92400e;padding:0.8rem;border-radius:10px;font-size:0.9rem;'><i class='fa-solid fa-shield-halved' style='margin-right:0.4rem;'></i> <b>Medical Disclaimer:</b> This AI tool is for educational purposes only. Always consult a qualified healthcare professional.</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
