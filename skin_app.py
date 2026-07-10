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
from groq import Groq
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

st.set_page_config(
    page_title="🔬 AI Skin Checker",
    page_icon="🔬",
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
        'name': 'Actinic Keratosis', 'emoji': '☀️', 'severity': 'Moderate',
        'description': 'A rough, scaly patch on the skin caused by years of sun exposure. It is considered precancerous and can develop into squamous cell carcinoma if left untreated.',
        'causes': '☀️ Cumulative UV radiation damage from sun exposure or tanning beds over many years. Risk factors include: fair skin, history of sunburns, age over 40, living in sunny climates.',
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
        'description': 'The most common type of skin cancer. It begins in the basal cells which produce new skin cells. Usually appears as a slightly transparent bump on sun-exposed skin.',
        'causes': '☀️ Primarily caused by long-term UV radiation exposure. Risk factors: fair skin, chronic sun exposure, radiation therapy, immunosuppression, arsenic exposure.',
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
        'description': "A common non-cancerous skin growth that appears as a waxy, wart-like spot. Often looks like it's stuck on the skin. These growths are harmless and don't become cancerous.",
        'causes': '🧬 The exact cause is unknown, but they tend to run in families. NOT caused by sun exposure or viral infections. More common after age 50.',
        'treatments': 'Usually no treatment needed. If desired: Cryotherapy, Curettage, Electrosurgery, Laser Treatment',
        'home_care': 'No special care required. Keep moisturized. Monitor for sudden changes. Avoid irritating or scratching the growth.',
        'youtube': 'https://www.youtube.com/results?search_query=seborrheic+keratosis',
        'articles': [
            {'title': 'Mayo Clinic Guide', 'url': 'https://www.mayoclinic.org/diseases-conditions/seborrheic-keratosis/symptoms-causes/syc-20353878'},
            {'title': 'AAD Overview', 'url': 'https://www.aad.org/public/diseases/a-z/seborrheic-keratoses-overview'},
        ]
    },
    'Dermatofibroma': {
        'name': 'Dermatofibroma', 'emoji': '🔵', 'severity': 'Low',
        'description': 'A common benign skin growth that feels like a hard bump under the skin. Usually brownish to red-purple. Often develops after minor injuries. Completely harmless.',
        'causes': '🦟 Often triggered by minor skin injuries such as insect bites, splinters, or small cuts. More common in women and adults aged 20-50.',
        'treatments': 'Usually no treatment needed. If bothersome: Surgical excision, Cryotherapy, Laser removal',
        'home_care': 'Protect from repeated trauma. Monitor for significant changes in size or color.',
        'youtube': 'https://www.youtube.com/results?search_query=dermatofibroma',
        'articles': [
            {'title': 'DermNet NZ', 'url': 'https://dermnetnz.org/topics/dermatofibroma'},
            {'title': 'Cleveland Clinic', 'url': 'https://my.clevelandclinic.org/health/diseases/24856-dermatofibroma'},
        ]
    },
    'Melanoma': {
        'name': 'Melanoma', 'emoji': '🚨', 'severity': 'Critical',
        'description': '⚠️ SERIOUS SKIN CANCER - The most dangerous form of skin cancer. Develops from melanocytes. Can spread to other organs if not caught early. Look for ABCDEs: Asymmetry, Border, Color, Diameter >6mm, Evolving.',
        'causes': '☀️🧬 Caused by DNA damage to melanocytes from UV radiation. Risk factors: intense sun exposure, many moles, fair skin, family history, weakened immune system.',
        'treatments': 'Wide Excision Surgery, Sentinel Lymph Node Biopsy, Immunotherapy (Keytruda, Opdivo), Targeted Therapy, Radiation, Chemotherapy',
        'home_care': '🚨 SEEK IMMEDIATE MEDICAL ATTENTION! Do not delay - early detection saves lives. Document with photos. Avoid sun exposure.',
        'youtube': 'https://www.youtube.com/results?search_query=melanoma+warning+signs+ABCDE',
        'articles': [
            {'title': 'National Cancer Institute', 'url': 'https://www.cancer.gov/types/skin/patient/melanoma-treatment-pdq'},
            {'title': 'American Cancer Society', 'url': 'https://www.cancer.org/cancer/types/melanoma-skin-cancer.html'},
            {'title': 'Melanoma Research Foundation', 'url': 'https://melanoma.org/'},
        ]
    },
    'Melanocytic Nevus': {
        'name': 'Melanocytic Nevus (Common Mole)', 'emoji': '🟤', 'severity': 'Low',
        'description': 'A common benign growth on the skin, known as a mole. Formed by clusters of melanocytes. Most people have 10-40 moles. Can be flat or raised, pink to dark brown.',
        'causes': '🧬☀️ Moles form when melanocytes grow in clusters. Caused by genetics and sun exposure. Most develop during childhood and adolescence.',
        'treatments': 'Usually no treatment needed. Removal: Surgical excision, Shave removal. Remove if suspicious (ABCDE criteria).',
        'home_care': 'Monthly self-exams using ABCDE rule. Protect from sun. Take photos to track changes. See a dermatologist annually.',
        'youtube': 'https://www.youtube.com/results?search_query=mole+skin+check+ABCDE',
        'articles': [
            {'title': 'Mayo Clinic Guide', 'url': 'https://www.mayoclinic.org/diseases-conditions/moles/symptoms-causes/syc-20375200'},
            {'title': 'AAD Mole Guide', 'url': 'https://www.aad.org/public/diseases/a-z/moles-overview'},
        ]
    },
    'Squamous Cell Carcinoma': {
        'name': 'Squamous Cell Carcinoma', 'emoji': '⚠️', 'severity': 'High',
        'description': 'The second most common skin cancer. Develops in squamous cells. Usually caused by UV exposure. Can spread if not treated.',
        'causes': '☀️ Cumulative UV exposure from sun or tanning beds. Risk factors: fair skin, chronic sun exposure, weakened immune system, HPV infection.',
        'treatments': 'Mohs Surgery, Excisional Surgery, Curettage & Electrodesiccation, Radiation Therapy, Topical Chemotherapy (5-FU)',
        'home_care': 'See a dermatologist urgently. Protect from sun. Document changes with photos.',
        'youtube': 'https://www.youtube.com/results?search_query=squamous+cell+carcinoma+skin',
        'articles': [
            {'title': 'Mayo Clinic Guide', 'url': 'https://www.mayoclinic.org/diseases-conditions/squamous-cell-carcinoma/symptoms-causes/syc-20352480'},
            {'title': 'Skin Cancer Foundation', 'url': 'https://www.skincancer.org/skin-cancer-information/squamous-cell-carcinoma/'},
        ]
    },
    'Vascular Lesion': {
        'name': 'Vascular Lesion', 'emoji': '❤️', 'severity': 'Low',
        'description': 'An abnormality of blood vessels in or under the skin. Includes cherry angiomas, spider veins, hemangiomas. Almost always benign and primarily cosmetic.',
        'causes': '🧬 Develop due to aging, genetics, or hormonal changes. Cherry angiomas increase with age. Spider veins caused by sun, hormones, or prolonged standing.',
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
<style>
    .main-title {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-size: 2rem; font-weight: 700; text-align: center;
    }
    .result-danger { background: linear-gradient(135deg, #dc2626, #b91c1c); color: white; padding: 1rem; border-radius: 12px; }
    .result-success { background: linear-gradient(135deg, #059669, #047857); color: white; padding: 1rem; border-radius: 12px; }
    .result-warning { background: linear-gradient(135deg, #d97706, #b45309); color: white; padding: 1rem; border-radius: 12px; }
    .resource-card { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1rem; margin: 0.5rem 0; }
    .resource-card a { color: #6366f1; text-decoration: none; font-weight: 600; }
    .resource-card a:hover { text-decoration: underline; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    try:
        if not os.path.exists(LOCAL_MODEL_PATH):
            st.error("❌ Model files not found.")
            st.stop()
        processor = AutoImageProcessor.from_pretrained(LOCAL_MODEL_PATH, local_files_only=True)
        model = AutoModelForImageClassification.from_pretrained(LOCAL_MODEL_PATH, local_files_only=True)
        model.eval()
        return processor, model
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
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
        return "⚠️ Add your Groq API key to .env file. Get free key at: console.groq.com/keys"
    try:
        client = Groq(api_key=GROQ_API_KEY)
        system = f"""You are SkinBot, a friendly dermatology assistant.
Condition: {disease_info.get('name','Unknown')}, Severity: {disease_info.get('severity','Unknown')}.
Description: {disease_info.get('description','')}. Treatments: {disease_info.get('treatments','')}.
Rules: Be friendly with emojis, keep responses 2-4 sentences, always recommend seeing a dermatologist."""
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "system", "content": system}, {"role": "user", "content": message}],
            max_tokens=300, temperature=0.7
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"❌ Error: {str(e)[:100]}"


def get_disease_info(disease_label):
    if disease_label in DISEASE_INFO:
        return DISEASE_INFO[disease_label]
    for key in DISEASE_INFO:
        if key.lower() == disease_label.lower():
            return DISEASE_INFO[key]
    return {
        'name': disease_label, 'emoji': '🔍', 'severity': 'Unknown',
        'description': 'Skin condition detected. Please consult a dermatologist.',
        'treatments': 'Consult a dermatologist', 'home_care': 'Monitor for changes',
        'youtube': f'https://www.youtube.com/results?search_query={disease_label.replace(" ", "+")}',
        'articles': []
    }


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
    """
    Use Nominatim (OpenStreetMap) to search for hospitals and clinics near lat/lon.
    This avoids the Overpass API which is often blocked or slow.
    """
    headers = {"User-Agent": "SkinDiseaseChecker/1.0"}
    found = []
    seen = set()

    # Search for hospitals and clinics using viewbox bounding box
    # Try progressively larger boxes until we get 3 results
    for delta in [0.2, 0.5, 1.0, 2.0]:
        viewbox = f"{lon-delta},{lat+delta},{lon+delta},{lat-delta}"
        for amenity in ["hospital", "clinic"]:
            try:
                resp = requests.get(
                    "https://nominatim.openstreetmap.org/search",
                    params={
                        "q": amenity,
                        "format": "json",
                        "limit": 10,
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
        # Deduplicate by proximity (skip if another entry is within 0.1 km)
        deduped = []
        for h in found:
            if not any(haversine_km(h["lat"], h["lon"], d["lat"], d["lon"]) < 0.1 for d in deduped):
                deduped.append(h)
        found = deduped

        if len(found) >= 3:
            break

    return found[:3]


def make_hospital_map(hospital, user_lat, user_lon):
    m = folium.Map(location=[hospital["lat"], hospital["lon"]], zoom_start=14, tiles="OpenStreetMap")
    folium.Marker(
        location=[hospital["lat"], hospital["lon"]],
        popup=folium.Popup(f"<b>{hospital['name']}</b><br>{hospital['address']}<br>📏 {hospital['distance_km']} km away", max_width=220),
        tooltip=hospital["name"],
        icon=folium.Icon(color="red", icon="plus-sign", prefix="glyphicon"),
    ).add_to(m)
    folium.Marker(
        location=[user_lat, user_lon], popup="📍 Your Location", tooltip="Your Location",
        icon=folium.Icon(color="blue", icon="home", prefix="glyphicon"),
    ).add_to(m)
    folium.PolyLine([[user_lat, user_lon], [hospital["lat"], hospital["lon"]]],
                    color="#6366f1", weight=2, dash_array="6", opacity=0.7).add_to(m)
    return m


def show_hospitals_section(disease_name):
    st.markdown("---")
    st.subheader("🏥 Find Nearest Hospitals / Dermatology Clinics")
    st.caption("Locate nearby facilities that can help diagnose and treat your skin condition.")

    if not FOLIUM_AVAILABLE:
        st.warning("⚠️ Map libraries not installed. Run:\n```\npip install folium streamlit-folium\n```\nthen restart the app.")
        return

    col_in, col_btn = st.columns([4, 1])
    with col_in:
        location_text = st.text_input(
            "loc_input", value=st.session_state.location_query,
            placeholder="Enter your city or address  e.g. New York, NY",
            label_visibility="collapsed",
        )
    with col_btn:
        search_clicked = st.button("🔍 Find Hospitals", type="primary", use_container_width=True)

    if search_clicked and location_text.strip():
        with st.spinner("📍 Locating and searching for nearby hospitals…"):
            lat, lon, display = geocode_location(location_text.strip())
            if lat is None:
                st.error("❌ Could not find that location. Try a more specific city or address.")
                return
            st.session_state.user_location = (lat, lon, display)
            st.session_state.location_query = location_text.strip()
            st.session_state.hospitals = fetch_nearby_hospitals(lat, lon)

    if st.session_state.user_location and st.session_state.hospitals is not None:
        lat, lon, display = st.session_state.user_location
        hospitals = st.session_state.hospitals
        st.success(f"📍 Showing results near: **{display}**")
        if not hospitals:
            st.warning("No hospitals found within 100 km. Try a different location.")
            return

        st.markdown(f"### 3 Nearest Facilities for **{disease_name}** Treatment")
        TYPE_COLOR = {"Hospital": "#dc2626", "Clinic": "#059669", "Dermatologist": "#6366f1", "Skin Care": "#a855f7"}
        RANKS = ["🥇", "🥈", "🥉"]

        map_cols = st.columns(3)
        for i, (col, hosp) in enumerate(zip(map_cols, hospitals)):
            badge_color = TYPE_COLOR.get(hosp["type"], "#6366f1")
            phone_html = f'<div style="font-size:0.75rem;margin-top:0.3rem;">📞 {hosp["phone"]}</div>' if hosp.get("phone") else ""
            web_html = f'<div style="font-size:0.75rem;"><a href="{hosp["website"]}" target="_blank">🌐 Website</a></div>' if hosp.get("website") else ""
            with col:
                st.markdown(f"""<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:0.8rem;margin-bottom:0.5rem;">
                    <span style="font-size:1.3rem;">{RANKS[i]}</span>
                    <span style="font-weight:700;font-size:0.95rem;color:#1e293b;margin-left:0.4rem;">{hosp['name']}</span>
                    <div style="margin:0.3rem 0;"><span style="background:{badge_color};color:white;font-size:0.7rem;padding:2px 8px;border-radius:999px;">{hosp['type']}</span></div>
                    <div style="font-size:0.8rem;color:#475569;">📏 <b>{hosp['distance_km']} km</b> away</div>
                    <div style="font-size:0.75rem;color:#94a3b8;margin-top:0.2rem;">{hosp['address']}</div>
                    {phone_html}{web_html}</div>""", unsafe_allow_html=True)
                m = make_hospital_map(hosp, lat, lon)
                st_folium(m, width=None, height=260, returned_objects=[], key=f"hosp_map_{i}")

        st.markdown("#### 🗺️ Get Directions (opens Google Maps)")
        dir_cols = st.columns(3)
        for i, (col, hosp) in enumerate(zip(dir_cols, hospitals)):
            gmaps = f"https://www.google.com/maps/dir/?api=1&origin={lat},{lon}&destination={hosp['lat']},{hosp['lon']}&travelmode=driving"
            short = hosp["name"][:26] + ("…" if len(hosp["name"]) > 26 else "")
            with col:
                st.markdown(f'<a href="{gmaps}" target="_blank" style="display:block;text-align:center;background:#6366f1;color:white;padding:0.5rem;border-radius:8px;text-decoration:none;font-weight:600;font-size:0.85rem;">{RANKS[i]} {short}</a>', unsafe_allow_html=True)
    else:
        st.info("👆 Enter your location above and click **Find Hospitals** to see the 3 nearest facilities on interactive maps.")


# ===== MAIN APP =====
def main():
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
                st.markdown("👋 **SkinBot:** Hi! Upload a skin image and I'll help explain the results!")
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

    st.markdown('<h1 class="main-title">🔬 AI Skin Disease Checker</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center;color:#64748b;">Upload a skin image • Get AI analysis • Chat with SkinBot</p>', unsafe_allow_html=True)

    with st.spinner("Loading AI model..."):
        processor, model = load_model()
    if processor is None or model is None:
        st.error("❌ Failed to load model.")
        return

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
            emoji = info.get('emoji', '🔍')
            if severity == 'Critical':
                st.markdown(f'<div class="result-danger"><h3>{emoji} {info.get("name", disease)}</h3><p>{info.get("description", "")}</p><p><b>Confidence: {confidence:.1%}</b></p></div>', unsafe_allow_html=True)
            elif severity == 'High':
                st.markdown(f'<div class="result-warning"><h3>{emoji} {info.get("name", disease)}</h3><p>{info.get("description", "")}</p><p><b>Confidence: {confidence:.1%}</b></p></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-success"><h3>{emoji} {info.get("name", disease)}</h3><p>{info.get("description", "")}</p><p><b>Confidence: {confidence:.1%}</b></p></div>', unsafe_allow_html=True)
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
        st.markdown("### ❓ Why Does This Occur?")
        st.warning(info.get('causes', 'Causes vary. Please consult a dermatologist.'))

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 💊 Treatments")
            st.info(info.get('treatments', 'Consult a dermatologist'))
            st.markdown("### 🏠 Home Care")
            st.success(info.get('home_care', 'Monitor for changes'))
        with col2:
            st.markdown("### 📺 Video Resources")
            youtube_url = info.get('youtube', f"https://www.youtube.com/results?search_query={disease.replace(' ', '+')}")
            st.markdown(f'<div class="resource-card"><p>🎥 <b>Educational Videos</b></p><a href="{youtube_url}" target="_blank">▶️ Watch on YouTube →</a></div>', unsafe_allow_html=True)
        with col3:
            st.markdown("### 📖 Articles")
            articles = info.get('articles', [])
            if articles:
                for article in articles:
                    st.markdown(f'<div class="resource-card"><a href="{article["url"]}" target="_blank">📄 {article["title"]} →</a></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="resource-card"><a href="https://www.google.com/search?q={disease.replace(" ", "+")}+skin" target="_blank">🔍 Search for more →</a></div>', unsafe_allow_html=True)

        # ===== NEAREST HOSPITALS =====
        show_hospitals_section(info.get('name', disease))

        # ===== DOWNLOAD REPORT =====
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
Why Does This Occur? {info.get('causes', '')}
Treatments: {info.get('treatments', '')}
Home Care: {info.get('home_care', '')}

DISCLAIMER: This AI analysis is for educational purposes only.
Always consult a qualified healthcare professional.
"""
        st.download_button("📥 Download Report", report, f"skin_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt", use_container_width=True)

    st.markdown("---")
    st.warning("⚠️ **Medical Disclaimer:** This AI tool is for educational purposes only. Always consult a qualified healthcare professional.")


if __name__ == "__main__":
    main()
