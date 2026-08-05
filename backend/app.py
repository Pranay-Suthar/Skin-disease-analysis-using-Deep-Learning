#!/usr/bin/env python
"""Flask API for AI Skin Disease Detection"""

import os
import math
import json
import requests
from io import BytesIO
from datetime import datetime
from PIL import Image
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
from groq import Groq
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

# ===== CONFIGURATION =====
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max upload
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize SQLite database for users
DB_PATH = os.path.join(os.path.dirname(__file__), 'users.db')

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# Enable CORS
CORS(app, resources={r"/*": {"origins": "*"}})

# ===== ML MODEL SETUP =====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_MODEL_PATH = os.path.join(BASE_DIR, "models", "final_model_optimized")

LABEL_MAP = {
    'AK': 'Actinic Keratosis', 'BCC': 'Basal Cell Carcinoma',
    'BKL': 'Benign Keratosis', 'DF': 'Dermatofibroma', 'MEL': 'Melanoma',
    'NV': 'Melanocytic Nevus', 'SCC': 'Squamous Cell Carcinoma', 'VASC': 'Vascular Lesion'
}

DISEASE_INFO = {
    'Actinic Keratosis': {
        'name': 'Actinic Keratosis', 'severity': 'Moderate',
        'description': 'A rough, scaly patch on the skin caused by years of sun exposure. It is considered precancerous and can develop into squamous cell carcinoma if left untreated.',
        'causes': 'Cumulative UV radiation damage from sun exposure or tanning beds over many years. Risk factors include: fair skin, history of sunburns, age over 40, living in sunny climates.',
        'treatments': 'Cryotherapy (freezing), 5-Fluorouracil Cream, Imiquimod, Photodynamic Therapy, Chemical Peels',
        'home_care': 'Apply SPF 50+ sunscreen daily, wear protective clothing and hats, avoid sun during peak hours (10am-4pm), perform regular skin self-exams'
    },
    'Basal Cell Carcinoma': {
        'name': 'Basal Cell Carcinoma', 'severity': 'High',
        'description': 'The most common type of skin cancer. It begins in the basal cells which produce new skin cells. Usually appears as a slightly transparent bump on sun-exposed skin.',
        'causes': 'Primarily caused by long-term UV radiation exposure. Risk factors: fair skin, chronic sun exposure, radiation therapy, immunosuppression, arsenic exposure.',
        'treatments': 'Mohs Micrographic Surgery (most effective), Excisional Surgery, Curettage & Electrodesiccation, Radiation Therapy, Topical Medications',
        'home_care': 'See a dermatologist as soon as possible, document any changes with photos, protect the area from sun, avoid picking or scratching the lesion'
    },
    'Benign Keratosis': {
        'name': 'Benign Keratosis (Seborrheic Keratosis)', 'severity': 'Low',
        'description': "A common non-cancerous skin growth that appears as a waxy, wart-like spot. Often looks like it's stuck on the skin. These growths are harmless and don't become cancerous.",
        'causes': 'The exact cause is unknown, but they tend to run in families. NOT caused by sun exposure or viral infections. More common after age 50.',
        'treatments': 'Usually no treatment needed. If desired: Cryotherapy, Curettage, Electrosurgery, Laser Treatment',
        'home_care': 'No special care required. Keep moisturized. Monitor for sudden changes. Avoid irritating or scratching the growth.'
    },
    'Dermatofibroma': {
        'name': 'Dermatofibroma', 'severity': 'Low',
        'description': 'A common benign skin growth that feels like a hard bump under the skin. Usually brownish to red-purple. Often develops after minor injuries. Completely harmless.',
        'causes': 'Often triggered by minor skin injuries such as insect bites, splinters, or small cuts. More common in women and adults aged 20-50.',
        'treatments': 'Usually no treatment needed. If bothersome: Surgical excision, Cryotherapy, Laser removal',
        'home_care': 'Protect from repeated trauma. Monitor for significant changes in size or color.'
    },
    'Melanoma': {
        'name': 'Melanoma', 'severity': 'Critical',
        'description': 'SERIOUS SKIN CANCER - The most dangerous form of skin cancer. Develops from melanocytes. Can spread to other organs if not caught early. Look for ABCDEs: Asymmetry, Border, Color, Diameter >6mm, Evolving.',
        'causes': 'Caused by DNA damage to melanocytes from UV radiation. Risk factors: intense sun exposure, many moles, fair skin, family history, weakened immune system.',
        'treatments': 'Wide Excision Surgery, Sentinel Lymph Node Biopsy, Immunotherapy (Keytruda, Opdivo), Targeted Therapy, Radiation, Chemotherapy',
        'home_care': 'SEEK IMMEDIATE MEDICAL ATTENTION! Do not delay - early detection saves lives. Document with photos. Avoid sun exposure.'
    },
    'Melanocytic Nevus': {
        'name': 'Melanocytic Nevus (Common Mole)', 'severity': 'Low',
        'description': 'A common benign growth on the skin, known as a mole. Formed by clusters of melanocytes. Most people have 10-40 moles. Can be flat or raised, pink to dark brown.',
        'causes': 'Moles form when melanocytes grow in clusters. Caused by genetics and sun exposure. Most develop during childhood and adolescence.',
        'treatments': 'Usually no treatment needed. Removal: Surgical excision, Shave removal. Remove if suspicious (ABCDE criteria).',
        'home_care': 'Monthly self-exams using ABCDE rule. Protect from sun. Take photos to track changes. See a dermatologist annually.'
    },
    'Squamous Cell Carcinoma': {
        'name': 'Squamous Cell Carcinoma', 'severity': 'High',
        'description': 'The second most common skin cancer. Develops in squamous cells. Usually caused by UV exposure. Can spread if not treated.',
        'causes': 'Cumulative UV exposure from sun or tanning beds. Risk factors: fair skin, chronic sun exposure, weakened immune system, HPV infection.',
        'treatments': 'Mohs Surgery, Excisional Surgery, Curettage & Electrodesiccation, Radiation Therapy, Topical Chemotherapy (5-FU)',
        'home_care': 'See a dermatologist urgently. Protect from sun. Document changes with photos.'
    },
    'Vascular Lesion': {
        'name': 'Vascular Lesion', 'severity': 'Low',
        'description': 'An abnormality of blood vessels in or under the skin. Includes cherry angiomas, spider veins, hemangiomas. Almost always benign and primarily cosmetic.',
        'causes': 'Develop due to aging, genetics, or hormonal changes. Cherry angiomas increase with age. Spider veins caused by sun, hormones, or prolonged standing.',
        'treatments': 'Usually no treatment needed. Cosmetic: Pulsed Dye Laser, IPL, Sclerotherapy, Electrocautery',
        'home_care': 'Protect from injury. No special care required. Monitor for rapid growth or changes.'
    },
}

class SkinModelService:
    _instance = None
    _processor = None
    _model = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._load_model()
        return cls._instance

    def _load_model(self):
        try:
            if not os.path.exists(LOCAL_MODEL_PATH):
                print(f"[WARNING] Model path not found: {LOCAL_MODEL_PATH}")
                self._model = None
                return
            self._processor = AutoImageProcessor.from_pretrained(LOCAL_MODEL_PATH, local_files_only=True)
            self._model = AutoModelForImageClassification.from_pretrained(LOCAL_MODEL_PATH, local_files_only=True)
            self._model.eval()
            print("[OK] ML Model loaded successfully")
        except Exception as e:
            print(f"[WARNING] Model loading failed (using demo mode): {str(e)[:100]}")
            self._model = None

    def predict(self, image_pil):
        if self._model is None or self._processor is None:
            # Demo mode - return mock prediction
            print("[WARNING] Model not available - returning demo prediction")
            return self._demo_predict(image_pil)
        
        image = image_pil.convert('RGB')
        inputs = self._processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)[0]
        
        predicted_idx = logits.argmax(-1).item()
        predicted_abbr = self._model.config.id2label[predicted_idx]
        predicted_label = LABEL_MAP.get(predicted_abbr, predicted_abbr)
        confidence = float(probs[predicted_idx].item())
        
        top_3 = []
        for idx in torch.argsort(probs, descending=True)[:3]:
            abbr = self._model.config.id2label[idx.item()]
            top_3.append({
                "disease": LABEL_MAP.get(abbr, abbr),
                "probability": float(probs[idx].item())
            })
            
        info = DISEASE_INFO.get(predicted_label, DISEASE_INFO.get(predicted_label.split(' ')[0], {}))
            
        return {
            'disease': predicted_label,
            'confidence': confidence,
            'top_3': top_3,
            'severity': info.get('severity', 'Unknown'),
            'info': info
        }
    
    def _demo_predict(self, image_pil):
        """Demo mode when model is not available"""
        import random
        disease_list = list(DISEASE_INFO.keys())
        disease = random.choice(disease_list)
        info = DISEASE_INFO[disease]
        
        return {
            'disease': disease,
            'confidence': round(random.uniform(0.7, 0.95), 3),
            'top_3': [
                {'disease': disease, 'probability': round(random.uniform(0.7, 0.95), 3)},
                {'disease': random.choice(disease_list), 'probability': round(random.uniform(0.05, 0.15), 3)},
                {'disease': random.choice(disease_list), 'probability': round(random.uniform(0.05, 0.10), 3)},
            ],
            'severity': info.get('severity', 'Low'),
            'info': info
        }

def get_disease_info(disease_label):
    if disease_label in DISEASE_INFO:
        return DISEASE_INFO[disease_label]
    for key in DISEASE_INFO:
        if key.lower() == disease_label.lower():
            return DISEASE_INFO[key]
    return {
        'name': disease_label, 'severity': 'Unknown',
        'description': 'Skin condition detected. Please consult a dermatologist.',
        'treatments': 'Consult a dermatologist', 'home_care': 'Monitor for changes',
    }

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ===== HELPER FUNCTIONS =====
def haversine_km(lat1, lon1, lat2, lon2):
    """Calculate distance between two coordinates in kilometers"""
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

# ===== API ENDPOINTS =====

@app.route('/', methods=['GET'])
def home():
    """Home endpoint showing API information"""
    return jsonify({
        'name': 'AI Skin Disease Detection API',
        'version': '1.0.0',
        'status': 'running',
        'endpoints': {
            'POST /api/predict/': 'Upload image for skin disease prediction',
            'POST /api/hospitals/': 'Find nearby hospitals/clinics',
            'POST /api/chat/': 'Chat with SkinBot AI assistant',
            'GET /api/health/': 'Health check',
            'POST /api/signup/': 'Register a new user',
            'POST /api/login/': 'Login an existing user'
        }
    }), 200

@app.route('/api/signup/', methods=['POST'])
def signup():
    """Register a new user"""
    try:
        data = request.get_json()
        name = data.get('name')
        email = data.get('email')
        password = data.get('password')
        
        if not all([name, email, password]):
            return jsonify({'error': 'Name, email, and password are required'}), 400
            
        hashed_password = generate_password_hash(password)
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        try:
            cursor.execute('INSERT INTO users (name, email, password) VALUES (?, ?, ?)',
                         (name, email, hashed_password))
            conn.commit()
            return jsonify({'message': 'User registered successfully', 'user': {'name': name, 'email': email}}), 201
        except sqlite3.IntegrityError:
            return jsonify({'error': 'Email already exists'}), 400
        finally:
            conn.close()
            
    except Exception as e:
        return jsonify({'error': f'Registration failed: {str(e)}'}), 500

@app.route('/api/login/', methods=['POST'])
def login():
    """Login user"""
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        
        if not all([email, password]):
            return jsonify({'error': 'Email and password are required'}), 400
            
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
        user = cursor.fetchone()
        conn.close()
        
        if user and check_password_hash(user['password'], password):
            return jsonify({
                'message': 'Login successful',
                'user': {'id': user['id'], 'name': user['name'], 'email': user['email']}
            }), 200
        else:
            return jsonify({'error': 'Invalid credentials'}), 401
            
    except Exception as e:
        return jsonify({'error': f'Login failed: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    model_status = "loaded" if SkinModelService.get_instance()._model is not None else "failed"
    return jsonify({'status': 'healthy', 'model': model_status}), 200

@app.route('/api/predict/', methods=['POST'])
def predict():
    """Predict skin disease from uploaded image"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Use PNG, JPG, or JPEG'}), 400
        
        # Read and process image
        image_pil = Image.open(file.stream)
        
        # Get predictions
        ml_service = SkinModelService.get_instance()
        results = ml_service.predict(image_pil)
        
        if not results:
            return jsonify({'error': 'Prediction failed. Please try again.'}), 500
        
        # Save uploaded image
        filename = secure_filename(f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        image_pil.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        results['scan_id'] = filename
        results['mode'] = 'demo' if ml_service._model is None else 'production'
        return jsonify(results), 200
        
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

@app.route('/api/hospitals/', methods=['POST'])
def find_hospitals():
    """Find nearby hospitals and clinics within 100km using Overpass API"""
    try:
        data = request.get_json()
        location_query = data.get('location', '').strip()
        lat = data.get('lat')
        lon = data.get('lon')
        
        if not location_query and (lat is None or lon is None):
            return jsonify({'error': 'Location or coordinates required'}), 400
        
        display_name = "Your Location"
        
        # Step 1: Geocode if lat/lon not provided
        if lat is None or lon is None:
            try:
                geocode_response = requests.get(
                    "https://nominatim.openstreetmap.org/search",
                    params={
                        "q": location_query,
                        "format": "json",
                        "limit": 1
                    },
                    headers={"User-Agent": "SkinAppFlask/1.0"},
                    timeout=8
                )
                geo_data = geocode_response.json()
                
                if not geo_data:
                    return jsonify({'error': 'Location not found'}), 404
                
                lat = float(geo_data[0]["lat"])
                lon = float(geo_data[0]["lon"])
                display_name = geo_data[0].get("display_name", location_query)
            except Exception as e:
                return jsonify({'error': f'Failed to geocode location: {str(e)}'}), 500
        else:
            lat, lon = float(lat), float(lon)
            if location_query:
                display_name = location_query
        
        # Step 2: Find hospitals and clinics using Overpass API
        # Use multiple amenity types — in India many are tagged as 'doctors' or 'healthcare'
        overpass_url = "https://overpass-api.de/api/interpreter"
        overpass_query = f"""
        [out:json][timeout:30];
        (
          node["amenity"="hospital"](around:100000,{lat},{lon});
          way["amenity"="hospital"](around:100000,{lat},{lon});
          node["amenity"="clinic"](around:100000,{lat},{lon});
          way["amenity"="clinic"](around:100000,{lat},{lon});
          node["amenity"="doctors"](around:100000,{lat},{lon});
          way["amenity"="doctors"](around:100000,{lat},{lon});
          node["amenity"="health_post"](around:100000,{lat},{lon});
          node["healthcare"="hospital"](around:100000,{lat},{lon});
          way["healthcare"="hospital"](around:100000,{lat},{lon});
          node["healthcare"="clinic"](around:100000,{lat},{lon});
          node["healthcare"="doctor"](around:100000,{lat},{lon});
          node["healthcare"="dermatologist"](around:100000,{lat},{lon});
          node["healthcare:speciality"="dermatology"](around:100000,{lat},{lon});
          node["specialty"="dermatology"](around:100000,{lat},{lon});
          node["specialty"="skin"](around:100000,{lat},{lon});
        );
        out center;
        """
        
        found = []
        try:
            response = requests.post(
                overpass_url,
                data={'data': overpass_query},
                timeout=30,
                headers={"User-Agent": "SkinAppFlask/1.0"}
            )
            data = response.json()
            
            seen = set()
            for element in data.get('elements', []):
                tags = element.get('tags', {})
                name = tags.get('name') or tags.get('name:en') or tags.get('operator')
                if not name or name in seen:
                    continue
                
                el_lat = element.get('lat') or (element.get('center', {}).get('lat'))
                el_lon = element.get('lon') or (element.get('center', {}).get('lon'))
                
                if el_lat is None or el_lon is None:
                    continue
                
                seen.add(name)
                distance = haversine_km(lat, lon, el_lat, el_lon)
                
                addr_parts = [
                    tags.get('addr:housenumber', ''),
                    tags.get('addr:street', ''),
                    tags.get('addr:suburb', ''),
                    tags.get('addr:city', '') or tags.get('addr:district', ''),
                    tags.get('addr:state', '')
                ]
                address = ", ".join([p for p in addr_parts if p]) or "Address not available"
                
                found.append({
                    "id": str(element.get('id')),
                    "name": name,
                    "lat": el_lat,
                    "lon": el_lon,
                    "type": (tags.get('healthcare') or tags.get('amenity', 'clinic')).title(),
                    "distance": round(distance, 2),
                    "address": address,
                    "phone": tags.get('phone', tags.get('contact:phone', tags.get('contact:mobile', ''))),
                    "website": tags.get('website', tags.get('contact:website', '')),
                    "rating": 4.2,
                    "reviews": 0,
                    "openNow": None
                })
        except Exception as e:
            print(f"Overpass API error: {e}")
            
        # Sort by distance
        found.sort(key=lambda x: x["distance"])
        
        # Return top 20 nearest
        return jsonify({
            'user_location': {'lat': lat, 'lon': lon, 'display_name': display_name},
            'hospitals': found[:20]
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Error finding hospitals: {str(e)}'}), 500

@app.route('/api/chat/', methods=['POST'])
def chat():
    """Chat with SkinBot AI assistant"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        disease_name = data.get('disease', '')
        history = data.get('history', [])  # list of {role, content}
        
        if not message:
            return jsonify({'error': 'Message required'}), 400
        
        groq_api_key = os.getenv('GROQ_API_KEY')
        if not groq_api_key:
            return jsonify({'reply': "Error: Groq API Key missing in environment."}), 500
        
        disease_info = get_disease_info(disease_name) if disease_name else {}
        condition_context = ""
        if disease_info and disease_name:
            condition_context = f"""
The user has been diagnosed with: {disease_info.get('name', disease_name)}
Severity: {disease_info.get('severity', 'Unknown')}
Description: {disease_info.get('description', '')}
Recommended Treatments: {disease_info.get('treatments', '')}
Home Care: {disease_info.get('home_care', '')}
"""
        
        system_prompt = f"""You are SkinBot, a highly knowledgeable and friendly AI assistant specializing in dermatology, skincare, and skin health. You work for DermaAI, an AI-powered skin analysis platform.

{condition_context}

YOUR EXPERTISE INCLUDES:
- Skin diseases and conditions (eczema, acne, psoriasis, rosacea, dermatitis, fungal infections, etc.)
- Skincare routines for all skin types (oily, dry, combination, sensitive)
- Skincare products — you CAN and SHOULD recommend real, specific products (e.g. CeraVe, La Roche-Posay, The Ordinary, Neutrogena, Cetaphil, Paula's Choice, Differin, Vanicream, etc.)
- Ingredients to look for and avoid (retinol, niacinamide, hyaluronic acid, AHAs, BHAs, vitamin C, etc.)
- Face cleansers, moisturizers, serums, sunscreens, toners, exfoliants
- Facial exercises and massage techniques that improve circulation and skin health
- Diet, hydration, and lifestyle factors affecting skin
- Sun protection (SPF, UVA/UVB, mineral vs chemical sunscreen)
- Anti-aging skincare
- Natural remedies for skin conditions
- When to see a dermatologist vs. when home care is sufficient
- Skin concerns by age and skin tone

GUIDELINES:
- Be helpful, warm, and detailed. Users rely on you for real guidance.
- Recommend specific, real product names when asked about products. Include approximate price ranges when helpful.
- Use bullet points and emojis to make responses readable and engaging.
- Format responses with clear sections when answering multi-part questions.
- Always note that severe conditions should be checked by a dermatologist, but don't refuse to help with any skin-related topic.
- STAY on-topic: only answer questions related to skin, skincare, dermatology, beauty routines, facial health, and wellness topics that directly affect skin.
- If asked something completely unrelated to skin/beauty (like math, coding, politics), politely redirect them to skin topics.
- Keep responses under 400 words unless complex treatment plans are needed.
- If a condition is detected, proactively offer to explain treatments, products, prevention, or what to expect."""

        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history (last 8 turns max)
        for h in history[-8:]:
            if h.get('role') in ('user', 'assistant') and h.get('content'):
                messages.append({"role": h['role'], "content": h['content']})
        
        messages.append({"role": "user", "content": message})
        
        try:
            client = Groq(api_key=groq_api_key)
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                max_tokens=600,
                temperature=0.7
            )
            reply = response.choices[0].message.content
        except Exception as e:
            reply = f"I'm having trouble connecting right now. Please try again in a moment. ({str(e)[:80]})"
        
        return jsonify({'reply': reply}), 200
        
    except Exception as e:
        return jsonify({'error': f'Chat error: {str(e)}'}), 500


# ===== ERROR HANDLERS =====
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# ===== MAIN =====
if __name__ == '__main__':
    print("=" * 45)
    print("  AI Skin Disease Detection - Flask API")
    print("  Starting Server...")
    print("=" * 45)
    
    # Load model on startup
    SkinModelService.get_instance()
    
    app.run(
        host='127.0.0.1',
        port=5000,
        debug=False,
        use_reloader=False
    )
