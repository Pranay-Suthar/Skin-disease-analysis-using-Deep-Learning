import os
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image

# Use the root models directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
                raise FileNotFoundError(f"Model path not found: {LOCAL_MODEL_PATH}")
            self._processor = AutoImageProcessor.from_pretrained(LOCAL_MODEL_PATH, local_files_only=True)
            self._model = AutoModelForImageClassification.from_pretrained(LOCAL_MODEL_PATH, local_files_only=True)
            self._model.eval()
        except Exception as e:
            print(f"Error loading model: {e}")
            self._model = None

    def predict(self, image_pil):
        if self._model is None or self._processor is None:
            return None
        
        image = image_pil.convert('RGB')
        inputs = self._processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)[0]
        
        predicted_idx = logits.argmax(-1).item()
        predicted_abbr = self._model.config.id2label[predicted_idx]
        predicted_label = LABEL_MAP.get(predicted_abbr, predicted_abbr)
        confidence = probs[predicted_idx].item()
        
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
