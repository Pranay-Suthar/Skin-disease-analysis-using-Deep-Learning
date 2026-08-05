import os
import math
import requests
from rest_framework import generics, status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, AllowAny
from django.contrib.auth.models import User
from groq import Groq
from PIL import Image

from .models import ScanHistory, ChatMessage
from .serializers import RegisterSerializer, UserSerializer, ScanHistorySerializer, ChatMessageSerializer
from .ml_service import SkinModelService, get_disease_info

class RegisterView(generics.CreateAPIView):
    queryset = User.objects.all()
    permission_classes = (AllowAny,)
    serializer_class = RegisterSerializer

class UserProfileView(generics.RetrieveAPIView):
    permission_classes = (IsAuthenticated,)
    serializer_class = UserSerializer

    def get_object(self):
        return self.request.user

class PredictSkinView(APIView):
    permission_classes = (AllowAny,)

    def post(self, request):
        if 'image' not in request.FILES:
            return Response({'error': 'No image provided'}, status=status.HTTP_400_BAD_REQUEST)
        
        image_file = request.FILES['image']
        try:
            image_pil = Image.open(image_file)
        except Exception:
            return Response({'error': 'Invalid image file'}, status=status.HTTP_400_BAD_REQUEST)
        
        ml_service = SkinModelService.get_instance()
        results = ml_service.predict(image_pil)
        
        if not results:
            return Response({'error': 'Model failed to load or predict'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        # Save history if logged in
        if request.user.is_authenticated:
            history = ScanHistory.objects.create(
                user=request.user,
                image=image_file,
                disease=results['disease'],
                confidence=results['confidence'],
                top_3=results['top_3'],
                severity=results['severity']
            )
            results['scan_id'] = history.id
            
        return Response(results)

class ScanHistoryListView(generics.ListAPIView):
    serializer_class = ScanHistorySerializer
    permission_classes = (IsAuthenticated,)

    def get_queryset(self):
        return ScanHistory.objects.filter(user=self.request.user).order_by('-created_at')

# --- Hospital Finder logic ---
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

class HospitalFinderView(APIView):
    permission_classes = (AllowAny,)

    def post(self, request):
        query = request.data.get('location')
        if not query:
            return Response({'error': 'Location required'}, status=status.HTTP_400_BAD_REQUEST)
        
        # 1. Geocode
        lat, lon, display = None, None, None
        try:
            r = requests.get("https://nominatim.openstreetmap.org/search",
                             params={"q": query, "format": "json", "limit": 1},
                             headers={"User-Agent": "SkinDiseaseCheckerDjango/1.0"}, timeout=8)
            data = r.json()
            if data:
                lat, lon, display = float(data[0]["lat"]), float(data[0]["lon"]), data[0].get("display_name", query)
        except Exception:
            return Response({'error': 'Failed to geocode location'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        if lat is None:
            return Response({'error': 'Location not found'}, status=status.HTTP_404_NOT_FOUND)
            
        # 2. Find hospitals
        headers = {"User-Agent": "SkinDiseaseCheckerDjango/1.0"}
        found = []
        seen = set()
        for delta in [0.2, 0.5, 1.0, 2.0]:
            viewbox = f"{lon-delta},{lat+delta},{lon+delta},{lat-delta}"
            for amenity in ["hospital", "clinic"]:
                try:
                    resp = requests.get(
                        "https://nominatim.openstreetmap.org/search",
                        params={"q": amenity, "format": "json", "limit": 15, "viewbox": viewbox, "bounded": 1, "addressdetails": 1},
                        headers=headers, timeout=10)
                    for item in resp.json():
                        name = item.get("display_name", "").split(",")[0].strip()
                        if not name or name in seen: continue
                        seen.add(name)
                        h_lat, h_lon = float(item["lat"]), float(item["lon"])
                        found.append({
                            "name": name, "lat": h_lat, "lon": h_lon,
                            "type": amenity.title(),
                            "distance_km": round(haversine_km(lat, lon, h_lat, h_lon), 2),
                            "address": item.get("display_name", "").split(",", 1)[-1].strip()
                        })
                except Exception:
                    continue
            found.sort(key=lambda x: x["distance_km"])
            deduped = []
            for h in found:
                if not any(haversine_km(h["lat"], h["lon"], d["lat"], d["lon"]) < 0.1 for d in deduped):
                    deduped.append(h)
            found = deduped
            if len(found) >= 6: break
            
        return Response({
            'user_location': {'lat': lat, 'lon': lon, 'display_name': display},
            'hospitals': found[:6]
        })

class SkinBotChatView(APIView):
    permission_classes = (AllowAny,)
    
    def post(self, request):
        message = request.data.get('message')
        disease_name = request.data.get('disease', 'Unknown')
        scan_id = request.data.get('scan_id')
        
        if not message:
            return Response({'error': 'Message required'}, status=status.HTTP_400_BAD_REQUEST)
            
        groq_api_key = os.getenv('GROQ_API_KEY')
        if not groq_api_key:
            return Response({'reply': "Error: Groq API Key missing in environment."})
            
        disease_info = get_disease_info(disease_name)
        system = f"""You are SkinBot, a friendly dermatology assistant.
Condition: {disease_info.get('name','Unknown')}, Severity: {disease_info.get('severity','Unknown')}.
Description: {disease_info.get('description','')}. Treatments: {disease_info.get('treatments','')}.
Rules: Keep responses professional, clear, 2-4 sentences, and always recommend seeing a dermatologist."""
        
        try:
            client = Groq(api_key=groq_api_key)
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "system", "content": system}, {"role": "user", "content": message}],
                max_tokens=300, temperature=0.7
            )
            reply = resp.choices[0].message.content
        except Exception as e:
            reply = f"Error communicating with AI: {str(e)[:100]}"
            
        if request.user.is_authenticated and scan_id:
            try:
                scan = ScanHistory.objects.get(id=scan_id, user=request.user)
                ChatMessage.objects.create(user=request.user, scan=scan, role='user', content=message)
                ChatMessage.objects.create(user=request.user, scan=scan, role='bot', content=reply)
            except ScanHistory.DoesNotExist:
                pass
                
        return Response({'reply': reply})
