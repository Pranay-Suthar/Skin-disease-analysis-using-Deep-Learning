from django.urls import path
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
from .views import (
    RegisterView, UserProfileView, PredictSkinView, 
    ScanHistoryListView, HospitalFinderView, SkinBotChatView
)

urlpatterns = [
    path('auth/register/', RegisterView.as_view(), name='register'),
    path('auth/login/', TokenObtainPairView.as_view(), name='login'),
    path('auth/refresh/', TokenRefreshView.as_view(), name='refresh'),
    path('auth/me/', UserProfileView.as_view(), name='me'),
    
    path('predict/', PredictSkinView.as_view(), name='predict'),
    path('history/', ScanHistoryListView.as_view(), name='history'),
    path('hospitals/', HospitalFinderView.as_view(), name='hospitals'),
    path('chat/', SkinBotChatView.as_view(), name='chat'),
]
