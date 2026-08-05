from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.http import HttpResponse
from django.shortcuts import redirect

def home_view(request):
    """Simple home view that redirects to API or shows welcome message"""
    return HttpResponse("""
    <html>
    <head><title>Skin App Backend</title></head>
    <body style="font-family: Arial, sans-serif; text-align: center; padding: 50px;">
        <h1>Skin App Backend API</h1>
        <p>Welcome to the Skin Disease Detection API</p>
        <p>Available endpoints:</p>
        <ul style="list-style: none; padding: 0;">
            <li><a href="/admin/">Admin Panel</a></li>
            <li><a href="/api/">API Endpoints</a></li>
            <li><a href="/api/auth/login/">Login</a></li>
            <li><a href="/api/predict/">Skin Prediction</a></li>
        </ul>
        <p>Frontend is running at: <a href="http://localhost:5173">http://localhost:5173</a></p>
    </body>
    </html>
    """)

urlpatterns = [
    path('', home_view, name='home'),
    path('admin/', admin.site.urls),
    path('api/', include('api.urls')),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
