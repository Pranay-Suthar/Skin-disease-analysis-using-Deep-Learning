from django.db import models
from django.contrib.auth.models import User

class ScanHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='scans', null=True, blank=True)
    image = models.ImageField(upload_to='scans/')
    disease = models.CharField(max_length=100)
    confidence = models.FloatField()
    top_3 = models.JSONField(default=list)
    severity = models.CharField(max_length=20, default='Unknown')
    location_query = models.CharField(max_length=255, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.disease} - {self.created_at.strftime('%Y-%m-%d %H:%M')}"

class ChatMessage(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='chats', null=True, blank=True)
    scan = models.ForeignKey(ScanHistory, on_delete=models.SET_NULL, null=True, blank=True, related_name='messages')
    role = models.CharField(max_length=10) # 'user' or 'bot'
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.role}: {self.content[:30]}"
