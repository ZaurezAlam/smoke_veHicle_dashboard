from rest_framework import serializers
from .models import VideoUpload

class VideoUploadSerializer(serializers.ModelSerializer):
    class Meta:
        model = VideoUpload
        fields = '__all__'
