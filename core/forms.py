# core/forms.py (Example)
from django import forms
from .models import VideoUpload

class VideoUploadForm(forms.ModelForm):
    class Meta:
        model = VideoUpload
        fields = ['video_file'] # Or '__all__' if you want all fields