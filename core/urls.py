from django.urls import path
from .views import (
    VideoUploadView,
    video_list_view,
    clear_all_videos,
    VehicleVideoCreateAPIView,
    ProcessingStatusAPIView,
    send_email_summary
)
from . import views

app_name = 'core'

urlpatterns = [
    # Web views (Django templates)
    path('', VideoUploadView.as_view(), name='upload_video'),              # Main upload page (GET/POST)
    path('videos/', video_list_view, name='video_list'),                  # Optional: list of all videos
    path('clear/', clear_all_videos, name='clear_all_videos'),           # Clear all uploaded/processed videos

    # API endpoints (used by frontend JS or other clients)
    path('api/upload/', VehicleVideoCreateAPIView.as_view(), name='api_upload_video'),  # API upload endpoint
    path('api/processing-status/<int:video_id>/', ProcessingStatusAPIView.as_view(), name='api_processing_status'),
    path('api/send-email-summary/', views.send_email_summary, name='send_email_summary'),   # Polling endpoint
]
