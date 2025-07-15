# core/models.py
from django.db import models

class VideoUpload(models.Model):
    video_file = models.FileField(upload_to='videos/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    # Status and progress for real-time updates
    status = models.CharField(max_length=50, default='Uploaded') # e.g., Uploaded, Processing, Completed, Failed
    progress = models.IntegerField(default=0) # Percentage progress (0-100)
    celery_task_id = models.CharField(max_length=255, null=True, blank=True, help_text="Celery task ID for video processing")

    # Fields to store processed outputs
    processed_video = models.FileField(upload_to='processed_videos/', null=True, blank=True)
    processed_csv = models.FileField(upload_to='processed_csvs/', null=True, blank=True) # For output_csv (main plates.csv)
    detection_csv = models.FileField(upload_to='detection_logs/', null=True, blank=True) # For all_detections_csv
    summary_excel = models.FileField(upload_to='summaries/', null=True, blank=True)
    bar_chart_image = models.FileField(upload_to='charts/', null=True, blank=True) # Renamed to _image for clarity
    pie_chart_image = models.FileField(upload_to='charts/', null=True, blank=True) # Renamed to _image for clarity

    # Fields for aggregated statistics from processing
    total_frames = models.IntegerField(null=True, blank=True)
    total_plates_detected = models.IntegerField(null=True, blank=True)
    successful_ocr = models.IntegerField(null=True, blank=True)
    unique_plates = models.IntegerField(null=True, blank=True)
    smoke_vehicles = models.IntegerField(null=True, blank=True)
    non_smoke_vehicles = models.IntegerField(null=True, blank=True)

    def __str__(self):
        return f"Video {self.id} - {self.status} (Task ID: {self.celery_task_id or 'N/A'})"

    class Meta:
        ordering = ['-uploaded_at']