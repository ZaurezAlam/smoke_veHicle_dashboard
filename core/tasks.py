# core/tasks.py

import os
import time
import shutil
import logging

from celery import shared_task
from django.conf import settings
from .models import VideoUpload
from .processing import process_video # Make sure this import is correct

logger = logging.getLogger(__name__)

@shared_task(bind=True)
def process_video_task(self, video_upload_id, original_video_path):
    video_obj = None
    processing_output_dir = None

    try:
        video_obj = VideoUpload.objects.get(id=video_upload_id)

        # Create a unique temporary directory for this task's outputs
        processing_output_dir = os.path.join(
            settings.MEDIA_ROOT, 'temp_processing_outputs', str(video_obj.id)
        )
        os.makedirs(processing_output_dir, exist_ok=True)
        logger.info(f"Task {self.request.id}: Created temporary dir: {processing_output_dir}")

        # --- Stage 1: Initializing (5%) ---
        self.update_state(state='PROGRESS', meta={'percent': 5, 'status': 'Starting video analysis...'})
        logger.info(f"Task {self.request.id}: Status: Starting video analysis (5%).")

        if not os.path.exists(original_video_path):
            raise FileNotFoundError(f"Original video file not found: {original_video_path}")

        # --- Stage 2: Core Video Processing (20% - 60%) ---
        # The `process_video` function now handles updates from 20% to 60% internally.
        # This initial 20% update in tasks.py signals the start of the heavy lifting.
        self.update_state(state='PROGRESS', meta={'percent': 20, 'status': 'Running object detection...'})
        logger.info(f"Task {self.request.id}: Status: Running object detection and frame processing (20%).")

        # Pass 'self' (the Celery task instance) to process_video
# core/tasks.py

# ... (previous code) ...

        processing_results = process_video(original_video_path, processing_output_dir, celery_task_instance=self)
        self.update_state(state='PROGRESS', meta={'percent': 85, 'status': 'Reports and charts generated.'})
        logger.info(f"Task {self.request.id}: Status: Reports and charts generated (85%).")

        # Extract file names from the full paths returned by process_video
        # Use os.path.basename to get just the file name
        processed_video_file_name = os.path.basename(processing_results.get('processed_video', ''))
        # IMPORTANT: Ensure 'csv' and 'excel' keys match actual keys from process_video's return dict
        # Based on your snippet, these are 'csv', 'excel', 'bar_chart', 'pie_chart'
        csv_file_name = os.path.basename(processing_results.get('csv', ''))
        excel_file_name = os.path.basename(processing_results.get('excel', ''))
        bar_chart_file_name = os.path.basename(processing_results.get('bar_chart', ''))
        pie_chart_file_name = os.path.basename(processing_results.get('pie_chart', ''))

        # Add the new 'all_detections_csv' path (assuming 'detection_log' is the key from process_video)
        detection_csv_file_name = os.path.basename(processing_results.get('detection_log', '')) # Using detection_log as discussed earlier

        # --- Stage 4: Moving Processed Files to Final Media Locations and Updating DB (85% - 95%) ---

        # Update VideoUpload object fields
        # IMPORTANT: Use the `upload_to` paths defined in your models.py
        # Django's FileField will handle joining with MEDIA_ROOT
        if processed_video_file_name:
            video_obj.processed_video.name = os.path.join(video_obj.processed_video.field.upload_to, processed_video_file_name)
        
        # processed_csv for the main output CSV
        if csv_file_name:
            video_obj.processed_csv.name = os.path.join(video_obj.processed_csv.field.upload_to, csv_file_name)
            
        # detection_csv for the all_detections_csv
        if detection_csv_file_name:
            video_obj.detection_csv.name = os.path.join(video_obj.detection_csv.field.upload_to, detection_csv_file_name)

        if excel_file_name:
            video_obj.summary_excel.name = os.path.join(video_obj.summary_excel.field.upload_to, excel_file_name)

        if bar_chart_file_name:
            video_obj.bar_chart_image.name = os.path.join(video_obj.bar_chart_image.field.upload_to, bar_chart_file_name) # FIX: bar_chart_image
            
        if pie_chart_file_name:
            video_obj.pie_chart_image.name = os.path.join(video_obj.pie_chart_image.field.upload_to, pie_chart_file_name) # FIX: pie_chart_image

        # --- Update the statistical fields ---
        video_obj.total_frames = processing_results.get('total_frames')
        video_obj.total_plates_detected = processing_results.get('total_plates_detected')
        video_obj.successful_ocr = processing_results.get('successful_ocr')
        video_obj.unique_plates = processing_results.get('unique_plates')
        video_obj.smoke_vehicles = processing_results.get('smoke_vehicles')
        video_obj.non_smoke_vehicles = processing_results.get('non_smoke_vehicles')

        # Ensure target directories exist *for each specific upload_to path*
        # These will be subdirectories of MEDIA_ROOT
        upload_dirs = [
            video_obj.processed_video.field.upload_to,
            video_obj.processed_csv.field.upload_to,
            video_obj.detection_csv.field.upload_to,
            video_obj.summary_excel.field.upload_to,
            video_obj.bar_chart_image.field.upload_to,
            video_obj.pie_chart_image.field.upload_to,
        ]
        for upload_dir in upload_dirs:
            full_path = os.path.join(settings.MEDIA_ROOT, upload_dir)
            os.makedirs(full_path, exist_ok=True)


        # Move processed files to their final media locations
        # Use the actual paths from processing_results, and join them with settings.MEDIA_ROOT
        # and the base file name obtained earlier.
        if processed_video_file_name and processing_results.get('processed_video'):
            shutil.move(
                processing_results['processed_video'], 
                os.path.join(settings.MEDIA_ROOT, video_obj.processed_video.name) # Use the .name attribute
            )
        
        if csv_file_name and processing_results.get('csv'):
            shutil.move(
                processing_results['csv'], 
                os.path.join(settings.MEDIA_ROOT, video_obj.processed_csv.name)
            )
        
        if detection_csv_file_name and processing_results.get('detection_log'):
             shutil.move(
                processing_results['detection_log'], 
                os.path.join(settings.MEDIA_ROOT, video_obj.detection_csv.name)
            )

        if excel_file_name and processing_results.get('excel'):
            shutil.move(
                processing_results['excel'], 
                os.path.join(settings.MEDIA_ROOT, video_obj.summary_excel.name)
            )
        
        if bar_chart_file_name and processing_results.get('bar_chart'):
            shutil.move(
                processing_results['bar_chart'], 
                os.path.join(settings.MEDIA_ROOT, video_obj.bar_chart_image.name)
            )
        
        if pie_chart_file_name and processing_results.get('pie_chart'):
            shutil.move(
                processing_results['pie_chart'], 
                os.path.join(settings.MEDIA_ROOT, video_obj.pie_chart_image.name)
            )

        self.update_state(state='PROGRESS', meta={'percent': 95, 'status': 'Files moved. Cleaning up...'})
        logger.info(f"Task {self.request.id}: Status: Files moved (95%).")

        # --- IMPORTANT: Update status and progress, then save ---
        video_obj.status = 'Completed' # Use the new status field
        video_obj.progress = 100     # Set to 100% on completion
        video_obj.save()             # Save all the changes to the DB
        logger.info(f"Task {self.request.id}: VideoUpload object updated and saved for {video_upload_id}.")

        # --- Stage 5: Cleanup and Finalization (95% - 100%) ---
        if os.path.exists(processing_output_dir):
            shutil.rmtree(processing_output_dir)
            logger.info(f"Task {self.request.id}: Cleaned up temporary directory: {processing_output_dir}")

        self.update_state(state='SUCCESS', meta={'percent': 100, 'status': 'Video processing complete!', 'video_id': video_upload_id})
        logger.info(f"Task {self.request.id}: Processing successfully completed for video {video_upload_id} (100%).")

    except VideoUpload.DoesNotExist:
        logger.error(f"Task {self.request.id}: VideoUpload object with ID {video_upload_id} not found.", exc_info=True)
        self.update_state(state='FAILURE', meta={'percent': 0, 'status': 'Video not found.'})
        raise
    except Exception as e:
        logger.error(f"Task {self.request.id}: Video processing failed: {e}", exc_info=True)
        if video_obj and hasattr(video_obj, 'pk') and video_obj.pk:
            try:
                # On failure, set status to 'Failed'
                video_obj.status = 'Failed'
                video_obj.progress = 0 # Or a value indicating failure, like 0
                video_obj.save(update_fields=['status', 'progress'])
            except Exception as save_error:
                logger.error(f"Task {self.request.id}: Failed to update video object on error: {save_error}")
        if processing_output_dir and os.path.exists(processing_output_dir):
            shutil.rmtree(processing_output_dir, ignore_errors=True)
        self.update_state(state='FAILURE', meta={'percent': 0, 'status': f'Processing failed: {str(e)}'})
        raise


