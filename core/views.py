# core/views.py
import os
import shutil
import logging

from django.shortcuts import render, redirect
from django.conf import settings
from django.views import View
from django.http import JsonResponse, HttpResponseNotFound
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
from celery.result import AsyncResult
from smoke_vehicle_dashboard_project.celery import app as celery_app 
from django.core.mail import EmailMessage # Keep this import
from django.conf import settings # Keep this import
import os # Keep this import
from .forms import VideoUploadForm
from .models import VideoUpload
from .serializers import VideoUploadSerializer
from core.tasks import process_video_task
from django.utils import timezone


logger = logging.getLogger(__name__)

# --- API View for Uploading and Triggering Processing (DRF) ---
class VehicleVideoCreateAPIView(APIView):
    parser_classes = (MultiPartParser, FormParser) # Necessary for file uploads

    def post(self, request, *args, **kwargs):
        serializer = VideoUploadSerializer(data=request.data)
        if serializer.is_valid():
            video_obj = serializer.save()
            original_video_path = video_obj.video_file.path
            task = process_video_task.delay(video_obj.id, original_video_path)
            video_obj.celery_task_id = task.id
            # Set initial status and progress when task is triggered
            video_obj.status = 'Queued'
            video_obj.progress = 0
            video_obj.save(update_fields=['celery_task_id', 'status', 'progress'])

            return Response({
                "message": "Video uploaded successfully! Processing started.",
                "status": "success",
                "id": video_obj.id,
                "task_id": task.id,
                "original_video_url": video_obj.video_file.url if video_obj.video_file else None
            }, status=status.HTTP_200_OK)
        else:
            return Response({
                "message": "Upload failed.",
                "errors": serializer.errors
            }, status=status.HTTP_400_BAD_REQUEST)


class ProcessingStatusAPIView(APIView):
    def get(self, request, video_id):
        try:
            video_obj = VideoUpload.objects.get(id=video_id)
            task_id = request.query_params.get('task_id') or video_obj.celery_task_id
            
            if not task_id:
                return Response({"error": "No Celery task ID found for this video."}, status=status.HTTP_400_BAD_REQUEST)

            task = AsyncResult(task_id, app=celery_app)
            
            print(f"DEBUG_VIEWS: Processing status request for video_id={video_id}, task_id={task_id}")
            print(f"DEBUG_VIEWS: Celery backend being used by Django server: {celery_app.backend}")
            print(f"DEBUG_VIEWS: Task state: {task.state}")

            current_status = task.state
            # Ensure initial current_meta is always a dict for consistency
            current_meta = task.info if isinstance(task.info, dict) else {} 
            if 'percent' not in current_meta:
                current_meta['percent'] = 0
            if 'status' not in current_meta:
                current_meta['status'] = 'Queued for processing...'


            response_data = {
                'status': current_status,
                'meta': current_meta,
                'video_id': video_obj.id,
                'task_id': task_id,
            }
            
            # --- FIX STARTS HERE ---
            # Replace `video_obj.processed` with `video_obj.status == 'Completed'`
            if video_obj.status == 'Completed' or task.state == 'SUCCESS':
                response_data['processed_video_url'] = video_obj.processed_video.url if video_obj.processed_video else None
                response_data['summary_excel_url'] = video_obj.summary_excel.url if video_obj.summary_excel else None
                response_data['bar_chart_url'] = video_obj.bar_chart_image.url if video_obj.bar_chart_image else None # Use bar_chart_image
                response_data['pie_chart_url'] = video_obj.pie_chart_image.url if video_obj.pie_chart_image else None # Use pie_chart_image
                response_data['detection_csv_url'] = video_obj.detection_csv.url if video_obj.detection_csv else None # Add detection_csv
                response_data['processed_csv_url'] = video_obj.processed_csv.url if video_obj.processed_csv else None # Add processed_csv

                # Add stats if they are available in the model
                response_data['stats'] = {
                    'total_frames': video_obj.total_frames,
                    'total_plates_detected': video_obj.total_plates_detected,
                    'successful_ocr': video_obj.successful_ocr,
                    'unique_plates': video_obj.unique_plates,
                    'smoke_vehicles': video_obj.smoke_vehicles,
                    'non_smoke_vehicles': video_obj.non_smoke_vehicles,
                }

                # This logic is good, keep it
                if video_obj.status == 'Completed' and current_status != 'SUCCESS':
                    response_data['status'] = 'SUCCESS'
                    response_data['meta'] = {'percent': 100, 'status': 'Processing complete! (from DB)'}
                elif task.state == 'SUCCESS': # If Celery reports SUCCESS but DB isn't updated yet,
                                              # assume success and return URLs
                     response_data['meta'] = {'percent': 100, 'status': 'Processing complete!'}


            elif task.state == 'PENDING':
                # This ensures the model's status and progress are updated even before the task explicitly reports PROGRESS
                if video_obj.status != 'Queued':
                    video_obj.status = 'Queued'
                    video_obj.progress = 0
                    video_obj.save(update_fields=['status', 'progress'])
                response_data['meta']['status'] = 'Queued for processing...'


            elif task.state == 'FAILURE':
                response_data['status'] = 'FAILURE'
                
                error_info = task.info
                error_message = "An unknown error occurred during processing."

                if isinstance(error_info, Exception):
                    error_message = str(error_info)
                elif isinstance(error_info, dict):
                    if 'exc_type' in error_info and 'exc_message' in error_info:
                        error_message = f"{error_info.get('exc_type', 'UnknownError')}: {error_info.get('exc_message', 'No specific message.')}"
                    elif 'message' in error_info:
                        error_message = error_info['message']
                    else:
                        error_message = str(error_info)
                else:
                    error_message = str(error_info)
                
                response_data['meta'] = {'percent': 100, 'status': 'Processing failed', 'error_details': error_message}
                
                # Update DB status to Failed if the task failed
                if video_obj.status != 'Failed':
                    video_obj.status = 'Failed'
                    video_obj.progress = 100 # Set progress to 100 on failure for UI consistency
                    video_obj.save(update_fields=['status', 'progress'])

            elif task.state == 'PROGRESS':
                if isinstance(task.info, dict):
                    response_data['meta']['percent'] = task.info.get('percent', video_obj.progress)
                    response_data['meta']['status'] = task.info.get('status', 'Processing in progress...')
                    
                    # Update model based on task info
                    video_obj.progress = response_data['meta']['percent']
                    video_obj.status = response_data['meta']['status']
                    video_obj.save(update_fields=['progress', 'status'])
                else:
                    # If task.info is not a dict for PROGRESS state, set generic values
                    if video_obj.status not in ['Processing', 'Completed', 'Failed']:
                        video_obj.status = 'Processing'
                        video_obj.save(update_fields=['status'])
                    response_data['meta']['status'] = video_obj.status
                    response_data['meta']['percent'] = video_obj.progress


            return Response(response_data)
        except VideoUpload.DoesNotExist:
            return Response({"error": "Video not found."}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            logger.error(f"Error fetching processing status for video {video_id}: {e}", exc_info=True)
            return Response({"error": "Internal server error.", "details": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# --- Traditional Django View for Upload Page ---
class VideoUploadView(View):
    def get(self, request):
        form = VideoUploadForm()
        videos = VideoUpload.objects.all().order_by('-uploaded_at')[:20]
        return render(request, 'core/upload_video.html', {'form': form, 'videos': videos})

    def post(self, request):
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            video_obj = form.save(commit=False)
            # Remove this line: video_obj.processed = False
            video_obj.status = 'Uploaded' # Set initial status
            video_obj.progress = 0 # Set initial progress
            video_obj.save()

            original_video_path = video_obj.video_file.path
            
            task = process_video_task.delay(video_obj.id, original_video_path)
            
            video_obj.celery_task_id = task.id
            video_obj.status = 'Queued' # Set status to Queued right after task creation
            video_obj.save(update_fields=['celery_task_id', 'status']) # Save both fields
            logger.info(f"WEB: Celery task {task.id} started for video ID {video_obj.id}")

            return JsonResponse({
                'status': 'success',
                'message': 'Video uploaded successfully! Processing started.',
                'video_id': video_obj.id,
                'task_id': task.id
            })
        else:
            logger.warning(f"WEB: Video upload form invalid: {form.errors}")
            return JsonResponse({
                'status': 'error',
                'message': 'Invalid form data.',
                'errors': form.errors
            }, status=400)

# --- Other Utility Views ---
def video_list_view(request):
    videos = VideoUpload.objects.order_by('-uploaded_at')
    return render(request, 'core/video_list.html', {'videos': videos})

def clear_all_videos(request):
    if request.method == 'POST':
        logger.info("Starting to clear all videos and associated files...")
        
        all_videos = VideoUpload.objects.all() 
        
        deleted_count = 0
        error_count = 0

        for video in list(all_videos):
            video_pk_for_log = video.pk

            # --- Delete associated files first ---
            # Updated field names to match the new model
            for field_name in ['video_file', 'processed_video', 'processed_csv', 'detection_csv', 'summary_excel', 'bar_chart_image', 'pie_chart_image']:
                file_field = getattr(video, field_name, None)
                if file_field and file_field.name:
                    file_path = os.path.join(settings.MEDIA_ROOT, file_field.name)
                    if os.path.isfile(file_path):
                        try:
                            os.remove(file_path)
                            logger.info(f"Deleted file: {file_path} for video PK {video_pk_for_log}")
                        except OSError as e:
                            logger.error(f"Error deleting file {file_path} for video PK {video_pk_for_log}: {e}")
            
            # --- Then delete the database object ---
            try:
                video.delete()
                logger.info(f"Deleted VideoUpload object: {video_pk_for_log}")
                deleted_count += 1
            except ValueError as e:
                logger.error(f"ValueError: Could not delete VideoUpload object (PK was {video_pk_for_log}): {e}")
                error_count += 1
            except Exception as e:
                logger.error(f"Unexpected error deleting VideoUpload object (PK was {video_pk_for_log}): {e}", exc_info=True)
                error_count += 1

        # --- After iterating and deleting individual objects, clean up empty directories ---
        media_subdirs_to_check = {
            VideoUpload._meta.get_field('video_file').upload_to.strip('/'),
            VideoUpload._meta.get_field('processed_video').upload_to.strip('/'),
            VideoUpload._meta.get_field('processed_csv').upload_to.strip('/'),    # New
            VideoUpload._meta.get_field('detection_csv').upload_to.strip('/'),    # New
            VideoUpload._meta.get_field('summary_excel').upload_to.strip('/'),
            VideoUpload._meta.get_field('bar_chart_image').upload_to.strip('/'), # Renamed
            VideoUpload._meta.get_field('pie_chart_image').upload_to.strip('/'), # Renamed
            'temp_processing_outputs', # This is a custom path, ensure it's correct
        }
        
        for dir_name in media_subdirs_to_check:
            dir_path = os.path.join(settings.MEDIA_ROOT, dir_name)
            if os.path.exists(dir_path) and not os.listdir(dir_path):
                try:
                    os.rmdir(dir_path)
                    logger.info(f"Removed empty directory: {dir_path}")
                except OSError as e:
                    logger.warning(f"Could not remove directory {dir_path} (might not be empty or in use): {e}")

        logger.info(f"Finished clearing all videos. Successfully deleted {deleted_count} objects, {error_count} failed.")
        return redirect('core:video_list')
    
    return redirect('core:video_list')


# --- FIXED BACKEND EMAIL SENDING FUNCTION ---
def send_email_summary(request):
    if request.method == 'POST':
        video_id = request.POST.get('video_id')
        recipient = request.POST.get('recipient_email', 'huzaifakamboh102@gmail.com')

        if not video_id:
            logger.warning("Email send request received without video_id.")
            return JsonResponse({"status": "error", "message": "Video ID is required."}, status=400)

        try:
            video_obj = VideoUpload.objects.get(id=video_id)
        except VideoUpload.DoesNotExist:
            logger.warning(f"Email send request: Video with ID {video_id} not found.")
            return JsonResponse({"status": "error", "message": "Video not found."}, status=404)

        excel_file_path = None
        if video_obj.summary_excel and video_obj.summary_excel.name:
            excel_file_path = video_obj.summary_excel.path
            if not os.path.exists(excel_file_path):
                logger.error(f"Excel file not found on disk at path: {excel_file_path}")
                excel_file_path = None

        email_subject = f'Video Analysis Report - Video ID: {video_id}'
        email_body = f'''Dear Authorities,

The analysis report for the video with ID {video_id} has been successfully generated and processed.

Processed on: {timezone.now().strftime('%Y-%m-%d %H:%M:%S')}

Best regards,
Your FYP Project Team'''
        
        try:
            email = EmailMessage(
                subject=email_subject,
                body=email_body,
                from_email=settings.DEFAULT_FROM_EMAIL,
                to=[recipient],
            )
            
            if excel_file_path:
                email.attach_file(excel_file_path)
            
            email.send()
            logger.info(f"Email for video ID {video_id} sent to {recipient} with attachment {bool(excel_file_path)}")
            return JsonResponse({"status": "success", "message": "Email sent successfully!"})
        except Exception as e:
            logger.error(f"Error sending email for video ID {video_id}: {e}", exc_info=True)
            return JsonResponse({"status": "error", "message": f"Failed to send email: {str(e)}"}, status=500)
    
    return JsonResponse({"status": "error", "message": "Invalid request method. Please use POST."}, status=405)
    