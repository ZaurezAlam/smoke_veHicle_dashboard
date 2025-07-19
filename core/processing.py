import os
import cv2
import torch
import easyocr
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
import matplotlib
matplotlib.use('Agg') # Use 'Agg' backend for non-interactive plotting
import matplotlib.pyplot as plt
import re
import numpy as np
import subprocess
import time # Import time for simulating delays

# Define a logger specifically for this module if you want to log to Django's/Celery's configured loggers
# import logging
# logger = logging.getLogger(__name__) # You could use this if you set up logging for core.processing

def process_video(video_path, output_dir, celery_task_instance=None):
    """
    Processes a video for smoke and number plate detection.
    Can report progress to a Celery task instance if provided.
    """
    total_frames = 0
    cap_temp = None
    try:
        cap_temp = cv2.VideoCapture(video_path)
        if cap_temp.isOpened():
            total_frames = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
            # Fallback for some codecs/containers that might report 0 frames initially
            if total_frames == 0:
                print("Warning: Initial frame count is 0. Attempting to count frames manually (may be slow).")
                temp_frame_count = 0
                while True:
                    ret_temp, _ = cap_temp.read()
                    if not ret_temp:
                        break
                    temp_frame_count += 1
                total_frames = temp_frame_count
                print(f"Manual frame count: {total_frames}")

            if total_frames == 0:
                print("Warning: Could not reliably determine total frame count. Progress bar will be less granular.")
            else:
                print(f"Total frames detected: {total_frames}")
        else:
            print(f"Warning: Could not open video for initial frame count: {video_path}")
            total_frames = 0
    except Exception as e:
        print(f"Warning: Error getting total frame count: {e}")
        total_frames = 0
    finally:
        if cap_temp:
            cap_temp.release()


    # --- Stage 2.1: Initial Setup & Model Loading (20% - 25%) ---
    if celery_task_instance:
        celery_task_instance.update_state(
            state='PROGRESS',
            meta={'percent': 20, 'status': 'Initializing models and video capture...'}
        )
        # Change this line
        print(f"Task {celery_task_instance.request.id}: Initializing models (20%).")
        # Or, if you want Celery's logger:
        # celery_task_instance.request.get_logger().info(f"Task {celery_task_instance.request.id}: Initializing models (20%).")


    # Load YOLO model
    model = YOLO("/home/zaurez/smoke_vehicle_dashboard_project/core/best.pt")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(f"YOLO model loaded successfully. Using device: {device}")

    # EasyOCR setup with improved configuration
    reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    print("EasyOCR initialized successfully")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Error: Could not open video file at {video_path}")

    # Video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Video properties: {frame_width}x{frame_height}, FPS: {fps}")

    # Validate video properties
    if frame_width == 0 or frame_height == 0 or fps == 0:
        cap.release()
        raise ValueError(f"Invalid video properties: {frame_width}x{frame_height}, FPS: {fps}. Video might be corrupt or empty.")

    if celery_task_instance:
        celery_task_instance.update_state(
            state='PROGRESS',
            meta={'percent': 25, 'status': 'Video loaded. Starting frame processing...'}
        )
        # Change this line
        print(f"Task {celery_task_instance.request.id}: Video loaded (25%).")
        # Or, if you want Celery's logger:
        # celery_task_instance.request.get_logger().info(f"Task {celery_task_instance.request.id}: Video loaded (25%).")


    # Output paths
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    processed_video_temp_path = os.path.join(output_dir, f"{base_name}_processed_temp.mp4")
    output_csv = os.path.join(output_dir, f"{base_name}_plates.csv")
    all_detections_csv = os.path.join(output_dir, f"{base_name}_all_log.csv")
    summary_excel = os.path.join(output_dir, f"{base_name}_summary.xlsx")
    bar_chart_path = os.path.join(output_dir, f"{base_name}_bar_chart.png")
    pie_chart_path = os.path.join(output_dir, f"{base_name}_pie_chart.png")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # VIDEO WRITER setup
    fourcc_mp4 = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(processed_video_temp_path, fourcc_mp4, fps, (frame_width, frame_height))

    avi_fallback_used = False
    if not out.isOpened():
        print(f"Error: Could not create MP4 video writer for {processed_video_temp_path}. Trying AVI fallback.")
        processed_video_temp_path = os.path.join(output_dir, f"{base_name}_processed_temp.avi")
        fourcc_avi = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(processed_video_temp_path, fourcc_avi, fps, (frame_width, frame_height))
        avi_fallback_used = True

        if not out.isOpened():
            cap.release()
            raise Exception("Error: Both MP4 and AVI video writers failed. Check OpenCV installation, codecs, and directory permissions.")
        else:
            print(f"Successfully created fallback AVI video writer to {processed_video_temp_path}")
    else:
        print(f"Successfully created MP4 video writer to {processed_video_temp_path}")


    # DataFrames for storing results
    df = pd.DataFrame(columns=["Timestamp", "Plate Number", "Detection Type", "Confidence", "Vehicle Type"])
    all_detections_df = pd.DataFrame(columns=[
        "Timestamp", "Frame_Number", "Object_Type", "Confidence", "Bounding_Box",
        "Track_ID", "Plate_Number_OCR", "Plate_OCR_Status", "Is_Logged_Plate", "Vehicle_Type"
    ])

    # Enhanced tracking variables
    unique_smoke_ids = set()
    unique_non_smoke_ids = set()
    vehicle_types = {}
    logged_normalized_plates_in_video = set()
    plate_cooldown_tracker = {}
    COOLDOWN_SECONDS = 5
    MIN_PLATE_DETECTION_CONFIDENCE = 0.65
    MIN_OCR_CONFIDENCE = 0.3
    vehicle_plate_processed_status = {}
    vehicle_bboxes = {}
    PLATE_RESIZE_FACTOR = 2.0

    frame_count = 0
    total_plates_detected = 0
    successful_ocr_count = 0

    print("Starting video processing loop...")

    # --- Stage 2.2: Frame Processing Loop (25% - 55%) ---
    last_reported_percent = 25

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"Finished processing at frame {frame_count}")
            break

        frame_count += 1
        current_time = datetime.now()
        timestamp_str = current_time.strftime('%Y-%m-%d %H:%M:%S')

        # Update progress based on a calculated percentage
        if celery_task_instance and total_frames > 0:
            current_stage_percent = (frame_count / total_frames) * 30
            current_overall_percent = 25 + current_stage_percent
            current_overall_percent = min(current_overall_percent, 55)

            if int(current_overall_percent) > last_reported_percent:
                last_reported_percent = int(current_overall_percent)
                celery_task_instance.update_state(
                    state='PROGRESS',
                    meta={'percent': last_reported_percent, 'status': f'Processing frames: {frame_count}/{total_frames}'}
                )
                # Change this line
                print(f"Task {celery_task_instance.request.id}: Processing frames ({last_reported_percent}%).")
                # Or, if you want Celery's logger:
                # celery_task_instance.request.get_logger().info(f"Task {celery_task_instance.request.id}: Processing frames ({last_reported_percent}%).")
        elif celery_task_instance and frame_count % 100 == 0:
             celery_task_instance.update_state(
                state='PROGRESS',
                meta={'percent': min(50, 25 + (frame_count // 100)), 'status': f'Processing frames: {frame_count} (approx)'}
            )
             # Change this line
             print(f"Task {celery_task_instance.request.id}: Processing frames ({min(50, 25 + (frame_count // 100))}% approx).")
             # Or, if you want Celery's logger:
             # celery_task_instance.request.get_logger().info(f"Task {celery_task_instance.request.id}: Processing frames ({min(50, 25 + (frame_count // 100))}% approx).")


        # ... (rest of your existing frame processing logic) ...
        # Run YOLO detection with tracking
        results = model.track(frame, persist=True, verbose=False)

        if not results or len(results) == 0:
            out.write(frame)
            continue

        results = results[0]

        if results.boxes is None:
            out.write(frame)
            continue

        current_frame_vehicle_ids = set()
        pending_plate_detections_info = []

# Process all detections
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]
            # These are the coordinates for the CURRENT box being processed in the loop
            current_x1, current_y1, current_x2, current_y2 = map(int, box.xyxy[0])

            track_id = int(box.id[0]) if box.id is not None and len(box.id) > 0 else None

            if label.lower() in ["smoke emission", "smoke vehicle"]:
                if track_id is not None:
                    if track_id not in unique_smoke_ids:
                        unique_smoke_ids.add(track_id)
                        # print(f"New smoke vehicle detected: ID {track_id}")
                    vehicle_types[track_id] = "Smoke Emitting"

            elif label.lower() == "non smoke vehicle":
                if track_id is not None:
                    if track_id not in unique_non_smoke_ids:
                        unique_non_smoke_ids.add(track_id)
                        # print(f"New non-smoke vehicle detected: ID {track_id}")
                    vehicle_types[track_id] = "Non-Smoke"

            if label.lower() in ["smoke emission", "smoke vehicle", "non smoke vehicle"]:
                if track_id is not None:
                    vehicle_bboxes[track_id] = (current_x1, current_y1, current_x2, current_y2) # Use current_coords
                    if track_id not in vehicle_plate_processed_status:
                        vehicle_plate_processed_status[track_id] = False
                    current_frame_vehicle_ids.add(track_id)

            if label.lower() == "number plate":
                if conf >= MIN_PLATE_DETECTION_CONFIDENCE:
                    total_plates_detected += 1
                    # *** FIX IS HERE ***
                    # Use the current_x1, current_y1, current_x2, current_y2
                    # which are the bounding box coordinates for this specific number plate detection.
                    pending_plate_detections_info.append(((current_x1, current_y1, current_x2, current_y2), conf, track_id))

        # The loop below, which processes pending_plate_detections_info, already correctly
        # unpacks (px1, py1, px2, py2) from its elements.
        # Process plate detections with enhanced OCR
        for (px1, py1, px2, py2), p_conf, p_track_id in pending_plate_detections_info:
            associated_vehicle_id = p_track_id
            # ... rest of your code ...
            if associated_vehicle_id is None:
                best_overlap = 0
                for v_id, (vx1, vy1, vx2, vy2) in vehicle_bboxes.items():
                    overlap_x = max(0, min(px2, vx2) - max(px1, vx1))
                    overlap_y = max(0, min(py2, vy2) - max(py1, vy1))
                    overlap_area = overlap_x * overlap_y
                    plate_area = (px2 - px1) * (py2 - py1)

                    if plate_area > 0:
                        overlap_ratio = overlap_area / plate_area
                        if overlap_ratio > best_overlap and overlap_ratio > 0.1:
                            best_overlap = overlap_ratio
                            associated_vehicle_id = v_id

            ocr_plate_num = ""
            ocr_status = "N/A"
            is_logged_plate_flag = False
            vehicle_type = vehicle_types.get(associated_vehicle_id, "Unknown") if associated_vehicle_id else "Unknown"

            if associated_vehicle_id is not None and not vehicle_plate_processed_status.get(associated_vehicle_id, False):
                plate_crop = frame[py1:py2, px1:px2]

                if plate_crop.shape[0] > 0 and plate_crop.shape[1] > 0:
                    try:
                        enhanced_plate = enhance_plate_image(plate_crop, PLATE_RESIZE_FACTOR)
                        result = reader.readtext(enhanced_plate, detail=1)

                        if result:
                            best_result = max(result, key=lambda x: x[2])
                            raw_plate_text = best_result[1].strip()
                            ocr_confidence = best_result[2]

                            if ocr_confidence >= MIN_OCR_CONFIDENCE:
                                normalized_plate_text = clean_plate_text(raw_plate_text)

                                if normalized_plate_text and len(normalized_plate_text) >= 3:
                                    ocr_plate_num = normalized_plate_text
                                    ocr_status = f"Success (Conf: {ocr_confidence:.2f})"
                                    successful_ocr_count += 1

                                    should_log = False
                                    if normalized_plate_text not in logged_normalized_plates_in_video:
                                        should_log = True
                                    elif normalized_plate_text in plate_cooldown_tracker:
                                        time_diff = (current_time - plate_cooldown_tracker[normalized_plate_text]).total_seconds()
                                        if time_diff > COOLDOWN_SECONDS:
                                            should_log = True

                                    if should_log:
                                        # print(f"[{timestamp_str}] Frame {frame_count} - LOGGED Plate: {normalized_plate_text} (Vehicle: {vehicle_type})")
                                        new_row = pd.DataFrame([[
                                            timestamp_str, normalized_plate_text, "number plate", p_conf, vehicle_type
                                        ]], columns=["Timestamp", "Plate Number", "Detection Type", "Confidence", "Vehicle Type"])
                                        df = pd.concat([df, new_row], ignore_index=True)

                                        logged_normalized_plates_in_video.add(normalized_plate_text)
                                        plate_cooldown_tracker[normalized_plate_text] = current_time
                                        vehicle_plate_processed_status[associated_vehicle_id] = True
                                        is_logged_plate_flag = True
                                else:
                                    ocr_status = f"Text too short (Conf: {ocr_confidence:.2f})"
                            else:
                                ocr_status = f"Low OCR confidence ({ocr_confidence:.2f})"
                        else:
                            ocr_status = "No text detected by OCR"

                    except Exception as e:
                        ocr_status = f"OCR Error: {str(e)[:50]}"
                        print(f"OCR Error on frame {frame_count}: {e}")
                else:
                    ocr_status = "Empty plate crop"
            else:
                if associated_vehicle_id is not None and vehicle_plate_processed_status.get(associated_vehicle_id, False):
                    ocr_status = "Vehicle already processed"
                else:
                    ocr_status = "No associated vehicle"

            new_detection_row = pd.DataFrame([[
                timestamp_str, frame_count, "number plate", p_conf,
                f"{px1},{py1},{px2},{py2}",
                associated_vehicle_id if associated_vehicle_id is not None else "N/A",
                ocr_plate_num, ocr_status, is_logged_plate_flag, vehicle_type
            ]], columns=[
                "Timestamp", "Frame_Number", "Object_Type", "Confidence", "Bounding_Box",
                "Track_ID", "Plate_Number_OCR", "Plate_OCR_Status", "Is_Logged_Plate", "Vehicle_Type"
            ])
            all_detections_df = pd.concat([all_detections_df, new_detection_row], ignore_index=True)

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            track_id = int(box.id[0]) if box.id is not None and len(box.id) > 0 else None

            color = (255, 255, 255)
            draw_box = True

            if label.lower() == "number plate":
                if track_id is not None and vehicle_plate_processed_status.get(track_id, False):
                    draw_box = False
                else:
                    color = (0, 255, 0)
            elif label.lower() in ["smoke emission", "smoke vehicle"]:
                color = (0, 0, 255)
                if track_id is not None and vehicle_plate_processed_status.get(track_id, False):
                    color = (0, 0, 150)
            elif label.lower() == "non smoke vehicle":
                color = (255, 0, 0)
                if track_id is not None and vehicle_plate_processed_status.get(track_id, False):
                    color = (150, 0, 0)

            if draw_box:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                display_text = f"{label} {conf:.2f}"
                if track_id is not None:
                    display_text += f" ID:{track_id}"

                text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                text_y = max(20, y1 - 5)
                text_bg_y = max(0, y1 - text_size[1] - 5)
                cv2.rectangle(frame, (x1, text_bg_y), (x1 + text_size[0], y1), color, -1)
                cv2.putText(frame, display_text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        keys_to_delete = [tid for tid in vehicle_bboxes if tid not in current_frame_vehicle_ids]
        for tid in keys_to_delete:
            vehicle_bboxes.pop(tid, None)

        out.write(frame)


    # Ensure final progress update for the frame processing stage if loop finished
    if celery_task_instance:
        celery_task_instance.update_state(
            state='PROGRESS',
            meta={'percent': 55, 'status': f'Finished frame processing ({frame_count}/{total_frames}).'}
        )
        # Change this line
        print(f"Task {celery_task_instance.request.id}: Finished frame processing (55%).")
        # Or, if you want Celery's logger:
        # celery_task_instance.request.get_logger().info(f"Task {celery_task_instance.request.id}: Finished frame processing (55%).")
    elif celery_task_instance:
        celery_task_instance.update_state(
            state='PROGRESS',
            meta={'percent': 55, 'status': f'Finished frame processing (approx {frame_count} frames).'}
        )
        # Change this line
        print(f"Task {celery_task_instance.request.id}: Finished frame processing (55% approx).")
        # Or, if you want Celery's logger:
        # celery_task_instance.request.get_logger().info(f"Task {celery_task_instance.request.id}: Finished frame processing (55% approx).")


    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"\nVideo processing loop completed!")
    print(f"Total frames processed: {frame_count}")
    print(f"Total plates detected: {total_plates_detected}")
    print(f"Successful OCR readings: {successful_ocr_count}")
    print(f"Unique plates logged: {len(logged_normalized_plates_in_video)}")

    # --- Stage 2.3: Post-processing (55% - 60%) ---
    if celery_task_instance:
        celery_task_instance.update_state(
            state='PROGRESS',
            meta={'percent': 55, 'status': 'Saving detection logs and generating summaries...'}
        )
        # Change this line
        print(f"Task {celery_task_instance.request.id}: Saving detection logs (55%).")
        # Or, if you want Celery's logger:
        # celery_task_instance.request.get_logger().info(f"Task {celery_task_instance.request.id}: Saving detection logs (55%).")

    df.to_csv(output_csv, index=False)
    all_detections_df.to_csv(all_detections_csv, index=False)

    summary_df = pd.DataFrame({
        "Vehicle Type": ["Smoke Emitting", "Non-Smoke"],
        "Count": [len(unique_smoke_ids), len(unique_non_smoke_ids)]
    })

    plate_summary = pd.DataFrame({
        "Detection Summary": ["Total Plates Detected", "Successful OCR", "Unique Plates Logged"],
        "Count": [total_plates_detected, successful_ocr_count, len(logged_normalized_plates_in_video)]
    })

    with pd.ExcelWriter(summary_excel, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Vehicle Summary', index=False)
        plate_summary.to_excel(writer, sheet_name='Plate Summary', index=False)
        df.to_excel(writer, sheet_name='Detected Plates', index=False)
    print(f"Summary Excel saved to: {summary_excel}")

    # Create enhanced visualizations
    create_enhanced_charts(summary_df, plate_summary, bar_chart_path, pie_chart_path)
    print(f"Charts saved to: {bar_chart_path}, {pie_chart_path}")

    # --- Stage 2.4: Final Video Encoding (60%) ---
    if celery_task_instance:
        celery_task_instance.update_state(
            state='PROGRESS',
            meta={'percent': 60, 'status': 'Finalizing video encoding...'}
        )
        # Change this line
        print(f"Task {celery_task_instance.request.id}: Finalizing video encoding (60%).")
        # Or, if you want Celery's logger:
        # celery_task_instance.request.get_logger().info(f"Task {celery_task_instance.request.id}: Finalizing video encoding (60%).")


    final_output_video_path = os.path.join(output_dir, f"{base_name}_processed.mp4")

    if avi_fallback_used:
        print(f"Attempting to convert AVI to MP4 using ffmpeg: {processed_video_temp_path} -> {final_output_video_path}")
        try:
            reencode_video_for_web(processed_video_temp_path, final_output_video_path)
            os.remove(processed_video_temp_path)
            print(f"Successfully converted AVI to MP4: {final_output_video_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error converting AVI to MP4 with ffmpeg: {e}")
            print(f"Keeping temporary AVI file for debugging: {processed_video_temp_path}")
            final_output_video_path = processed_video_temp_path
        except FileNotFoundError:
            print("ffmpeg command not found. Please ensure ffmpeg is installed and in your system's PATH.")
            print(f"Keeping temporary AVI file for debugging: {processed_video_temp_path}")
            final_output_video_path = processed_video_temp_path
    else:
        print(f"Re-encoding MP4 for web compatibility: {processed_video_temp_path} -> {final_output_video_path}")
        try:
            temp_final_path = os.path.join(output_dir, f"{base_name}_processed_reencoded.mp4")
            reencode_video_for_web(processed_video_temp_path, temp_final_path)
            os.remove(processed_video_temp_path)
            final_output_video_path = temp_final_path
            print(f"Successfully re-encoded MP4 for web: {final_output_video_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error re-encoding MP4 with ffmpeg: {e}")
            print(f"Keeping original temporary MP4 file for debugging: {processed_video_temp_path}")
            final_output_video_path = processed_video_temp_path
        except FileNotFoundError:
            print("ffmpeg command not found. Please ensure ffmpeg is installed and in your system's PATH.")
            print(f"Keeping original temporary MP4 file for debugging: {processed_video_temp_path}")
            final_output_video_path = processed_video_temp_path

    print(f"\nFinal results paths:")
    print(f"- Processed video: {final_output_video_path}")
    print(f"- Detected plates: {output_csv}")
    print(f"- Detection log: {all_detections_csv}")
    print(f"- Summary Excel: {summary_excel}")
    print(f"- Charts: {bar_chart_path}, {pie_chart_path}")

    return {
        "processed_video": final_output_video_path,
        "csv": output_csv,
        "detection_log": all_detections_csv,
        "excel": summary_excel,
        "bar_chart": bar_chart_path,
        "pie_chart": pie_chart_path,
        "stats": {
            "total_frames": frame_count,
            "total_plates_detected": total_plates_detected,
            "successful_ocr": successful_ocr_count,
            "unique_plates": len(logged_normalized_plates_in_video),
            "smoke_vehicles": len(unique_smoke_ids),
            "non_smoke_vehicles": len(unique_non_smoke_ids)
        }
    }


def enhance_plate_image(plate_crop, resize_factor=2.0):
    """Enhanced plate image preprocessing for better OCR"""
    try:
        height, width = plate_crop.shape[:2]
        new_height, new_width = int(height * resize_factor), int(width * resize_factor)
        resized = cv2.resize(plate_crop, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        if len(resized.shape) == 3:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = resized

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    except Exception as e:
        print(f"Error in plate enhancement: {e}")
        return plate_crop


def clean_plate_text(raw_text):
    """Enhanced plate text cleaning"""
    cleaned = re.sub(r'[^a-zA-Z0-9]', '', raw_text).upper()
    return cleaned


def create_enhanced_charts(summary_df, plate_summary, bar_chart_path, pie_chart_path):
    """Create enhanced visualization charts"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    bars1 = ax1.bar(summary_df["Vehicle Type"], summary_df["Count"], color=["red", "blue"], alpha=0.7)
    ax1.set_title("Vehicle Type Summary", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Count")
    ax1.grid(axis='y', alpha=0.3)
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}', ha='center', va='bottom')

    if summary_df["Count"].sum() > 0:
        ax2.pie(summary_df["Count"], labels=summary_df["Vehicle Type"],
                autopct='%1.1f%%', colors=["red", "blue"])
        ax2.set_title("Vehicle Type Distribution", fontsize=14, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'No vehicle data', ha='center', va='center', fontsize=12, color='gray')
        ax2.set_title("Vehicle Type Distribution - No Data", fontsize=14, fontweight='bold')
        ax2.set_aspect('equal')

    bars3 = ax3.bar(plate_summary["Detection Summary"], plate_summary["Count"],
                    color=["green", "orange", "purple"], alpha=0.7)
    ax3.set_title("Plate Detection Summary", fontsize=14, fontweight='bold')
    ax3.set_ylabel("Count")
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(axis='y', alpha=0.3)
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}', ha='center', va='bottom')

    total_plates = plate_summary["Count"].iloc[0]
    successful_ocr = plate_summary["Count"].iloc[1]

    if total_plates > 0:
        efficiency = (successful_ocr / total_plates) * 100
        ax4.bar(['OCR Success Rate'], [efficiency], color='green', alpha=0.7)
        ax4.set_title("OCR Success Rate", fontsize=14, fontweight='bold')
        ax4.set_ylabel("Percentage (%)")
        ax4.set_ylim(0, 100)
        ax4.grid(axis='y', alpha=0.3)
        ax4.text(0, efficiency + 2, f'{efficiency:.1f}%', ha='center', va='bottom')
    else:
        ax4.text(0.5, 0.5, 'No plate detections', ha='center', va='center', fontsize=12, color='gray')
        ax4.set_title("OCR Success Rate - No Data", fontsize=14, fontweight='bold')
        ax4.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(bar_chart_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    plt.figure(figsize=(8, 8))
    if summary_df["Count"].sum() > 0:
        plt.pie(summary_df["Count"], labels=summary_df["Vehicle Type"],
                autopct='%1.1f%%', colors=["red", "blue"],
                explode=(0.05, 0.05))
        plt.title("Vehicle Type Distribution", fontsize=16, fontweight='bold')
    else:
        plt.text(0.5, 0.5, 'No vehicles detected', ha='center', va='center',
                 transform=plt.gca().transAxes, fontsize=14)
        plt.title("Vehicle Type Distribution - No Data", fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig(pie_chart_path, dpi=300, bbox_inches='tight')
    plt.close()


def reencode_video_for_web(input_path, output_path):
    """Re-encode video to H.264/AAC MP4 using GPU (NVIDIA NVENC) for faster encoding."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    command = [
        'ffmpeg', '-y', '-hwaccel', 'cuda', '-i', input_path,
        '-c:v', 'h264_nvenc', '-preset', 'fast', '-cq', '23',
        '-c:a', 'aac', '-b:a', '128k',
        '-movflags', '+faststart',
        output_path
    ]
    print(f"Executing ffmpeg GPU command: {' '.join(command)}")
    subprocess.run(command, check=True, capture_output=True, text=True)
    print("GPU ffmpeg encoding complete.")

if __name__ == "__main__":
    process_video("media/test_vehicles.mp4", "media/processed/")