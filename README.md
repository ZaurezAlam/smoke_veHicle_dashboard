Smoke Vehicle Detection and Dashboard
This project provides a web-based dashboard for uploading videos, processing them using AI models (YOLO for object detection, EasyOCR for number plate recognition), and generating analytics reports and processed videos. It leverages Django for the web framework, Celery for asynchronous task processing, and RabbitMQ as the message broker.

Features
Video Upload: Easily upload video files through a user-friendly web interface.

Asynchronous Processing: Videos are processed in the background using Celery, preventing UI blocking.

Real-time Progress Tracking: A progress bar updates in real-time to show the status of video processing.

Smoke and Vehicle Detection: Utilizes YOLOv8 for detecting smoke emissions, smoke-emitting vehicles, and non-smoke vehicles.

Number Plate Recognition (OCR): Integrates EasyOCR to identify and extract number plates from detected vehicles.

Processed Video Output: Generates a new video with bounding boxes and labels for detected objects.

Detailed Reports: Produces CSV and Excel summary reports of detections.

Visual Analytics: Generates bar and pie charts summarizing vehicle types and plate detection statistics.

Email Integration: Option to send processed reports via email.

Data Management: Clear all uploaded and processed video data with a single click, or delete individual video logs.

Technologies Used
Backend:

Django: Web framework

Celery: Asynchronous task queue

RabbitMQ: Message broker for Celery

Django-Celery-Results: Stores Celery task results in the Django database

OpenCV (cv2): Video processing and frame manipulation

Ultralytics YOLOv8: Object detection (requires a pre-trained best.pt model)

EasyOCR: Optical Character Recognition for number plates

Pandas: Data manipulation and CSV/Excel generation

Matplotlib: Chart generation

ffmpeg: Video re-encoding (for web compatibility and AVI conversion)

Frontend:

HTML, CSS, JavaScript

Font Awesome: Icons

Setup and Installation
Follow these steps to get the project up and running on your local machine.

Prerequisites
Before you begin, ensure you have the following installed:

Python 3.8+

pip (Python package installer)

RabbitMQ Server: This is essential for Celery.

Ubuntu/Debian:

sudo apt update
sudo apt install rabbitmq-server
sudo systemctl start rabbitmq-server
sudo systemctl enable rabbitmq-server
sudo systemctl status rabbitmq-server

macOS (using Homebrew):

brew install rabbitmq
brew services start rabbitmq

Windows: Download and install from the RabbitMQ website.

ffmpeg: Required for video re-encoding.

Ubuntu/Debian: sudo apt install ffmpeg

macOS (using Homebrew): brew install ffmpeg

Windows: Download binaries and add to PATH, or use a package manager like Chocolatey.

YOLOv8 Model: You need a pre-trained YOLOv8 model (best.pt). Place this file in the specified path within your core/processing.py (currently /home/zaurez/smoke_vehicle_dashboard_project/core/migrations/media/best.pt). Adjust this path in core/processing.py if your model is located elsewhere.

1. Clone the Repository
git clone https://github.com/your-username/smoke_vehicle_dashboard_project.git
cd smoke_vehicle_dashboard_project

2. Create and Activate a Virtual Environment
It's highly recommended to use a virtual environment to manage project dependencies.

python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

3. Install Dependencies
Install all required Python packages using pip:

pip install -r requirements.txt

(If you don't have a requirements.txt, you can generate one after installing all packages: pip freeze > requirements.txt. For this project, you'll need Django, djangorestframework, celery, django-celery-results, opencv-python, torch, ultralytics, easyocr, pandas, matplotlib, openpyxl.)

4. Database Setup
Apply Django database migrations:

python manage.py makemigrations core
python manage.py migrate

5. Celery Configuration
Ensure your smoke_vehicle_dashboard_project/settings.py is configured for Celery and RabbitMQ. It should look something like this (verify BROKER_URL and RESULT_BACKEND):

# smoke_vehicle_dashboard_project/settings.py

# ... other settings ...

# Celery Configuration
CELERY_BROKER_URL = 'amqp://guest:guest@localhost:5672//' # Default RabbitMQ URL
CELERY_RESULT_BACKEND = 'django-db' # Store results in Django database
CELERY_ACCEPT_CONTENT = ['json']
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
CELERY_TIMEZONE = 'Asia/Karachi' # Or your local timezone

# Django-Celery-Results
CELERY_CACHE_BACKEND = 'default' # Use Django's default cache

# ... other settings ...

6. Running the Application
You need to run three separate processes:

a. Start RabbitMQ Server
(If not already running from prerequisites step)

sudo systemctl start rabbitmq-server # For Linux/WSL
# or `brew services start rabbitmq` for macOS

b. Start Celery Worker
Open a new terminal window/tab, activate your virtual environment, and start the Celery worker:

source venv/bin/activate
celery -A smoke_vehicle_dashboard_project worker -l info --pool=solo

celery -A smoke_vehicle_dashboard_project: Specifies your Django project as the Celery app.

worker: Starts a worker process.

-l info: Sets the logging level to info (for detailed output).

--pool=solo: Runs the worker in a single-threaded, synchronous mode. This is good for debugging but for production, you might use prefork or gevent.

c. Start Django Development Server
Open another new terminal window/tab, activate your virtual environment, and start the Django development server:

source venv/bin/activate
python manage.py runserver

7. Access the Dashboard
Once all three processes are running, open your web browser and navigate to:

http://127.0.0.1:8000/app/ (or http://localhost:8000/app/)

You should see the video upload dashboard.

Usage
Upload Video: Click "Choose Video File" or drag and drop a video.

Monitor Progress: The dashboard will show the video in a "Processing" state with a progress bar.

View Results: Once processing is complete, the video card will update to show the processed video, download links for reports (Excel, CSVs), and generated charts.

Email Reports: Use the "Send via Server" button to email the Excel summary.

Manage Data:

Use the "Back to Upload" button to return to the main upload page.

Click "Clear All Data" (with confirmation) to delete all videos and associated files.

Individual "Delete" buttons are available on each video card to remove specific entries.

Contributing
Feel free to fork the repository, open issues, and submit pull requests.

License
This project is licensed under the MIT License - see the LICENSE file for details.
