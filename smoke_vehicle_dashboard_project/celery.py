# smoke_vehicle_dashboard_project/celery.py

import os
from celery import Celery
from django.conf import settings # Import Django settings module

# Set default Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'smoke_vehicle_dashboard_project.settings')

app = Celery('smoke_vehicle_dashboard_project')

# Load settings from Django's settings.py
app.config_from_object('django.conf:settings', namespace='CELERY')

# Auto-discover tasks in all apps
app.autodiscover_tasks()

# --- ADD THESE DEBUG PRINT LINES ---
try:
    print(f"DEBUG: Celery app's configured result backend (from app.conf): {app.conf.result_backend}")
    print(f"DEBUG: Celery app's configured broker URL (from app.conf): {app.conf.broker_url}")
except AttributeError:
    print("DEBUG: Celery app.conf attributes not yet fully loaded or accessible during startup.")
# --- END DEBUG PRINT LINES ---

@app.task(bind=True)
def debug_task(self):
    print(f'Request: {self.request!r}')