"""
URL configuration for smoke_vehicle_dashboard_project project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
# In your project's urls.py


# In your project's urls.py

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

# Import the view you want to serve at the root
from core.views import VideoUploadView # <--- ADD THIS LINE

urlpatterns = [
    path('admin/', admin.site.urls),
    # Add a specific path for the root URL
    path('', VideoUploadView.as_view(), name='home'), # <--- ADD THIS LINE
    # All other core app URLs will now be under /app/
    path('app/', include('core.urls', namespace='core')),
]

# Serve media files only during development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)