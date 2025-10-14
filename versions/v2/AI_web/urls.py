"""
URL configuration for AI_web project.

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
from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from image_generator import views

urlpatterns = [
    path('', views.index, name='index'),
    path('gallery/', views.gallery, name='gallery'),
    path('archive/', views.archive, name='archive'),
    path('history/', views.history, name='history'),
    path('generate-ajax/', views.generate_ajax, name='generate_ajax'),
    path('delete-image/<str:filename>/', views.delete_image, name='delete_image'),
    path('restore-image/<str:filename>/', views.restore_image, name='restore_image'),
    path('delete-permanent/<str:filename>/', views.delete_permanent, name='delete_permanent'),
    path('get-progress/', views.get_progress, name='get_progress'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
