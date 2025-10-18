from django.urls import path
from django.contrib.auth.decorators import login_required
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('gallery/', login_required(views.gallery), name='gallery'),
    path('archive/', login_required(views.archive), name='archive'),
    path('history/', login_required(views.history), name='history'),

    # Аутентификация
    path('register/', views.register_view, name='register'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('profile/', views.profile_view, name='profile'),

    # Админка
    path('user-lists/', views.user_list_view, name='user_list'),
    path('user/<int:user_id>/', views.user_detail_view, name='user_detail'),

    # API
    path('generate-ajax/', login_required(views.generate_ajax), name='generate_ajax'),
    path('delete-image/<str:filename>/', views.delete_image, name='delete_image'),
    path('restore-image/<str:filename>/', views.restore_image, name='restore_image'),
    path('delete-permanent/<str:filename>/', views.delete_permanent, name='delete_permanent'),
    path('get-progress/', login_required(views.get_progress), name='get_progress'),
    path('api/stats/', views.get_stats, name='get_stats'),
    path('api/stats/increment/', views.increment_stats, name='increment_stats'),
]
