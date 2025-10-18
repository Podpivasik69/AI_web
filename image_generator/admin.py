from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import CustomUser, UserProfile, Task, GeneratedImage, PromptHistory, ModelSettings, GenerationSettings, \
    SiteStats
from django import forms

@admin.register(CustomUser)
class CustomUserAdmin(UserAdmin):
    list_display = ['username', 'email', 'first_name', 'last_name', 'is_staff', 'created_at']
    list_filter = ['is_staff', 'is_superuser', 'is_active']
    fieldsets = UserAdmin.fieldsets + (
        ('Дополнительная информация', {'fields': ('avatar', 'bio', 'created_at')}),
    )
    readonly_fields = ['created_at']


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ['user', 'created_at']
    readonly_fields = ['created_at']


@admin.register(Task)
class TaskAdmin(admin.ModelAdmin):
    list_display = ['id', 'user', 'status', 'created_at']
    list_filter = ['status', 'created_at']
    readonly_fields = ['created_at', 'completed_at']


@admin.register(GeneratedImage)
class GeneratedImageAdmin(admin.ModelAdmin):
    list_display = ['id', 'task', 'model_used', 'created_at']
    readonly_fields = ['created_at']


@admin.register(PromptHistory)
class PromptHistoryAdmin(admin.ModelAdmin):
    list_display = ['prompt_text', 'model_used', 'use_count', 'last_used_at']
    list_filter = ['model_used', 'quality_used']
    readonly_fields = ['created_at', 'last_used_at']


# @admin.register(ModelSettings)
# class ModelSettingsAdmin(admin.ModelAdmin):
#     list_display = ['name', 'model_id', 'style', 'is_active', 'created_at']
#     list_filter = ['is_active', 'style']
#     readonly_fields = ['created_at']
#     # Добавим явное указание полей для формы
#     fields = ['name', 'model_id', 'style', 'is_active', 'created_at']


@admin.register(GenerationSettings)
class GenerationSettingsAdmin(admin.ModelAdmin):
    list_display = ['name', 'quality_type', 'steps', 'width', 'height']
    list_filter = ['quality_type']


class ModelSettingsForm(forms.ModelForm):
    class Meta:
        model = ModelSettings
        fields = '__all__'

    def clean(self):
        cleaned_data = super().clean()
        # Добавим валидацию если нужно
        return cleaned_data


@admin.register(ModelSettings)
class ModelSettingsAdmin(admin.ModelAdmin):
    form = ModelSettingsForm
    list_display = ['name', 'model_id', 'style', 'is_active', 'created_at']
    list_filter = ['is_active', 'style']
    readonly_fields = ['created_at']
    fields = ['name', 'model_id', 'style', 'is_active', 'created_at']


@admin.register(SiteStats)
class SiteStatsAdmin(admin.ModelAdmin):
    list_display = ['total_generated', 'last_updated']
    readonly_fields = ['last_updated']
