from django.db import models
from django.contrib.auth.models import User


class Task(models.Model):
    """Задачи для генерации изображений"""
    STATUS_CHOICES = [
        ('pending', 'Ожидание'),
        ('processing', 'В процессе'),
        ('completed', 'Завершено'),
        ('failed', 'Ошибка'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    prompt = models.TextField()
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    task_id = models.CharField(max_length=100, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"Task {self.id}: {self.prompt[:50]}..."


class GeneratedImage(models.Model):
    """Сгенерированные изображения"""
    task = models.ForeignKey(Task, on_delete=models.CASCADE, related_name='images')
    image = models.ImageField(upload_to='generated_images/')
    model_used = models.CharField(max_length=100)
    quality_used = models.CharField(max_length=20)
    negative_prompt_used = models.CharField(max_length=100, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Image for: {self.task.prompt[:50]}..."


class PromptHistory(models.Model):
    """История использованных промптов"""
    prompt_text = models.TextField()
    translated_prompt = models.TextField(blank=True)
    model_used = models.CharField(max_length=100)
    quality_used = models.CharField(max_length=20)
    negative_prompt_used = models.CharField(max_length=100, blank=True)
    use_count = models.IntegerField(default=1)  # Счетчик использования
    created_at = models.DateTimeField(auto_now_add=True)
    last_used_at = models.DateTimeField(auto_now=True)  # Дата последнего использования

    class Meta:
        unique_together = ['prompt_text', 'model_used', 'quality_used', 'negative_prompt_used']

    def __str__(self):
        return f"{self.prompt_text[:50]}... (использован {self.use_count} раз)"


class ModelSettings(models.Model):
    """Настройки моделей"""
    name = models.CharField(max_length=100)
    model_id = models.CharField(max_length=200)
    style = models.CharField(max_length=50)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name


class GenerationSettings(models.Model):
    """Настройки генерации"""
    QUALITY_CHOICES = [
        ('fast', 'Быстрое'),
        ('medium', 'Среднее'),
        ('high', 'Высокое'),
    ]

    name = models.CharField(max_length=100)
    quality_type = models.CharField(max_length=20, choices=QUALITY_CHOICES)
    steps = models.IntegerField()
    width = models.IntegerField()
    height = models.IntegerField()
    guidance_scale = models.FloatField(default=7.5)

    def __str__(self):
        return f"{self.name} ({self.width}x{self.height})"



class SiteStats(models.Model):
    total_generated = models.IntegerField(default=0)
    last_updated = models.DateTimeField(auto_now=True)

    @classmethod
    def get_stats(cls):
        stats, created = cls.objects.get_or_create(pk=1)
        return stats

    @classmethod
    def increment_generated(cls):
        stats = cls.get_stats()
        stats.total_generated += 1
        stats.save()
        return stats.total_generated