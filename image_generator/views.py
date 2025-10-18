from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse, JsonResponse
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth.forms import AuthenticationForm
from django.contrib import messages
from django.views.decorators.csrf import csrf_exempt
from django.contrib.admin.views.decorators import staff_member_required
from django.conf import settings
from io import StringIO
from .forms import CustomUserCreationForm, UserProfileForm
from diffusers import StableDiffusionPipeline
from googletrans import Translator
from .models import (
    SiteStats, PromptHistory, ModelSettings, GenerationSettings,
    CustomUser, UserProfile, Task, GeneratedImage
)

import torch
import time
import json
import re
import sys
import glob
import os

generation_progress = 0
is_generating = False
current_pipe = None
current_model = None


def archive(request):
    """Показывает архив удаленных изображений ТОЛЬКО текущего пользователя"""
    if not request.user.is_authenticated:
        return redirect('login')

    archive_dir = os.path.join(settings.MEDIA_ROOT, 'bad')
    archive_files = []

    if os.path.exists(archive_dir):
        for file_path in glob.glob(os.path.join(archive_dir, "*.png")):
            filename = os.path.basename(file_path)

            # Фильтруем по ID пользователя в имени файла
            parts = filename.split('_')
            if len(parts) >= 3 and parts[1] == str(request.user.id):
                file_time = os.path.getmtime(file_path)
                archive_files.append({
                    'filename': filename,
                    'url': f'/media/bad/{filename}',
                    'time': file_time,
                    'date': time.strftime('%Y-%m-%d %H:%M', time.localtime(file_time))
                })

    archive_files.sort(key=lambda x: x['time'], reverse=True)

    context = {
        'images': archive_files,
    }
    return render(request, 'archive.html', context)


def restore_image(request, filename):
    """Восстанавливает изображение из архива в галерею с проверкой прав"""
    if request.method == 'POST':
        try:
            # Проверяем, что файл принадлежит пользователю
            parts = filename.split('_')
            if len(parts) < 3:
                return JsonResponse({'success': False, 'error': 'Некорректное имя файла'})

            file_user_id = parts[1]
            current_user_id = str(request.user.id)

            # Проверяем принадлежность файла (кроме админов)
            if file_user_id != current_user_id and not request.user.is_superuser:
                return JsonResponse({'success': False, 'error': 'Нет прав для восстановления этого файла'})

            # Путь к файлу в архиве
            archive_path = os.path.join(settings.MEDIA_ROOT, 'bad', filename)
            # Путь для восстановления
            restore_path = os.path.join(settings.MEDIA_ROOT, 'img', filename)

            if os.path.exists(archive_path):
                # Перемещаем файл обратно
                os.rename(archive_path, restore_path)
                return JsonResponse({'success': True, 'message': 'Изображение восстановлено'})
            else:
                return JsonResponse({'success': False, 'error': 'Файл не найден в архиве'})

        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})

    return JsonResponse({'success': False, 'error': 'Invalid method'})


def delete_permanent(request, filename):
    """Полностью удаляет изображение из архива с проверкой прав"""
    if request.method == 'POST':
        try:
            # Проверяем, что файл принадлежит пользователю
            parts = filename.split('_')
            if len(parts) < 3:
                return JsonResponse({'success': False, 'error': 'Некорректное имя файла'})

            file_user_id = parts[1]
            current_user_id = str(request.user.id)

            # Проверяем принадлежность файла (кроме админов)
            if file_user_id != current_user_id and not request.user.is_superuser:
                return JsonResponse({'success': False, 'error': 'Нет прав для удаления этого файла'})

            file_path = os.path.join(settings.MEDIA_ROOT, 'bad', filename)

            if os.path.exists(file_path):
                os.remove(file_path)
                return JsonResponse({'success': True, 'message': 'Изображение удалено навсегда'})
            else:
                return JsonResponse({'success': False, 'error': 'Файл не найден'})

        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})

    return JsonResponse({'success': False, 'error': 'Invalid method'})


def delete_image(request, filename):
    """Перемещает изображение в папку bad вместо удаления с проверкой прав"""
    if request.method == 'POST':
        try:
            # Проверяем, что файл принадлежит пользователю
            parts = filename.split('_')
            if len(parts) < 3:
                return JsonResponse({'success': False, 'error': 'Некорректное имя файла'})

            file_user_id = parts[1]
            current_user_id = str(request.user.id)

            # Проверяем принадлежность файла (кроме админов)
            if file_user_id != current_user_id and not request.user.is_superuser:
                return JsonResponse({'success': False, 'error': 'Нет прав для удаления этого файла'})

            # Исходный путь к изображению
            source_path = os.path.join(settings.MEDIA_ROOT, 'img', filename)

            # Путь к папке bad
            bad_dir = os.path.join(settings.MEDIA_ROOT, 'bad')

            # Создаем папку bad если ее нет
            os.makedirs(bad_dir, exist_ok=True)

            # Новый путь для изображения
            destination_path = os.path.join(bad_dir, filename)

            if os.path.exists(source_path):
                # Перемещаем файл
                os.rename(source_path, destination_path)
                return JsonResponse({'success': True, 'message': 'Изображение перемещено в архив'})
            else:
                return JsonResponse({'success': False, 'error': 'Файл не найден'})

        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})

    return JsonResponse({'success': False, 'error': 'Invalid method'})


@login_required
def gallery(request):
    # Получаем только изображения текущего пользователя
    image_dir = os.path.join(settings.MEDIA_ROOT, 'img')
    image_files = []

    if os.path.exists(image_dir):
        for file_path in glob.glob(os.path.join(image_dir, "*.png")):
            filename = os.path.basename(file_path)

            # Фильтруем по ID пользователя в имени файла
            parts = filename.split('_')
            if len(parts) >= 3 and parts[1] == str(request.user.id):
                file_time = os.path.getmtime(file_path)

                # Определяем качество из имени файла
                quality = 'unknown'
                if 'fast_' in filename:
                    quality = 'fast'
                elif 'medium_' in filename:
                    quality = 'medium'
                elif 'high_' in filename:
                    quality = 'high'

                image_files.append({
                    'filename': filename,
                    'url': f'/media/img/{filename}',
                    'time': file_time,
                    'date': time.strftime('%Y-%m-%d %H:%M', time.localtime(file_time)),
                    'quality': quality
                })

    # Сортируем по дате (новые сверху)
    image_files.sort(key=lambda x: x['time'], reverse=True)

    # Фильтрация по качеству из GET параметров
    quality_filter = request.GET.get('quality', 'all')
    if quality_filter != 'all':
        image_files = [img for img in image_files if quality_filter in img['filename']]

    context = {
        'images': image_files,
        'quality_filter': quality_filter,
    }
    return render(request, 'gallery.html', context)


class ProgressCapture:
    def __init__(self):
        self.progress = 0
        self.old_stdout = sys.stdout
        self.captured_output = StringIO()

    def write(self, text):
        self.captured_output.write(text)
        # Парсим прогресс из строк типа " 15%|#5        | 3/20"
        match = re.search(r'(\d+)%', text)
        if match:
            self.progress = int(match.group(1))
        return len(text)

    def flush(self):
        self.captured_output.flush()


progress_capture = ProgressCapture()

# Модели
# MODELS = {
#     '1': {'name': 'Waifu Diffusion', 'model_id': 'hakurei/waifu-diffusion', 'style': 'anime'},
#     '2': {'name': 'MeinaMix', 'model_id': 'Meina/MeinaMix_V11', 'style': 'anime'},
#     '3': {'name': 'Anything V5', 'model_id': 'stablediffusionapi/anything-v5', 'style': 'anime'},
#     '4': {'name': 'Stable Diffusion v1.5', 'model_id': 'runwayml/stable-diffusion-v1-5', 'style': 'realistic'}
# }

# Настройки качества
# QUALITY_PRESETS = {
#     'fast': {'steps': 20, 'width': 512, 'height': 512, 'name': 'Быстрое'},
#     'medium': {'steps': 30, 'width': 640, 'height': 640, 'name': 'Среднее'},
#     'high': {'steps': 40, 'width': 768, 'height': 768, 'name': 'Высокое'}
# }

# Негативные промпты
NEGATIVE_PROMPTS = {
    'none': {'name': 'Без негативных', 'prompt': ''},
    'standard': {'name': 'Стандартные', 'prompt': 'ugly, poorly drawn, deformed, blurry, low quality, worst quality'},
    'anime': {'name': 'Для аниме',
              'prompt': 'ugly, poorly drawn, deformed, blurry, bad anatomy, bad hands, missing fingers, extra limbs'}
}


def get_models():
    """Получает модели из базы данных или возвращает дефолтные"""
    try:
        models_dict = {}
        models_db = ModelSettings.objects.filter(is_active=True)
        if models_db.exists():
            for i, model in enumerate(models_db, 1):
                models_dict[str(i)] = {
                    'name': model.name,
                    'model_id': model.model_id,
                    'style': model.style
                }
        else:
            # Дефолтные модели если база пустая
            models_dict = {
                '1': {'name': 'Waifu Diffusion', 'model_id': 'hakurei/waifu-diffusion', 'style': 'anime'},
                '2': {'name': 'MeinaMix', 'model_id': 'Meina/MeinaMix_V11', 'style': 'anime'},
                '3': {'name': 'Anything V5', 'model_id': 'stablediffusionapi/anything-v5', 'style': 'anime'},
                '4': {'name': 'Stable Diffusion v1.5', 'model_id': 'runwayml/stable-diffusion-v1-5',
                      'style': 'realistic'}

            }
        return models_dict
    except Exception as e:
        print(f"Error getting models: {e}")
        # Fallback если база не готова
        return {
            '1': {'name': 'Waifu Diffusion', 'model_id': 'hakurei/waifu-diffusion', 'style': 'anime'},
            '2': {'name': 'MeinaMix', 'model_id': 'Meina/MeinaMix_V11', 'style': 'anime'},
            '3': {'name': 'Anything V5', 'model_id': 'stablediffusionapi/anything-v5', 'style': 'anime'},
            '4': {'name': 'Stable Diffusion v1.5', 'model_id': 'runwayml/stable-diffusion-v1-5', 'style': 'realistic'}
        }


def get_quality_presets():
    """Получает настройки качества из базы данных или возвращает дефолтные"""
    try:
        quality_dict = {}
        qualities_db = GenerationSettings.objects.all()
        if qualities_db.exists():
            for quality in qualities_db:
                quality_dict[quality.quality_type] = {
                    'steps': quality.steps,
                    'width': quality.width,
                    'height': quality.height,
                    'name': quality.name,
                    'guidance_scale': quality.guidance_scale
                }
        else:
            # Дефолтные настройки если база пустая
            quality_dict = {
                'fast': {'steps': 20, 'width': 512, 'height': 512, 'name': 'Быстрое'},
                'medium': {'steps': 30, 'width': 640, 'height': 640, 'name': 'Среднее'},
                'high': {'steps': 40, 'width': 768, 'height': 768, 'name': 'Высокое'}
            }
        return quality_dict
    except:
        # Fallback если база не готова
        return {
            'fast': {'steps': 20, 'width': 512, 'height': 512, 'name': 'Быстрое'},
            'medium': {'steps': 30, 'width': 640, 'height': 640, 'name': 'Среднее'},
            'high': {'steps': 40, 'width': 768, 'height': 768, 'name': 'Высокое'}
        }


def load_model(model_id):
    """Загружает модель"""
    global current_pipe
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        # Простые оптимизации
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
        current_pipe = pipe
        return True
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return False


def translate_to_english(text):
    """Переводит текст на английский"""
    try:
        translator = Translator()
        translated = translator.translate(text, src='ru', dest='en').text
        return translated
    except Exception as e:
        # Логируем ошибку без эмодзи для Windows
        print(f"Ошибка перевода: {e}")
        return text


@csrf_exempt
def generate_ajax(request):
    global current_pipe, current_model, is_generating, generation_progress

    if request.method == 'POST':
        try:
            data = json.loads(request.body)
        except:
            return JsonResponse({'error': 'Invalid JSON'})

        prompt = data.get('prompt', '')
        model_choice = data.get('model', '1')
        quality_choice = data.get('quality', 'fast')
        negative_choice = data.get('negative', 'none')

        print(f"[DEBUG AJAX] Получен промпт: {prompt}")
        print(f"[DEBUG AJAX] Пользователь: {request.user}")
        print(f"[DEBUG AJAX] Аутентифицирован: {request.user.is_authenticated}")

        if prompt and not is_generating:
            is_generating = True
            generation_progress = 10  # Начало

            try:
                # Загрузка модели
                models_dict = get_models()
                model_info = models_dict.get(model_choice)
                if not current_pipe or current_model != model_choice:
                    generation_progress = 20
                    if not load_model(model_info['model_id']):
                        is_generating = False
                        generation_progress = 0
                        return JsonResponse({'error': 'Ошибка загрузки модели'})
                    current_model = model_choice

                generation_progress = 30  # Модель загружена

                # Настройки
                quality_presets = get_quality_presets()
                quality_settings = quality_presets[quality_choice]
                negative_prompt = NEGATIVE_PROMPTS[negative_choice]['prompt']
                english_prompt = translate_to_english(prompt)

                # СОХРАНЯЕМ ПРОМПТ В ИСТОРИЮ (ДОБАВЬТЕ ЭТОТ БЛОК)
                try:
                    print(f"[DEBUG AJAX] Попытка сохранения промпта в историю")

                    # Создаем или обновляем запись в истории промптов
                    prompt_obj, created = PromptHistory.objects.get_or_create(
                        prompt_text=prompt,
                        model_used=model_info['name'],
                        quality_used=quality_choice,
                        negative_prompt_used=negative_choice,
                        defaults={
                            'translated_prompt': english_prompt,
                            'use_count': 1,
                            'user': request.user if request.user.is_authenticated else None
                        }
                    )

                    if not created:
                        # Если запись уже существует - обновляем счетчик и перевод
                        prompt_obj.use_count += 1
                        prompt_obj.translated_prompt = english_prompt
                        prompt_obj.user = request.user if request.user.is_authenticated else None
                        prompt_obj.save()
                        print(f"[SUCCESS AJAX] Промпт обновлен, использован {prompt_obj.use_count} раз")
                    else:
                        print("[SUCCESS AJAX] Новый промпт сохранен в историю")

                except Exception as e:
                    print(f"[ERROR AJAX] Ошибка сохранения в историю: {e}")
                    # Если не получается сохранить через get_or_create, пробуем простой способ
                    try:
                        prompt_obj = PromptHistory(
                            prompt_text=prompt,
                            translated_prompt=english_prompt,
                            model_used=model_info['name'],
                            quality_used=quality_choice,
                            negative_prompt_used=negative_choice,
                            use_count=1,
                            user=request.user if request.user.is_authenticated else None
                        )
                        prompt_obj.save()
                        print("[SUCCESS AJAX] Промпт сохранен (альтернативный метод)")
                    except Exception as e2:
                        print(f"[ERROR AJAX] Альтернативное сохранение тоже не удалось: {e2}")

                generation_progress = 50  # Подготовка завершена

                # Генерация изображения
                image = current_pipe(
                    prompt=english_prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=quality_settings['steps'],
                    guidance_scale=7.5,
                    width=quality_settings['width'],
                    height=quality_settings['height']
                ).images[0]

                generation_progress = 90  # Генерация завершена

                # Сохранение с ID пользователя
                user_id = request.user.id if request.user.is_authenticated else 'anonymous'
                filename = f"{quality_choice}_{user_id}_{int(time.time())}.png"
                image_path = os.path.join('media', 'img', filename)
                image.save(image_path)

                generation_progress = 100  # Полное завершение
                is_generating = False

                return JsonResponse({'success': True, 'image_url': f'/media/img/{filename}'})

            except Exception as e:
                is_generating = False
                generation_progress = 0
                return JsonResponse({'error': f'Ошибка генерации: {e}'})

    return JsonResponse({'error': 'Invalid request'})


def history(request):
    """Показывает историю промптов"""
    try:
        # Сортируем по дате последнего использования и берем последние 50
        # Фильтруем по текущему пользователю
        prompts = PromptHistory.objects.filter(user=request.user).order_by('-last_used_at')[:50]
    except:
        prompts = []

    context = {
        'prompts': prompts
    }
    return render(request, 'history.html', context)


@csrf_exempt
def get_progress(request):
    global generation_progress, is_generating
    return JsonResponse({
        'progress': generation_progress,
        'generating': is_generating
    })


def get_stats(request):
    stats = SiteStats.get_stats()
    return JsonResponse({'total_generated': stats.total_generated})


def increment_stats(request):
    if request.method == 'POST':
        total = SiteStats.increment_generated()
        return JsonResponse({'total_generated': total})


def initialize_stats():
    """Устанавливает начальное значение счетчика на основе существующих изображений"""
    stats = SiteStats.get_stats()

    # Считаем существующие изображения
    img_dir = os.path.join(settings.MEDIA_ROOT, 'img')
    if os.path.exists(img_dir):
        png_files = glob.glob(os.path.join(img_dir, "*.png"))
        existing_count = len(png_files)
    else:
        existing_count = 0

    # Устанавливаем счетчик
    stats.total_generated = existing_count
    stats.save()
    print(f"Статистика инициализирована: {existing_count} изображений")

    return existing_count


# Функция для проверки админа
def is_admin(user):
    return user.is_authenticated and user.is_superuser


# Регистрация
def register_view(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST, request.FILES)
        if form.is_valid():
            user = form.save()

            # Сохраняем аватар если он есть (теперь в модель CustomUser)
            if 'avatar' in request.FILES:
                user.avatar = request.FILES['avatar']
                user.save()

            # Создаем профиль пользователя (без аватарки)
            UserProfile.objects.create(user=user)

            # Автоматически логиним пользователя после регистрации
            login(request, user)
            messages.success(request, 'Регистрация прошла успешно!')
            return redirect('/')
        else:
            messages.error(request, 'Пожалуйста, исправьте ошибки в форме.')
    else:
        form = CustomUserCreationForm()

    return render(request, 'registration/register.html', {'form': form})


# Вход
def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                messages.success(request, f'Добро пожаловать, {username}!')
                next_url = request.GET.get('next', '/')
                return redirect(next_url)
        messages.error(request, 'Неверное имя пользователя или пароль.')
    else:
        form = AuthenticationForm()

    return render(request, 'registration/login.html', {'form': form})


# Выход
def logout_view(request):
    logout(request)
    messages.success(request, 'Вы успешно вышли из системы.')
    return redirect('/')


# Профиль пользователя
@login_required
def profile_view(request):
    # Используем CustomUser вместо UserProfile для формы
    if request.method == 'POST':
        form = UserProfileForm(request.POST, request.FILES, instance=request.user)
        if form.is_valid():
            form.save()
            messages.success(request, 'Профиль успешно обновлен!')
            return redirect('profile')
    else:
        form = UserProfileForm(instance=request.user)

    # Получаем историю промптов ТОЛЬКО текущего пользователя
    user_prompts = PromptHistory.objects.filter(user=request.user).order_by('-last_used_at')[:20]

    # Получаем изображения пользователя из галереи с правильной фильтрацией
    user_images = []
    image_dir = os.path.join(settings.MEDIA_ROOT, 'img')
    if os.path.exists(image_dir):
        for file_path in glob.glob(os.path.join(image_dir, "*.png")):
            filename = os.path.basename(file_path)
            # Фильтруем по ID пользователя в имени файла
            parts = filename.split('_')
            if len(parts) >= 3 and parts[1] == str(request.user.id):
                file_time = os.path.getmtime(file_path)
                user_images.append({
                    'filename': filename,
                    'url': f'/media/img/{filename}',
                    'time': file_time,
                    'date': time.strftime('%Y-%m-%d %H:%M', time.localtime(file_time))
                })

    user_images.sort(key=lambda x: x['time'], reverse=True)

    context = {
        'form': form,
        'user_prompts': user_prompts,
        'user_images': user_images[:12]  # Последние 12 изображений
    }
    return render(request, 'registration/profile.html', context)


# Список пользователей (только для админа)
@staff_member_required
@login_required
def user_list_view(request):
    if not request.user.is_superuser:
        messages.error(request, 'У вас нет доступа к этой странице')
        return redirect('/')

    users = CustomUser.objects.all().order_by('-date_joined')
    user_stats = []

    for user in users:
        # Считаем промпты пользователя
        prompt_count = PromptHistory.objects.filter(user=user).count()

        # Считаем ТОЛЬКО изображения пользователя
        image_count = 0
        image_dir = os.path.join(settings.MEDIA_ROOT, 'img')
        if os.path.exists(image_dir):
            for file_path in glob.glob(os.path.join(image_dir, "*.png")):
                filename = os.path.basename(file_path)
                # Проверяем, что файл принадлежит пользователю
                # Формат: quality_userID_timestamp.png
                parts = filename.split('_')
                if len(parts) >= 3 and parts[1] == str(user.id):
                    image_count += 1

        user_stats.append({
            'user': user,
            'prompt_count': prompt_count,
            'image_count': image_count,
        })

    context = {'user_stats': user_stats}
    return render(request, 'admin/user_list.html', context)


# Детальная информация о пользователе (только для админа)
@staff_member_required
@login_required
def user_detail_view(request, user_id):
    if not request.user.is_superuser:
        messages.error(request, 'У вас нет доступа к этой странице')
        return redirect('/')

    user = get_object_or_404(CustomUser, id=user_id)

    # История промптов пользователя
    user_prompts = PromptHistory.objects.filter(user=user).order_by('-last_used_at')

    # Галерея пользователя - ТОЛЬКО его изображения
    user_images = []
    image_dir = os.path.join(settings.MEDIA_ROOT, 'img')
    if os.path.exists(image_dir):
        for file_path in glob.glob(os.path.join(image_dir, "*.png")):
            filename = os.path.basename(file_path)
            # Проверяем принадлежность пользователю
            parts = filename.split('_')
            if len(parts) >= 3 and parts[1] == str(user.id):
                file_time = os.path.getmtime(file_path)
                user_images.append({
                    'filename': filename,
                    'url': f'/media/img/{filename}',
                    'time': file_time,
                    'date': time.strftime('%Y-%m-%d %H:%M', time.localtime(file_time))
                })

    user_images.sort(key=lambda x: x['time'], reverse=True)

    context = {
        'target_user': user,
        'user_prompts': user_prompts,
        'user_images': user_images
    }
    return render(request, 'admin/user_detail.html', context)


# Обновляем функцию index для учета пользователей
def index(request):
    global current_pipe, current_model

    image_url = None
    error = None

    if request.method == 'POST':
        # Получаем данные из формы
        prompt = request.POST.get('prompt', '')
        model_choice = request.POST.get('model', '1')
        quality_choice = request.POST.get('quality', 'fast')
        negative_choice = request.POST.get('negative', 'none')

        print(f"[DEBUG] Получен промпт: {prompt}")
        print(f"[DEBUG] Пользователь: {request.user}")
        print(f"[DEBUG] Аутентифицирован: {request.user.is_authenticated}")

        if prompt:
            try:
                # Загружаем модель если нужно
                models_dict = get_models()
                model_info = models_dict.get(model_choice)
                if not current_pipe or current_model != model_choice:
                    if load_model(model_info['model_id']):
                        current_model = model_choice
                    else:
                        error = "Ошибка загрузки модели"

                if current_pipe and not error:
                    # Получаем настройки
                    quality_presets = get_quality_presets()
                    quality_settings = quality_presets[quality_choice]
                    negative_prompt = NEGATIVE_PROMPTS[negative_choice]['prompt']

                    # Генерируем изображение
                    english_prompt = translate_to_english(prompt)

                    # Сохраняем в историю промптов С ПРИВЯЗКОЙ К ПОЛЬЗОВАТЕЛЮ
                    try:
                        print(f"[DEBUG] Попытка сохранения промпта в историю")

                        # Создаем или обновляем запись в истории промптов
                        prompt_obj, created = PromptHistory.objects.get_or_create(
                            prompt_text=prompt,
                            model_used=model_info['name'],
                            quality_used=quality_choice,
                            negative_prompt_used=negative_choice,
                            defaults={
                                'translated_prompt': english_prompt,
                                'use_count': 1,
                                'user': request.user if request.user.is_authenticated else None
                            }
                        )

                        if not created:
                            # Если запись уже существует - обновляем счетчик и перевод
                            prompt_obj.use_count += 1
                            prompt_obj.translated_prompt = english_prompt
                            prompt_obj.user = request.user if request.user.is_authenticated else None
                            prompt_obj.save()
                            print(f"[SUCCESS] Промпт обновлен, использован {prompt_obj.use_count} раз")
                        else:
                            print("[SUCCESS] Новый промпт сохранен в историю")

                    except Exception as e:
                        print(f"[ERROR] Ошибка сохранения в историю: {e}")
                        # Если не получается сохранить через get_or_create, пробуем простой способ
                        try:
                            prompt_obj = PromptHistory(
                                prompt_text=prompt,
                                translated_prompt=english_prompt,
                                model_used=model_info['name'],
                                quality_used=quality_choice,
                                negative_prompt_used=negative_choice,
                                use_count=1,
                                user=request.user if request.user.is_authenticated else None
                            )
                            prompt_obj.save()
                            print("[SUCCESS] Промпт сохранен (альтернативный метод)")
                        except Exception as e2:
                            print(f"[ERROR] Альтернативное сохранение тоже не удалось: {e2}")

                    image = current_pipe(
                        prompt=english_prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=quality_settings['steps'],
                        guidance_scale=7.5,
                        width=quality_settings['width'],
                        height=quality_settings['height']
                    ).images[0]

                    # Сохраняем изображение с ID пользователя в имени файла
                    user_id = request.user.id if request.user.is_authenticated else 'anonymous'
                    filename = f"{quality_choice}_{user_id}_{int(time.time())}.png"
                    image_path = os.path.join('media', 'img', filename)
                    image.save(image_path)

                    image_url = f'/media/img/{filename}'

            except Exception as e:
                error = f"Ошибка генерации: {e}"

    stats = SiteStats.get_stats()

    # Контекст для шаблона
    context = {
        'models': get_models(),
        'quality_presets': get_quality_presets(),
        'negative_prompts': NEGATIVE_PROMPTS,
        'image_url': image_url,
        'error': error,
        'total_generated': stats.total_generated
    }
    return render(request, 'index.html', context)
