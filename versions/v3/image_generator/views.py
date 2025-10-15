from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
from .models import SiteStats
from django.views.decorators.csrf import csrf_exempt
from .models import PromptHistory, ModelSettings, GenerationSettings
from diffusers import StableDiffusionPipeline
from googletrans import Translator
from contextlib import redirect_stdout
from django.conf import settings

from io import StringIO
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
    """Показывает архив удаленных изображений"""
    archive_dir = os.path.join(settings.MEDIA_ROOT, 'bad')
    archive_files = []

    if os.path.exists(archive_dir):
        for file_path in glob.glob(os.path.join(archive_dir, "*.png")):
            filename = os.path.basename(file_path)
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
    """Восстанавливает изображение из архива в галерею"""
    if request.method == 'POST':
        try:
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
    """Полностью удаляет изображение из архива"""
    if request.method == 'POST':
        try:
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
    """Перемещает изображение в папку bad вместо удаления"""
    if request.method == 'POST':
        try:
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


def gallery(request):
    # Получаем все изображения из media/img
    image_dir = os.path.join(settings.MEDIA_ROOT, 'img')
    image_files = []

    if os.path.exists(image_dir):
        # Получаем все PNG файлы
        for file_path in glob.glob(os.path.join(image_dir, "*.png")):
            filename = os.path.basename(file_path)
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
                'quality': quality  # Добавляем качество
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

        if prompt:
            try:
                # Загружаем модель если нужно
                models_dict = get_models()  # ИСПОЛЬЗУЕМ ФУНКЦИЮ ИЗ БАЗЫ ДАННЫХ
                model_info = models_dict.get(model_choice)
                if not current_pipe or current_model != model_choice:
                    if load_model(model_info['model_id']):
                        current_model = model_choice
                    else:
                        error = "Ошибка загрузки модели"

                if current_pipe and not error:
                    # Получаем настройки
                    quality_presets = get_quality_presets()  # ИСПОЛЬЗУЕМ ФУНКЦИЮ ИЗ БАЗЫ ДАННЫХ
                    quality_settings = quality_presets[quality_choice]
                    negative_prompt = NEGATIVE_PROMPTS[negative_choice]['prompt']

                    # Генерируем изображение
                    english_prompt = translate_to_english(prompt)

                    # СОХРАНЯЕМ В ИСТОРИЮ ПРОМПТОВ - ЭТО НОВАЯ ФУНКЦИОНАЛЬНОСТЬ
                    # СОХРАНЯЕМ В ИСТОРИЮ ПРОМПТОВ
                    # СОХРАНЯЕМ В ИСТОРИЮ ПРОМПТОВ (ОБНОВЛЯЕМ СУЩЕСТВУЮЩИЙ ИЛИ СОЗДАЕМ НОВЫЙ)
                    try:
                        # Используем get_or_create для атомарной операции
                        prompt_obj, created = PromptHistory.objects.get_or_create(
                            prompt_text=prompt,
                            model_used=model_info['name'],
                            quality_used=quality_choice,
                            negative_prompt_used=negative_choice,
                            defaults={
                                'translated_prompt': english_prompt,
                                'use_count': 1
                            }
                        )

                        if not created:
                            # Если промпт уже существует - увеличиваем счетчик и обновляем время
                            prompt_obj.use_count += 1
                            prompt_obj.translated_prompt = english_prompt  # Обновляем перевод на случай изменений
                            prompt_obj.save()
                            print(f"[SUCCESS] Промпт обновлен, использован {prompt_obj.use_count} раз")
                        else:
                            print("[SUCCESS] Новый промпт сохранен в историю")

                    except Exception as e:
                        print(f"[ERROR] Ошибка сохранения в историю: {e}")

                    image = current_pipe(
                        prompt=english_prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=quality_settings['steps'],
                        guidance_scale=7.5,
                        width=quality_settings['width'],
                        height=quality_settings['height']
                    ).images[0]

                    # Сохраняем изображение
                    filename = f"{quality_choice}_{int(time.time())}.png"
                    image_path = os.path.join('media', 'img', filename)
                    image.save(image_path)

                    image_url = f'/media/img/{filename}'

            except Exception as e:
                error = f"Ошибка генерации: {e}"

    stats = SiteStats.get_stats()

    # Контекст для шаблона
    context = {
        'models': get_models(),  # ИСПОЛЬЗУЕМ ФУНКЦИЮ ИЗ БАЗЫ ДАННЫХ
        'quality_presets': get_quality_presets(),  # ИСПОЛЬЗУЕМ ФУНКЦИЮ ИЗ БАЗЫ ДАННЫХ
        'negative_prompts': NEGATIVE_PROMPTS,
        'image_url': image_url,
        'error': error,
        'total_generated': stats.total_generated
    }
    return render(request, 'index.html', context)


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

                # Сохранение
                filename = f"{quality_choice}_{int(time.time())}.png"
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
        prompts = PromptHistory.objects.all().order_by('-last_used_at')[:50]
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