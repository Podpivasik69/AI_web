from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
import torch
from diffusers import StableDiffusionPipeline
from googletrans import Translator
import os
import time
import json
from django.views.decorators.csrf import csrf_exempt
from .models import PromptHistory, ModelSettings, GenerationSettings
import io
import sys
from contextlib import redirect_stdout
import re
import sys
from io import StringIO
import glob
import os
from django.conf import settings
from django.http import HttpResponse

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
            image_files.append({
                'filename': filename,
                'url': f'/media/img/{filename}',
                'time': file_time,
                'date': time.strftime('%Y-%m-%d %H:%M', time.localtime(file_time))
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
MODELS = {
    '1': {'name': 'Waifu Diffusion', 'model_id': 'hakurei/waifu-diffusion', 'style': 'anime'},
    '2': {'name': 'MeinaMix', 'model_id': 'Meina/MeinaMix_V11', 'style': 'anime'},
    '3': {'name': 'Anything V5', 'model_id': 'stablediffusionapi/anything-v5', 'style': 'anime'},
    '4': {'name': 'Stable Diffusion v1.5', 'model_id': 'runwayml/stable-diffusion-v1-5', 'style': 'realistic'}
}

# Настройки качества
QUALITY_PRESETS = {
    'fast': {'steps': 20, 'width': 512, 'height': 512, 'name': 'Быстрое'},
    'medium': {'steps': 30, 'width': 640, 'height': 640, 'name': 'Среднее'},
    'high': {'steps': 40, 'width': 768, 'height': 768, 'name': 'Высокое'}
}

# Негативные промпты
NEGATIVE_PROMPTS = {
    'none': {'name': 'Без негативных', 'prompt': ''},
    'standard': {'name': 'Стандартные', 'prompt': 'ugly, poorly drawn, deformed, blurry, low quality, worst quality'},
    'anime': {'name': 'Для аниме',
              'prompt': 'ugly, poorly drawn, deformed, blurry, bad anatomy, bad hands, missing fingers, extra limbs'}
}

# Глобальные переменные
current_pipe = None
current_model = None


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
        return translator.translate(text, src='ru', dest='en').text
    except:
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
                model_info = MODELS.get(model_choice)
                if not current_pipe or current_model != model_choice:
                    if load_model(model_info['model_id']):
                        current_model = model_choice
                    else:
                        error = "Ошибка загрузки модели"

                if current_pipe and not error:
                    # Получаем настройки
                    quality_settings = QUALITY_PRESETS[quality_choice]
                    negative_prompt = NEGATIVE_PROMPTS[negative_choice]['prompt']

                    # Генерируем изображение
                    english_prompt = translate_to_english(prompt)

                    image = current_pipe(
                        prompt=english_prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=quality_settings['steps'],
                        guidance_scale=7.5,
                        width=quality_settings['width'],
                        height=quality_settings['height']
                    ).images[0]

                    # Сохраняем изображение
                    filename = f"img_{int(time.time())}.png"
                    image_path = os.path.join('media', 'img', filename)
                    image.save(image_path)

                    image_url = f'/media/img/{filename}'

            except Exception as e:
                error = f"Ошибка генерации: {e}"

    # Контекст для шаблона
    context = {
        'models': MODELS,
        'quality_presets': QUALITY_PRESETS,
        'negative_prompts': NEGATIVE_PROMPTS,
        'image_url': image_url,
        'error': error
    }
    return render(request, 'index.html', context)


@csrf_exempt
def generate_ajax(request):
    global current_pipe, current_model, is_generating

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

            try:
                # Загрузка модели
                model_info = MODELS.get(model_choice)
                if not current_pipe or current_model != model_choice:
                    if not load_model(model_info['model_id']):
                        return JsonResponse({'error': 'Ошибка загрузки модели'})
                    current_model = model_choice

                # Генерация
                quality_settings = QUALITY_PRESETS[quality_choice]
                negative_prompt = NEGATIVE_PROMPTS[negative_choice]['prompt']
                english_prompt = translate_to_english(prompt)

                image = current_pipe(
                    prompt=english_prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=quality_settings['steps'],
                    guidance_scale=7.5,
                    width=quality_settings['width'],
                    height=quality_settings['height']
                ).images[0]

                # Сохранение с качеством в имени файла
                filename = f"{quality_choice}_{int(time.time())}.png"  # ИЗМЕНЕНИЕ ЗДЕСЬ
                image_path = os.path.join('media', 'img', filename)
                image.save(image_path)

                is_generating = False
                return JsonResponse({'success': True, 'image_url': f'/media/img/{filename}'})

            except Exception as e:
                is_generating = False
                return JsonResponse({'error': f'Ошибка генерации: {e}'})

    return JsonResponse({'error': 'Invalid request'})


@csrf_exempt
def get_progress(request):
    return JsonResponse({'progress': 0, 'generating': False})  # Не используется


@csrf_exempt
def get_progress(request):
    return JsonResponse({'progress': 0, 'generating': False})  # Не используется
