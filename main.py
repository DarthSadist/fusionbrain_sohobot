import os
# Настройки для onnxruntime
os.environ['ONNXRUNTIME_PROVIDERS'] = 'CPUExecutionProvider'
os.environ['ORT_LOGGING_LEVEL'] = '3'  # Только критические ошибки
os.environ['ORT_DISABLE_TENSORRT'] = '1'
os.environ['ORT_DISABLE_CUDA'] = '1'

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="onnxruntime")

import logging
import logging.handlers

# Создаем форматтер для логов
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [USER_ID:%(user_id)s] - %(message)s')

# Создаем файловый обработчик
file_handler = logging.handlers.RotatingFileHandler(
    'bot.log',
    maxBytes=10485760,  # 10MB
    backupCount=5,
    encoding='utf-8'
)
file_handler.setFormatter(log_formatter)

# Создаем консольный обработчик
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

# Настраиваем корневой логгер
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Добавляем фильтр для user_id
class UserIDFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, 'user_id'):
            record.user_id = 'N/A'
        return True

logger.addFilter(UserIDFilter())

import base64
import json
import time
import requests
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types, F
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.client.default import DefaultBotProperties
from aiogram.types import CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.exceptions import TelegramBadRequest
import asyncio
import io
import uuid as uuid_lib
from PIL import Image, ImageEnhance, ImageFilter
from rembg import remove
from collections import defaultdict

# Загрузка переменных окружения из файла .env
load_dotenv()

# Конфигурация
API_TOKEN = os.getenv('API_TOKEN')
FUSIONBRAIN_API_KEY = os.getenv('FUSIONBRAIN_API_KEY')
FUSIONBRAIN_SECRET_KEY = os.getenv('FUSIONBRAIN_SECRET_KEY')

START_IMAGE_URL = 'https://ваша ссылка на картинку'

# Инициализация бота и диспетчера
bot = Bot(token=API_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()

class CensorshipError(Exception):
    pass

class Text2ImageAPI:
    MAX_PROMPT_LENGTH = 500  # Максимальная длина промпта

    def __init__(self, url, api_key, secret_key):
        self.URL = url
        self.AUTH_HEADERS = {
            'X-Key': f'Key {api_key}',
            'X-Secret': f'Secret {secret_key}'
        }

    def _prepare_prompt(self, prompt):
        """Подготовка промпта: обрезка до максимальной длины"""
        if len(prompt) > self.MAX_PROMPT_LENGTH:
            logger.warning(f"Prompt too long ({len(prompt)} chars), truncating to {self.MAX_PROMPT_LENGTH}")
            return prompt[:self.MAX_PROMPT_LENGTH] + "..."
        return prompt

    def get_model(self):
        response = requests.get(f'{self.URL}/key/api/v1/models', headers=self.AUTH_HEADERS)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f'Failed to get models: {response.text}')

    def generate(self, prompt, model, images=1, width=1024, height=1024, reference_image=None):
        url = f'{self.URL}/key/api/v1/text2image/run'
        logger.info(f"Sending request to: {url}")
        
        # Подготавливаем промпт
        prepared_prompt = self._prepare_prompt(prompt)
        
        # Создаем параметры запроса
        params_json = {
            "type": "GENERATE",
            "numImages": images,
            "width": width,
            "height": height,
            "generateParams": {
                "query": prepared_prompt
            }
        }
        
        if reference_image:
            params_json["generateParams"]["reference_image"] = reference_image
        
        # Создаем multipart/form-data
        files = {
            'model_id': (None, str(model)),
            'params': (None, json.dumps(params_json), 'application/json')
        }
        
        logger.info(f"Request files: {files}")
        logger.info(f"Request params: {params_json}")
        logger.info(f"Request headers: {self.AUTH_HEADERS}")
        
        response = requests.post(url, headers=self.AUTH_HEADERS, files=files)
        logger.info(f"Response status: {response.status_code}")
        logger.info(f"Response headers: {dict(response.headers)}")
        logger.info(f"Response text: {response.text}")
        
        if response.status_code in [200, 201]:
            data = response.json()
            return data.get('uuid')
        else:
            raise Exception(f'Failed to generate: {response.text}')

    def check_generation(self, request_id):
        response = requests.get(f'{self.URL}/key/api/v1/text2image/status/{request_id}', headers=self.AUTH_HEADERS)
        logger.info(f"Check status response: {response.text}")
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            raise Exception(f'Failed to check generation: {response.text}')

# Константы для эмодзи
class Emoji:
    SETTINGS = "⚙️"
    CREATE = "🎨"
    BACK = "◀️"
    HELP = "❓"
    REMOVE_BG = "🎭"
    WAIT = "⏳"
    ERROR = "❌"
    SUCCESS = "✅"
    EDIT = "✏️"

# Константы для текстов
class Messages:
    WELCOME = f"""
Привет! Я бот для создания изображений с помощью нейросети Kandinsky.

Что я умею:
• Создавать изображения по текстовому описанию
• Удалять фон с готовых изображений
• Создавать изображения разных размеров
• Генерировать изображения в разных стилях

Нажмите кнопку {Emoji.CREATE} <b>Создать</b>, чтобы начать!
"""
    
    HELP = f"""
{Emoji.CREATE} <b>Как использовать бота:</b>

1. Нажмите кнопку {Emoji.CREATE} <b>Создать</b>
2. Выберите желаемый стиль изображения
3. Опишите желаемое изображение
4. Дождитесь результата
5. Используйте кнопку {Emoji.REMOVE_BG} для удаления фона

{Emoji.SETTINGS} <b>Настройки:</b>
• Выберите размер будущего изображения
• Выберите стиль изображения
"""
    
    SETTINGS = f"""{Emoji.SETTINGS} <b>Текущие настройки</b>

Размер изображения: <b>{{width}}x{{height}}</b>

Выберите новый размер:"""
    
    STYLES = f"""{Emoji.SETTINGS} <b>Выберите стиль изображения</b>

Текущий стиль: <b>{{style_label}}</b>

Выберите новый стиль:"""
    
    STYLE_CHANGED = f"{Emoji.SUCCESS} Установлен стиль: <b>{{style}}</b>"
    
    CURRENT_SETTINGS = f"""{Emoji.SETTINGS} <b>Текущие настройки</b>

Стиль: <b>{{style}}</b>
Размер: <b>{{size}}</b>

{Emoji.EDIT} Теперь опишите желаемое изображение или измените настройки:"""
    
    PROMPT = f"""{Emoji.EDIT} <b>Опишите изображение</b>

Напишите, что бы вы хотели увидеть на изображении. 
Чем подробнее описание, тем лучше результат!

Текущие настройки:
• Стиль: <b>{{style}}</b>
• Размер: <b>{{size}}</b>

Примеры:
• "Космический корабль в стиле киберпанк"
• "Котенок играет с клубком, акварель"
• "Закат на море, масляная живопись"
"""
    
    GENERATING = f"{Emoji.WAIT} <b>Генерирую изображение...</b>"
    REMOVING_BG = f"{Emoji.WAIT} <b>Удаляю фон...</b>"
    SIZE_CHANGED = f"{Emoji.SUCCESS} Установлен размер: <b>{{size}}</b>"
    ERROR_GEN = f"{Emoji.ERROR} Ошибка при генерации: {{error}}"
    ERROR_SIZE = f"{Emoji.ERROR} Ошибка: неверный размер"
    ERROR_CRITICAL = f"{Emoji.ERROR} Произошла критическая ошибка"
    BG_REMOVED = f"{Emoji.SUCCESS} Фон успешно удален!"
    MAIN_MENU = "Выберите действие:"

# Константы для колбэков
class CallbackData:
    SETTINGS = "settings"
    GENERATE = "generate"
    SIZE_PREFIX = "size_"
    HELP = "help"
    BACK = "back_to_main"
    REMOVE_BG = "remove_bg"
    STYLES = "styles"  # Новый callback для меню стилей
    STYLE_PREFIX = "style_"  # Префикс для выбора стиля

# Доступные размеры изображений
IMAGE_SIZES = {
    "square": {
        "width": 1024, 
        "height": 1024, 
        "label": "Квадратное 1024×1024",
        "description": "Идеально для портретов и симметричных композиций"
    },
    "wide": {
        "width": 1024, 
        "height": 576, 
        "label": "Широкое 1024×576",
        "description": "Отлично подходит для пейзажей и панорамных сцен"
    },
    "tall": {
        "width": 576, 
        "height": 1024, 
        "label": "Вертикальное 576×1024",
        "description": "Лучший выбор для портретов в полный рост"
    }
}

# Доступные стили изображений
IMAGE_STYLES = {
    "realistic": {
        "label": "📸 Фотореалистичный",
        "description": "Максимально реалистичные изображения",
        "prompt_prefix": "photorealistic, highly detailed, 8k uhd, high quality"
    },
    "anime": {
        "label": "🎨 Аниме",
        "description": "В стиле японской анимации",
        "prompt_prefix": "anime style, manga, detailed anime illustration"
    },
    "oil": {
        "label": "🖼 Масляная живопись",
        "description": "Классическая масляная живопись",
        "prompt_prefix": "oil painting, detailed brushstrokes, canvas texture, artistic"
    },
    "watercolor": {
        "label": "💦 Акварель",
        "description": "Нежная акварельная техника",
        "prompt_prefix": "watercolor painting, soft colors, watercolor paper texture"
    },
    "digital": {
        "label": "💻 Цифровое искусство",
        "description": "Современное цифровое искусство",
        "prompt_prefix": "digital art, concept art, highly detailed digital illustration"
    }
}

# Состояния пользователя
class UserState:
    def __init__(self):
        self.width = 1024
        self.height = 1024
        self.awaiting_prompt = False
        self.last_image = None  # Хранение последнего сгенерированного изображения
        self.last_image_id = None  # ID последнего изображения для callback

# Словарь для хранения пользовательских настроек
class UserSettings:
    def __init__(self):
        self.width = 1024
        self.height = 1024
        self.style = "realistic"  # Стиль по умолчанию

user_states = defaultdict(UserState)
user_settings = defaultdict(UserSettings)

# Добавляем класс для работы с изображениями
class ImageProcessor:
    # Максимальный размер изображения для обработки
    MAX_SIZE = 1500
    # Храним экземпляр модели
    _model = None
    
    @classmethod
    def _get_model(cls):
        """Получает или создает экземпляр модели"""
        if cls._model is None:
            from rembg.bg import remove as remove_bg
            cls._model = remove_bg
        return cls._model

    @classmethod
    def _resize_if_needed(cls, image: Image.Image) -> tuple[Image.Image, tuple[int, int]]:
        """Уменьшает изображение, если оно слишком большое"""
        original_size = image.size
        width, height = original_size
        
        # Проверяем, нужно ли уменьшать
        if max(width, height) <= cls.MAX_SIZE:
            return image, None
            
        # Вычисляем новый размер с сохранением пропорций
        if width > height:
            new_width = cls.MAX_SIZE
            new_height = int(height * (cls.MAX_SIZE / width))
        else:
            new_height = cls.MAX_SIZE
            new_width = int(width * (cls.MAX_SIZE / height))
            
        # Уменьшаем изображение
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return resized_image, original_size

    @classmethod
    def _restore_size(cls, image: Image.Image, original_size: tuple[int, int]) -> Image.Image:
        """Возвращает изображение к исходному размеру"""
        if original_size:
            return image.resize(original_size, Image.Resampling.LANCZOS)
        return image

    @classmethod
    async def remove_background(cls, image_data: bytes) -> bytes:
        """Удаляет фон с изображения"""
        try:
            # Создаем объект изображения из байтов
            input_image = Image.open(io.BytesIO(image_data))
            
            # Конвертируем в RGB, если нужно
            if input_image.mode not in ('RGB', 'RGBA'):
                input_image = input_image.convert('RGB')
            
            # Уменьшаем размер, если нужно
            resized_image, original_size = cls._resize_if_needed(input_image)
            
            # Получаем модель и удаляем фон
            remove_bg = cls._get_model()
            output_image = remove_bg(resized_image)
            
            # Возвращаем к исходному размеру, если изображение было уменьшено
            if original_size:
                output_image = cls._restore_size(output_image, original_size)
            
            # Сохраняем результат в байты
            output_buffer = io.BytesIO()
            output_image.save(output_buffer, format='PNG', optimize=True)
            return output_buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Error removing background: {str(e)}")
            raise

# Регистрируем обработчики
@dp.message(Command("start"))
async def send_welcome(message: types.Message):
    user_id = message.from_user.id
    logger.info(f"Новый пользователь начал работу с ботом", extra={'user_id': user_id})
    keyboard = get_main_keyboard()
    await message.answer(
        Messages.WELCOME,
        reply_markup=keyboard
    )

async def update_message(message: types.Message, text: str, reply_markup: InlineKeyboardMarkup = None) -> None:
    """Обновляет или отправляет новое сообщение в зависимости от типа текущего сообщения"""
    try:
        # Пробуем отредактировать текущее сообщение
        await message.edit_text(text, reply_markup=reply_markup)
    except (TelegramBadRequest, AttributeError):
        # Если не получилось (например, это сообщение с фото), отправляем новое
        await message.answer(text, reply_markup=reply_markup)

@dp.callback_query(lambda c: c.data == CallbackData.HELP)
async def show_help(callback_query: CallbackQuery):
    user_id = callback_query.from_user.id
    logger.info(f"Пользователь запросил помощь", extra={'user_id': user_id})
    help_text = Messages.HELP
    await update_message(callback_query.message, help_text, get_main_keyboard())
    await callback_query.answer()

@dp.callback_query(lambda c: c.data == CallbackData.SETTINGS)
async def show_settings(callback_query: CallbackQuery):
    user_id = callback_query.from_user.id
    logger.info(f"Пользователь открыл настройки", extra={'user_id': user_id})
    settings = user_settings[user_id]
    await update_message(
        callback_query.message,
        Messages.SETTINGS.format(width=settings.width, height=settings.height),
        get_settings_keyboard()
    )
    await callback_query.answer()

@dp.callback_query(lambda c: c.data.startswith(CallbackData.SIZE_PREFIX))
async def process_size_change(callback_query: CallbackQuery):
    try:
        user_id = callback_query.from_user.id
        size_key = callback_query.data.replace(CallbackData.SIZE_PREFIX, "")
        
        if size_key not in IMAGE_SIZES:
            logger.error(f"Неверный ключ размера: {size_key}", extra={'user_id': user_id})
            await callback_query.answer("❌ Ошибка: неверный размер")
            return
            
        size_config = IMAGE_SIZES[size_key]
        user_settings[user_id].width = size_config["width"]
        user_settings[user_id].height = size_config["height"]
        
        logger.info(f"Пользователь изменил размер изображения на {size_config['label']}", extra={'user_id': user_id})
        
        if user_states[user_id].awaiting_prompt:
            # Если ожидаем промпт, показываем обновленные настройки
            await update_message(
                callback_query.message,
                Messages.CURRENT_SETTINGS.format(
                    style=IMAGE_STYLES[user_settings[user_id].style]["label"],
                    size=size_config['label']
                ),
                get_prompt_keyboard()
            )
        else:
            # Иначе показываем сообщение об изменении размера
            await update_message(
                callback_query.message,
                Messages.SIZE_CHANGED.format(size=size_config['label']),
                get_main_keyboard()
            )
        await callback_query.answer()
        
    except Exception as e:
        logger.error(f"Ошибка при изменении размера: {str(e)}", exc_info=True, extra={'user_id': user_id})
        await callback_query.answer("❌ Произошла ошибка")

@dp.callback_query(lambda c: c.data == CallbackData.GENERATE)
async def start_generation(callback_query: CallbackQuery):
    user_id = callback_query.from_user.id
    user_states[user_id].awaiting_prompt = True
    settings = user_settings[user_id]
    
    logger.info(f"Пользователь начал процесс генерации", extra={'user_id': user_id})
    
    # Показываем текущие настройки и запрашиваем промпт
    await update_message(
        callback_query.message,
        Messages.CURRENT_SETTINGS.format(
            style=IMAGE_STYLES[settings.style]["label"],
            size=f"{settings.width}x{settings.height}"
        ),
        get_prompt_keyboard()
    )
    await callback_query.answer()

@dp.callback_query(lambda c: c.data == CallbackData.BACK)
async def back_to_main(callback_query: CallbackQuery):
    await update_message(
        callback_query.message,
        Messages.MAIN_MENU,
        get_main_keyboard()
    )
    await callback_query.answer()

@dp.message()
async def generate_image(message: types.Message):
    try:
        user_id = message.from_user.id
        
        if not user_states[user_id].awaiting_prompt:
            logger.info(f"Получено сообщение без ожидания промпта", extra={'user_id': user_id})
            return

        prompt = message.text
        logger.info(f"Получен промпт для генерации: {prompt}", extra={'user_id': user_id})

        # Сбрасываем флаг ожидания промпта
        user_states[user_id].awaiting_prompt = False
        
        # Получаем настройки пользователя
        settings = user_settings[user_id]
        
        # Получаем стиль и добавляем его к промпту
        style_config = IMAGE_STYLES[settings.style]
        full_prompt = f"{style_config['prompt_prefix']}, {prompt}"
        
        processing_message = await message.answer(Messages.GENERATING)
        
        try:
            # Инициализация API
            api = Text2ImageAPI('https://api-key.fusionbrain.ai/', FUSIONBRAIN_API_KEY, FUSIONBRAIN_SECRET_KEY)
            
            # Получение модели
            logger.info("Запрос списка моделей", extra={'user_id': user_id})
            models = api.get_model()
            model_id = models[0]["id"]
            
            # Подготовка параметров генерации
            params = {
                "width": settings.width,
                "height": settings.height
            }
            
            # Генерация изображения
            logger.info("Отправка запроса на генерацию", extra={'user_id': user_id})
            uuid = api.generate(full_prompt, model_id, **params)
            
            # Ожидание результата
            while True:
                await asyncio.sleep(1)
                status = api.check_generation(uuid)
                
                if status["status"] == "DONE":
                    images = status["images"]
                    
                    if not images:
                        raise CensorshipError("Контент не прошел модерацию")
                    
                    # Сохраняем изображение для возможности удаления фона
                    image_data = base64.b64decode(images[0])
                    user_states[user_id].last_image = image_data
                    user_states[user_id].last_image_id = str(uuid_lib.uuid4())
                    
                    logger.info("Изображение успешно сгенерировано", extra={'user_id': user_id})
                    
                    # Отправляем изображение
                    await message.answer_photo(
                        types.BufferedInputFile(
                            image_data,
                            filename="generated_image.png"
                        ),
                        reply_markup=get_image_keyboard(user_states[user_id].last_image_id)
                    )
                    break
                
                elif status["status"] == "FAILED":
                    error_msg = status.get("error", "Неизвестная ошибка")
                    logger.error(f"Ошибка генерации: {error_msg}", extra={'user_id': user_id})
                    await message.answer(Messages.ERROR_GEN.format(error=error_msg))
                    break
        
        except CensorshipError as e:
            logger.warning(f"Ошибка цензуры: {str(e)}", extra={'user_id': user_id})
            await message.answer(str(e))
        except Exception as e:
            logger.error(f"Неожиданная ошибка: {str(e)}", exc_info=True, extra={'user_id': user_id})
            await message.answer(Messages.ERROR_CRITICAL)
        finally:
            await processing_message.delete()
            
    except Exception as e:
        logger.error(f"Критическая ошибка: {str(e)}", exc_info=True, extra={'user_id': user_id})
        await message.answer(Messages.ERROR_CRITICAL)

@dp.callback_query(lambda c: c.data.startswith(CallbackData.REMOVE_BG))
async def process_remove_background(callback_query: CallbackQuery):
    try:
        user_id = callback_query.from_user.id
        image_id = callback_query.data.replace(CallbackData.REMOVE_BG, "")
        
        logger.info(f"Запрос на удаление фона для изображения {image_id}", extra={'user_id': user_id})
        
        # Проверяем, есть ли сохраненное изображение
        if (not user_states[user_id].last_image or 
            user_states[user_id].last_image_id != image_id):
            logger.warning("Изображение не найдено в кэше", extra={'user_id': user_id})
            await callback_query.answer("❌ Изображение не найдено")
            return
        
        # Отправляем сообщение о начале обработки
        processing_message = await callback_query.message.answer(Messages.REMOVING_BG)
        
        try:
            # Удаляем фон
            image_without_bg = await ImageProcessor.remove_background(user_states[user_id].last_image)
            
            logger.info("Фон успешно удален", extra={'user_id': user_id})
            
            # Отправляем обработанное изображение
            await callback_query.message.answer_photo(
                types.BufferedInputFile(
                    image_without_bg,
                    filename="image_without_bg.png"
                ),
                caption=Messages.BG_REMOVED,
                reply_markup=get_main_keyboard()
            )
            
        except Exception as e:
            logger.error(f"Ошибка при удалении фона: {str(e)}", exc_info=True, extra={'user_id': user_id})
            await callback_query.message.answer(Messages.ERROR_CRITICAL)
            
        finally:
            await processing_message.delete()
            
    except Exception as e:
        logger.error(f"Критическая ошибка при удалении фона: {str(e)}", exc_info=True, extra={'user_id': user_id})
        await callback_query.message.answer(Messages.ERROR_CRITICAL)

@dp.callback_query(lambda c: c.data == CallbackData.STYLES)
async def show_styles(callback_query: CallbackQuery):
    user_id = callback_query.from_user.id
    current_style = user_settings[user_id].style
    style_label = IMAGE_STYLES[current_style]["label"]
    
    logger.info(f"Пользователь открыл меню стилей", extra={'user_id': user_id})
    
    await update_message(
        callback_query.message,
        Messages.STYLES.format(style_label=style_label),
        get_styles_keyboard()
    )
    await callback_query.answer()

@dp.callback_query(lambda c: c.data.startswith(CallbackData.STYLE_PREFIX))
async def process_style_change(callback_query: CallbackQuery):
    try:
        user_id = callback_query.from_user.id
        style_key = callback_query.data.replace(CallbackData.STYLE_PREFIX, "")
        
        if style_key not in IMAGE_STYLES:
            logger.error(f"Неверный ключ стиля: {style_key}", extra={'user_id': user_id})
            await callback_query.answer("❌ Ошибка: неверный стиль")
            return
            
        user_settings[user_id].style = style_key
        style_config = IMAGE_STYLES[style_key]
        
        logger.info(f"Пользователь изменил стиль на {style_config['label']}", extra={'user_id': user_id})
        
        if user_states[user_id].awaiting_prompt:
            # Если ожидаем промпт, показываем обновленные настройки
            await update_message(
                callback_query.message,
                Messages.CURRENT_SETTINGS.format(
                    style=style_config['label'],
                    size=f"{user_settings[user_id].width}x{user_settings[user_id].height}"
                ),
                get_prompt_keyboard()
            )
        else:
            # Иначе показываем сообщение об изменении стиля
            await update_message(
                callback_query.message,
                Messages.STYLE_CHANGED.format(style=style_config['label']),
                get_main_keyboard()
            )
        await callback_query.answer()
        
    except Exception as e:
        logger.error(f"Ошибка при изменении стиля: {str(e)}", exc_info=True, extra={'user_id': user_id})
        await callback_query.answer("❌ Произошла ошибка")

def get_image_keyboard(image_id: str) -> InlineKeyboardMarkup:
    """Создает клавиатуру для изображения"""
    keyboard = [
        [InlineKeyboardButton(
            text=f"{Emoji.REMOVE_BG} Удалить фон",
            callback_data=f"{CallbackData.REMOVE_BG}{image_id}"
        )],
        [InlineKeyboardButton(
            text=f"{Emoji.CREATE} Создать новое",
            callback_data=CallbackData.GENERATE
        )],
        [InlineKeyboardButton(
            text=f"{Emoji.BACK} В главное меню",
            callback_data=CallbackData.BACK
        )]
    ]
    return InlineKeyboardMarkup(inline_keyboard=keyboard)

def get_main_keyboard() -> InlineKeyboardMarkup:
    """Создает основную клавиатуру с главным меню"""
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=f"{Emoji.CREATE} Создать", callback_data=CallbackData.GENERATE)],
        [InlineKeyboardButton(text=f"{Emoji.SETTINGS} Стили", callback_data=CallbackData.STYLES)],
        [InlineKeyboardButton(text=f"{Emoji.SETTINGS} Настройки", callback_data=CallbackData.SETTINGS)],
        [InlineKeyboardButton(text=f"{Emoji.HELP} Помощь", callback_data=CallbackData.HELP)]
    ])

def get_settings_keyboard() -> InlineKeyboardMarkup:
    """Создает клавиатуру с настройками размеров"""
    keyboard = []
    
    # Добавляем кнопки для каждого размера
    for size_key, size_config in IMAGE_SIZES.items():
        keyboard.append([
            InlineKeyboardButton(
                text=f"{size_config['label']} - {size_config['description']}",
                callback_data=f"{CallbackData.SIZE_PREFIX}{size_key}"
            )
        ])
    
    # Добавляем кнопку возврата
    keyboard.append([
        InlineKeyboardButton(
            text=f"{Emoji.BACK} В главное меню",
            callback_data=CallbackData.BACK
        )
    ])
    
    return InlineKeyboardMarkup(inline_keyboard=keyboard)

def get_styles_keyboard() -> InlineKeyboardMarkup:
    """Создает клавиатуру с выбором стилей"""
    keyboard = []
    
    # Добавляем кнопки для каждого стиля
    for style_key, style_config in IMAGE_STYLES.items():
        keyboard.append([
            InlineKeyboardButton(
                text=f"{style_config['label']} - {style_config['description']}",
                callback_data=f"{CallbackData.STYLE_PREFIX}{style_key}"
            )
        ])
    
    # Добавляем кнопку возврата
    keyboard.append([
        InlineKeyboardButton(
            text=f"{Emoji.BACK} В главное меню",
            callback_data=CallbackData.BACK
        )
    ])
    
    return InlineKeyboardMarkup(inline_keyboard=keyboard)

def get_prompt_keyboard() -> InlineKeyboardMarkup:
    """Создает клавиатуру для режима ввода промпта"""
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=f"{Emoji.SETTINGS} Стиль", callback_data=CallbackData.STYLES)],
        [InlineKeyboardButton(text=f"{Emoji.SETTINGS} Размер", callback_data=CallbackData.SETTINGS)],
        [InlineKeyboardButton(text=f"{Emoji.BACK} В главное меню", callback_data=CallbackData.BACK)]
    ])

def get_back_keyboard() -> InlineKeyboardMarkup:
    """Создает клавиатуру для возврата"""
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=f"{Emoji.BACK} В главное меню", callback_data=CallbackData.BACK)]
    ])

async def main():
    # Запуск бота
    await dp.start_polling(bot, skip_updates=True)

if __name__ == '__main__':
    asyncio.run(main())
