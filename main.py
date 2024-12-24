import os
import sys
import logging
import asyncio
import base64
from datetime import datetime
from aiogram import Bot, Dispatcher, types, Router, F
from aiogram.enums import ParseMode
from aiogram.types import BufferedInputFile
from aiogram.filters import Command
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    FSInputFile
)
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.exceptions import TelegramBadRequest
import aiohttp
import io
import uuid as uuid_lib
from PIL import Image, ImageEnhance, ImageFilter
from rembg import remove
from collections import defaultdict
import json
import time
import requests
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="onnxruntime")

import logging
import logging.handlers

# Настройки для onnxruntime
os.environ['ONNXRUNTIME_PROVIDERS'] = 'CPUExecutionProvider'
os.environ['ORT_LOGGING_LEVEL'] = '3'  # Только критические ошибки
os.environ['ORT_DISABLE_TENSORRT'] = '1'
os.environ['ORT_DISABLE_CUDA'] = '1'

# Создаем форматтер для логов с дополнительной информацией
log_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - [USER_ID:%(user_id)s] - [OPERATION:%(operation)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Создаем файловый обработчик с ротацией по размеру и времени
file_handler = logging.handlers.TimedRotatingFileHandler(
    'logs/bot.log',
    when='midnight',
    interval=1,
    backupCount=7,
    encoding='utf-8'
)
file_handler.setFormatter(log_formatter)

# Создаем консольный обработчик с цветным выводом
class ColoredConsoleHandler(logging.StreamHandler):
    colors = {
        'DEBUG': '\033[0;36m',  # Cyan
        'INFO': '\033[0;32m',   # Green
        'WARNING': '\033[0;33m', # Yellow
        'ERROR': '\033[0;31m',   # Red
        'CRITICAL': '\033[0;35m' # Purple
    }
    reset = '\033[0m'

    def emit(self, record):
        try:
            message = self.format(record)
            color = self.colors.get(record.levelname, self.reset)
            self.stream.write(f'{color}{message}{self.reset}\n')
            self.flush()
        except Exception:
            self.handleError(record)

console_handler = ColoredConsoleHandler()
console_handler.setFormatter(log_formatter)

# Настраиваем корневой логгер
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Добавляем расширенный фильтр для контекстной информации
class ContextFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, 'user_id'):
            record.user_id = 'N/A'
        if not hasattr(record, 'operation'):
            record.operation = 'SYSTEM'
        return True

logger.addFilter(ContextFilter())

# Создаем директорию для логов, если она не существует
os.makedirs('logs', exist_ok=True)

from dotenv import load_dotenv
from src.api.text2image import Text2ImageAPI
from src.models.image_info import ImageInfo
from src.constants.messages import MessageTemplate, MessageKey
from src.constants.bot_constants import (
    IMAGE_STYLES,
    IMAGE_SIZES,
    StyleType,
    EmojiEnum,
    CallbackEnum,
    ImageSize
)
from src.utils.image_processor import ImageProcessor

# Загрузка переменных окружения из файла .env
load_dotenv()

# Конфигурация
API_TOKEN = os.getenv('API_TOKEN')
FUSIONBRAIN_API_KEY = os.getenv('FUSIONBRAIN_API_KEY')
FUSIONBRAIN_SECRET_KEY = os.getenv('FUSIONBRAIN_SECRET_KEY')

# Проверяем наличие всех необходимых переменных окружения
if not all([API_TOKEN, FUSIONBRAIN_API_KEY, FUSIONBRAIN_SECRET_KEY]):
    logger.error("Не все необходимые переменные окружения установлены!")
    if not API_TOKEN:
        logger.error("Отсутствует API_TOKEN")
    if not FUSIONBRAIN_API_KEY:
        logger.error("Отсутствует FUSIONBRAIN_API_KEY")
    if not FUSIONBRAIN_SECRET_KEY:
        logger.error("Отсутствует FUSIONBRAIN_SECRET_KEY")
    sys.exit(1)

# Проверяем формат ключей
if any([' ' in key for key in [FUSIONBRAIN_API_KEY, FUSIONBRAIN_SECRET_KEY]]):
    logger.error("API ключи не должны содержать пробелов!")
    sys.exit(1)

if any(['"' in key or "'" in key for key in [FUSIONBRAIN_API_KEY, FUSIONBRAIN_SECRET_KEY]]):
    logger.error("API ключи не должны содержать кавычек!")
    sys.exit(1)

logger.info("Конфигурация загружена успешно")
logger.debug(f"API Key length: {len(FUSIONBRAIN_API_KEY)}, Secret Key length: {len(FUSIONBRAIN_SECRET_KEY)}")

START_IMAGE_URL = 'https://ваша ссылка на картинку'

# Инициализация бота и диспетчера
bot = Bot(token=API_TOKEN, parse_mode=ParseMode.HTML)
dp = Dispatcher()
router = Router()
dp.include_router(router)

class CensorshipError(Exception):
    pass

class Text2ImageAPI:
    MAX_PROMPT_LENGTH = 500

    def __init__(self, api_key, secret_key):
        self.URL = 'https://api-key.fusionbrain.ai'
        self.api_key = api_key
        self.secret_key = secret_key
        self.logger = logging.getLogger(__name__)

    async def _make_request(self, method, url, **kwargs):
        """Выполняет запрос к API с правильной авторизацией"""
        headers = {
            "X-Key": f"Key {self.api_key}",
            "X-Secret": f"Secret {self.secret_key}",
        }
        if "headers" in kwargs:
            kwargs["headers"].update(headers)
        else:
            kwargs["headers"] = headers

        async with aiohttp.ClientSession() as session:
            async with session.request(method, url, **kwargs) as response:
                response_text = await response.text()
                self.logger.info(
                    f"API Response: url={url}, status={response.status}, response={response_text}",
                    extra={'operation': 'API_REQUEST'}
                )
                
                # Проверяем статус ответа
                if response.status == 401:
                    self.logger.error(
                        "Ошибка авторизации: неверные ключи API",
                        extra={'operation': 'AUTH_ERROR'}
                    )
                    raise Exception("Ошибка авторизации. Проверьте правильность ключей API.")
                elif response.status == 403:
                    raise Exception("Доступ запрещен. Проверьте права доступа.")
                elif response.status == 429:
                    raise Exception("Превышен лимит запросов. Пожалуйста, подождите немного.")
                elif response.status >= 500:
                    raise Exception("Сервер временно недоступен. Попробуйте позже.")
                elif response.status not in [200, 201]:  # Добавляем 201 как допустимый статус
                    raise Exception(f"Ошибка API: {response.status}")
                
                try:
                    return json.loads(response_text)
                except json.JSONDecodeError:
                    raise Exception("Некорректный ответ от сервера")

    def _prepare_prompt(self, prompt: str) -> str:
        """Подготовка промпта: обрезка до максимальной длины"""
        if len(prompt) > self.MAX_PROMPT_LENGTH:
            self.logger.warning(
                f"Промпт превышает максимальную длину ({len(prompt)} > {self.MAX_PROMPT_LENGTH})",
                extra={'operation': 'PROMPT_PREPARATION'}
            )
            return prompt[:self.MAX_PROMPT_LENGTH]
        return prompt

    async def get_model(self) -> list:
        """Получение списка доступных моделей"""
        self.logger.info("Запрос списка моделей", extra={'operation': 'GET_MODELS'})
        try:
            response = await self._make_request('GET', f'{self.URL}/key/api/v1/models')
            if not response:
                raise Exception("Не удалось получить список моделей")
            return response
        except Exception as e:
            self.logger.error(f"Ошибка при получении списка моделей: {str(e)}", 
                            extra={'operation': 'GET_MODELS_ERROR'})
            raise

    async def generate(self, prompt: str, model_id: int, width: int = 1024, height: int = 1024) -> str:
        """Запуск генерации изображения"""
        self.logger.info(
            f"Запуск генерации изображения: prompt='{prompt}', model_id={model_id}, size={width}x{height}",
            extra={'operation': 'GENERATION_START'}
        )
        
        try:
            # Подготовка параметров
            params = {
                "type": "GENERATE",
                "numImages": 1,
                "width": width,
                "height": height,
                "generateParams": {
                    "query": self._prepare_prompt(prompt)
                }
            }

            # Создаем форму для отправки
            form = aiohttp.FormData()
            form.add_field('model_id', str(model_id))
            form.add_field('params', json.dumps(params), content_type='application/json')

            # Отправляем запрос
            response = await self._make_request(
                'POST',
                f'{self.URL}/key/api/v1/text2image/run',
                data=form
            )

            # Проверяем ответ
            if not response:
                raise Exception("Пустой ответ от сервера")
            
            uuid = response.get('uuid')
            if not uuid:
                raise Exception("UUID не найден в ответе сервера")

            self.logger.info(
                f"Генерация запущена успешно: uuid={uuid}",
                extra={'operation': 'GENERATION_STARTED'}
            )
            return uuid

        except Exception as e:
            self.logger.error(
                f"Ошибка при запуске генерации: {str(e)}", 
                extra={'operation': 'GENERATION_START_ERROR'}
            )
            raise

    async def check_generation(self, uuid: str) -> dict:
        """Проверка статуса генерации"""
        try:
            url = f"{self.URL}/key/api/v1/text2image/status/{uuid}"
            self.logger.info(f"Проверка статуса генерации: uuid={uuid}", extra={'operation': 'CHECK_STATUS'})
            
            response = await self._make_request("GET", url)
            
            if not response:
                self.logger.error("Получен пустой ответ от сервера", extra={
                    'operation': 'CHECK_STATUS_ERROR',
                    'uuid': uuid
                })
                raise Exception("Пустой ответ при проверке статуса")
            
            status = response.get("status")
            self.logger.info(f"Статус генерации: {status}", extra={
                'operation': 'GENERATION_STATUS',
                'uuid': uuid,
                'status': status
            })
            
            if status == "DONE":
                images = response.get("images")
                if not images:
                    self.logger.error("Изображения отсутствуют в ответе", extra={
                        'operation': 'CHECK_STATUS_ERROR',
                        'uuid': uuid,
                        'status': status
                    })
                    raise Exception("Изображения отсутствуют в ответе")
                
                self.logger.info("Генерация завершена успешно", extra={
                    'operation': 'GENERATION_DONE',
                    'uuid': uuid,
                    'images_count': len(images)
                })
                return response
                
            elif status in ["INITIAL", "PROCESSING"]:
                self.logger.info("Генерация все еще выполняется", extra={
                    'operation': 'GENERATION_IN_PROGRESS',
                    'uuid': uuid,
                    'status': status
                })
                raise Exception("Generation still in progress")
                
            elif status == "FAILED":
                error = response.get("error", "Неизвестная ошибка")
                self.logger.error(f"Генерация не удалась: {error}", extra={
                    'operation': 'GENERATION_FAILED',
                    'uuid': uuid,
                    'error': error
                })
                raise Exception(f"Генерация не удалась: {error}")
                
            else:
                self.logger.error(f"Получен неизвестный статус: {status}", extra={
                    'operation': 'UNKNOWN_STATUS',
                    'uuid': uuid,
                    'status': status
                })
                raise Exception(f"Неизвестный статус генерации: {status}")
                
        except Exception as e:
            self.logger.error(f"Ошибка при проверке статуса генерации: {str(e)}", extra={
                'operation': 'CHECK_STATUS_ERROR',
                'uuid': uuid
            })
            raise

# Константы для эмодзи
class EmojiEnum:
    """Эмодзи для кнопок и сообщений"""
    SETTINGS = "⚙️"
    BACK = "↩️"
    CREATE = "🎨"
    HELP = "❓"
    CHECK = "✅"
    REMOVE_BG = "🖼"
    WAIT = "⏳"
    ERROR = "❌"
    SUCCESS = "✅"
    GALLERY = "🗂"
    STYLE = "🎭"
    SIZE = "📏"
    HOME = "🏠"

# Константы для текстов
class Messages:
    """Шаблоны сообщений бота"""
    WELCOME = (
        "Привет! Я бот для генерации изображений.\n\n"
        "🎨 Текущий стиль: <b>{current_style}</b>\n"
        "Нажмите кнопку <b>Создать</b>, чтобы начать."
    )
    
    PROMPT = (
        "Опишите изображение, которое хотите создать.\n\n"
        "🎨 Стиль: <b>{style}</b>\n"
        "📏 Размер: <b>{size}</b>\n\n"
        "💡 <b>Советы по составлению описания:</b>\n"
        "• Используйте конкретные детали\n"
        "• Указывайте время суток и освещение\n"
        "• Описывайте настроение и атмосферу"
    )
    
    GENERATING = (
        "⏳ <b>Генерация изображения...</b>\n\n"
        "Это может занять некоторое время.\n"
        "🎨 Стиль: <b>{style}</b>"
    )
    
    REMOVING_BG = (
        "⏳ <b>Удаление фона...</b>\n\n"
        "Это может занять некоторое время."
    )
    
    REMOVE_BG_SUCCESS = "✅ <b>Фон успешно удален!</b>"
    
    REMOVE_BG_ERROR = (
        "❌ <b>Ошибка при удалении фона</b>\n\n"
        "{error}"
    )
    
    ERROR_GEN = (
        "❌ <b>Ошибка при генерации изображения</b>\n\n"
        "{error}\n\n"
        "Попробуйте:\n"
        "• Изменить описание\n"
        "• Выбрать другой стиль\n"
        "• Уменьшить размер изображения"
    )
    
    ERROR_CRITICAL = (
        "❌ Произошла критическая ошибка.\n"
        "Попробуйте еще раз или обратитесь к администратору."
    )
    
    SIZE_CHANGED = "✅ <b>Размер изменен на {size}</b>"
    
    STYLE_CHANGED = "✅ <b>Стиль изменен на {style}</b>"
    
    HELP = (
        "<b>Как пользоваться ботом:</b>\n\n"
        "1. Нажмите кнопку <b>Создать</b>\n"
        "2. Введите описание желаемого изображения\n"
        "3. Дождитесь результата\n\n"
        "<b>Дополнительные возможности:</b>\n"
        "• <b>Стили</b> - выбор стиля изображения\n"
        "• <b>Настройки</b> - изменение размера изображения\n"
        "• <b>Повторить</b> - повторная генерация с тем же промптом\n"
        "• <b>Удалить фон</b> - удаление фона с изображения"
    )
    
    STYLES = (
        "🎨 <b>Выберите стиль изображения</b>\n\n"
        "Текущий стиль: <b>{current_style}</b>"
    )
    
    SETTINGS = (
        "⚙️ <b>Настройки</b>\n\n"
        "Текущий размер: <b>{current_size}</b>"
    )
    
    MAIN_MENU = "Выберите действие:"
    
    CURRENT_SETTINGS = (
        "🎨 <b>Создание изображения</b>\n\n"
        "🎨 Стиль: <b>{style}</b>\n"
        "📏 Размер: <b>{size}</b>\n"
        "✍️ Введите описание желаемого изображения:"
    )

# Константы для колбэков
class CallbackEnum:
    """Callback-данные для кнопок"""
    BACK = "back"
    SETTINGS = "settings"
    STYLES = "styles"
    GENERATE = "generate"
    REGENERATE = "regenerate"
    STYLE_PREFIX = "style_"
    HELP = "help"
    REMOVE_BG = "remove_bg"
    SIZE_PREFIX = "size_"

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
    "DEFAULT": {
        "label": "Обычный",
        "prompt_prefix": "",
        "description": "Стандартный стиль без дополнительных модификаций. Подходит для общих изображений.",
        "example": "Например: 'Красивый закат над морем'",
        "model_id": 1
    },
    "ANIME": {
        "label": "Аниме",
        "prompt_prefix": "anime style, anime art, high quality anime art, ",
        "description": "Стиль японской анимации в духе современного аниме и манги. Яркие цвета и характерные черты персонажей.",
        "example": "Например: 'Девушка с длинными волосами в школьной форме'",
        "model_id": 1
    },
    "REALISTIC": {
        "label": "Реалистичный",
        "prompt_prefix": "realistic, photorealistic, hyperrealistic, 8k uhd, high quality, detailed, ",
        "description": "Максимально реалистичное изображение с детальной проработкой и фотографическим качеством.",
        "example": "Например: 'Портрет пожилого человека с морщинами'",
        "model_id": 1
    },
    "PORTRAIT": {
        "label": "Портрет",
        "prompt_prefix": "portrait style, professional portrait, detailed face features, studio lighting, ",
        "description": "Профессиональный портретный стиль с акцентом на черты лица и студийное освещение.",
        "example": "Например: 'Деловой портрет молодой женщины'",
        "model_id": 1
    },
    "STUDIO_GHIBLI": {
        "label": "Студия Гибли",
        "prompt_prefix": "studio ghibli style, ghibli anime, hayao miyazaki style, ",
        "description": "В стиле анимационных фильмов Студии Гибли. Мягкие цвета и сказочная атмосфера.",
        "example": "Например: 'Волшебный лес с духами природы'",
        "model_id": 1
    },
    "CYBERPUNK": {
        "label": "Киберпанк",
        "prompt_prefix": "cyberpunk style, neon lights, futuristic city, high tech low life, ",
        "description": "Футуристический стиль киберпанка с неоновыми огнями и высокотехнологичным окружением.",
        "example": "Например: 'Ночной город будущего с летающими машинами'",
        "model_id": 1
    },
    "WATERCOLOR": {
        "label": "Акварель",
        "prompt_prefix": "watercolor painting, watercolor art style, soft colors, flowing paint, ",
        "description": "Акварельная живопись с мягкими переходами цветов и характерными разводами краски.",
        "example": "Например: 'Весенний пейзаж с цветущей сакурой'",
        "model_id": 1
    },
    "OIL_PAINTING": {
        "label": "Масло",
        "prompt_prefix": "oil painting style, classical art, detailed brush strokes, ",
        "description": "Масляная живопись с детальными мазками кисти и классическим стилем.",
        "example": "Например: 'Портрет молодой женщины в стиле Ренуара'",
        "model_id": 1
    },
    "DIGITAL_ART": {
        "label": "Цифровое искусство",
        "prompt_prefix": "digital art, digital painting, concept art, highly detailed digital illustration, ",
        "description": "Современное цифровое искусство с высоким уровнем детализации и концептуальными идеями.",
        "example": "Например: 'Фантастический город будущего'",
        "model_id": 1
    },
    "PENCIL_SKETCH": {
        "label": "Карандашный эскиз",
        "prompt_prefix": "pencil sketch, graphite drawing, detailed line art, black and white sketch, ",
        "description": "Карандашный рисунок с детальными линиями и черно-белой палитрой.",
        "example": "Например: 'Портрет пожилого человека'",
        "model_id": 1
    },
    "POP_ART": {
        "label": "Поп-арт",
        "prompt_prefix": "pop art style, bright colors, bold patterns, comic book style, ",
        "description": "Яркий стиль поп-арт с яркими цветами и смелыми узорами.",
        "example": "Например: 'Портрет знаменитости в стиле поп-арт'",
        "model_id": 1
    },
    "STEAMPUNK": {
        "label": "Стимпанк",
        "prompt_prefix": "steampunk style, victorian era, brass and copper, mechanical parts, steam-powered machinery, ",
        "description": "Стиль альтернативной викторианской эпохи с механическими деталями и паровыми машинами.",
        "example": "Например: 'Город будущего в стиле стимпанк'",
        "model_id": 1
    },
    "FANTASY": {
        "label": "Фэнтези",
        "prompt_prefix": "fantasy art style, magical, mystical, ethereal atmosphere, ",
        "description": "Фэнтезийный стиль с магическими элементами и мистической атмосферой.",
        "example": "Например: 'Волшебный лес с драконами'",
        "model_id": 1
    },
    "MINIMALIST": {
        "label": "Минимализм",
        "prompt_prefix": "minimalist style, simple shapes, clean lines, minimal color palette, ",
        "description": "Минималистичный стиль с простыми формами и ограниченной цветовой палитрой.",
        "example": "Например: 'Простой пейзаж с минималистичным дизайном'",
        "model_id": 1
    },
    "IMPRESSIONIST": {
        "label": "Импрессионизм",
        "prompt_prefix": "impressionist painting style, loose brush strokes, light and color focus, plein air, ",
        "description": "Стиль импрессионизма с свободными мазками кисти и акцентом на свете и цвете.",
        "example": "Например: 'Пейзаж в стиле импрессионизма'",
        "model_id": 1
    },
    "SURREALISM": {
        "label": "Сюрреализм",
        "prompt_prefix": "surrealist art style, dreamlike, abstract elements, symbolic imagery, ",
        "description": "Сюрреалистический стиль с абстрактными элементами и символическими образами.",
        "example": "Например: 'Сюрреалистический пейзаж с абстрактными формами'",
        "model_id": 1
    },
    "COMIC": {
        "label": "Комикс",
        "prompt_prefix": "comic book style, bold outlines, cel shading, action lines, ",
        "description": "Стиль комиксов с яркими контурами и динамичными линиями.",
        "example": "Например: 'Комикс о супергероях'",
        "model_id": 1
    },
    "PIXEL_ART": {
        "label": "Пиксель-арт",
        "prompt_prefix": "pixel art style, retro gaming, 8-bit graphics, pixelated, ",
        "description": "Пиксельная графика в стиле ретро-игр и 8-битной графики.",
        "example": "Например: 'Пиксель-арт персонажа из ретро-игры'",
        "model_id": 1
    },
    "GOTHIC": {
        "label": "Готика",
        "prompt_prefix": "gothic art style, dark atmosphere, medieval architecture, dramatic lighting, ",
        "description": "Готический стиль с темной атмосферой и драматическим освещением.",
        "example": "Например: 'Готический собор в темном лесу'",
        "model_id": 1
    },
    "RETRO": {
        "label": "Ретро",
        "prompt_prefix": "retro style, vintage aesthetics, old school design, nostalgic feel, ",
        "description": "Ретро стиль с винтажной эстетикой и старомодным дизайном.",
        "example": "Например: 'Ретро-пейзаж с винтажными автомобилями'",
        "model_id": 1
    }
}

# Состояния пользователя
class UserState:
    def __init__(self):
        self.width = ImageSize.DEFAULT_SIZE
        self.height = ImageSize.DEFAULT_SIZE
        self.awaiting_prompt = False
        self.last_image = None  # Хранение последнего сгенерированного изображения
        self.last_image_id = None  # ID последнего изображения для callback
        self.last_prompt = None  # Последний использованный промпт

# Словарь для хранения пользовательских настроек
class UserSettings:
    def __init__(self):
        self.width = ImageSize.DEFAULT_SIZE
        self.height = ImageSize.DEFAULT_SIZE
        self.style = StyleType.DEFAULT.name  # Стиль по умолчанию

user_states = defaultdict(UserState)
user_settings = defaultdict(UserSettings)

# Добавляем класс для работы с изображениями
class ImageProcessor:
    """Класс для обработки изображений"""
    MAX_SIZE = 1500
    _model = None

    @classmethod
    def _get_model(cls):
        """Получает или создает экземпляр модели"""
        if cls._model is None:
            cls._model = remove
        return cls._model

    @classmethod
    def _resize_if_needed(cls, image: Image.Image) -> Image.Image:
        """Уменьшает изображение, если оно слишком большое"""
        width, height = image.size
        if width > cls.MAX_SIZE or height > cls.MAX_SIZE:
            # Вычисляем новый размер, сохраняя пропорции
            if width > height:
                new_width = cls.MAX_SIZE
                new_height = int(height * (cls.MAX_SIZE / width))
            else:
                new_height = cls.MAX_SIZE
                new_width = int(width * (cls.MAX_SIZE / height))
            
            logger.info(
                f"Изменение размера изображения с {width}x{height} на {new_width}x{new_height}",
                extra={'operation': 'RESIZE_IMAGE'}
            )
            return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return image

    @classmethod
    def _restore_size(cls, image: Image.Image, original_size: tuple[int, int]) -> Image.Image:
        """Возвращает изображение к исходному размеру"""
        if image.size != original_size:
            logger.info(
                f"Восстановление исходного размера {original_size[0]}x{original_size[1]}",
                extra={'operation': 'RESTORE_SIZE'}
            )
            return image.resize(original_size, Image.Resampling.LANCZOS)
        return image

    @classmethod
    def remove_background(cls, image_data: bytes) -> bytes:
        """Удаляет фон с изображения"""
        try:
            logger.info("Начало удаления фона", extra={'operation': 'REMOVE_BG_START'})
            
            # Загружаем изображение
            image = Image.open(io.BytesIO(image_data))
            original_size = image.size
            
            # Конвертируем в RGB, если нужно
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Изменяем размер, если нужно
            image = cls._resize_if_needed(image)
            
            # Удаляем фон
            model = cls._get_model()
            image_without_bg = model(image)
            
            # Восстанавливаем исходный размер
            if image.size != original_size:
                image_without_bg = cls._restore_size(image_without_bg, original_size)
            
            # Сохраняем результат в bytes
            output = io.BytesIO()
            image_without_bg.save(output, format='PNG')
            result = output.getvalue()
            
            logger.info("Фон успешно удален", extra={'operation': 'REMOVE_BG_SUCCESS'})
            return result
            
        except Exception as e:
            logger.error(f"Ошибка при удалении фона: {str(e)}", 
                        extra={'operation': 'REMOVE_BG_ERROR'})
            raise

from aiogram.filters.callback_data import CallbackData as BaseCallbackData

class StyleCallback(BaseCallbackData, prefix="style"):
    style: str

@router.message(Command("start"))
async def send_welcome(message: types.Message):
    """Обработчик команды /start"""
    try:
        await message.answer(
            MessageTemplate.get(MessageKey.WELCOME),
            reply_markup=get_main_keyboard(message.from_user.id),
            parse_mode=ParseMode.HTML
        )
    except Exception as e:
        logger.error(f"Ошибка в send_welcome: {str(e)}", extra={
            'user_id': message.from_user.id,
            'operation': 'WELCOME'
        })
        await message.answer(MessageTemplate.get(MessageKey.ERROR_CRITICAL))

@router.callback_query(F.data == CallbackEnum.HELP)
async def show_help(callback_query: CallbackQuery):
    """Обработчик кнопки помощи"""
    try:
        user_id = callback_query.from_user.id
        logger.info("Показываем справку", extra={
            'user_id': user_id,
            'operation': 'SHOW_HELP'
        })
        
        if callback_query.message.photo:
            await callback_query.message.edit_caption(
                caption=MessageTemplate.get(MessageKey.HELP),
                reply_markup=get_back_keyboard(user_id),
                parse_mode=ParseMode.HTML
            )
        else:
            await callback_query.message.edit_text(
                text=MessageTemplate.get(MessageKey.HELP),
                reply_markup=get_back_keyboard(user_id),
                parse_mode=ParseMode.HTML
            )
        await callback_query.answer()
        
    except Exception as e:
        logger.error(f"Ошибка при показе справки: {str(e)}", extra={
            'user_id': user_id,
            'operation': 'HELP_ERROR'
        })
        await callback_query.answer(MessageTemplate.get(MessageKey.ERROR_CRITICAL), show_alert=True)

@router.callback_query(F.data == CallbackEnum.SETTINGS)
async def show_settings(callback_query: CallbackQuery):
    """Обработчик кнопки настроек"""
    try:
        user_id = callback_query.from_user.id
        settings = user_settings[user_id]
        
        if callback_query.message.photo:
            await callback_query.message.edit_caption(
                caption=MessageTemplate.get(
                    MessageKey.SETTINGS,
                    size=f"{settings.width}x{settings.height}",
                    style=IMAGE_STYLES[settings.style]['label']
                ),
                reply_markup=get_settings_keyboard(user_id)
            )
        else:
            await callback_query.message.edit_text(
                text=MessageTemplate.get(
                    MessageKey.SETTINGS,
                    size=f"{settings.width}x{settings.height}",
                    style=IMAGE_STYLES[settings.style]['label']
                ),
                reply_markup=get_settings_keyboard(user_id)
            )
        await callback_query.answer()
    except Exception as e:
        logger.error(f"Ошибка в show_settings: {str(e)}", extra={
            'user_id': callback_query.from_user.id,
            'operation': 'SETTINGS'
        })
        await callback_query.answer(MessageTemplate.get(MessageKey.ERROR_CRITICAL), show_alert=True)

@router.callback_query(F.data.startswith(CallbackEnum.SIZE_PREFIX))
async def process_size_change(callback_query: CallbackQuery):
    """Обработчик изменения размера изображения"""
    user_id = callback_query.from_user.id
    size_key = callback_query.data.replace(CallbackEnum.SIZE_PREFIX, "")
    
    try:
        # Получаем размеры из словаря
        if size_key not in IMAGE_SIZES:
            logger.error("Неверный размер", extra={
                'user_id': user_id,
                'operation': 'INVALID_SIZE',
                'size_key': size_key
            })
            await callback_query.answer(
                MessageTemplate.get(MessageKey.ERROR_SIZE),
                show_alert=True
            )
            return
            
        # Обновляем настройки пользователя
        user_settings[user_id].width = IMAGE_SIZES[size_key]['width']
        user_settings[user_id].height = IMAGE_SIZES[size_key]['height']
        
        logger.info("Размер изменен", extra={
            'user_id': user_id,
            'operation': 'SIZE_CHANGED',
            'size': f"{IMAGE_SIZES[size_key]['width']}x{IMAGE_SIZES[size_key]['height']}"
        })
        
        # Отправляем подтверждение
        await callback_query.message.edit_text(
            MessageTemplate.get(
                MessageKey.SIZE_CHANGED,
                size=IMAGE_SIZES[size_key]['label']
            ),
            reply_markup=get_back_keyboard(user_id),
            parse_mode=ParseMode.HTML
        )
        
    except Exception as e:
        logger.error(f"Ошибка при изменении размера: {str(e)}", extra={
            'user_id': user_id,
            'operation': 'SIZE_CHANGE_ERROR'
        })
        await callback_query.answer(
            MessageTemplate.get(MessageKey.ERROR_CRITICAL),
            show_alert=True
        )

@router.callback_query(F.data == CallbackEnum.REMOVE_BG)
async def process_remove_background(callback_query: CallbackQuery):
    """Обработчик удаления фона с изображения"""
    try:
        user_id = callback_query.from_user.id
        user_state = user_states[user_id]

        if not user_state.last_image:
            await callback_query.answer("Нет доступного изображения для обработки")
            return

        # Отправляем сообщение о начале обработки
        status_message = await callback_query.message.answer(
            MessageTemplate.get(MessageKey.REMOVING_BG),
            reply_markup=get_back_keyboard(user_id),
            parse_mode=ParseMode.HTML
        )

        logger.info("Начало удаления фона", extra={
            'user_id': user_id,
            'operation': 'REMOVE_BG_START'
        })

        try:
            # Засекаем время начала обработки
            start_time = datetime.now()
            
            # Удаляем фон
            image_without_bg = await ImageProcessor.remove_background(user_state.last_image)
            
            # Вычисляем время обработки
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Создаем объект с информацией об изображении
            image_info = ImageInfo(
                id=user_state.last_image_id,
                prompt=user_state.last_prompt,
                style=user_settings[user_id].style,
                style_prompt=IMAGE_STYLES[user_settings[user_id].style]['prompt_prefix'],
                width=user_settings[user_id].width,
                height=user_settings[user_id].height,
                model_id=IMAGE_STYLES[user_settings[user_id].style].get('model_id', 1),
                created_at=datetime.now(),
                generation_time=0,  # Для удаления фона не учитываем время генерации
                user_id=user_id,
                has_removed_bg=True,
                bg_removal_time=processing_time
            )

            # Формируем сообщение с информацией
            message_text = MessageTemplate.get_image_info(image_info)

            # Отправляем обработанное изображение
            await callback_query.message.answer_photo(
                BufferedInputFile(
                    image_without_bg,
                    filename=f"nobg_{user_state.last_image_id}.png"
                ),
                caption=message_text,
                reply_markup=get_image_keyboard(user_state.last_image_id, user_id),
                parse_mode=ParseMode.HTML
            )

            # Удаляем сообщение о процессе
            await status_message.delete()

            logger.info("Фон успешно удален", extra={
                'user_id': user_id,
                'operation': 'REMOVE_BG_SUCCESS',
                'processing_time': processing_time
            })

        except Exception as e:
            logger.error(f"Ошибка при удалении фона: {str(e)}", extra={
                'user_id': user_id,
                'operation': 'REMOVE_BG_ERROR',
                'error': str(e)
            })
            
            await status_message.edit_text(
                MessageTemplate.get(
                    MessageKey.REMOVE_BG_ERROR,
                    error="Не удалось обработать изображение"
                ),
                reply_markup=get_back_keyboard(user_id),
                parse_mode=ParseMode.HTML
            )

    except Exception as e:
        logger.error(f"Критическая ошибка при удалении фона: {str(e)}", extra={
            'user_id': user_id,
            'operation': 'REMOVE_BG_CRITICAL_ERROR',
            'error': str(e)
        })
        await callback_query.message.answer(
            MessageTemplate.get(MessageKey.ERROR_CRITICAL),
            reply_markup=get_back_keyboard(user_id),
            parse_mode=ParseMode.HTML
        )

@router.callback_query(F.data == CallbackEnum.GENERATE)
async def start_generation(callback_query: CallbackQuery):
    """Начинает процесс генерации изображения"""
    try:
        user_id = callback_query.from_user.id
        logger.info(f"Пользователь начал процесс генерации", extra={
            'user_id': user_id,
            'operation': 'SYSTEM'
        })

        user_state = user_states[user_id]
        user_setting = user_settings[user_id]
        
        # Получаем текущий стиль
        style_info = IMAGE_STYLES[user_setting.style]
        current_size = f"{user_setting.width}x{user_setting.height}"
        
        # Устанавливаем флаг ожидания промпта
        user_state.awaiting_prompt = True
        
        # Отправляем сообщение с текущими настройками
        if callback_query.message.photo:
            await callback_query.message.answer(
                text=MessageTemplate.get(
                    MessageKey.CURRENT_SETTINGS,
                    style=style_info['label'],
                    size=current_size
                ),
                reply_markup=get_prompt_keyboard(user_id)
            )
        else:
            await callback_query.message.edit_text(
                text=MessageTemplate.get(
                    MessageKey.CURRENT_SETTINGS,
                    style=style_info['label'],
                    size=current_size
                ),
                reply_markup=get_prompt_keyboard(user_id)
            )
        
        await callback_query.answer()
        
    except Exception as e:
        user_id = callback_query.from_user.id if callback_query.from_user else "N/A"
        logger.error(f"Ошибка при начале генерации: {str(e)}", extra={
            'user_id': user_id,
            'operation': 'ERROR'
        })
        await callback_query.message.edit_text(
            text=MessageTemplate.get(MessageKey.ERROR_CRITICAL),
            reply_markup=get_back_keyboard(user_id)
        )

@router.callback_query(F.data == CallbackEnum.BACK)
async def back_to_main(callback_query: CallbackQuery):
    """Возврат в главное меню"""
    user_id = callback_query.from_user.id
    try:
        # Если сообщение содержит фото, отправляем новое
        if callback_query.message.photo:
            await callback_query.message.answer(
                "Выберите действие:",
                reply_markup=get_main_keyboard(user_id)
            )
        else:
            # Иначе редактируем текущее
            await callback_query.message.edit_text(
                "Выберите действие:",
                reply_markup=get_main_keyboard(user_id)
            )
    except Exception as e:
        logger.error(f"Ошибка при возврате в главное меню: {str(e)}", extra={
            'user_id': user_id,
            'operation': 'BACK_TO_MAIN_ERROR'
        })
        await callback_query.answer(MessageTemplate.get(MessageKey.ERROR_CRITICAL), show_alert=True)

@router.callback_query(F.data == CallbackEnum.STYLES)
async def show_styles(callback_query: CallbackQuery):
    """Показывает меню выбора стиля"""
    user_id = callback_query.from_user.id
    settings = user_settings[user_id]
    current_style = IMAGE_STYLES[settings.style]["label"]
    
    # Создаем текст с описанием текущего стиля
    current_style_data = IMAGE_STYLES[settings.style]
    style_info = (
        f"🎨 <b>Текущий стиль:</b> {current_style}\n"
        f"📝 {current_style_data['description']}\n"
        f"💡 {current_style_data['example']}\n\n"
        f"Выберите новый стиль:"
    )
    
    try:
        # Если сообщение содержит фото, отправляем новое сообщение
        if callback_query.message.photo:
            await callback_query.message.answer(
                style_info,
                reply_markup=get_styles_keyboard(user_id),
                parse_mode=ParseMode.HTML
            )
        else:
            # Иначе редактируем текущее сообщение
            await callback_query.message.edit_text(
                style_info,
                reply_markup=get_styles_keyboard(user_id),
                parse_mode=ParseMode.HTML
            )
    except TelegramBadRequest as e:
        if "message is not modified" not in str(e).lower():
            logger.error(f"Error editing message: {e}", extra={
                'user_id': user_id,
                'operation': 'show_styles'
            })
            await callback_query.answer("Произошла ошибка при обновлении стилей")
    except Exception as e:
        logger.error(f"Error in show_styles: {e}", extra={
            'user_id': user_id,
            'operation': 'show_styles'
        })
        await callback_query.answer("Произошла ошибка при показе стилей")

@router.callback_query(F.data.startswith(CallbackEnum.STYLE_PREFIX))
async def process_style_change(callback_query: CallbackQuery):
    """Обработчик изменения стиля изображения"""
    user_id = callback_query.from_user.id
    
    try:
        # Получаем ключ стиля из callback data
        style_key = callback_query.data.replace(f"{CallbackEnum.STYLE_PREFIX}", "")
        
        if style_key not in IMAGE_STYLES:
            logger.error(f"Неверный ключ стиля: {style_key}", extra={
                'user_id': user_id,
                'operation': 'style_change'
            })
            await callback_query.answer("Ошибка: неверный стиль")
            return
        
        # Обновляем стиль в настройках пользователя
        user_settings[user_id].style = style_key
        style_data = IMAGE_STYLES[style_key]
        
        # Формируем сообщение с информацией о новом стиле
        style_info = (
            f"✅ <b>Стиль успешно изменен!</b>\n\n"
            f"🎨 <b>Новый стиль:</b> {style_data['label']}\n"
            f"📝 {style_data['description']}\n"
            f"💡 {style_data['example']}\n\n"
            f"Теперь вы можете создать изображение в этом стиле!"
        )
        
        try:
            if callback_query.message.photo:
                await callback_query.message.answer(
                    style_info,
                    reply_markup=get_main_keyboard(user_id),
                    parse_mode=ParseMode.HTML
                )
            else:
                await callback_query.message.edit_text(
                    style_info,
                    reply_markup=get_main_keyboard(user_id),
                    parse_mode=ParseMode.HTML
                )
        except TelegramBadRequest as e:
            if "message is not modified" not in str(e).lower():
                logger.error(f"Error editing message: {e}", extra={
                    'user_id': user_id,
                    'operation': 'style_change'
                })
                await callback_query.answer("Произошла ошибка при обновлении сообщения")
                return
        
        await callback_query.answer(f"Выбран стиль: {style_data['label']}")
        
    except Exception as e:
        logger.error(f"Error in process_style_change: {e}", extra={
            'user_id': user_id,
            'operation': 'style_change'
        })
        await callback_query.answer("Произошла ошибка при изменении стиля")

@router.callback_query(F.data == CallbackEnum.REGENERATE)
async def regenerate_image(callback_query: CallbackQuery):
    """Обработчик повторной генерации изображения"""
    user_id = callback_query.from_user.id
    user_state = user_states[user_id]
    
    try:
        if not user_state.last_prompt:
            logger.warning("Попытка регенерации без сохраненного промпта", extra={
                'user_id': user_id,
                'operation': 'REGENERATION_NO_PROMPT'
            })
            await callback_query.answer("Нет сохранённого промпта для повторной генерации", show_alert=True)
            return

        if not callback_query.message:
            logger.error("Отсутствует сообщение для регенерации", extra={
                'user_id': user_id,
                'operation': 'REGENERATION_NO_MESSAGE'
            })
            await callback_query.answer("Ошибка: невозможно выполнить регенерацию", show_alert=True)
            return

        logger.info("Запуск повторной генерации", extra={
            'user_id': user_id,
            'operation': 'REGENERATION_START',
            'prompt': user_state.last_prompt
        })

        # Отправляем сообщение о начале генерации
        status_message = await callback_query.message.answer(
            MessageTemplate.get(MessageKey.GENERATING, style=IMAGE_STYLES[user_settings[user_id].style]['label']),
            reply_markup=get_back_keyboard(user_id),
            parse_mode=ParseMode.HTML
        )

        # Проверяем наличие и валидность ключей API
        api_key = os.getenv('FUSIONBRAIN_API_KEY')
        secret_key = os.getenv('FUSIONBRAIN_SECRET_KEY')

        if not api_key or not secret_key:
            logger.error("Отсутствуют ключи API", extra={
                'user_id': user_id,
                'operation': 'MISSING_API_KEYS'
            })
            await status_message.edit_text(
                "⚠️ Ошибка конфигурации: отсутствуют ключи API. Обратитесь к администратору.",
                reply_markup=get_back_keyboard(user_id)
            )
            return

        # Создаем экземпляр API
        api = Text2ImageAPI(api_key, secret_key)
        
        # Получаем настройки пользователя
        user_settings_data = user_settings[user_id]
        width = user_settings_data.width
        height = user_settings_data.height
        style = user_settings_data.style
        
        try:
            # Получаем доступные модели
            models = await api.get_model()
            if not models:
                raise Exception("Список моделей пуст")
            model_id = models[0]["id"]
            
            logger.info("Получена модель", extra={
                'user_id': user_id,
                'operation': 'MODEL_INFO',
                'model_id': model_id
            })
            
            # Получаем стиль и добавляем префикс к промпту
            style_data = IMAGE_STYLES[style]
            styled_prompt = f"{style_data['prompt_prefix']}{user_state.last_prompt}"
            
            # Запускаем генерацию
            uuid = await api.generate(styled_prompt, model_id, width, height)
            
            # Проверяем статус генерации
            await check_generation_status(api, uuid, status_message, user_id)

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Ошибка при генерации: {str(e)}", extra={
                'user_id': user_id,
                'operation': 'GENERATION_ERROR',
                'error': str(e)
            })
            
            # Преобразуем технические ошибки в понятные пользователю сообщения
            user_message = str(e)
            if "Generation still in progress" in str(e):
                user_message = "Генерация все еще выполняется. Пожалуйста, подождите."
            elif "Превышено время ожидания" in str(e):
                user_message = "Генерация заняла слишком много времени. Попробуйте еще раз."
            elif "авторизации" in str(e).lower():
                user_message = "Ошибка доступа к сервису. Обратитесь к администратору."
            elif "модели" in str(e).lower():
                user_message = "Сервис временно недоступен. Попробуйте позже."
            elif "Изображение не было сгенерировано" in str(e):
                user_message = "Не удалось сгенерировать изображение. Попробуйте другой промпт или стиль."
            
            await status_message.edit_text(
                MessageTemplate.get(MessageKey.ERROR_GEN, error=user_message),
                reply_markup=get_back_keyboard(user_id),
                parse_mode=ParseMode.HTML
            )
            
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Ошибка при запуске регенерации: {error_msg}", extra={
            'user_id': user_id,
            'operation': 'REGENERATION_ERROR',
            'error': error_msg
        })
        await callback_query.answer(
            "Произошла ошибка при запуске регенерации. Попробуйте позже.", 
            show_alert=True
        )

@router.message(F.text)
async def handle_text(message: types.Message):
    """Обработчик текстовых сообщений для генерации изображений"""
    user_id = message.from_user.id
    
    logger.info("Получено текстовое сообщение", extra={
        'user_id': user_id,
        'operation': 'TEXT_RECEIVED',
        'text': message.text
    })
    
    # Проверяем состояние ожидания промпта
    user_state = user_states[user_id]
    if not user_state.awaiting_prompt:
        logger.warning("Получен текст без ожидания промпта", extra={
            'user_id': user_id,
            'operation': 'UNEXPECTED_TEXT',
            'text': message.text,
            'awaiting_prompt': user_state.awaiting_prompt
        })
        await message.answer(
            "Для генерации изображения нажмите кнопку 'Создать' и введите описание изображения.",
            reply_markup=get_main_keyboard(user_id)
        )
        return

    logger.info("Начало обработки промпта", extra={
        'user_id': user_id,
        'operation': 'PROMPT_PROCESSING',
        'prompt': message.text
    })

    # Сбрасываем флаг ожидания промпта
    user_state.awaiting_prompt = False

    # Проверяем наличие ключей API
    if not all([FUSIONBRAIN_API_KEY, FUSIONBRAIN_SECRET_KEY]):
        logger.error("Отсутствуют ключи API", extra={
            'user_id': user_id,
            'operation': 'MISSING_API_KEYS'
        })
        await message.answer(
            "⚠️ Ошибка конфигурации: отсутствуют ключи API. Обратитесь к администратору.",
            reply_markup=get_back_keyboard(user_id)
        )
        return

    # Проверяем длину промпта
    prompt = message.text
    if len(prompt) > Text2ImageAPI.MAX_PROMPT_LENGTH:
        logger.warning(f"Промпт превышает максимальную длину: {len(prompt)}", extra={
            'user_id': user_id,
            'operation': 'PROMPT_TOO_LONG',
            'prompt_length': len(prompt)
        })
        prompt = prompt[:Text2ImageAPI.MAX_PROMPT_LENGTH]
        await message.answer(
            f"⚠️ Ваш промпт слишком длинный и был сокращен до {Text2ImageAPI.MAX_PROMPT_LENGTH} символов.",
            reply_markup=None
        )

    # Сохраняем промпт
    user_state = user_states[user_id]
    user_state.last_prompt = prompt

    # Отправляем сообщение о начале генерации
    status_message = await message.answer(
        MessageTemplate.get(MessageKey.GENERATING, style=IMAGE_STYLES[user_settings[user_id].style]['label']),
        reply_markup=get_back_keyboard(user_id),
        parse_mode=ParseMode.HTML
    )

    logger.info("Начало генерации изображения", extra={
        'user_id': user_id,
        'operation': 'GENERATION_START',
        'prompt': prompt
    })

    try:
        # Инициализируем API и запускаем генерацию
        api = Text2ImageAPI(FUSIONBRAIN_API_KEY, FUSIONBRAIN_SECRET_KEY)
        
        # Получаем настройки пользователя
        user_settings_data = user_settings[user_id]
        width = user_settings_data.width
        height = user_settings_data.height
        style = user_settings_data.style
        
        try:
            # Получаем доступные модели
            models = await api.get_model()
            if not models:
                raise Exception("Список моделей пуст")
            model_id = models[0]["id"]
            
            logger.info("Получена модель", extra={
                'user_id': user_id,
                'operation': 'MODEL_INFO',
                'model_id': model_id
            })
            
            # Получаем стиль и добавляем префикс к промпту
            style_data = IMAGE_STYLES[style]
            styled_prompt = f"{style_data['prompt_prefix']}{prompt}"
            
            logger.info("Подготовленный промпт", extra={
                'user_id': user_id,
                'operation': 'STYLED_PROMPT',
                'original_prompt': prompt,
                'styled_prompt': styled_prompt
            })
            
            # Запускаем генерацию
            uuid = await api.generate(styled_prompt, model_id, width, height)
            
            # Проверяем статус генерации
            await check_generation_status(api, uuid, status_message, user_id)

        except Exception as e:
            logger.error(f"Ошибка при генерации: {str(e)}", extra={
                'user_id': user_id,
                'operation': 'GENERATION_ERROR',
                'error': str(e)
            })
            
            # Преобразуем технические ошибки в понятные пользователю сообщения
            user_message = str(e)
            if "Generation still in progress" in str(e):
                user_message = "Генерация все еще выполняется. Пожалуйста, подождите."
            elif "Превышено время ожидания" in str(e):
                user_message = "Генерация заняла слишком много времени. Попробуйте еще раз."
            elif "авторизации" in str(e).lower():
                user_message = "Ошибка доступа к сервису. Обратитесь к администратору."
            elif "модели" in str(e).lower():
                user_message = "Сервис временно недоступен. Попробуйте позже."
            elif "Изображение не было сгенерировано" in str(e):
                user_message = "Не удалось сгенерировать изображение. Попробуйте другой промпт или стиль."
            
            await status_message.edit_text(
                MessageTemplate.get(MessageKey.ERROR_GEN, error=user_message),
                reply_markup=get_back_keyboard(user_id),
                parse_mode=ParseMode.HTML
            )
            
    except Exception as e:
        logger.error(f"Критическая ошибка в generate_image: {str(e)}", extra={
            'user_id': user_id if 'user_id' in locals() else 'N/A',
            'operation': 'CRITICAL_ERROR',
            'error': str(e)
        })
        if 'status_message' in locals():
            await status_message.edit_text(
                MessageTemplate.get(MessageKey.ERROR_CRITICAL),
                reply_markup=get_back_keyboard(user_id),
                parse_mode=ParseMode.HTML
            )
        else:
            await message.answer(
                MessageTemplate.get(MessageKey.ERROR_CRITICAL),
                reply_markup=get_back_keyboard(user_id),
                parse_mode=ParseMode.HTML
            )

async def generate_image_with_prompt(message: types.Message, prompt: str):
    user_id = message.from_user.id
    user_state = user_states[user_id]
    
    # Сохраняем промпт перед генерацией
    user_state.last_prompt = prompt
    
    try:
        # Отправляем сообщение о начале генерации
        status_message = await message.answer(
            MessageTemplate.get(MessageKey.GENERATING, style=IMAGE_STYLES[user_settings[user_id].style]['label']),
            reply_markup=get_back_keyboard(user_id),
            parse_mode=ParseMode.HTML
        )

        # Проверяем наличие и валидность ключей API
        api_key = os.getenv('FUSIONBRAIN_API_KEY')
        secret_key = os.getenv('FUSIONBRAIN_SECRET_KEY')

        # Проверяем наличие ключей API
        if not all([api_key, secret_key]):
            logger.error("Отсутствуют ключи API", extra={
                'user_id': user_id,
                'operation': 'MISSING_API_KEYS'
            })
            await message.answer(
                MessageTemplate.get(MessageKey.ERROR, error_message="Отсутствуют ключи API. Обратитесь к администратору."),
                reply_markup=get_back_keyboard(user_id),
                parse_mode=ParseMode.HTML
            )
            return

        # Инициализируем API и запускаем генерацию
        api = Text2ImageAPI(api_key, secret_key)
        
        width = user_settings[user_id].width
        height = user_settings[user_id].height
        style = user_settings[user_id].style
        
        logger.info(f"Параметры генерации", extra={
            'user_id': user_id,
            'operation': 'GENERATION_PARAMS',
            'width': width,
            'height': height,
            'style': style
        })

        # Получаем модель
        try:
            models = await api.get_model()
            if not models:
                raise Exception("Список моделей пуст")
            model_id = models[0]["id"]
            
            logger.info(f"Получена модель", extra={
                'user_id': user_id,
                'operation': 'MODEL_INFO',
                'model_id': model_id
            })
            
        except Exception as e:
            logger.error(f"Ошибка при получении модели: {str(e)}", extra={
                'user_id': user_id,
                'operation': 'MODEL_ERROR'
            })
            raise Exception("Не удалось получить доступ к модели генерации. Попробуйте позже.")
        
        # Формируем промпт с учетом стиля
        styled_prompt = f"{prompt}, {IMAGE_STYLES[style]['prompt_prefix']}" if style != StyleType.DEFAULT.name else prompt
        
        # Запускаем генерацию
        start_time = datetime.now()  # Засекаем время начала генерации
        uuid = await api.generate(styled_prompt, model_id, width, height)
        
        # Проверяем статус генерации
        await check_generation_status(api, uuid, status_message, user_id, start_time)

    except Exception as e:
        logger.error(f"Ошибка при генерации: {str(e)}", extra={
            'user_id': user_id,
            'operation': 'GENERATION_ERROR'
        })
        
        # Преобразуем технические ошибки в понятные пользователю сообщения
        user_message = str(e)
        if "Generation still in progress" in str(e):
            user_message = "Генерация все еще выполняется. Пожалуйста, подождите."
        elif "Превышено время ожидания" in str(e):
            user_message = "Генерация заняла слишком много времени. Попробуйте еще раз."
        elif "авторизации" in str(e).lower():
            user_message = "Ошибка доступа к сервису. Обратитесь к администратору."
        elif "модели" in str(e).lower():
            user_message = "Сервис временно недоступен. Попробуйте позже."
        elif "Изображение не было сгенерировано" in str(e):
            user_message = "Не удалось сгенерировать изображение. Попробуйте другой промпт или стиль."
        
        await status_message.edit_text(
            MessageTemplate.get(MessageKey.ERROR_GEN, error=user_message),
            reply_markup=get_back_keyboard(user_id),
            parse_mode=ParseMode.HTML
        )
        
async def generate_image(message: types.Message):
    """Генерирует изображение на основе промпта"""
    try:
        user_id = message.from_user.id
        logger.info("Получен запрос на генерацию изображения", extra={
            'user_id': user_id,
            'operation': 'IMAGE_GENERATION_START',
            'prompt': message.text
        })
        
        if not message.text:
            logger.warning("Получено не текстовое сообщение", extra={
                'user_id': user_id,
                'operation': 'INVALID_MESSAGE_TYPE'
            })
            return

        if not user_states[user_id].awaiting_prompt:
            logger.warning("Получен промпт без ожидания", extra={
                'user_id': user_id,
                'operation': 'UNEXPECTED_PROMPT',
                'prompt': message.text,
                'awaiting_prompt': user_states[user_id].awaiting_prompt
            })
            await message.answer(
                "Для генерации изображения нажмите кнопку 'Создать' и введите описание изображения.",
                reply_markup=get_main_keyboard(user_id)
            )
            return

        logger.info("Получен промпт для генерации", extra={
            'user_id': user_id,
            'operation': 'PROMPT_RECEIVED',
            'prompt': message.text
        })

        # Сбрасываем флаг ожидания промпта
        user_states[user_id].awaiting_prompt = False

        # Проверяем наличие ключей API
        if not all([FUSIONBRAIN_API_KEY, FUSIONBRAIN_SECRET_KEY]):
            logger.error("Отсутствуют ключи API", extra={
                'user_id': user_id,
                'operation': 'MISSING_API_KEYS'
            })
            await message.answer(
                "⚠️ Ошибка конфигурации: отсутствуют ключи API. Обратитесь к администратору.",
                reply_markup=get_back_keyboard(user_id)
            )
            return

        # Проверяем длину промпта
        prompt = message.text
        if len(prompt) > Text2ImageAPI.MAX_PROMPT_LENGTH:
            logger.warning(f"Промпт превышает максимальную длину: {len(prompt)}", extra={
                'user_id': user_id,
                'operation': 'PROMPT_TOO_LONG',
                'prompt_length': len(prompt)
            })
            prompt = prompt[:Text2ImageAPI.MAX_PROMPT_LENGTH]
            await message.answer(
                f"⚠️ Ваш промпт слишком длинный и был сокращен до {Text2ImageAPI.MAX_PROMPT_LENGTH} символов.",
                reply_markup=None
            )

        # Сохраняем промпт
        user_state = user_states[user_id]
        user_state.last_prompt = prompt

        # Отправляем сообщение о начале генерации
        status_message = await message.answer(
            MessageTemplate.get(MessageKey.GENERATING, style=IMAGE_STYLES[user_settings[user_id].style]['label']),
            reply_markup=get_back_keyboard(user_id),
            parse_mode=ParseMode.HTML
        )

        logger.info("Начало генерации изображения", extra={
            'user_id': user_id,
            'operation': 'GENERATION_START',
            'prompt': prompt
        })

        try:
            # Инициализируем API и запускаем генерацию
            api = Text2ImageAPI(FUSIONBRAIN_API_KEY, FUSIONBRAIN_SECRET_KEY)
            
            # Получаем настройки пользователя
            user_settings_data = user_settings[user_id]
            width = user_settings_data.width
            height = user_settings_data.height
            style = user_settings_data.style
            
            logger.info("Параметры генерации", extra={
                'user_id': user_id,
                'operation': 'GENERATION_PARAMS',
                'width': width,
                'height': height,
                'style': style
            })

            # Получаем модель
            try:
                models = await api.get_model()
                if not models:
                    raise Exception("Список моделей пуст")
                model_id = models[0]["id"]
                
                logger.info("Получена модель", extra={
                    'user_id': user_id,
                    'operation': 'MODEL_INFO',
                    'model_id': model_id
                })
                
            except Exception as e:
                logger.error(f"Ошибка при получении модели: {str(e)}", extra={
                    'user_id': user_id,
                    'operation': 'MODEL_ERROR'
                })
                raise Exception("Не удалось получить доступ к модели генерации. Попробуйте позже.")
            
            # Формируем промпт с учетом стиля
            style_data = IMAGE_STYLES[style]
            styled_prompt = f"{style_data['prompt_prefix']}{prompt}"
            
            logger.info("Подготовленный промпт", extra={
                'user_id': user_id,
                'operation': 'STYLED_PROMPT',
                'original_prompt': prompt,
                'styled_prompt': styled_prompt
            })
            
            # Запускаем генерацию
            uuid = await api.generate(styled_prompt, model_id, width, height)
            
            # Проверяем статус генерации
            await check_generation_status(api, uuid, status_message, user_id)

        except Exception as e:
            logger.error(f"Ошибка при генерации: {str(e)}", extra={
                'user_id': user_id,
                'operation': 'GENERATION_ERROR',
                'error': str(e)
            })
            
            # Преобразуем технические ошибки в понятные пользователю сообщения
            user_message = str(e)
            if "Generation still in progress" in str(e):
                user_message = "Генерация все еще выполняется. Пожалуйста, подождите."
            elif "Превышено время ожидания" in str(e):
                user_message = "Генерация заняла слишком много времени. Попробуйте еще раз."
            elif "авторизации" in str(e).lower():
                user_message = "Ошибка доступа к сервису. Обратитесь к администратору."
            elif "модели" in str(e).lower():
                user_message = "Сервис временно недоступен. Попробуйте позже."
            elif "Изображение не было сгенерировано" in str(e):
                user_message = "Не удалось сгенерировать изображение. Попробуйте другой промпт или стиль."
            
            await status_message.edit_text(
                MessageTemplate.get(MessageKey.ERROR_GEN, error=user_message),
                reply_markup=get_back_keyboard(user_id),
                parse_mode=ParseMode.HTML
            )
            
    except Exception as e:
        logger.error(f"Критическая ошибка в generate_image: {str(e)}", extra={
            'user_id': user_id if 'user_id' in locals() else 'N/A',
            'operation': 'CRITICAL_ERROR',
            'error': str(e)
        })
        if 'status_message' in locals():
            await status_message.edit_text(
                MessageTemplate.get(MessageKey.ERROR_CRITICAL),
                reply_markup=get_back_keyboard(user_id),
                parse_mode=ParseMode.HTML
            )
        else:
            await message.answer(
                MessageTemplate.get(MessageKey.ERROR_CRITICAL),
                reply_markup=get_back_keyboard(user_id),
                parse_mode=ParseMode.HTML
            )

async def check_generation_status(api, uuid, status_message, user_id, start_time=None):
    """Проверяет статус генерации изображения"""
    try:
        max_attempts = 60  # Максимальное количество попыток
        attempt = 0
        
        while attempt < max_attempts:
            try:
                # Проверяем статус генерации
                response = await api.check_generation(uuid)
                
                logger.info("Получен ответ от API", extra={
                    'user_id': user_id,
                    'operation': 'API_RESPONSE',
                    'response': str(response)
                })
                
                if isinstance(response, list) and response:
                    # Если ответ - список с изображением
                    logger.info("Изображение успешно сгенерировано", extra={
                        'user_id': user_id,
                        'operation': 'GENERATION_SUCCESS'
                    })
                    
                    # Сохраняем изображение
                    image_data = base64.b64decode(response[0])
                    
                    # Создаем объект с информацией об изображении
                    generation_time = (datetime.now() - start_time).total_seconds() if start_time else 0
                    image_info = ImageInfo(
                        id=uuid,
                        prompt=user_states[user_id].last_prompt,
                        style=user_settings[user_id].style,
                        style_prompt=IMAGE_STYLES[user_settings[user_id].style]['prompt_prefix'],
                        width=user_settings[user_id].width,
                        height=user_settings[user_id].height,
                        model_id=IMAGE_STYLES[user_settings[user_id].style].get('model_id', 1),
                        created_at=datetime.now(),
                        generation_time=generation_time,
                        user_id=user_id
                    )
                    
                    # Отправляем изображение пользователю с полной информацией
                    message_text = MessageTemplate.get_image_info(image_info)
                    
                    if status_message.photo:
                        await status_message.answer_photo(
                            BufferedInputFile(
                                image_data,
                                filename=f"generation_{uuid}.png"
                            ),
                            caption=message_text,
                            reply_markup=get_image_keyboard(uuid, user_id),
                            parse_mode=ParseMode.HTML
                        )
                    else:
                        await status_message.answer_photo(
                            BufferedInputFile(
                                image_data,
                                filename=f"generation_{uuid}.png"
                            ),
                            caption=message_text,
                            reply_markup=get_image_keyboard(uuid, user_id),
                            parse_mode=ParseMode.HTML
                        )
                    
                    # Сохраняем информацию о последнем изображении
                    user_states[user_id].last_image = image_data
                    user_states[user_id].last_image_id = uuid
                    
                    return True
                    
                elif isinstance(response, dict):
                    # Если ответ - словарь со статусом
                    status = response.get('status')
                    
                    if status == "DONE":
                        images = response.get('images')
                        if not images:
                            raise Exception("Изображение не было сгенерировано")
                            
                        logger.info("Изображение успешно сгенерировано", extra={
                            'user_id': user_id,
                            'operation': 'GENERATION_SUCCESS'
                        })
                        
                        # Сохраняем изображение
                        image_data = base64.b64decode(images[0])
                        
                        # Создаем объект с информацией об изображении
                        generation_time = (datetime.now() - start_time).total_seconds() if start_time else 0
                        image_info = ImageInfo(
                            id=uuid,
                            prompt=user_states[user_id].last_prompt,
                            style=user_settings[user_id].style,
                            style_prompt=IMAGE_STYLES[user_settings[user_id].style]['prompt_prefix'],
                            width=user_settings[user_id].width,
                            height=user_settings[user_id].height,
                            model_id=IMAGE_STYLES[user_settings[user_id].style].get('model_id', 1),
                            created_at=datetime.now(),
                            generation_time=generation_time,
                            user_id=user_id
                        )
                        
                        # Отправляем изображение пользователю с полной информацией
                        message_text = MessageTemplate.get_image_info(image_info)
                        
                        if status_message.photo:
                            await status_message.answer_photo(
                                BufferedInputFile(
                                    image_data,
                                    filename=f"generation_{uuid}.png"
                                ),
                                caption=message_text,
                                reply_markup=get_image_keyboard(uuid, user_id),
                                parse_mode=ParseMode.HTML
                            )
                        else:
                            await status_message.answer_photo(
                                BufferedInputFile(
                                    image_data,
                                    filename=f"generation_{uuid}.png"
                                ),
                                caption=message_text,
                                reply_markup=get_image_keyboard(uuid, user_id),
                                parse_mode=ParseMode.HTML
                            )
                        
                        # Сохраняем информацию о последнем изображении
                        user_states[user_id].last_image = image_data
                        user_states[user_id].last_image_id = uuid
                        
                        return True
                        
                    elif status in ["INITIAL", "PROCESSING"]:
                        logger.info("Генерация все еще выполняется", extra={
                            'user_id': user_id,
                            'operation': 'GENERATION_IN_PROGRESS',
                            'uuid': uuid,
                            'status': status
                        })
                        raise Exception("Generation still in progress")
                        
                    elif status == "FAIL":
                        error = response.get("error", "Неизвестная ошибка")
                        raise Exception(f"Ошибка генерации: {error}")
                
            except Exception as e:
                if "Generation still in progress" in str(e):
                    attempt += 1
                    await asyncio.sleep(1)
                    continue
                raise e
            
        raise Exception("Превышено время ожидания генерации")
        
    except Exception as e:
        logger.error(f"Ошибка при проверке статуса генерации: {str(e)}", extra={
            'user_id': user_id,
            'operation': 'CHECK_STATUS_ERROR',
            'error': str(e)
        })
        raise e

def get_image_keyboard(image_id: str, user_id: int) -> InlineKeyboardMarkup:
    """Клавиатура для работы с изображением"""
    keyboard = InlineKeyboardBuilder()
    
    # Основные кнопки для работы с изображением
    keyboard.button(text=f"{EmojiEnum.REMOVE_BG} Удалить фон", callback_data=f"{CallbackEnum.REMOVE_BG}_{image_id}")
    
    # Добавляем кнопку регенерации, если есть сохраненный промпт
    if user_states[user_id].last_prompt:
        keyboard.button(text=f"{EmojiEnum.CREATE} Повторить", callback_data=CallbackEnum.REGENERATE)
    
    keyboard.button(text=f"{EmojiEnum.STYLE} Стиль", callback_data=CallbackEnum.STYLES)
    keyboard.button(text=f"{EmojiEnum.SIZE} Размер", callback_data=CallbackEnum.SETTINGS)
    keyboard.button(text=f"{EmojiEnum.BACK} В меню", callback_data=CallbackEnum.BACK)
    
    keyboard.adjust(2)
    return keyboard.as_markup()

def get_main_keyboard(user_id: int) -> InlineKeyboardMarkup:
    """Основная клавиатура"""
    keyboard = InlineKeyboardBuilder()
    
    # Основные кнопки
    keyboard.button(text=f"{EmojiEnum.CREATE} Создать", callback_data=CallbackEnum.GENERATE)
    
    # Добавляем кнопку регенерации, если есть сохраненный промпт
    if user_states[user_id].last_prompt:
        keyboard.button(text=f"{EmojiEnum.CREATE} Повторить", callback_data=CallbackEnum.REGENERATE)
    
    keyboard.button(text=f"{EmojiEnum.STYLE} Стиль", callback_data=CallbackEnum.STYLES)
    keyboard.button(text=f"{EmojiEnum.SIZE} Размер", callback_data=CallbackEnum.SETTINGS)
    keyboard.button(text=f"{EmojiEnum.HELP} Помощь", callback_data=CallbackEnum.HELP)
    
    keyboard.adjust(2)
    return keyboard.as_markup()

def get_settings_keyboard(user_id: int) -> InlineKeyboardMarkup:
    """Клавиатура выбора размера"""
    keyboard = InlineKeyboardBuilder()
    
    # Кнопки размеров
    for size_key, size_data in IMAGE_SIZES.items():
        keyboard.button(
            text=f"{size_data['label']} ({size_data['width']}x{size_data['height']})",
            callback_data=f"{CallbackEnum.SIZE_PREFIX}{size_key}"
        )
    
    # Добавляем кнопку регенерации, если есть сохраненный промпт
    if user_states[user_id].last_prompt:
        keyboard.button(text=f"{EmojiEnum.CREATE} Повторить", callback_data=CallbackEnum.REGENERATE)
    
    keyboard.button(text=f"{EmojiEnum.BACK} Назад", callback_data=CallbackEnum.BACK)
    
    keyboard.adjust(2)
    return keyboard.as_markup()

def get_styles_keyboard(user_id: int) -> InlineKeyboardMarkup:
    """Клавиатура выбора стиля изображения"""
    keyboard = InlineKeyboardBuilder()
    
    # Добавляем кнопки стилей
    current_style = user_settings[user_id].style
    
    logger.info("Создание клавиатуры стилей", extra={
        'user_id': user_id,
        'operation': 'CREATING_STYLES_KEYBOARD',
        'current_style': current_style
    })
    
    for style_key, style_data in IMAGE_STYLES.items():
        # Добавляем маркер к текущему стилю
        button_text = f"{EmojiEnum.CHECK if style_key == current_style else ''} {style_data['label']}"
        callback_data = f"{CallbackEnum.STYLE_PREFIX}{style_key}"
        
        logger.info("Добавление кнопки стиля", extra={
            'user_id': user_id,
            'operation': 'ADDING_STYLE_BUTTON',
            'style_key': style_key,
            'button_text': button_text,
            'callback_data': callback_data
        })
        
        keyboard.button(
            text=button_text,
            callback_data=callback_data
        )
    
    # Добавляем кнопку "Назад"
    keyboard.button(
        text=f"{EmojiEnum.BACK} Назад",
        callback_data=CallbackEnum.BACK
    )
    
    # Добавляем кнопку "Повторить", если есть последний промпт
    if user_states[user_id].last_prompt:
        keyboard.button(
            text=f"{EmojiEnum.CREATE} Повторить",
            callback_data=CallbackEnum.REGENERATE
        )
    
    # Настраиваем размещение кнопок
    keyboard.adjust(2)
    
    return keyboard.as_markup()

def get_prompt_keyboard(user_id: int) -> InlineKeyboardMarkup:
    """Клавиатура для режима ввода промпта"""
    keyboard = InlineKeyboardBuilder()
    
    # Добавляем кнопку регенерации, если есть сохраненный промпт
    if user_states[user_id].last_prompt:
        keyboard.button(text=f"{EmojiEnum.CREATE} Повторить", callback_data=CallbackEnum.REGENERATE)
    
    keyboard.button(text=f"{EmojiEnum.BACK} Назад", callback_data=CallbackEnum.BACK)
    
    keyboard.adjust(2)
    return keyboard.as_markup()

def get_back_keyboard(user_id: int) -> InlineKeyboardMarkup:
    """Клавиатура с кнопкой возврата"""
    keyboard = InlineKeyboardBuilder()
    
    # Добавляем кнопку регенерации, если есть сохраненный промпт
    if user_states[user_id].last_prompt:
        keyboard.button(text=f"{EmojiEnum.CREATE} Повторить", callback_data=CallbackEnum.REGENERATE)
    
    keyboard.button(text=f"{EmojiEnum.BACK} Назад", callback_data=CallbackEnum.BACK)
    
    keyboard.adjust(2)
    return keyboard.as_markup()

async def main():
    """Запуск бота"""
    logger.info("Запуск бота", extra={'operation': 'STARTUP'})
    try:
        await dp.start_polling(bot)
    except Exception as e:
        logger.error(f"Ошибка при запуске бота: {str(e)}", extra={'operation': 'STARTUP_ERROR'})
        sys.exit(1)

if __name__ == '__main__':
    asyncio.run(main())
