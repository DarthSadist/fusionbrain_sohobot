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

import base64
import json
import time
import requests
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types, F, Router
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    FSInputFile,
    BufferedInputFile
)
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.exceptions import TelegramBadRequest
import asyncio
import aiohttp
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
class Emoji:
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
        "📏 Размер: <b>{size}</b>"
    )
    
    GENERATING = (
        "⏳ <b>Генерация изображения...</b>\n\n"
        "Это может занять некоторое время."
    )
    
    REMOVING_BG = (
        "⏳ <b>Удаление фона...</b>\n\n"
        "Это может занять некоторое время."
    )
    
    REMOVE_BG_SUCCESS = (
        "✅ <b>Фон успешно удален!</b>"
    )
    
    REMOVE_BG_ERROR = (
        "❌ <b>Ошибка при удалении фона</b>\n\n"
        "{error}"
    )
    
    ERROR_GEN = (
        "❌ <b>Ошибка при генерации изображения</b>\n\n"
        "{error}"
    )
    
    ERROR_CRITICAL = (
        "❌ Произошла критическая ошибка.\n"
        "Попробуйте еще раз или обратитесь к администратору."
    )
    
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
    
    STYLE_CHANGED = (
        "✅ <b>Стиль изменен</b>\n\n"
        "Текущий стиль: <b>{style}</b>"
    )
    
    SETTINGS = (
        "⚙️ <b>Настройки</b>\n\n"
        "Текущий размер: <b>{current_size}</b>"
    )
    
    SIZE_CHANGED = (
        "✅ <b>Размер изменен</b>\n\n"
        "Текущий размер: <b>{size}</b>"
    )
    
    MAIN_MENU = "Выберите действие:"
    
    CURRENT_SETTINGS = (
        "🎨 <b>Создание изображения</b>\n\n"
        "🎨 Стиль: <b>{style}</b>\n"
        "📏 Размер: <b>{size}</b>\n"
        "✍️ Введите описание желаемого изображения:"
    )

# Константы для колбэков
class CallbackData:
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
        "description": "Стандартный стиль без дополнительных модификаций",
        "model_id": 1
    },
    "ANIME": {
        "label": "Аниме",
        "prompt_prefix": "anime style, anime art, high quality anime art, ",
        "description": "Стиль японской анимации",
        "model_id": 1
    },
    "REALISTIC": {
        "label": "Реалистичный",
        "prompt_prefix": "realistic, photorealistic, hyperrealistic, 8k uhd, high quality, detailed, ",
        "description": "Максимально реалистичное изображение",
        "model_id": 1
    },
    "PORTRAIT": {
        "label": "Портрет",
        "prompt_prefix": "portrait style, professional portrait, detailed face features, studio lighting, ",
        "description": "Профессиональный портретный стиль",
        "model_id": 1
    },
    "STUDIO_GHIBLI": {
        "label": "Студия Гибли",
        "prompt_prefix": "studio ghibli style, ghibli anime, hayao miyazaki style, ",
        "description": "В стиле анимационных фильмов Студии Гибли",
        "model_id": 1
    },
    "CYBERPUNK": {
        "label": "Киберпанк",
        "prompt_prefix": "cyberpunk style, neon lights, futuristic city, high tech low life, ",
        "description": "Футуристический стиль киберпанка",
        "model_id": 1
    },
    "WATERCOLOR": {
        "label": "Акварель",
        "prompt_prefix": "watercolor painting, watercolor art style, soft colors, flowing paint, ",
        "description": "Акварельная живопись",
        "model_id": 1
    },
    "OIL_PAINTING": {
        "label": "Масло",
        "prompt_prefix": "oil painting style, classical art, detailed brush strokes, ",
        "description": "Масляная живопись",
        "model_id": 1
    },
    "DIGITAL_ART": {
        "label": "Цифровое искусство",
        "prompt_prefix": "digital art, digital painting, concept art, highly detailed digital illustration, ",
        "description": "Современное цифровое искусство",
        "model_id": 1
    },
    "PENCIL_SKETCH": {
        "label": "Карандашный эскиз",
        "prompt_prefix": "pencil sketch, graphite drawing, detailed line art, black and white sketch, ",
        "description": "Карандашный рисунок",
        "model_id": 1
    },
    "POP_ART": {
        "label": "Поп-арт",
        "prompt_prefix": "pop art style, bright colors, bold patterns, comic book style, ",
        "description": "Яркий стиль поп-арт",
        "model_id": 1
    },
    "STEAMPUNK": {
        "label": "Стимпанк",
        "prompt_prefix": "steampunk style, victorian era, brass and copper, mechanical parts, steam-powered machinery, ",
        "description": "Стиль альтернативной викторианской эпохи",
        "model_id": 1
    },
    "FANTASY": {
        "label": "Фэнтези",
        "prompt_prefix": "fantasy art style, magical, mystical, ethereal atmosphere, ",
        "description": "Фэнтезийный стиль",
        "model_id": 1
    },
    "MINIMALIST": {
        "label": "Минимализм",
        "prompt_prefix": "minimalist style, simple shapes, clean lines, minimal color palette, ",
        "description": "Минималистичный стиль",
        "model_id": 1
    },
    "IMPRESSIONIST": {
        "label": "Импрессионизм",
        "prompt_prefix": "impressionist painting style, loose brush strokes, light and color focus, plein air, ",
        "description": "Стиль импрессионизма",
        "model_id": 1
    },
    "SURREALISM": {
        "label": "Сюрреализм",
        "prompt_prefix": "surrealist art style, dreamlike, abstract elements, symbolic imagery, ",
        "description": "Сюрреалистический стиль",
        "model_id": 1
    },
    "COMIC": {
        "label": "Комикс",
        "prompt_prefix": "comic book style, bold outlines, cel shading, action lines, ",
        "description": "Стиль комиксов",
        "model_id": 1
    },
    "PIXEL_ART": {
        "label": "Пиксель-арт",
        "prompt_prefix": "pixel art style, retro gaming, 8-bit graphics, pixelated, ",
        "description": "Пиксельная графика",
        "model_id": 1
    },
    "GOTHIC": {
        "label": "Готика",
        "prompt_prefix": "gothic art style, dark atmosphere, medieval architecture, dramatic lighting, ",
        "description": "Готический стиль",
        "model_id": 1
    },
    "RETRO": {
        "label": "Ретро",
        "prompt_prefix": "retro style, vintage aesthetics, old school design, nostalgic feel, ",
        "description": "Ретро стиль",
        "model_id": 1
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
        self.last_prompt = None  # Хранение последнего промпта

# Словарь для хранения пользовательских настроек
class UserSettings:
    def __init__(self):
        self.width = 1024
        self.height = 1024
        self.style = "DEFAULT"  # Стиль по умолчанию

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
            Messages.WELCOME.format(current_style=IMAGE_STYLES[user_settings[message.from_user.id].style]['label']),
            reply_markup=get_main_keyboard(message.from_user.id),
            parse_mode=ParseMode.HTML
        )
    except Exception as e:
        logger.error(f"Ошибка в send_welcome: {str(e)}", extra={
            'user_id': message.from_user.id,
            'operation': 'WELCOME'
        })
        await message.answer(Messages.ERROR_CRITICAL)

@router.callback_query(F.data == CallbackData.HELP)
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
                caption=Messages.HELP,
                reply_markup=get_back_keyboard(user_id),
                parse_mode=ParseMode.HTML
            )
        else:
            await callback_query.message.edit_text(
                text=Messages.HELP,
                reply_markup=get_back_keyboard(user_id),
                parse_mode=ParseMode.HTML
            )
        await callback_query.answer()
        
    except Exception as e:
        logger.error(f"Ошибка при показе справки: {str(e)}", extra={
            'user_id': user_id if 'user_id' in locals() else 'N/A',
            'operation': 'HELP_ERROR'
        })
        await callback_query.answer("Произошла ошибка. Попробуйте еще раз.")

@router.callback_query(F.data == CallbackData.SETTINGS)
async def show_settings(callback_query: CallbackQuery):
    """Обработчик кнопки настроек"""
    try:
        user_id = callback_query.from_user.id
        settings = user_settings[user_id]
        
        if callback_query.message.photo:
            await callback_query.message.edit_caption(
                caption=Messages.SETTINGS.format(
                    current_size=f"{settings.width}x{settings.height}"
                ),
                reply_markup=get_settings_keyboard(user_id)
            )
        else:
            await callback_query.message.edit_text(
                text=Messages.SETTINGS.format(
                    current_size=f"{settings.width}x{settings.height}"
                ),
                reply_markup=get_settings_keyboard(user_id)
            )
        await callback_query.answer()
    except Exception as e:
        logger.error(f"Ошибка в show_settings: {str(e)}", extra={
            'user_id': callback_query.from_user.id,
            'operation': 'SETTINGS'
        })
        await callback_query.answer("Произошла ошибка. Попробуйте еще раз.")

@router.callback_query(F.data.startswith(CallbackData.SIZE_PREFIX))
async def process_size_change(callback_query: CallbackQuery):
    """Обработчик изменения размера изображения"""
    user_id = callback_query.from_user.id
    size_key = callback_query.data.replace(CallbackData.SIZE_PREFIX, "")
    
    try:
        # Получаем размеры из словаря
        size_data = IMAGE_SIZES.get(size_key)
        if not size_data:
            logger.error(f"Неверный размер: {size_key}", extra={
                'user_id': user_id,
                'operation': 'INVALID_SIZE'
            })
            await callback_query.answer(Messages.ERROR_SIZE, show_alert=True)
            return

        # Обновляем настройки пользователя
        user_settings[user_id].width = size_data["width"]
        user_settings[user_id].height = size_data["height"]
        
        size_label = f"{size_data['width']}x{size_data['height']}"
        
        logger.info(f"Изменен размер изображения", extra={
            'user_id': user_id,
            'operation': 'SIZE_CHANGED',
            'new_size': size_label
        })

        # Если есть последний промпт, добавляем кнопку регенерации
        keyboard = InlineKeyboardBuilder()
        keyboard.button(text=f"{Emoji.BACK} Назад", callback_data=CallbackData.BACK)
        
        if user_states[user_id].last_prompt:
            keyboard.button(text=f"{Emoji.CREATE} Повторить", callback_data=CallbackData.REGENERATE)
        
        keyboard.adjust(2)

        if callback_query.message.photo:
            await callback_query.message.edit_caption(
                caption=Messages.SIZE_CHANGED.format(size=size_label),
                reply_markup=keyboard.as_markup(),
                parse_mode=ParseMode.HTML
            )
        else:
            await callback_query.message.edit_text(
                text=Messages.SIZE_CHANGED.format(size=size_label),
                reply_markup=keyboard.as_markup(),
                parse_mode=ParseMode.HTML
            )
        
    except Exception as e:
        logger.error(f"Ошибка при изменении размера: {str(e)}", extra={
            'user_id': user_id,
            'operation': 'SIZE_CHANGE_ERROR'
        })
        await callback_query.answer(Messages.ERROR_CRITICAL, show_alert=True)

@router.callback_query(F.data.startswith(CallbackData.REMOVE_BG))
async def process_remove_background(callback_query: CallbackQuery):
    """Обработчик удаления фона с изображения"""
    try:
        user_id = callback_query.from_user.id
        logger.info("Получен запрос на удаление фона", extra={
            'user_id': user_id,
            'operation': 'REMOVE_BG_START'
        })

        # Проверяем наличие последнего изображения
        if not user_states[user_id].last_image:
            logger.warning("Попытка удаления фона без изображения", extra={
                'user_id': user_id,
                'operation': 'NO_IMAGE'
            })
            await callback_query.answer("Нет доступного изображения")
            return

        # Отправляем сообщение о начале обработки
        if callback_query.message.photo:
            await callback_query.message.edit_caption(
                caption=Messages.REMOVING_BG,
                reply_markup=None
            )
        else:
            await callback_query.message.edit_text(
                text=Messages.REMOVING_BG,
                reply_markup=None
            )

        try:
            # Удаляем фон в отдельном потоке
            loop = asyncio.get_event_loop()
            result_image = await loop.run_in_executor(
                None,
                ImageProcessor.remove_background,
                user_states[user_id].last_image
            )

            # Сохраняем результат
            user_states[user_id].last_image = result_image
            image_id = str(uuid_lib.uuid4())
            user_states[user_id].last_image_id = image_id

            # Отправляем обработанное изображение
            if callback_query.message.photo:
                await callback_query.message.answer_photo(
                    BufferedInputFile(
                        result_image,
                        filename=f"nobg_{image_id}.png"
                    ),
                    caption=Messages.REMOVE_BG_SUCCESS,
                    reply_markup=get_image_keyboard(image_id, user_id)
                )
            else:
                await callback_query.message.answer_photo(
                    BufferedInputFile(
                        result_image,
                        filename=f"nobg_{image_id}.png"
                    ),
                    caption=Messages.REMOVE_BG_SUCCESS,
                    reply_markup=get_image_keyboard(image_id, user_id)
                )

            # Удаляем старое сообщение
            await callback_query.message.delete()

            logger.info("Фон успешно удален", extra={
                'user_id': user_id,
                'operation': 'REMOVE_BG_SUCCESS'
            })

        except Exception as e:
            error_message = str(e)
            logger.error(f"Ошибка при удалении фона: {error_message}", extra={
                'user_id': user_id,
                'operation': 'REMOVE_BG_ERROR'
            })
            if callback_query.message.photo:
                await callback_query.message.edit_caption(
                    caption=Messages.ERROR_GEN.format(error=error_message),
                    reply_markup=get_image_keyboard(user_states[user_id].last_image_id, user_id)
                )
            else:
                await callback_query.message.edit_text(
                    text=Messages.ERROR_GEN.format(error=error_message),
                    reply_markup=get_image_keyboard(user_states[user_id].last_image_id, user_id)
                )

    except Exception as e:
        logger.error(f"Критическая ошибка в process_remove_background: {str(e)}", extra={
            'user_id': user_id if 'user_id' in locals() else 'N/A',
            'operation': 'CRITICAL_ERROR'
        })
        await callback_query.answer("Произошла критическая ошибка")

@router.callback_query(F.data == CallbackData.GENERATE)
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
                text=Messages.CURRENT_SETTINGS.format(
                    style=style_info['label'],
                    size=current_size
                ),
                reply_markup=get_prompt_keyboard(user_id)
            )
        else:
            await callback_query.message.edit_text(
                text=Messages.CURRENT_SETTINGS.format(
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
            text=Messages.ERROR_CRITICAL,
            reply_markup=get_back_keyboard(user_id)
        )

@router.callback_query(F.data == CallbackData.BACK)
async def back_to_main(callback_query: CallbackQuery):
    try:
        if callback_query.message.photo:
            await callback_query.message.edit_caption(
                caption=Messages.MAIN_MENU,
                reply_markup=get_main_keyboard(callback_query.from_user.id)
            )
        else:
            await callback_query.message.edit_text(
                text=Messages.MAIN_MENU,
                reply_markup=get_main_keyboard(callback_query.from_user.id)
            )
        await callback_query.answer()
    except Exception as e:
        logger.error(f"Ошибка при возврате в главное меню: {str(e)}")
        await callback_query.answer("Произошла ошибка. Попробуйте еще раз.")

@router.callback_query(F.data == CallbackData.STYLES)
async def show_styles(callback_query: CallbackQuery):
    """Показывает меню выбора стиля"""
    user_id = callback_query.from_user.id
    
    try:
        logger.info("Открытие меню стилей", extra={
            'user_id': user_id,
            'operation': 'STYLES'
        })
        
        # Получаем текущий стиль
        current_style = user_settings[user_id].style
        style_label = IMAGE_STYLES[current_style]["label"]
        
        logger.info("Текущий стиль", extra={
            'user_id': user_id,
            'operation': 'CURRENT_STYLE',
            'style_key': current_style,
            'style_label': style_label
        })
        
        # Формируем текст сообщения
        message_text = Messages.STYLES.format(current_style=style_label)
        
        # Проверяем тип сообщения и редактируем соответственно
        if callback_query.message.photo:
            logger.info("Редактирование подписи фото", extra={
                'user_id': user_id,
                'operation': 'EDIT_PHOTO_CAPTION'
            })
            await callback_query.message.edit_caption(
                caption=message_text,
                reply_markup=get_styles_keyboard(user_id),
                parse_mode=ParseMode.HTML
            )
        else:
            logger.info("Редактирование текстового сообщения", extra={
                'user_id': user_id,
                'operation': 'EDIT_TEXT_MESSAGE'
            })
            await callback_query.message.edit_text(
                text=message_text,
                reply_markup=get_styles_keyboard(user_id),
                parse_mode=ParseMode.HTML
            )
        
        await callback_query.answer()
        
    except Exception as e:
        logger.error(f"Ошибка при показе стилей: {str(e)}", extra={
            'user_id': user_id,
            'operation': 'STYLES_ERROR',
            'error': str(e)
        })
        
        # Если не удалось отредактировать, отправляем новое сообщение
        try:
            logger.info("Попытка отправки нового сообщения", extra={
                'user_id': user_id,
                'operation': 'SEND_NEW_MESSAGE'
            })
            await callback_query.message.answer(
                Messages.STYLES.format(current_style=style_label),
                reply_markup=get_styles_keyboard(user_id),
                parse_mode=ParseMode.HTML
            )
            # Удаляем старое сообщение
            await callback_query.message.delete()
        except Exception as send_error:
            logger.error(f"Не удалось отправить новое сообщение: {str(send_error)}", extra={
                'user_id': user_id,
                'operation': 'SEND_NEW_MESSAGE_ERROR',
                'error': str(send_error)
            })
            await callback_query.answer(
                text="❌ Не удалось показать стили",
                show_alert=True
            )

@router.callback_query(F.data.startswith(CallbackData.STYLE_PREFIX))
async def process_style_change(callback_query: CallbackQuery):
    """Обработчик изменения стиля изображения"""
    user_id = callback_query.from_user.id
    style_key = callback_query.data.replace(CallbackData.STYLE_PREFIX, "")
    
    logger.info("Получен callback изменения стиля", extra={
        'user_id': user_id,
        'operation': 'STYLE_CALLBACK_RECEIVED',
        'callback_data': callback_query.data,
        'style_key': style_key
    })
    
    try:
        # Проверяем существование стиля
        if style_key not in IMAGE_STYLES:
            logger.error(f"Неверный стиль: {style_key}", extra={
                'user_id': user_id,
                'operation': 'INVALID_STYLE'
            })
            await callback_query.answer(
                text="❌ Неверный стиль",
                show_alert=True
            )
            return
        
        # Обновляем стиль в настройках пользователя
        user_settings[user_id].style = style_key
        style_label = IMAGE_STYLES[style_key]["label"]
        
        logger.info("Стиль успешно изменен", extra={
            'user_id': user_id,
            'operation': 'STYLE_CHANGED',
            'new_style': style_key,
            'style_label': style_label
        })
        
        # Отправляем новое сообщение
        if callback_query.message.photo:
            await callback_query.message.answer(
                Messages.STYLE_CHANGED.format(style=style_label),
                reply_markup=get_styles_keyboard(user_id),
                parse_mode=ParseMode.HTML
            )
        else:
            await callback_query.message.answer(
                Messages.STYLE_CHANGED.format(style=style_label),
                reply_markup=get_styles_keyboard(user_id),
                parse_mode=ParseMode.HTML
            )
        
        # Удаляем старое сообщение
        await callback_query.message.delete()
        
        # Подтверждаем callback
        await callback_query.answer(
            text=f"✅ Стиль изменен на {style_label}",
            show_alert=False
        )
        
    except Exception as e:
        logger.error(f"Ошибка при изменении стиля: {str(e)}", extra={
            'user_id': user_id,
            'operation': 'STYLE_CHANGE_ERROR',
            'error': str(e)
        })
        await callback_query.answer(
            text="❌ Не удалось изменить стиль",
            show_alert=True
        )

@router.callback_query(F.data == CallbackData.REGENERATE)
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
            Messages.GENERATING,
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
                Messages.ERROR_GEN.format(error=user_message),
                reply_markup=get_back_keyboard(user_id),
                parse_mode=ParseMode.HTML
            )
            return
        
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
        Messages.GENERATING,
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
            Messages.ERROR_GEN.format(error=user_message),
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
            Messages.GENERATING,
            reply_markup=get_back_keyboard(user_id),
            parse_mode=ParseMode.HTML
        )

        # Проверяем наличие и валидность ключей API
        api_key = os.getenv('FUSIONBRAIN_API_KEY')
        secret_key = os.getenv('FUSIONBRAIN_SECRET_KEY')

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

        # Сохраняем последний промпт
        user_states[user_id].last_prompt = prompt

        # Отправляем сообщение о начале генерации
        status_message = await message.answer(
            Messages.GENERATING,
            reply_markup=get_back_keyboard(user_id),
            parse_mode=ParseMode.HTML
        )

        logger.info(f"Начало генерации изображения", extra={
            'user_id': user_id,
            'operation': 'GENERATION_PROCESS',
            'prompt': prompt
        })

        try:
            # Инициализируем API и запускаем генерацию
            api = Text2ImageAPI(FUSIONBRAIN_API_KEY, FUSIONBRAIN_SECRET_KEY)
            
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
            styled_prompt = f"{prompt}, {IMAGE_STYLES[style]['prompt_prefix']}" if style != "DEFAULT" else prompt
            
            # Запускаем генерацию
            uuid = await api.generate(styled_prompt, model_id, width, height)
            
            # Проверяем статус генерации
            await check_generation_status(api, uuid, status_message, user_id)

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
                Messages.ERROR_GEN.format(error=user_message),
                reply_markup=get_back_keyboard(user_id),
                parse_mode=ParseMode.HTML
            )
            
    except Exception as e:
        logger.error(f"Критическая ошибка в generate_image_with_prompt: {str(e)}", extra={
            'user_id': user_id,
            'operation': 'CRITICAL_ERROR',
            'error': str(e)
        })
        if 'status_message' in locals():
            await status_message.edit_text(
                Messages.ERROR_CRITICAL,
                reply_markup=get_back_keyboard(user_id),
                parse_mode=ParseMode.HTML
            )
        else:
            await message.answer(
                Messages.ERROR_CRITICAL,
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
            Messages.GENERATING,
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
                Messages.ERROR_GEN.format(error=user_message),
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
                Messages.ERROR_CRITICAL,
                reply_markup=get_back_keyboard(user_id),
                parse_mode=ParseMode.HTML
            )
        else:
            await message.answer(
                Messages.ERROR_CRITICAL,
                reply_markup=get_back_keyboard(user_id),
                parse_mode=ParseMode.HTML
            )

async def check_generation_status(api, uuid, status_message, user_id):
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
                    
                    # Отправляем изображение пользователю
                    if status_message.photo:
                        await status_message.answer_photo(
                            BufferedInputFile(
                                image_data,
                                filename=f"generation_{uuid}.png"
                            ),
                            caption=(
                                f"🎨 Стиль: <b>{IMAGE_STYLES[user_settings[user_id].style]['label']}</b>\n"
                                f"📏 Размер: <b>{user_settings[user_id].width}x{user_settings[user_id].height}</b>\n"
                                f"💭 Промпт: <i>{user_states[user_id].last_prompt}</i>"
                            ),
                            reply_markup=get_image_keyboard(uuid, user_id),
                            parse_mode=ParseMode.HTML
                        )
                    else:
                        await status_message.answer_photo(
                            BufferedInputFile(
                                image_data,
                                filename=f"generation_{uuid}.png"
                            ),
                            caption=(
                                f"🎨 Стиль: <b>{IMAGE_STYLES[user_settings[user_id].style]['label']}</b>\n"
                                f"📏 Размер: <b>{user_settings[user_id].width}x{user_settings[user_id].height}</b>\n"
                                f"💭 Промпт: <i>{user_states[user_id].last_prompt}</i>"
                            ),
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
                        images = response.get('images', [])
                        if not images:
                            raise Exception("Изображение не было сгенерировано")
                            
                        logger.info("Изображение успешно сгенерировано", extra={
                            'user_id': user_id,
                            'operation': 'GENERATION_SUCCESS'
                        })
                        
                        # Сохраняем изображение
                        image_data = base64.b64decode(images[0])
                        
                        # Отправляем изображение пользователю
                        if status_message.photo:
                            await status_message.answer_photo(
                                BufferedInputFile(
                                    image_data,
                                    filename=f"generation_{uuid}.png"
                                ),
                                caption=(
                                    f"🎨 Стиль: <b>{IMAGE_STYLES[user_settings[user_id].style]['label']}</b>\n"
                                    f"📏 Размер: <b>{user_settings[user_id].width}x{user_settings[user_id].height}</b>\n"
                                    f"💭 Промпт: <i>{user_states[user_id].last_prompt}</i>"
                                ),
                                reply_markup=get_image_keyboard(uuid, user_id),
                                parse_mode=ParseMode.HTML
                            )
                        else:
                            await status_message.answer_photo(
                                BufferedInputFile(
                                    image_data,
                                    filename=f"generation_{uuid}.png"
                                ),
                                caption=(
                                    f"🎨 Стиль: <b>{IMAGE_STYLES[user_settings[user_id].style]['label']}</b>\n"
                                    f"📏 Размер: <b>{user_settings[user_id].width}x{user_settings[user_id].height}</b>\n"
                                    f"💭 Промпт: <i>{user_states[user_id].last_prompt}</i>"
                                ),
                                reply_markup=get_image_keyboard(uuid, user_id),
                                parse_mode=ParseMode.HTML
                            )
                        
                        # Сохраняем информацию о последнем изображении
                        user_states[user_id].last_image = image_data
                        user_states[user_id].last_image_id = uuid
                        
                        return True
                        
                    elif status in ["INITIAL", "PROCESSING"]:
                        self.logger.info("Генерация все еще выполняется", extra={
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
                else:
                    raise e
            
            attempt += 1
            await asyncio.sleep(1)
        
        # Если превышено максимальное количество попыток
        raise Exception("Превышено время ожидания генерации")
        
    except Exception as e:
        logger.error(f"Ошибка при проверке статуса генерации: {str(e)}", extra={
            'user_id': user_id,
            'operation': 'STATUS_CHECK_ERROR',
            'error': str(e)
        })
        raise e

def get_image_keyboard(image_id: str, user_id: int) -> InlineKeyboardMarkup:
    """Создает клавиатуру для изображения"""
    keyboard = InlineKeyboardBuilder()
    
    # Основные кнопки для работы с изображением
    keyboard.button(text=f"{Emoji.REMOVE_BG} Удалить фон", callback_data=f"{CallbackData.REMOVE_BG}_{image_id}")
    
    # Добавляем кнопку регенерации, если есть сохраненный промпт
    if user_states[user_id].last_prompt:
        keyboard.button(text=f"{Emoji.CREATE} Повторить", callback_data=CallbackData.REGENERATE)
    
    keyboard.button(text=f"{Emoji.STYLE} Стиль", callback_data=CallbackData.STYLES)
    keyboard.button(text=f"{Emoji.SIZE} Размер", callback_data=CallbackData.SETTINGS)
    keyboard.button(text=f"{Emoji.BACK} В меню", callback_data=CallbackData.BACK)
    
    keyboard.adjust(2)
    return keyboard.as_markup()

def get_main_keyboard(user_id: int) -> InlineKeyboardMarkup:
    """Создает основную клавиатуру главного меню"""
    keyboard = InlineKeyboardBuilder()
    
    # Основные кнопки
    keyboard.button(text=f"{Emoji.CREATE} Создать", callback_data=CallbackData.GENERATE)
    
    # Добавляем кнопку регенерации, если есть сохраненный промпт
    if user_states[user_id].last_prompt:
        keyboard.button(text=f"{Emoji.CREATE} Повторить", callback_data=CallbackData.REGENERATE)
    
    keyboard.button(text=f"{Emoji.STYLE} Стиль", callback_data=CallbackData.STYLES)
    keyboard.button(text=f"{Emoji.SIZE} Размер", callback_data=CallbackData.SETTINGS)
    keyboard.button(text=f"{Emoji.HELP} Помощь", callback_data=CallbackData.HELP)
    
    keyboard.adjust(2)
    return keyboard.as_markup()

def get_settings_keyboard(user_id: int) -> InlineKeyboardMarkup:
    """Создает клавиатуру настроек размера"""
    keyboard = InlineKeyboardBuilder()
    
    # Кнопки размеров
    for size_key, size_data in IMAGE_SIZES.items():
        keyboard.button(
            text=f"{size_data['label']} ({size_data['width']}x{size_data['height']})",
            callback_data=f"{CallbackData.SIZE_PREFIX}{size_key}"
        )
    
    # Добавляем кнопку регенерации, если есть сохраненный промпт
    if user_states[user_id].last_prompt:
        keyboard.button(text=f"{Emoji.CREATE} Повторить", callback_data=CallbackData.REGENERATE)
    
    keyboard.button(text=f"{Emoji.BACK} Назад", callback_data=CallbackData.BACK)
    
    keyboard.adjust(2)
    return keyboard.as_markup()

def get_styles_keyboard(user_id: int) -> InlineKeyboardMarkup:
    """Создает клавиатуру выбора стиля изображения"""
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
        button_text = f"{Emoji.CHECK if style_key == current_style else ''} {style_data['label']}"
        callback_data = f"{CallbackData.STYLE_PREFIX}{style_key}"
        
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
        text=f"{Emoji.BACK} Назад",
        callback_data=CallbackData.BACK
    )
    
    # Добавляем кнопку "Повторить", если есть последний промпт
    if user_states[user_id].last_prompt:
        keyboard.button(
            text=f"{Emoji.CREATE} Повторить",
            callback_data=CallbackData.REGENERATE
        )
    
    # Настраиваем размещение кнопок
    keyboard.adjust(2)
    
    return keyboard.as_markup()

def get_prompt_keyboard(user_id: int) -> InlineKeyboardMarkup:
    """Создает клавиатуру для режима ввода промпта"""
    keyboard = InlineKeyboardBuilder()
    
    # Добавляем кнопку регенерации, если есть сохраненный промпт
    if user_states[user_id].last_prompt:
        keyboard.button(text=f"{Emoji.CREATE} Повторить", callback_data=CallbackData.REGENERATE)
    
    keyboard.button(text=f"{Emoji.BACK} Назад", callback_data=CallbackData.BACK)
    
    keyboard.adjust(2)
    return keyboard.as_markup()

def get_back_keyboard(user_id: int) -> InlineKeyboardMarkup:
    """Создает клавиатуру с кнопкой возврата"""
    keyboard = InlineKeyboardBuilder()
    
    # Добавляем кнопку регенерации, если есть сохраненный промпт
    if user_states[user_id].last_prompt:
        keyboard.button(text=f"{Emoji.CREATE} Повторить", callback_data=CallbackData.REGENERATE)
    
    keyboard.button(text=f"{Emoji.BACK} Назад", callback_data=CallbackData.BACK)
    
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
