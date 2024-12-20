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
        url = f"{self.URL}/key/api/v1/text2image/status/{uuid}"
        response = await self._make_request("GET", url)
        
        if not response:
            raise Exception("Пустой ответ при проверке статуса")
        
        status = response.get("status")
        self.logger.info(f"Generation status: {status}", extra={'operation': 'GENERATION_STATUS'})
        
        if status == "DONE":
            if not response.get("images"):
                raise Exception("Изображения отсутствуют в ответе")
            return response
        elif status in ["INITIAL", "PROCESSING"]:
            raise Exception("Generation still in progress")
        elif status == "FAILED":
            error = response.get("error", "Неизвестная ошибка")
            raise Exception(f"Генерация не удалась: {error}")
        else:
            raise Exception(f"Неизвестный статус генерации: {status}")

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
    "DEFAULT": {
        "label": "Обычный",
        "prompt_prefix": "",
        "description": "Стандартный стиль без дополнительных модификаций"
    },
    "ANIME": {
        "label": "Аниме",
        "prompt_prefix": "anime style, anime art, ",
        "description": "Стиль японской анимации"
    },
    "REALISTIC": {
        "label": "Реалистичный",
        "prompt_prefix": "realistic, photorealistic, hyperrealistic, ",
        "description": "Максимально реалистичное изображение"
    },
    "PORTRAIT": {
        "label": "Портрет",
        "prompt_prefix": "portrait style, professional portrait, ",
        "description": "Профессиональный портретный стиль"
    },
    "STUDIO_GHIBLI": {
        "label": "Студия Гибли",
        "prompt_prefix": "studio ghibli style, ghibli anime, ",
        "description": "В стиле анимационных фильмов Студии Гибли"
    },
    "CYBERPUNK": {
        "label": "Киберпанк",
        "prompt_prefix": "cyberpunk style, neon, futuristic, ",
        "description": "Футуристический стиль киберпанка"
    },
    "WATERCOLOR": {
        "label": "Акварель",
        "prompt_prefix": "watercolor painting, watercolor art style, ",
        "description": "Акварельная живопись"
    },
    "OIL_PAINTING": {
        "label": "Масло",
        "prompt_prefix": "oil painting style, classical art, ",
        "description": "Классическая масляная живопись"
    },
    "PENCIL_DRAWING": {
        "label": "Карандаш",
        "prompt_prefix": "pencil drawing, sketch style, ",
        "description": "Карандашный рисунок"
    },
    "DIGITAL_ART": {
        "label": "Цифровое искусство",
        "prompt_prefix": "digital art, digital painting, ",
        "description": "Современное цифровое искусство"
    },
    "POP_ART": {
        "label": "Поп-арт",
        "prompt_prefix": "pop art style, vibrant colors, ",
        "description": "Яркий стиль поп-арт"
    },
    "STEAMPUNK": {
        "label": "Стимпанк",
        "prompt_prefix": "steampunk style, victorian sci-fi, ",
        "description": "Викторианский научно-фантастический стиль"
    },
    "MINIMALIST": {
        "label": "Минимализм",
        "prompt_prefix": "minimalist style, simple, clean, ",
        "description": "Минималистичный стиль"
    },
    "FANTASY": {
        "label": "Фэнтези",
        "prompt_prefix": "fantasy art style, magical, mystical, ",
        "description": "Фэнтезийный стиль"
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

@router.message(Command("start"))
async def send_welcome(message: types.Message):
    """Обработчик команды /start"""
    try:
        await message.answer(
            Messages.WELCOME,
            reply_markup=get_main_keyboard(),
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
                reply_markup=get_back_keyboard(),
                parse_mode=ParseMode.HTML
            )
        else:
            await callback_query.message.edit_text(
                text=Messages.HELP,
                reply_markup=get_back_keyboard(),
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
                    width=settings.width,
                    height=settings.height
                ),
                reply_markup=get_settings_keyboard()
            )
        else:
            await callback_query.message.edit_text(
                text=Messages.SETTINGS.format(
                    width=settings.width,
                    height=settings.height
                ),
                reply_markup=get_settings_keyboard()
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
    try:
        user_id = callback_query.from_user.id
        size_key = callback_query.data.replace(CallbackData.SIZE_PREFIX, "")
        size_config = IMAGE_SIZES.get(size_key)
        
        if not size_config:
            logger.error(f"Неверный размер: {size_key}", extra={
                'user_id': user_id,
                'operation': 'SIZE_ERROR'
            })
            await callback_query.answer(Messages.ERROR_SIZE)
            return
        
        user_settings[user_id].width = size_config["width"]
        user_settings[user_id].height = size_config["height"]
        
        logger.info(f"Изменен размер изображения: {size_config['label']}", extra={
            'user_id': user_id,
            'operation': 'SIZE_CHANGE'
        })
        
        if callback_query.message.photo:
            await callback_query.message.edit_caption(
                caption=Messages.SIZE_CHANGED.format(size=size_config['label']),
                reply_markup=get_main_keyboard()
            )
        else:
            await callback_query.message.edit_text(
                text=Messages.SIZE_CHANGED.format(size=size_config['label']),
                reply_markup=get_main_keyboard()
            )
        await callback_query.answer()
    except Exception as e:
        logger.error(f"Ошибка при изменении размера: {str(e)}", extra={
            'user_id': user_id if 'user_id' in locals() else 'N/A',
            'operation': 'SIZE_ERROR'
        })
        await callback_query.answer("Произошла ошибка. Попробуйте еще раз.")

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
        await callback_query.message.edit_caption(
            caption=Messages.REMOVING_BG,
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
            await callback_query.message.answer_photo(
                BufferedInputFile(
                    result_image,
                    filename=f"nobg_{image_id}.png"
                ),
                caption=Messages.BG_REMOVED,
                reply_markup=get_image_keyboard(image_id)
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
            await callback_query.message.edit_caption(
                caption=Messages.ERROR_GEN.format(error=error_message),
                reply_markup=get_image_keyboard(user_states[user_id].last_image_id)
            )

    except Exception as e:
        logger.error(f"Критическая ошибка в process_remove_background: {str(e)}", extra={
            'user_id': user_id if 'user_id' in locals() else 'N/A',
            'operation': 'CRITICAL_ERROR'
        })
        await callback_query.answer("Произошла критическая ошибка")

@router.callback_query(F.data == CallbackData.GENERATE)
async def start_generation(callback_query: CallbackQuery):
    user_id = callback_query.from_user.id
    user_states[user_id].awaiting_prompt = True
    settings = user_settings[user_id]
    
    logger.info(f"Пользователь начал процесс генерации", extra={
        'user_id': user_id,
        'operation': 'SYSTEM'
    })
    
    try:
        # Проверяем тип сообщения и используем соответствующий метод
        if callback_query.message.photo:
            await callback_query.message.edit_caption(
                caption=Messages.CURRENT_SETTINGS.format(
                    style=IMAGE_STYLES[settings.style]["label"],
                    size=f"{settings.width}x{settings.height}"
                ),
                reply_markup=get_prompt_keyboard()
            )
        else:
            await callback_query.message.edit_text(
                text=Messages.CURRENT_SETTINGS.format(
                    style=IMAGE_STYLES[settings.style]["label"],
                    size=f"{settings.width}x{settings.height}"
                ),
                reply_markup=get_prompt_keyboard()
            )
        await callback_query.answer()
    except Exception as e:
        logger.error(f"Ошибка при начале генерации: {str(e)}", extra={
            'user_id': user_id,
            'operation': 'ERROR'
        })
        await callback_query.answer("Произошла ошибка. Попробуйте еще раз.")

@router.callback_query(F.data == CallbackData.BACK)
async def back_to_main(callback_query: CallbackQuery):
    try:
        if callback_query.message.photo:
            await callback_query.message.edit_caption(
                caption=Messages.MAIN_MENU,
                reply_markup=get_main_keyboard()
            )
        else:
            await callback_query.message.edit_text(
                text=Messages.MAIN_MENU,
                reply_markup=get_main_keyboard()
            )
        await callback_query.answer()
    except Exception as e:
        logger.error(f"Ошибка при возврате в главное меню: {str(e)}")
        await callback_query.answer("Произошла ошибка. Попробуйте еще раз.")

@router.callback_query(F.data == CallbackData.STYLES)
async def show_styles(callback_query: CallbackQuery):
    try:
        user_id = callback_query.from_user.id
        current_style = user_settings[user_id].style
        style_label = IMAGE_STYLES[current_style]["label"]
        
        logger.info(f"Пользователь открыл меню стилей", extra={
            'user_id': user_id,
            'operation': 'STYLES'
        })
        
        if callback_query.message.photo:
            await callback_query.message.edit_caption(
                caption=Messages.STYLES.format(style_label=style_label),
                reply_markup=get_styles_keyboard()
            )
        else:
            await callback_query.message.edit_text(
                text=Messages.STYLES.format(style_label=style_label),
                reply_markup=get_styles_keyboard()
            )
        await callback_query.answer()
    except Exception as e:
        logger.error(f"Ошибка при показе стилей: {str(e)}", extra={
            'user_id': user_id if 'user_id' in locals() else 'N/A',
            'operation': 'STYLES_ERROR'
        })
        await callback_query.answer("Произошла ошибка. Попробуйте еще раз.")

@router.callback_query(F.data.startswith(CallbackData.STYLE_PREFIX))
async def process_style_change(callback_query: CallbackQuery):
    """Обработчик изменения стиля изображения"""
    try:
        user_id = callback_query.from_user.id
        style_key = callback_query.data.replace(CallbackData.STYLE_PREFIX, "")
        
        if style_key not in IMAGE_STYLES:
            logger.error(f"Неверный стиль: {style_key}", extra={
                'user_id': user_id,
                'operation': 'STYLE_ERROR'
            })
            await callback_query.answer("❌ Ошибка: неверный стиль")
            return
            
        user_settings[user_id].style = style_key
        style_info = IMAGE_STYLES[style_key]
        
        logger.info(f"Изменен стиль изображения: {style_info['label']}", extra={
            'user_id': user_id,
            'operation': 'STYLE_CHANGE'
        })
        
        message_text = (
            f"{Emoji.SUCCESS} Выбран стиль: <b>{style_info['label']}</b>\n"
            f"\n"
            f"<i>{style_info['description']}</i>"
        )
        
        if callback_query.message.photo:
            await callback_query.message.edit_caption(
                caption=message_text,
                reply_markup=get_prompt_keyboard(),
                parse_mode=ParseMode.HTML
            )
        else:
            await callback_query.message.edit_text(
                text=message_text,
                reply_markup=get_prompt_keyboard(),
                parse_mode=ParseMode.HTML
            )
        
        # Устанавливаем флаг ожидания промпта
        user_states[callback_query.from_user.id].awaiting_prompt = True
        await callback_query.answer()
        
    except Exception as e:
        logger.error(f"Ошибка при изменении стиля: {str(e)}", extra={
            'user_id': user_id if 'user_id' in locals() else 'N/A',
            'operation': 'STYLE_ERROR'
        })
        await callback_query.answer("Произошла ошибка. Попробуйте еще раз.")

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
    
    # Создаем кнопки для каждого стиля, по 2 в ряд
    current_row = []
    for style_id, style_info in IMAGE_STYLES.items():
        button = InlineKeyboardButton(
            text=style_info["label"],
            callback_data=f"{CallbackData.STYLE_PREFIX}{style_id}"
        )
        current_row.append(button)
        
        if len(current_row) == 2:
            keyboard.append(current_row)
            current_row = []
    
    # Добавляем оставшиеся кнопки, если есть
    if current_row:
        keyboard.append(current_row)
    
    # Добавляем кнопку "Назад"
    keyboard.append([
        InlineKeyboardButton(
            text=f"{Emoji.BACK} Назад",
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

@router.message()
async def generate_image(message: types.Message):
    """Генерирует изображение на основе промпта"""
    try:
        user_id = message.from_user.id
        logger.info("Получен запрос на генерацию изображения", extra={
            'user_id': user_id,
            'operation': 'IMAGE_GENERATION_START'
        })
        
        if not user_states[user_id].awaiting_prompt:
            logger.warning("Получен неожиданный промпт", extra={
                'user_id': user_id,
                'operation': 'UNEXPECTED_PROMPT'
            })
            return

        # Проверяем наличие ключей API
        if not all([FUSIONBRAIN_API_KEY, FUSIONBRAIN_SECRET_KEY]):
            logger.error("Отсутствуют ключи API", extra={
                'user_id': user_id,
                'operation': 'MISSING_API_KEYS'
            })
            await message.answer(
                "⚠️ Ошибка конфигурации: отсутствуют ключи API. Обратитесь к администратору.",
                reply_markup=get_back_keyboard()
            )
            return

        prompt = message.text
        if len(prompt) > Text2ImageAPI.MAX_PROMPT_LENGTH:
            logger.warning(f"Промпт превышает максимальную длину: {len(prompt)}", extra={
                'user_id': user_id,
                'operation': 'PROMPT_TOO_LONG'
            })
            prompt = prompt[:Text2ImageAPI.MAX_PROMPT_LENGTH]

        # Отправляем сообщение о начале генерации
        status_message = await message.answer(
            Messages.GENERATING,
            reply_markup=get_back_keyboard()
        )

        logger.info(f"Начало генерации изображения с промптом: {prompt}", extra={
            'user_id': user_id,
            'operation': 'GENERATION_PROCESS'
        })

        try:
            # Инициализируем API и запускаем генерацию
            api = Text2ImageAPI(FUSIONBRAIN_API_KEY, FUSIONBRAIN_SECRET_KEY)
            
            width = user_settings[user_id].width
            height = user_settings[user_id].height
            style = user_settings[user_id].style
            
            logger.info(f"Параметры генерации: {width}x{height}, стиль: {style}", extra={
                'user_id': user_id,
                'operation': 'GENERATION_PARAMS'
            })

            # Получаем модель
            try:
                models = await api.get_model()
                if not models:
                    raise Exception("Список моделей пуст")
                model_id = models[0]["id"]
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
            
            if not uuid:
                raise Exception("Не удалось получить UUID для генерации")

            logger.info(f"Получен UUID генерации: {uuid}", extra={
                'user_id': user_id,
                'operation': 'GENERATION_UUID'
            })

            # Ожидаем результат
            max_attempts = 60  # Максимальное количество попыток
            for attempt in range(max_attempts):
                try:
                    response = await api.check_generation(uuid)
                    
                    if response.get("status") == "DONE":
                        images = response.get("images", [])
                        if not images:
                            raise Exception("Изображение не было сгенерировано")
                            
                        # Сохраняем изображение в памяти
                        image_data = base64.b64decode(images[0])
                        user_states[user_id].last_image = image_data
                        image_id = str(uuid_lib.uuid4())
                        user_states[user_id].last_image_id = image_id
                        
                        # Отправляем изображение
                        await message.answer_photo(
                            BufferedInputFile(
                                image_data,
                                filename=f"generation_{image_id}.png"
                            ),
                            caption=(
                                f"🎨 Стиль: <b>{IMAGE_STYLES[style]['label']}</b>\n"
                                f"📏 Размер: <b>{width}x{height}</b>\n"
                                f"💭 Промпт: <i>{message.text}</i>"
                            ),
                            reply_markup=get_image_keyboard(image_id),
                            parse_mode=ParseMode.HTML
                        )
                        
                        logger.info("Изображение успешно сгенерировано", extra={
                            'user_id': user_id,
                            'operation': 'GENERATION_SUCCESS'
                        })
                        break

                    await asyncio.sleep(1)
                        
                except Exception as e:
                    if "Generation still in progress" in str(e):
                        if attempt == max_attempts - 1:
                            raise Exception("Превышено время ожидания генерации")
                        await asyncio.sleep(1)
                        continue
                    raise e

            # Удаляем статусное сообщение
            await status_message.delete()
            
            # Сбрасываем состояние ожидания промпта
            user_states[user_id].awaiting_prompt = False

        except Exception as e:
            error_message = str(e)
            logger.error(f"Ошибка при генерации: {error_message}", extra={
                'user_id': user_id,
                'operation': 'GENERATION_ERROR'
            })
            
            # Преобразуем технические ошибки в понятные пользователю сообщения
            if "Generation still in progress" in error_message:
                error_message = "Генерация все еще выполняется. Пожалуйста, подождите."
            elif "Превышено время ожидания" in error_message:
                error_message = "Генерация заняла слишком много времени. Попробуйте еще раз."
            elif "авторизации" in error_message.lower():
                error_message = "Ошибка доступа к сервису. Обратитесь к администратору."
            
            await status_message.edit_text(
                Messages.ERROR_GEN.format(error=error_message),
                reply_markup=get_back_keyboard()
            )
            
    except Exception as e:
        logger.error(f"Критическая ошибка в generate_image: {str(e)}", extra={
            'user_id': user_id if 'user_id' in locals() else 'N/A',
            'operation': 'CRITICAL_ERROR'
        })
        if 'status_message' in locals():
            await status_message.edit_text(
                Messages.ERROR_CRITICAL,
                reply_markup=get_back_keyboard()
            )
        else:
            await message.answer(
                Messages.ERROR_CRITICAL,
                reply_markup=get_back_keyboard()
            )

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
