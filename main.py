import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import logging
import base64
import json
import time
import requests
import asyncio
import os
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types, F
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.client.default import DefaultBotProperties
import asyncio
import io
import uuid as uuid_lib
from PIL import Image, ImageEnhance, ImageFilter
from rembg import remove
from collections import defaultdict
from aiogram.utils.keyboard import InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.types import CallbackQuery

# Загрузка переменных окружения из файла .env
load_dotenv()

# Конфигурация
API_TOKEN = os.getenv('API_TOKEN')
FUSIONBRAIN_API_KEY = os.getenv('FUSIONBRAIN_API_KEY')
FUSIONBRAIN_SECRET_KEY = os.getenv('FUSIONBRAIN_SECRET_KEY')

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

    def generate(self, prompt, model, images=1, width=1024, height=1024):
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

# Константы для callback-данных
class CallbackData:
    SETTINGS = "settings"
    GENERATE = "generate"
    SIZE_PREFIX = "size_"
    HELP = "help"
    BACK = "back_to_main"

def get_main_keyboard() -> InlineKeyboardMarkup:
    """Создает основную клавиатуру с главным меню"""
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🎨 Сгенерировать изображение", callback_data=CallbackData.GENERATE)],
        [InlineKeyboardButton(text="⚙️ Настройки", callback_data=CallbackData.SETTINGS)],
        [InlineKeyboardButton(text="❓ Помощь", callback_data=CallbackData.HELP)]
    ])

def get_settings_keyboard() -> InlineKeyboardMarkup:
    """Создает клавиатуру с настройками размеров"""
    keyboard = [
        [InlineKeyboardButton(text=size_info["label"], callback_data=f"{CallbackData.SIZE_PREFIX}{size_key}")]
        for size_key, size_info in IMAGE_SIZES.items()
    ]
    keyboard.append([InlineKeyboardButton(text="◀️ Назад", callback_data=CallbackData.BACK)])
    return InlineKeyboardMarkup(inline_keyboard=keyboard)

# Состояния пользователя
class UserState:
    def __init__(self):
        self.width = 1024
        self.height = 1024
        self.awaiting_prompt = False

user_states = defaultdict(UserState)

# Словарь для хранения пользовательских настроек
class UserSettings:
    def __init__(self):
        self.width = 1024
        self.height = 1024

user_settings = defaultdict(UserSettings)

# Доступные размеры изображений
IMAGE_SIZES = {
    "square_small": {"width": 512, "height": 512, "label": "512x512"},
    "square_medium": {"width": 768, "height": 768, "label": "768x768"},
    "square_large": {"width": 1024, "height": 1024, "label": "1024x1024"},
    "wide": {"width": 1024, "height": 576, "label": "1024x576 (Wide)"},
    "tall": {"width": 576, "height": 1024, "label": "576x1024 (Tall)"}
}

# Регистрируем обработчики
@dp.message(Command("start"))
async def send_welcome(message: types.Message):
    welcome_text = (
        "Привет! Я бот для генерации изображений.\n\n"
        "Используйте кнопки ниже для управления:"
    )
    await message.reply(welcome_text, reply_markup=get_main_keyboard())

@dp.callback_query(lambda c: c.data == CallbackData.HELP)
async def show_help(callback_query: CallbackQuery):
    settings = user_states[callback_query.from_user.id]
    help_text = (
        "🎨 Как использовать бота:\n\n"
        "1. Нажмите кнопку '🎨 Сгенерировать изображение'\n"
        "2. Отправьте текстовое описание желаемого изображения\n"
        "3. Дождитесь генерации\n\n"
        "📐 Текущие настройки:\n"
        f"Размер изображения: {settings.width}x{settings.height}"
    )
    await callback_query.message.edit_text(help_text, reply_markup=get_main_keyboard())
    await callback_query.answer()

@dp.callback_query(lambda c: c.data == CallbackData.SETTINGS)
async def show_settings(callback_query: CallbackQuery):
    settings = user_states[callback_query.from_user.id]
    await callback_query.message.edit_text(
        f"📐 Текущий размер: {settings.width}x{settings.height}\n\n"
        "Выберите новый размер изображения:",
        reply_markup=get_settings_keyboard()
    )
    await callback_query.answer()

@dp.callback_query(lambda c: c.data.startswith(CallbackData.SIZE_PREFIX))
async def process_size_change(callback_query: CallbackQuery):
    size_key = callback_query.data.replace(CallbackData.SIZE_PREFIX, '')
    if size_key in IMAGE_SIZES:
        user_id = callback_query.from_user.id
        user_states[user_id].width = IMAGE_SIZES[size_key]["width"]
        user_states[user_id].height = IMAGE_SIZES[size_key]["height"]
        
        await callback_query.message.edit_text(
            f"✅ Размер изображения установлен: {IMAGE_SIZES[size_key]['label']}\n\n"
            "Вернитесь в главное меню:",
            reply_markup=get_main_keyboard()
        )
    await callback_query.answer()

@dp.callback_query(lambda c: c.data == CallbackData.BACK)
async def back_to_main(callback_query: CallbackQuery):
    await callback_query.message.edit_text(
        "Выберите действие:",
        reply_markup=get_main_keyboard()
    )
    await callback_query.answer()

@dp.callback_query(lambda c: c.data == CallbackData.GENERATE)
async def start_generation(callback_query: CallbackQuery):
    user_id = callback_query.from_user.id
    user_states[user_id].awaiting_prompt = True
    await callback_query.message.edit_text(
        "✏️ Отправьте текстовое описание желаемого изображения:",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="◀️ Назад", callback_data=CallbackData.BACK)]
        ])
    )
    await callback_query.answer()

@dp.message(lambda message: message.text and not message.text.startswith('/'))
async def generate_image(message: types.Message):
    user_id = message.from_user.id
    user_state = user_states[user_id]

    # Проверяем, ожидаем ли мы промпт от пользователя
    if not user_state.awaiting_prompt:
        await message.reply(
            "Пожалуйста, нажмите кнопку '🎨 Сгенерировать изображение' для начала:",
            reply_markup=get_main_keyboard()
        )
        return

    try:
        logger.info(f"Received text message: {message.text}")
        
        if len(message.text) > Text2ImageAPI.MAX_PROMPT_LENGTH:
            await message.reply(
                f"⚠️ Ваш запрос слишком длинный ({len(message.text)} символов). "
                f"Он будет сокращен до {Text2ImageAPI.MAX_PROMPT_LENGTH} символов."
            )
        
        # Отправляем сообщение о начале генерации
        progress_message = await message.reply(
            "🎨 Начинаю генерацию изображения...",
            reply_markup=InlineKeyboardMarkup(inline_keyboard=[])
        )
        
        # Инициализируем API
        api = Text2ImageAPI('https://api-key.fusionbrain.ai', FUSIONBRAIN_API_KEY, FUSIONBRAIN_SECRET_KEY)
        
        try:
            models = api.get_model()
            logger.info(f"Available models: {models}")
            model_id = models[0]['id']
        except Exception as e:
            logger.error(f"Failed to get models: {str(e)}")
            model_id = 4  # Используем известный ID модели Kandinsky 3.1
        
        # Запускаем генерацию с настройками пользователя
        request_id = api.generate(
            message.text,
            model=model_id,
            width=user_state.width,
            height=user_state.height
        )
        logger.info(f"Generation started with request_id: {request_id}")
        
        # Сбрасываем флаг ожидания промпта
        user_state.awaiting_prompt = False
        
        # Проверяем статус генерации
        while True:
            try:
                status = api.check_generation(request_id)
                logger.info(f"Generation status: {status}")
                
                if status.get('status') == 'DONE':
                    images = status.get('images', [])
                    if images:
                        # Отправляем изображение
                        image_data = base64.b64decode(images[0])
                        photo = types.BufferedInputFile(image_data, filename='generated_image.png')
                        await message.reply_photo(
                            photo,
                            caption="✨ Ваше изображение готово!",
                            reply_markup=get_main_keyboard()
                        )
                        await progress_message.delete()
                        break
                    else:
                        await progress_message.edit_text(
                            "❌ Изображение не было сгенерировано",
                            reply_markup=get_main_keyboard()
                        )
                        break
                elif status.get('status') == 'FAILED':
                    await progress_message.edit_text(
                        "❌ Произошла ошибка при генерации изображения",
                        reply_markup=get_main_keyboard()
                    )
                    break
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error checking generation status: {str(e)}")
                await progress_message.edit_text(
                    "❌ Произошла ошибка при проверке статуса генерации",
                    reply_markup=get_main_keyboard()
                )
                break
            
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        await message.reply(
            "❌ Произошла ошибка при генерации изображения",
            reply_markup=get_main_keyboard()
        )

async def main():
    # Запуск бота
    await dp.start_polling(bot, skip_updates=True)

if __name__ == '__main__':
    asyncio.run(main())
