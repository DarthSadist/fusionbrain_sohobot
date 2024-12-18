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

# Регистрируем обработчики
@dp.message(Command("start"))
async def send_welcome(message: types.Message):
    welcome_text = (
        "Привет! Я бот для генерации изображений.\n\n"
        "Просто отправь мне текстовое описание того, что ты хочешь увидеть на картинке, "
        "и я постараюсь это сгенерировать.\n\n"
        "Для получения справки используй команду /help"
    )
    await message.reply(welcome_text)

@dp.message(Command("help"))
async def send_help(message: types.Message):
    help_text = (
        "🎨 Как использовать бота:\n\n"
        "1. Отправьте текстовое описание желаемого изображения\n"
        "2. Дождитесь генерации\n"
        "3. Используйте кнопки под изображением для дополнительных действий\n\n"
        "Доступные команды:\n"
        "/start - Начать работу с ботом\n"
        "/help - Показать это сообщение"
    )
    await message.reply(help_text)

# Обработчик для всех текстовых сообщений, кроме команд
@dp.message(lambda message: message.text and not message.text.startswith('/'))
async def generate_image(message: types.Message):
    try:
        logger.info(f"Received text message: {message.text}")
        
        if len(message.text) > Text2ImageAPI.MAX_PROMPT_LENGTH:
            await message.reply(f"⚠️ Ваш запрос слишком длинный ({len(message.text)} символов). Он будет сокращен до {Text2ImageAPI.MAX_PROMPT_LENGTH} символов.")
        
        # Отправляем сообщение о начале генерации
        progress_message = await message.reply("🎨 Начинаю генерацию изображения...")
        
        # Инициализируем API
        api = Text2ImageAPI('https://api-key.fusionbrain.ai', FUSIONBRAIN_API_KEY, FUSIONBRAIN_SECRET_KEY)
        
        # Получаем список доступных моделей
        try:
            models = api.get_model()
            logger.info(f"Available models: {models}")
            model_id = models[0]['id']  # Используем ID первой доступной модели
        except Exception as e:
            logger.error(f"Failed to get models: {str(e)}")
            model_id = 4  # Используем известный ID модели Kandinsky 3.1
        
        # Запускаем генерацию
        request_id = api.generate(message.text, model=model_id)
        logger.info(f"Generation started with request_id: {request_id}")
        
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
                        await message.reply_photo(photo, caption="✨ Ваше изображение готово!")
                        await progress_message.delete()
                        break
                    else:
                        await progress_message.edit_text("❌ Изображение не было сгенерировано")
                        break
                elif status.get('status') == 'FAILED':
                    await progress_message.edit_text("❌ Произошла ошибка при генерации изображения")
                    break
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error checking generation status: {str(e)}")
                await progress_message.edit_text("❌ Произошла ошибка при проверке статуса генерации")
                break
            
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        await message.reply("❌ Произошла ошибка при генерации изображения")

async def main():
    # Запуск бота
    await dp.start_polling(bot, skip_updates=True)

if __name__ == '__main__':
    asyncio.run(main())
