from aiogram import types
from aiogram.types import CallbackQuery, BufferedInputFile
import logging
import base64
import os
from typing import Optional

from ..models.user_state import UserState, UserSettings
from ..utils.image_processor import ImageProcessor
from ..constants.messages import Messages
from ..constants.bot_constants import IMAGE_STYLES
from ..api.fusion_brain import Text2ImageAPI, CensorshipError
from .command_handlers import (
    user_states,
    user_settings,
    get_prompt_keyboard,
    get_image_keyboard,
    get_main_keyboard
)

logger = logging.getLogger(__name__)

async def start_generation(callback_query: CallbackQuery):
    """Начинает процесс генерации изображения"""
    user_id = callback_query.from_user.id
    settings = user_settings[user_id]
    state = user_states[user_id]
    
    # Устанавливаем флаг ожидания промпта
    state.awaiting_prompt = True
    
    # Получаем текущий стиль
    style_data = IMAGE_STYLES[settings.style]
    
    await callback_query.message.edit_text(
        Messages.PROMPT.format(
            style=style_data["label"],
            size=f"{settings.width}x{settings.height}"
        ),
        reply_markup=get_prompt_keyboard(),
        parse_mode="HTML"
    )
    await callback_query.answer()

async def back_to_main(callback_query: CallbackQuery):
    """Возвращает в главное меню"""
    user_id = callback_query.from_user.id
    state = user_states[user_id]
    
    # Сбрасываем флаг ожидания промпта
    state.awaiting_prompt = False
    
    await callback_query.message.edit_text(
        Messages.MAIN_MENU,
        reply_markup=get_main_keyboard(),
        parse_mode="HTML"
    )
    await callback_query.answer()

async def show_styles(callback_query: CallbackQuery):
    """Показывает меню выбора стилей"""
    user_id = callback_query.from_user.id
    settings = user_settings[user_id]
    
    # Получаем текущий стиль
    style_data = IMAGE_STYLES[settings.style]
    
    await callback_query.message.edit_text(
        Messages.STYLES.format(style_label=style_data["label"]),
        reply_markup=get_styles_keyboard(),
        parse_mode="HTML"
    )
    await callback_query.answer()

async def process_style_change(callback_query: CallbackQuery):
    """Обработчик изменения стиля изображения"""
    user_id = callback_query.from_user.id
    style_key = callback_query.data.replace("style_", "")
    
    if style_key in IMAGE_STYLES:
        user_settings[user_id].style = style_key
        style_data = IMAGE_STYLES[style_key]
        
        await callback_query.message.edit_text(
            Messages.STYLE_CHANGED.format(style=style_data["label"]),
            reply_markup=get_main_keyboard(),
            parse_mode="HTML"
        )
    else:
        await callback_query.message.edit_text(
            "Ошибка: неверный стиль",
            reply_markup=get_main_keyboard(),
            parse_mode="HTML"
        )
    
    await callback_query.answer()

async def regenerate_image(callback_query: CallbackQuery):
    """Обработчик повторной генерации изображения"""
    user_id = callback_query.from_user.id
    state = user_states[user_id]
    
    if not state.last_prompt:
        await callback_query.answer("Нет сохраненного промпта для регенерации")
        return
    
    # Генерируем изображение с тем же промптом
    await generate_image_with_prompt(callback_query.message, state.last_prompt)
    await callback_query.answer()

async def check_generation_status(api: Text2ImageAPI, uuid: str, status_message: types.Message, user_id: int):
    """Проверяет статус генерации изображения"""
    try:
        while True:
            result = await api.check_generation(uuid)
            if result:
                # Декодируем base64 в байты
                image_data = base64.b64decode(result)
                
                # Сохраняем изображение в состоянии пользователя
                user_states[user_id].last_image = image_data
                user_states[user_id].last_image_id = uuid
                
                # Отправляем изображение
                await status_message.answer_photo(
                    BufferedInputFile(
                        image_data,
                        filename=f"generation_{uuid}.png"
                    ),
                    reply_markup=get_image_keyboard(uuid)
                )
                
                # Удаляем статусное сообщение
                await status_message.delete()
                return True
    except Exception as e:
        logger.error(f"Error checking generation status: {str(e)}")
        await status_message.edit_text(
            Messages.ERROR_GEN.format(error=str(e)),
            parse_mode="HTML"
        )
        return False

async def generate_image_with_prompt(message: types.Message, prompt: str):
    """Генерирует изображение на основе промпта"""
    user_id = message.from_user.id
    settings = user_settings[user_id]
    
    # Получаем текущий стиль и его префикс
    style_data = IMAGE_STYLES[settings.style]
    full_prompt = f"{style_data['prompt_prefix']}{prompt}"
    
    # Создаем API клиент
    api = Text2ImageAPI(
        api_key=os.getenv('API_KEY'),
        secret_key=os.getenv('SECRET_KEY')
    )
    
    try:
        # Отправляем сообщение о начале генерации
        status_message = await message.answer(
            Messages.GENERATING,
            parse_mode="HTML"
        )
        
        # Запускаем генерацию
        uuid = await api.generate(
            prompt=full_prompt,
            model_id=style_data["model_id"],
            width=settings.width,
            height=settings.height
        )
        
        # Сохраняем промпт
        user_states[user_id].last_prompt = prompt
        
        # Проверяем статус генерации
        await check_generation_status(api, uuid, status_message, user_id)
        
    except CensorshipError:
        await status_message.edit_text(
            "❌ Контент не прошел модерацию. Пожалуйста, измените описание.",
            parse_mode="HTML"
        )
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        await status_message.edit_text(
            Messages.ERROR_GEN.format(error=str(e)),
            parse_mode="HTML"
        )

async def generate_image(message: types.Message):
    """Генерирует изображение"""
    user_id = message.from_user.id
    state = user_states[user_id]
    
    # Проверяем, ожидаем ли мы промпт от пользователя
    if not state.awaiting_prompt:
        return
    
    # Сбрасываем флаг ожидания промпта
    state.awaiting_prompt = False
    
    # Генерируем изображение
    await generate_image_with_prompt(message, message.text)
