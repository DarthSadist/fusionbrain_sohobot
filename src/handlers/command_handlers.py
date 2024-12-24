import logging
from aiogram import types
from aiogram.types import CallbackQuery, FSInputFile, BufferedInputFile
from aiogram.utils.keyboard import InlineKeyboardBuilder
import os
from collections import defaultdict
import uuid as uuid_lib

from ..models.user_state import UserState, UserSettings
from ..utils.image_processor import ImageProcessor
from ..constants.messages import Messages
from ..constants.bot_constants import Emoji, CallbackData, IMAGE_SIZES, IMAGE_STYLES
from ..api.fusion_brain import Text2ImageAPI, CensorshipError

# Словари для хранения состояний и настроек пользователей
user_states = defaultdict(UserState)
user_settings = defaultdict(UserSettings)

# Настройка логгера
logger = logging.getLogger(__name__)

async def send_welcome(message: types.Message):
    """Обработчик команды /start"""
    keyboard = get_main_keyboard()
    await message.answer(Messages.WELCOME, reply_markup=keyboard, parse_mode="HTML")

async def show_help(callback_query: CallbackQuery):
    """Обработчик кнопки помощи"""
    keyboard = get_back_keyboard()
    await callback_query.message.edit_text(
        Messages.HELP,
        reply_markup=keyboard,
        parse_mode="HTML"
    )
    await callback_query.answer()

async def show_settings(callback_query: CallbackQuery):
    """Обработчик кнопки настроек"""
    user_id = callback_query.from_user.id
    settings = user_settings[user_id]
    keyboard = get_settings_keyboard()
    
    await callback_query.message.edit_text(
        Messages.SETTINGS.format(
            width=settings.width,
            height=settings.height
        ),
        reply_markup=keyboard,
        parse_mode="HTML"
    )
    await callback_query.answer()

async def process_size_change(callback_query: CallbackQuery):
    """Обработчик изменения размера изображения"""
    user_id = callback_query.from_user.id
    size_key = callback_query.data.replace(CallbackData.SIZE_PREFIX, "")
    
    if size_key in IMAGE_SIZES:
        size_data = IMAGE_SIZES[size_key]
        user_settings[user_id].width = size_data["width"]
        user_settings[user_id].height = size_data["height"]
        
        await callback_query.message.edit_text(
            Messages.SIZE_CHANGED.format(size=size_data["label"]),
            reply_markup=get_back_keyboard(),
            parse_mode="HTML"
        )
    else:
        await callback_query.message.edit_text(
            Messages.ERROR_SIZE,
            reply_markup=get_back_keyboard(),
            parse_mode="HTML"
        )
    
    await callback_query.answer()

async def process_remove_background(callback_query: CallbackQuery):
    """Обработчик удаления фона с изображения"""
    user_id = callback_query.from_user.id
    logger = logging.getLogger(__name__)
    logger.info(f"[USER_ID:{user_id}] Начало обработки запроса на удаление фона")
    
    try:
        # Получаем состояние пользователя
        if user_id not in user_states:
            logger.error(f"[USER_ID:{user_id}] Состояние пользователя не найдено")
            await callback_query.answer("Ошибка: состояние пользователя не найдено")
            return
            
        state = user_states[user_id]
        logger.info(f"[USER_ID:{user_id}] Состояние пользователя получено")
        
        # Проверяем наличие изображения
        if not state.last_image:
            logger.warning(f"[USER_ID:{user_id}] Нет изображения для обработки")
            await callback_query.answer("Нет изображения для обработки")
            return
            
        logger.info(f"[USER_ID:{user_id}] Найдено изображение размером {len(state.last_image)} байт")
        
        # Отправляем сообщение о начале обработки
        try:
            status_message = await callback_query.message.edit_text(
                Messages.REMOVING_BG,
                parse_mode="HTML"
            )
            logger.info(f"[USER_ID:{user_id}] Отправлено сообщение о начале обработки")
        except Exception as e:
            logger.error(f"[USER_ID:{user_id}] Ошибка при отправке статусного сообщения: {str(e)}")
            return
            
        try:
            # Удаляем фон
            logger.info(f"[USER_ID:{user_id}] Начинаем процесс удаления фона")
            result = await ImageProcessor.remove_background(state.last_image)
            logger.info(f"[USER_ID:{user_id}] Фон успешно удален")
            
            # Отправляем результат
            await callback_query.message.answer_photo(
                BufferedInputFile(
                    result,
                    filename=f"no_bg_{uuid_lib.uuid4()}.png"
                ),
                caption=Messages.BG_REMOVED,
                reply_markup=get_image_keyboard(str(uuid_lib.uuid4()))
            )
            logger.info(f"[USER_ID:{user_id}] Результат успешно отправлен")
            
            await status_message.delete()
            
        except Exception as e:
            error_msg = f"Ошибка при удалении фона: {str(e)}"
            logger.error(f"[USER_ID:{user_id}] {error_msg}", exc_info=True)
            await status_message.edit_text(
                Messages.BG_REMOVE_ERROR,
                parse_mode="HTML"
            )
            
    except Exception as e:
        error_msg = f"Неожиданная ошибка в обработчике удаления фона: {str(e)}"
        logger.error(f"[USER_ID:{user_id}] {error_msg}", exc_info=True)
        try:
            await callback_query.message.edit_text(
                Messages.UNEXPECTED_ERROR,
                parse_mode="HTML"
            )
        except Exception as edit_error:
            logger.error(f"[USER_ID:{user_id}] Ошибка при отправке сообщения об ошибке: {str(edit_error)}")
    
    await callback_query.answer()

def get_main_keyboard():
    """Создает основную клавиатуру с главным меню"""
    builder = InlineKeyboardBuilder()
    builder.button(text=f"{Emoji.CREATE} Создать", callback_data=CallbackData.GENERATE)
    builder.button(text=f"{Emoji.SETTINGS} Настройки", callback_data=CallbackData.SETTINGS)
    builder.button(text=f"{Emoji.HELP} Помощь", callback_data=CallbackData.HELP)
    builder.adjust(2, 1)
    return builder.as_markup()

def get_settings_keyboard():
    """Создает клавиатуру с настройками размеров"""
    builder = InlineKeyboardBuilder()
    
    for size_key, size_data in IMAGE_SIZES.items():
        builder.button(
            text=size_data["label"],
            callback_data=f"{CallbackData.SIZE_PREFIX}{size_key}"
        )
    
    builder.button(text=f"{Emoji.BACK} Назад", callback_data=CallbackData.BACK)
    builder.adjust(1)
    return builder.as_markup()

def get_styles_keyboard():
    """Создает клавиатуру с выбором стилей"""
    builder = InlineKeyboardBuilder()
    
    for style_key, style_data in IMAGE_STYLES.items():
        builder.button(
            text=style_data["label"],
            callback_data=f"{CallbackData.STYLE_PREFIX}{style_key}"
        )
    
    builder.button(text=f"{Emoji.BACK} Назад", callback_data=CallbackData.BACK)
    builder.adjust(2)
    return builder.as_markup()

def get_prompt_keyboard():
    """Создает клавиатуру для режима ввода промпта"""
    builder = InlineKeyboardBuilder()
    builder.button(text=f"{Emoji.SETTINGS} Размер", callback_data=CallbackData.SETTINGS)
    builder.button(text=f"{Emoji.BACK} Назад", callback_data=CallbackData.BACK)
    builder.adjust(2)
    return builder.as_markup()

def get_back_keyboard():
    """Создает клавиатуру для возврата"""
    builder = InlineKeyboardBuilder()
    builder.button(text=f"{Emoji.BACK} Назад", callback_data=CallbackData.BACK)
    return builder.as_markup()

def get_image_keyboard(image_id: str):
    """Создает клавиатуру для изображения"""
    builder = InlineKeyboardBuilder()
    builder.button(text=f"{Emoji.REMOVE_BG} Удалить фон", callback_data=CallbackData.REMOVE_BG)
    builder.button(text="🔄 Повторить", callback_data=CallbackData.REGENERATE)
    builder.button(text=f"{Emoji.BACK} В меню", callback_data=CallbackData.BACK)
    builder.adjust(2, 1)
    return builder.as_markup()
