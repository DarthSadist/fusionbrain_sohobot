from typing import Optional, Dict, Any
import logging
from datetime import datetime
from functools import wraps
import traceback
import asyncio
from aiogram.types import Message, CallbackQuery

logger = logging.getLogger(__name__)

class BotError(Exception):
    """Базовый класс для ошибок бота"""
    def __init__(self, message: str, user_message: Optional[str] = None):
        super().__init__(message)
        self.user_message = user_message or message

class APIError(BotError):
    """Ошибка при работе с API"""
    pass

class ValidationError(BotError):
    """Ошибка валидации данных"""
    pass

class ProcessingError(BotError):
    """Ошибка обработки данных"""
    pass

class ErrorTracker:
    """Класс для отслеживания и анализа ошибок"""
    _errors: Dict[str, Dict[str, Any]] = {}
    MAX_ERRORS = 1000
    
    @classmethod
    def track_error(cls, error: Exception, context: Dict[str, Any]):
        """Отслеживание ошибки"""
        error_id = str(datetime.now().timestamp())
        cls._errors[error_id] = {
            'type': type(error).__name__,
            'message': str(error),
            'traceback': traceback.format_exc(),
            'context': context,
            'timestamp': datetime.now()
        }
        
        # Очистка старых ошибок
        if len(cls._errors) > cls.MAX_ERRORS:
            oldest_key = min(cls._errors.keys())
            cls._errors.pop(oldest_key)
    
    @classmethod
    def get_error_stats(cls) -> Dict[str, int]:
        """Получение статистики по ошибкам"""
        stats = {}
        for error in cls._errors.values():
            error_type = error['type']
            stats[error_type] = stats.get(error_type, 0) + 1
        return stats

def handle_error(func):
    """Декоратор для обработки ошибок в хендлерах"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            # Определяем контекст ошибки
            context = {
                'function': func.__name__,
                'args': str(args),
                'kwargs': str(kwargs)
            }
            
            # Получаем объект сообщения или колбэка
            message = None
            for arg in args:
                if isinstance(arg, (Message, CallbackQuery)):
                    message = arg
                    context['user_id'] = message.from_user.id
                    break
            
            # Логируем ошибку
            logger.error(
                f"Error in {func.__name__}: {str(e)}",
                extra={
                    'user_id': context.get('user_id', 'unknown'),
                    'operation': func.__name__
                },
                exc_info=True
            )
            
            # Отслеживаем ошибку
            ErrorTracker.track_error(e, context)
            
            # Формируем сообщение для пользователя
            user_message = "Произошла ошибка. Попробуйте позже."
            if isinstance(e, BotError):
                user_message = e.user_message
            
            # Отправляем сообщение пользователю
            if message:
                if isinstance(message, CallbackQuery):
                    await message.answer(user_message)
                else:
                    await message.reply(user_message)
            
            # Пробрасываем ошибку дальше
            raise
    
    return wrapper

async def retry_with_backoff(func, max_retries: int = 3, initial_delay: float = 1.0):
    """Повторение операции с экспоненциальной задержкой"""
    delay = initial_delay
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return await func()
        except Exception as e:
            last_exception = e
            if attempt == max_retries - 1:
                raise
            
            logger.warning(
                f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}. Retrying in {delay} seconds..."
            )
            await asyncio.sleep(delay)
            delay *= 2
    
    raise last_exception
