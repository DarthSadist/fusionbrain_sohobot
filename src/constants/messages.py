from typing import Final, Dict
from .bot_constants import EmojiEnum
from enum import Enum

class MessageKey(str, Enum):
    """Ключи для сообщений"""
    WELCOME = "welcome"
    HELP = "help"
    PROMPT = "prompt"
    GENERATING = "generating"
    ERROR = "error"
    SUCCESS = "success"
    SETTINGS = "settings"
    STYLE = "style"
    SIZE = "size"
    REMOVE_BG = "remove_bg"
    MAIN_MENU = "main_menu"
    CURRENT_SETTINGS = "current_settings"
    STYLE_CHANGED = "style_changed"
    SIZE_CHANGED = "size_changed"
    BG_REMOVED = "bg_removed"
    ERROR_GEN = "error_gen"
    ERROR_SIZE = "error_size"
    ERROR_CRITICAL = "error_critical"

class MessageTemplate:
    """Шаблоны сообщений с поддержкой форматирования и валидации"""
    
    _templates: Final[Dict[str, str]] = {
        MessageKey.WELCOME: f"""
🎨 <b>Добро пожаловать в FusionBrain Art Bot!</b>

Я помогу вам создавать удивительные изображения с помощью нейросети Kandinsky 2.2

<b>Мои возможности:</b>
✨ Создание изображений по описанию
🎭 Удаление фона с изображений
📏 Различные размеры (1024x1024, 1024x1536 и др.)
🎨 Множество художественных стилей

<b>Начать работу:</b>
1. Нажмите {EmojiEnum.CREATE} <b>Создать</b>
2. Выберите стиль и размер
3. Опишите желаемое изображение

Используйте {EmojiEnum.HELP} для получения подробной справки
""",
        MessageKey.HELP: f"""
<b>🎨 Подробное руководство по использованию бота</b>

<b>1. Создание изображения:</b>
• Нажмите {EmojiEnum.CREATE} <b>Создать</b>
• Выберите стиль изображения
• Опишите, что хотите увидеть
• Дождитесь результата

<b>2. Настройка параметров {EmojiEnum.SETTINGS}</b>
• Размер: выберите из доступных форматов
• Стиль: различные художественные стили

<b>3. Работа с результатом:</b>
• {EmojiEnum.REMOVE_BG} Удаление фона
• 🔄 Повторная генерация
• 💾 Сохранение изображения

<b>4. Советы по описанию:</b>
• Используйте детальные описания
• Указывайте стиль, цвета, настроение
• Пример: "Космический кит плывет среди звезд, неоновые цвета"
""",
        MessageKey.PROMPT: f"""
{EmojiEnum.EDIT} <b>Создание изображения</b>

🎨 Стиль: <b>{{style}}</b>
📏 Размер: <b>{{size}}</b>

✍️ Опишите желаемое изображение:
""",
        MessageKey.GENERATING: f"""
{EmojiEnum.WAIT} <b>Генерация изображения...</b>

Это может занять некоторое время.
Текущий стиль: <b>{{style}}</b>
""",
        MessageKey.ERROR: f"""
{EmojiEnum.ERROR} <b>Произошла ошибка</b>

{{error_message}}

Попробуйте еще раз или обратитесь к администратору.
""",
        MessageKey.SUCCESS: f"""
{EmojiEnum.SUCCESS} <b>Изображение готово!</b>

🎨 Стиль: <b>{{style}}</b>
📏 Размер: <b>{{size}}</b>
""",
        MessageKey.SETTINGS: f"""
{EmojiEnum.SETTINGS} <b>Настройки</b>

Текущие параметры:
📏 Размер: <b>{{size}}</b>
🎨 Стиль: <b>{{style}}</b>
""",
        MessageKey.STYLE: f"""
🎨 <b>Выбор стиля</b>

Текущий стиль: <b>{{style}}</b>
""",
        MessageKey.SIZE: f"""
📏 <b>Выбор размера</b>

Текущий размер: <b>{{size}}</b>
""",
        MessageKey.REMOVE_BG: f"""
{EmojiEnum.REMOVE_BG} <b>Удаление фона</b>

{{status}}
""",
        MessageKey.MAIN_MENU: "Выберите действие:",
        MessageKey.CURRENT_SETTINGS: f"""
{EmojiEnum.SETTINGS} <b>Текущие настройки</b>

🎨 Стиль: <b>{{style}}</b>
📏 Размер: <b>{{size}}</b>

{EmojiEnum.EDIT} Опишите желаемое изображение или измените настройки:
""",
        MessageKey.STYLE_CHANGED: f"{EmojiEnum.SUCCESS} Установлен стиль: <b>{{style}}</b>",
        MessageKey.SIZE_CHANGED: f"{EmojiEnum.SUCCESS} Установлен размер: <b>{{size}}</b>",
        MessageKey.BG_REMOVED: f"{EmojiEnum.SUCCESS} Фон успешно удален!",
        MessageKey.ERROR_GEN: f"{EmojiEnum.ERROR} Ошибка при генерации: {{error}}",
        MessageKey.ERROR_SIZE: f"{EmojiEnum.ERROR} Ошибка: неверный размер",
        MessageKey.ERROR_CRITICAL: f"{EmojiEnum.ERROR} Произошла критическая ошибка"
    }

    @classmethod
    def get(cls, key: MessageKey, **kwargs) -> str:
        """
        Получение отформатированного сообщения по ключу
        
        Args:
            key: Ключ сообщения
            **kwargs: Параметры для форматирования
            
        Returns:
            str: Отформатированное сообщение
            
        Raises:
            KeyError: Если ключ не найден
            ValueError: Если не хватает параметров для форматирования
        """
        if key not in cls._templates:
            raise KeyError(f"Message template not found: {key}")
            
        template = cls._templates[key]
        try:
            return template.format(**kwargs) if kwargs else template
        except KeyError as e:
            raise ValueError(f"Missing required parameter for message {key}: {e}")

    @classmethod
    def validate_all_templates(cls):
        """Проверка всех шаблонов на корректность"""
        for key in MessageKey:
            if key not in cls._templates:
                raise ValueError(f"Missing template for key: {key}")
            # Проверяем базовое форматирование
            try:
                cls._templates[key].format()
            except Exception as e:
                raise ValueError(f"Invalid template format for key {key}: {e}")

# Проверяем все шаблоны при импорте модуля
MessageTemplate.validate_all_templates()
