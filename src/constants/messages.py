from typing import Final, Dict
from enum import Enum

class MessageKey(str, Enum):
    """Ключи для сообщений"""
    WELCOME = "welcome"
    HELP = "help"
    PROMPT = "prompt"
    GENERATING = "generating"
    REMOVING_BG = "removing_bg"
    REMOVE_BG_SUCCESS = "remove_bg_success"
    REMOVE_BG_ERROR = "remove_bg_error"
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
    IMAGE_INFO = "image_info"

class MessageTemplate:
    """Шаблоны сообщений с поддержкой форматирования и валидации"""
    
    _templates: Final[Dict[str, str]] = {
        MessageKey.WELCOME: """
🎨 <b>Добро пожаловать в FusionBrain Art Bot!</b>

Я помогу вам создавать удивительные изображения с помощью нейросети Kandinsky 2.2

<b>Мои возможности:</b>
✨ Создание изображений по описанию
🎭 Удаление фона с изображений
📏 Различные размеры (1024x1024, 1024x1536 и др.)
🎨 Множество художественных стилей

<b>Начать работу:</b>
1. Нажмите 🎨 <b>Создать</b>
2. Выберите стиль и размер
3. Опишите желаемое изображение

Используйте ❓ для получения подробной справки
""",
        MessageKey.HELP: """
<b>🎨 Подробное руководство по использованию бота</b>

<b>1. Создание изображения:</b>
• Нажмите 🎨 <b>Создать</b>
• Выберите стиль изображения
• Опишите, что хотите увидеть
• Дождитесь результата

<b>2. Настройка параметров ⚙️</b>
• Размер: выберите из доступных форматов
• Стиль: различные художественные стили

<b>3. Работа с результатом:</b>
• 🖼 Удаление фона
• 🔄 Повторная генерация
• 💾 Сохранение изображения

<b>4. Советы по описанию:</b>
• Используйте детальные описания
• Указывайте стиль, цвета, настроение
• Пример: "Космический кит плывет среди звезд, неоновые цвета"
""",
        MessageKey.PROMPT: """
✍️ <b>Создание изображения</b>

🎨 Стиль: <b>{{style}}</b>
📏 Размер: <b>{{size}}</b>

✍️ Опишите желаемое изображение:
""",
        MessageKey.GENERATING: """
⏳ <b>Генерация изображения...</b>

Это может занять некоторое время.
🎨 Стиль: <b>{style}</b>
""",
        MessageKey.REMOVING_BG: """
⏳ <b>Удаление фона...</b>

Это может занять некоторое время.
""",
        MessageKey.REMOVE_BG_SUCCESS: """
✅ <b>Фон успешно удален!</b>
""",
        MessageKey.REMOVE_BG_ERROR: """
❌ <b>Ошибка при удалении фона</b>

{{error_message}}

Попробуйте еще раз или обратитесь к администратору.
""",
        MessageKey.ERROR: """
❌ <b>Произошла ошибка</b>

{{error_message}}

Попробуйте еще раз или обратитесь к администратору.
""",
        MessageKey.SUCCESS: """
✅ <b>Изображение готово!</b>

<b>📝 Информация о генерации:</b>
🎨 Стиль: <b>{{style}}</b>
📏 Размер: <b>{{size}}</b>
⚡️ Время генерации: <b>{{generation_time}}</b>

<b>✍️ Использованный промпт:</b>
{{prompt}}

<b>🔧 Технические детали:</b>
📊 ID модели: <b>{{model_id}}</b>
🔍 ID изображения: <code>{{image_id}}</code>
""",
        MessageKey.SETTINGS: """
⚙️ <b>Настройки</b>

Текущие параметры:
📏 Размер: <b>{{size}}</b>
🎨 Стиль: <b>{{style}}</b>
""",
        MessageKey.STYLE: """
🎨 <b>Выбор стиля</b>

Текущий стиль: <b>{{style}}</b>
""",
        MessageKey.SIZE: """
📏 <b>Выбор размера</b>

Текущий размер: <b>{{size}}</b>
""",
        MessageKey.REMOVE_BG: """
🖼 <b>Удаление фона</b>

{{status}}
""",
        MessageKey.MAIN_MENU: "Выберите действие:",
        MessageKey.CURRENT_SETTINGS: """
⚙️ <b>Текущие настройки</b>

🎨 Стиль: <b>{style}</b>
📏 Размер: <b>{size}</b>

✍️ Опишите желаемое изображение или измените настройки:
""",
        MessageKey.STYLE_CHANGED: """
✅ <b>Стиль изменен</b>

Текущий стиль: <b>{style}</b>
""",
        MessageKey.SIZE_CHANGED: """
✅ <b>Размер изменен</b>

Текущий размер: <b>{size}</b>
""",
        MessageKey.BG_REMOVED: "✅ Фон успешно удален!",
        MessageKey.ERROR_GEN: """
❌ <b>Ошибка при генерации изображения</b>

{error}

Попробуйте:
• Изменить описание
• Выбрать другой стиль
• Уменьшить размер изображения
""",
        MessageKey.ERROR_SIZE: "❌ Ошибка: неверный размер",
        MessageKey.ERROR_CRITICAL: "❌ Произошла критическая ошибка",
        MessageKey.IMAGE_INFO: """
<b>📝 Информация об изображении</b>

<b>Основные параметры:</b>
🎨 Стиль: <b>{style}</b>
📏 Размер: <b>{size}</b>
⚡️ Время генерации: <b>{generation_time}</b>
{bg_removal_info}

<b>Использованный промпт:</b>
✍️ <code>{prompt}</code>

<b>Префикс стиля:</b>
🎭 <code>{style_prompt}</code>

<b>Технические детали:</b>
📊 ID модели: <b>{model_id}</b>
🔍 ID изображения: <code>{image_id}</code>
⏰ Создано: <b>{created_at}</b>
""",
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
    def get_image_info(cls, image_info) -> str:
        """
        Форматирование информации об изображении
        
        Args:
            image_info: Объект ImageInfo
            
        Returns:
            str: Отформатированное сообщение с информацией
        """
        bg_removal_info = ""
        if image_info.has_removed_bg:
            bg_time = image_info.get_bg_removal_time_str()
            bg_removal_info = f"\n⚡️ Время удаления фона: <b>{bg_time}</b>"

        return cls.get(
            MessageKey.IMAGE_INFO,
            style=image_info.style,
            size=image_info.get_size_str(),
            generation_time=image_info.get_generation_time_str(),
            bg_removal_info=bg_removal_info,
            prompt=image_info.prompt,
            style_prompt=image_info.style_prompt,
            model_id=image_info.model_id,
            image_id=image_info.id,
            created_at=image_info.created_at.strftime("%Y-%m-%d %H:%M:%S")
        )

    @classmethod
    def validate_all_templates(cls):
        """Проверка всех шаблонов на корректность"""
        for key in MessageKey:
            if key not in cls._templates:
                raise ValueError(f"Missing template for key: {key}")
            # Проверяем только наличие шаблона и его базовую структуру
            template = cls._templates[key]
            if not isinstance(template, str):
                raise ValueError(f"Template for key {key} must be a string")

# Проверяем все шаблоны при импорте модуля
MessageTemplate.validate_all_templates()
