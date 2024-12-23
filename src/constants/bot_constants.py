from dataclasses import dataclass
from typing import Dict, Any, Final
from enum import Enum, auto

class EmojiEnum(str, Enum):
    """Эмодзи в виде перечисления для более безопасного использования"""
    SETTINGS = "⚙️"
    CREATE = "🎨"
    BACK = "◀️"
    HELP = "❓"
    REMOVE_BG = "🎭"
    WAIT = "⏳"
    ERROR = "❌"
    SUCCESS = "✅"
    EDIT = "✏️"

class CallbackEnum(str, Enum):
    """Колбэки в виде перечисления для более безопасного использования"""
    SETTINGS = "settings"
    GENERATE = "generate"
    SIZE_PREFIX = "size_"
    HELP = "help"
    BACK = "back_to_main"
    REMOVE_BG = "remove_bg"
    STYLES = "styles"
    STYLE_PREFIX = "style_"
    REGENERATE = "regenerate"

# Константы для размеров изображений
class ImageSize:
    """Константы для размеров изображений"""
    MIN_SIZE: Final[int] = 64
    MAX_SIZE: Final[int] = 2048
    DEFAULT_SIZE: Final[int] = 1024

# Размеры изображений с валидацией
IMAGE_SIZES: Final[Dict[str, Dict[str, Any]]] = {
    "square": {
        "width": 1024,
        "height": 1024,
        "label": "1024x1024 (квадрат)"
    },
    "portrait": {
        "width": 1024,
        "height": 1536,
        "label": "1024x1536 (портрет)"
    },
    "landscape": {
        "width": 1536,
        "height": 1024,
        "label": "1536x1024 (пейзаж)"
    }
}

# Проверка корректности размеров
for size_config in IMAGE_SIZES.values():
    if not (ImageSize.MIN_SIZE <= size_config["width"] <= ImageSize.MAX_SIZE and
            ImageSize.MIN_SIZE <= size_config["height"] <= ImageSize.MAX_SIZE):
        raise ValueError(f"Invalid image size configuration: {size_config}")

class StyleType(Enum):
    """Типы стилей для изображений"""
    DEFAULT = auto()
    ANIME = auto()
    REALISTIC = auto()
    ARTISTIC = auto()
    RETRO = auto()

# Стили изображений с валидацией
IMAGE_STYLES: Final[Dict[str, Dict[str, Any]]] = {
    StyleType.DEFAULT.name: {
        "label": "Обычный",
        "prompt_prefix": "",
        "description": "Стандартный стиль без дополнительных модификаций",
        "model_id": 1
    },
    StyleType.ANIME.name: {
        "label": "Аниме",
        "prompt_prefix": "anime style, manga, japanese animation, ",
        "description": "Аниме стиль",
        "model_id": 1
    },
    StyleType.REALISTIC.name: {
        "label": "Реалистичный",
        "prompt_prefix": "photorealistic, highly detailed, sharp focus, ",
        "description": "Реалистичный стиль",
        "model_id": 1
    },
    StyleType.ARTISTIC.name: {
        "label": "Художественный",
        "prompt_prefix": "artistic style, creative, expressive, ",
        "description": "Художественный стиль",
        "model_id": 1
    },
    StyleType.RETRO.name: {
        "label": "Ретро",
        "prompt_prefix": "retro style, vintage aesthetics, old school design, nostalgic feel, ",
        "description": "Ретро стиль",
        "model_id": 1
    }
}

# Проверка корректности стилей
required_style_keys = {"label", "prompt_prefix", "description", "model_id"}
for style_name, style_config in IMAGE_STYLES.items():
    if not all(key in style_config for key in required_style_keys):
        raise ValueError(f"Invalid style configuration for {style_name}")
    if not isinstance(style_config["model_id"], int):
        raise ValueError(f"Invalid model_id for style {style_name}")

# Константы для API
class APIConstants:
    """Константы для работы с API"""
    MAX_RETRIES: Final[int] = 3
    TIMEOUT: Final[int] = 30
    MAX_PROMPT_LENGTH: Final[int] = 500
    BASE_URL: Final[str] = "https://api-key.fusionbrain.ai"

# Константы для обработки изображений
class ImageProcessingConstants:
    """Константы для обработки изображений"""
    MAX_IMAGE_SIZE: Final[int] = 1500
    SUPPORTED_FORMATS: Final[tuple] = ("PNG", "JPEG", "JPG", "WEBP")
    MAX_FILE_SIZE: Final[int] = 10 * 1024 * 1024  # 10MB
