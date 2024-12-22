from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class Emoji:
    """Константы для эмодзи"""
    SETTINGS = "⚙️"
    CREATE = "🎨"
    BACK = "◀️"
    HELP = "❓"
    REMOVE_BG = "🎭"
    WAIT = "⏳"
    ERROR = "❌"
    SUCCESS = "✅"
    EDIT = "✏️"

@dataclass
class CallbackData:
    """Константы для колбэков"""
    SETTINGS = "settings"
    GENERATE = "generate"
    SIZE_PREFIX = "size_"
    HELP = "help"
    BACK = "back_to_main"
    REMOVE_BG = "remove_bg"
    STYLES = "styles"
    STYLE_PREFIX = "style_"
    REGENERATE = "regenerate"

# Доступные размеры изображений
IMAGE_SIZES: Dict[str, Dict[str, Any]] = {
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

# Стили изображений
IMAGE_STYLES: Dict[str, Dict[str, Any]] = {
    "DEFAULT": {
        "label": "Обычный",
        "prompt_prefix": "",
        "description": "Стандартный стиль без дополнительных модификаций",
        "model_id": 1
    },
    "ANIME": {
        "label": "Аниме",
        "prompt_prefix": "anime style, manga art, japanese animation, ",
        "description": "Аниме стиль",
        "model_id": 1
    },
    "CYBERPUNK": {
        "label": "Киберпанк",
        "prompt_prefix": "cyberpunk style, neon lights, futuristic, high tech, ",
        "description": "Киберпанк стиль",
        "model_id": 1
    },
    "WATERCOLOR": {
        "label": "Акварель",
        "prompt_prefix": "watercolor painting style, soft colors, artistic, ",
        "description": "Акварельный стиль",
        "model_id": 1
    },
    "OIL_PAINTING": {
        "label": "Масло",
        "prompt_prefix": "oil painting style, textured, classical art, ",
        "description": "Масляная живопись",
        "model_id": 1
    },
    "RETRO": {
        "label": "Ретро",
        "prompt_prefix": "retro style, vintage aesthetics, old school design, nostalgic feel, ",
        "description": "Ретро стиль",
        "model_id": 1
    }
}
