from dataclasses import dataclass
from typing import Optional

@dataclass
class UserState:
    """Класс для хранения состояния пользователя"""
    width: int = 1024
    height: int = 1024
    awaiting_prompt: bool = False
    last_image: Optional[bytes] = None
    last_image_id: Optional[str] = None
    last_prompt: Optional[str] = None

@dataclass
class UserSettings:
    """Класс для хранения настроек пользователя"""
    width: int = 1024
    height: int = 1024
    style: str = "DEFAULT"  # Стиль по умолчанию
