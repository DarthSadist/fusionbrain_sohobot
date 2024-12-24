from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class ImageInfo:
    """Класс для хранения информации об изображении"""
    id: str  # Уникальный идентификатор изображения
    prompt: str  # Промпт, использованный для генерации
    style: str  # Название стиля
    style_prompt: str  # Префикс стиля
    width: int  # Ширина изображения
    height: int  # Высота изображения
    model_id: int  # ID модели
    created_at: datetime  # Время создания
    generation_time: float  # Время генерации в секундах
    user_id: int  # ID пользователя
    has_removed_bg: bool = False  # Был ли удален фон
    bg_removal_time: Optional[float] = None  # Время удаления фона в секундах

    def get_size_str(self) -> str:
        """Получение строки с размером изображения"""
        return f"{self.width}x{self.height}"

    def get_full_prompt(self) -> str:
        """Получение полного промпта с учетом стиля"""
        return f"{self.style_prompt}{self.prompt}"

    def get_generation_time_str(self) -> str:
        """Получение строки с временем генерации"""
        return f"{self.generation_time:.1f} сек"

    def get_bg_removal_time_str(self) -> Optional[str]:
        """Получение строки с временем удаления фона"""
        if self.bg_removal_time is not None:
            return f"{self.bg_removal_time:.1f} сек"
        return None

    def to_dict(self) -> dict:
        """Преобразование в словарь для сохранения"""
        return {
            "id": self.id,
            "prompt": self.prompt,
            "style": self.style,
            "style_prompt": self.style_prompt,
            "width": self.width,
            "height": self.height,
            "model_id": self.model_id,
            "created_at": self.created_at.isoformat(),
            "generation_time": self.generation_time,
            "user_id": self.user_id,
            "has_removed_bg": self.has_removed_bg,
            "bg_removal_time": self.bg_removal_time
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ImageInfo':
        """Создание объекта из словаря"""
        data = data.copy()
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)
