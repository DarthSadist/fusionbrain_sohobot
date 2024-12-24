from dataclasses import dataclass, field
from typing import Optional, Dict
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class UserState:
    """Класс для хранения состояния пользователя с валидацией и оптимизацией памяти"""
    width: int = field(default=1024)
    height: int = field(default=1024)
    awaiting_prompt: bool = field(default=False)
    last_image: Optional[bytes] = field(default=None, repr=False)  # не включаем в repr для оптимизации памяти
    last_image_id: Optional[str] = field(default=None)
    last_prompt: Optional[str] = field(default=None)
    last_activity: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        self.validate()
    
    def validate(self):
        """Проверка корректности значений"""
        if not isinstance(self.width, int) or not isinstance(self.height, int):
            raise ValueError("Width and height must be integers")
        if self.width < 64 or self.width > 2048 or self.height < 64 or self.height > 2048:
            raise ValueError("Width and height must be between 64 and 2048")
    
    def update_activity(self):
        """Обновление времени последней активности"""
        self.last_activity = datetime.now()
    
    def clear_image(self):
        """Очистка данных изображения для освобождения памяти"""
        self.last_image = None
        self.last_image_id = None
    
    def is_inactive(self, timeout_minutes: int = 30) -> bool:
        """Проверка неактивности пользователя"""
        return datetime.now() - self.last_activity > timedelta(minutes=timeout_minutes)

@dataclass
class UserSettings:
    """Класс для хранения настроек пользователя с валидацией"""
    width: int = field(default=1024)
    height: int = field(default=1024)
    style: str = field(default="DEFAULT")
    created_at: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        self.validate()
    
    def validate(self):
        """Проверка корректности значений"""
        if not isinstance(self.width, int) or not isinstance(self.height, int):
            raise ValueError("Width and height must be integers")
        if self.width < 64 or self.width > 2048 or self.height < 64 or self.height > 2048:
            raise ValueError("Width and height must be between 64 and 2048")
        if not isinstance(self.style, str):
            raise ValueError("Style must be a string")
    
    def update(self, **kwargs):
        """Обновление настроек с валидацией"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.last_modified = datetime.now()
        self.validate()

class UserStateManager:
    """Менеджер состояний пользователей с автоочисткой неактивных сессий"""
    _states: Dict[int, UserState] = {}
    _settings: Dict[int, UserSettings] = {}
    _last_cleanup: datetime = datetime.now()
    CLEANUP_INTERVAL = timedelta(minutes=60)
    INACTIVE_TIMEOUT = timedelta(minutes=30)
    
    @classmethod
    def get_state(cls, user_id: int) -> UserState:
        """Получение состояния пользователя"""
        if user_id not in cls._states:
            cls._states[user_id] = UserState()
        cls._states[user_id].update_activity()
        cls._cleanup_if_needed()
        return cls._states[user_id]
    
    @classmethod
    def get_settings(cls, user_id: int) -> UserSettings:
        """Получение настроек пользователя"""
        if user_id not in cls._settings:
            cls._settings[user_id] = UserSettings()
        return cls._settings[user_id]
    
    @classmethod
    def _cleanup_if_needed(cls):
        """Очистка неактивных сессий"""
        now = datetime.now()
        if now - cls._last_cleanup > cls.CLEANUP_INTERVAL:
            inactive_users = [
                user_id for user_id, state in cls._states.items()
                if now - state.last_activity > cls.INACTIVE_TIMEOUT
            ]
            for user_id in inactive_users:
                cls._states.pop(user_id, None)
                logger.info(f"Cleaned up inactive session for user {user_id}")
            cls._last_cleanup = now
    
    @classmethod
    def clear_user_data(cls, user_id: int):
        """Полная очистка данных пользователя"""
        cls._states.pop(user_id, None)
        cls._settings.pop(user_id, None)
