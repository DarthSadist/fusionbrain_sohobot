from PIL import Image
from rembg import remove
import io
from typing import Tuple, Optional, Dict
import logging
from functools import lru_cache
import hashlib

class ImageProcessor:
    """Класс для обработки изображений с оптимизированным кэшированием и обработкой ошибок"""
    MAX_SIZE = 1500
    _model = None
    _logger = logging.getLogger(__name__)
    _cache: Dict[str, bytes] = {}
    MAX_CACHE_SIZE = 100  # Максимальное количество кэшированных результатов

    @classmethod
    def _get_model(cls):
        """Получает или создает экземпляр модели"""
        if cls._model is None:
            cls._model = remove
        return cls._model

    @staticmethod
    def _calculate_hash(image_data: bytes) -> str:
        """Вычисляет хеш изображения для кэширования"""
        return hashlib.md5(image_data).hexdigest()

    @classmethod
    def _resize_if_needed(cls, image: Image.Image) -> Tuple[Image.Image, Optional[Tuple[int, int]]]:
        """Уменьшает изображение, если оно слишком большое"""
        original_size = None
        width, height = image.size
        
        if width > cls.MAX_SIZE or height > cls.MAX_SIZE:
            cls._logger.info(f"Resizing image from {width}x{height}")
            ratio = min(cls.MAX_SIZE / width, cls.MAX_SIZE / height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            original_size = (width, height)
            try:
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                cls._logger.info(f"Image resized to {new_width}x{new_height}")
            except Exception as e:
                cls._logger.error(f"Error during image resize: {str(e)}")
                raise ValueError(f"Failed to resize image: {str(e)}")
            
        return image, original_size

    @classmethod
    @lru_cache(maxsize=32)
    def _restore_size(cls, width: int, height: int, original_width: int, original_height: int) -> Tuple[int, int]:
        """Кэшированный расчет параметров восстановления размера"""
        return (original_width, original_height)

    @classmethod
    def _manage_cache(cls):
        """Управление размером кэша"""
        if len(cls._cache) > cls.MAX_CACHE_SIZE:
            # Удаляем 20% старых записей
            items_to_remove = int(cls.MAX_CACHE_SIZE * 0.2)
            for _ in range(items_to_remove):
                cls._cache.pop(next(iter(cls._cache)))

    @classmethod
    async def remove_background(cls, image_data: bytes) -> bytes:
        """Удаляет фон с изображения с использованием кэширования"""
        image_hash = cls._calculate_hash(image_data)
        
        # Проверяем кэш
        if image_hash in cls._cache:
            cls._logger.info("Using cached result for background removal")
            return cls._cache[image_hash]

        try:
            # Открываем изображение
            image = Image.open(io.BytesIO(image_data))
            
            # Проверяем и конвертируем формат если нужно
            if image.mode not in ('RGB', 'RGBA'):
                image = image.convert('RGBA')
            
            # Уменьшаем размер если нужно
            image, original_size = cls._resize_if_needed(image)
            
            # Конвертируем в bytes для обработки
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # Удаляем фон
            model = cls._get_model()
            result = model(img_byte_arr)
            
            # Восстанавливаем размер если нужно
            if original_size:
                result_image = Image.open(io.BytesIO(result))
                restore_size = cls._restore_size(
                    result_image.width,
                    result_image.height,
                    original_size[0],
                    original_size[1]
                )
                result_image = result_image.resize(restore_size, Image.Resampling.LANCZOS)
                
                # Конвертируем обратно в bytes
                output = io.BytesIO()
                result_image.save(output, format='PNG')
                result = output.getvalue()
            
            # Сохраняем в кэш
            cls._cache[image_hash] = result
            cls._manage_cache()
            
            return result
            
        except Exception as e:
            cls._logger.error(f"Error during background removal: {str(e)}", 
                            exc_info=True,
                            extra={'image_size': image.size if 'image' in locals() else None})
            raise ValueError(f"Failed to remove background: {str(e)}")

    @classmethod
    def clear_cache(cls):
        """Очищает кэш обработанных изображений"""
        cls._cache.clear()
        cls._restore_size.cache_clear()
