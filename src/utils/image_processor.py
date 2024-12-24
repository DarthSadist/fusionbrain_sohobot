from PIL import Image
from rembg import remove
import io
from typing import Tuple, Optional, Dict
import logging
from functools import lru_cache
import hashlib

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Класс для обработки изображений с оптимизированным кэшированием и обработкой ошибок"""
    MAX_SIZE = 1500
    _model = None
    _cache: Dict[str, bytes] = {}
    MAX_CACHE_SIZE = 100  # Максимальное количество кэшированных результатов

    @classmethod
    def _get_model(cls):
        """Получает или создает экземпляр модели"""
        logger.info("Получение модели для удаления фона")
        if cls._model is None:
            logger.info("Инициализация новой модели")
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
            logger.info(f"Изображение требует уменьшения. Текущий размер: {width}x{height}")
            ratio = min(cls.MAX_SIZE / width, cls.MAX_SIZE / height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            original_size = (width, height)
            try:
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                logger.info(f"Изображение успешно уменьшено до {new_width}x{new_height}")
            except Exception as e:
                logger.error(f"Ошибка при изменении размера: {str(e)}")
                raise ValueError(f"Не удалось изменить размер изображения: {str(e)}")
            
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
            logger.info(f"Кэш превысил максимальный размер ({cls.MAX_CACHE_SIZE}). Очистка...")
            # Удаляем 20% старых записей
            items_to_remove = int(cls.MAX_CACHE_SIZE * 0.2)
            for _ in range(items_to_remove):
                cls._cache.pop(next(iter(cls._cache)))
            logger.info(f"Удалено {items_to_remove} элементов из кэша")

    @classmethod
    def remove_background(cls, image_data: bytes) -> bytes:
        """Удаляет фон с изображения с использованием кэширования"""
        logger.info("Начало процесса удаления фона")
        image_hash = cls._calculate_hash(image_data)
        
        # Проверяем кэш
        if image_hash in cls._cache:
            logger.info("Найден кэшированный результат")
            return cls._cache[image_hash]

        try:
            # Открываем изображение
            logger.info("Открытие изображения")
            image = Image.open(io.BytesIO(image_data))
            logger.info(f"Изображение открыто. Режим: {image.mode}, Размер: {image.size}")
            
            # Проверяем и конвертируем формат если нужно
            if image.mode not in ('RGB', 'RGBA'):
                logger.info(f"Конвертация изображения из {image.mode} в RGBA")
                image = image.convert('RGBA')
            
            # Уменьшаем размер если нужно
            logger.info("Проверка размера изображения")
            image, original_size = cls._resize_if_needed(image)
            
            # Конвертируем в bytes для обработки
            logger.info("Конвертация изображения в bytes")
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # Удаляем фон
            logger.info("Запуск процесса удаления фона")
            model = cls._get_model()
            result = model(img_byte_arr)
            logger.info("Фон успешно удален")
            
            # Восстанавливаем размер если нужно
            if original_size:
                logger.info(f"Восстановление исходного размера: {original_size}")
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
            
            logger.info("Процесс удаления фона успешно завершен")
            return result
            
        except Exception as e:
            logger.error(f"Ошибка при удалении фона: {str(e)}", 
                        exc_info=True,
                        extra={'image_size': image.size if 'image' in locals() else None,
                              'image_mode': image.mode if 'image' in locals() else None})
            raise ValueError(f"Не удалось удалить фон: {str(e)}")

    @classmethod
    def clear_cache(cls):
        """Очищает кэш обработанных изображений"""
        cls._cache.clear()
        cls._restore_size.cache_clear()
        logger.info("Кэш очищен")
