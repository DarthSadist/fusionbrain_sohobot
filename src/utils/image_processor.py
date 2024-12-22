from PIL import Image
from rembg import remove
import io
from typing import Tuple, Optional
import logging

class ImageProcessor:
    """Класс для обработки изображений"""
    MAX_SIZE = 1500
    _model = None
    _logger = logging.getLogger(__name__)

    @classmethod
    def _get_model(cls):
        """Получает или создает экземпляр модели"""
        if cls._model is None:
            cls._model = remove
        return cls._model

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
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            cls._logger.info(f"Image resized to {new_width}x{new_height}")
            
        return image, original_size

    @classmethod
    def _restore_size(cls, image: Image.Image, original_size: Tuple[int, int]) -> Image.Image:
        """Возвращает изображение к исходному размеру"""
        if original_size:
            cls._logger.info(f"Restoring image to original size {original_size}")
            return image.resize(original_size, Image.Resampling.LANCZOS)
        return image

    @classmethod
    async def remove_background(cls, image_data: bytes) -> bytes:
        """Удаляет фон с изображения"""
        try:
            # Открываем изображение
            image = Image.open(io.BytesIO(image_data))
            
            # Уменьшаем размер если нужно
            image, original_size = cls._resize_if_needed(image)
            
            # Конвертируем в bytes для обработки
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format=image.format or 'PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # Удаляем фон
            model = cls._get_model()
            result = model(img_byte_arr)
            
            # Восстанавливаем размер если нужно
            if original_size:
                result_image = Image.open(io.BytesIO(result))
                result_image = cls._restore_size(result_image, original_size)
                
                # Конвертируем обратно в bytes
                output = io.BytesIO()
                result_image.save(output, format='PNG')
                result = output.getvalue()
            
            return result
            
        except Exception as e:
            cls._logger.error(f"Error during background removal: {str(e)}")
            raise
