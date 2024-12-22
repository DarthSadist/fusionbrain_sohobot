import pytest
from unittest.mock import patch, MagicMock
from PIL import Image
import io
from src.utils.image_processor import ImageProcessor

@pytest.fixture
def sample_image():
    """Создает тестовое изображение"""
    img = Image.new('RGB', (100, 100), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()

def test_resize_if_needed_no_resize():
    """Тест: изображение не требует уменьшения"""
    image = Image.new('RGB', (100, 100), color='red')
    result, original_size = ImageProcessor._resize_if_needed(image)
    
    assert original_size is None
    assert result.size == (100, 100)

def test_resize_if_needed_with_resize():
    """Тест: изображение требует уменьшения"""
    image = Image.new('RGB', (2000, 2000), color='red')
    result, original_size = ImageProcessor._resize_if_needed(image)
    
    assert original_size == (2000, 2000)
    assert max(result.size) <= ImageProcessor.MAX_SIZE

def test_restore_size():
    """Тест восстановления размера изображения"""
    image = Image.new('RGB', (100, 100), color='red')
    original_size = (200, 200)
    
    result = ImageProcessor._restore_size(image, original_size)
    assert result.size == original_size

@pytest.mark.asyncio
async def test_remove_background(sample_image):
    """Тест удаления фона"""
    with patch('src.utils.image_processor.remove') as mock_remove:
        mock_remove.return_value = b"processed_image"
        
        result = await ImageProcessor.remove_background(sample_image)
        
        assert result == b"processed_image"
        mock_remove.assert_called_once()
