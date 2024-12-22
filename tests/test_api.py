import pytest
from unittest.mock import AsyncMock, patch
from src.api.fusion_brain import Text2ImageAPI, CensorshipError

@pytest.mark.asyncio
async def test_generate_success():
    """Тест успешной генерации изображения"""
    with patch('aiohttp.ClientSession') as MockSession:
        mock_session = AsyncMock()
        MockSession.return_value.__aenter__.return_value = mock_session
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {"uuid": "test-uuid"}
        mock_session.request.return_value.__aenter__.return_value = mock_response
        
        api = Text2ImageAPI("test-key", "test-secret")
        result = await api.generate("test prompt", 1)
        
        assert result == "test-uuid"

@pytest.mark.asyncio
async def test_generate_censorship():
    """Тест обработки ошибки цензуры"""
    with patch('aiohttp.ClientSession') as MockSession:
        mock_session = AsyncMock()
        MockSession.return_value.__aenter__.return_value = mock_session
        
        mock_response = AsyncMock()
        mock_response.status = 451
        mock_session.request.return_value.__aenter__.return_value = mock_response
        
        api = Text2ImageAPI("test-key", "test-secret")
        
        with pytest.raises(CensorshipError):
            await api.generate("test prompt", 1)

@pytest.mark.asyncio
async def test_check_generation_success():
    """Тест успешной проверки статуса генерации"""
    with patch('aiohttp.ClientSession') as MockSession:
        mock_session = AsyncMock()
        MockSession.return_value.__aenter__.return_value = mock_session
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "status": "DONE",
            "images": ["test-image-data"]
        }
        mock_session.request.return_value.__aenter__.return_value = mock_response
        
        api = Text2ImageAPI("test-key", "test-secret")
        result = await api.check_generation("test-uuid")
        
        assert result == "test-image-data"
