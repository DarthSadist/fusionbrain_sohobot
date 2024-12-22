import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.handlers.command_handlers import (
    send_welcome,
    show_help,
    show_settings,
    process_size_change
)
from src.handlers.generation_handlers import (
    start_generation,
    generate_image_with_prompt
)
from src.constants.bot_constants import CallbackData

@pytest.fixture
def mock_message():
    """Создает мок объекта сообщения"""
    message = AsyncMock()
    message.from_user.id = 12345
    return message

@pytest.fixture
def mock_callback_query():
    """Создает мок объекта callback query"""
    callback = AsyncMock()
    callback.from_user.id = 12345
    return callback

@pytest.mark.asyncio
async def test_send_welcome(mock_message):
    """Тест обработчика команды /start"""
    await send_welcome(mock_message)
    mock_message.answer.assert_called_once()

@pytest.mark.asyncio
async def test_show_help(mock_callback_query):
    """Тест показа справки"""
    await show_help(mock_callback_query)
    mock_callback_query.message.edit_text.assert_called_once()
    mock_callback_query.answer.assert_called_once()

@pytest.mark.asyncio
async def test_show_settings(mock_callback_query):
    """Тест показа настроек"""
    await show_settings(mock_callback_query)
    mock_callback_query.message.edit_text.assert_called_once()
    mock_callback_query.answer.assert_called_once()

@pytest.mark.asyncio
async def test_process_size_change_valid(mock_callback_query):
    """Тест изменения размера (валидный размер)"""
    mock_callback_query.data = f"{CallbackData.SIZE_PREFIX}square"
    await process_size_change(mock_callback_query)
    mock_callback_query.message.edit_text.assert_called_once()
    mock_callback_query.answer.assert_called_once()

@pytest.mark.asyncio
async def test_process_size_change_invalid(mock_callback_query):
    """Тест изменения размера (невалидный размер)"""
    mock_callback_query.data = f"{CallbackData.SIZE_PREFIX}invalid"
    await process_size_change(mock_callback_query)
    mock_callback_query.message.edit_text.assert_called_once()
    mock_callback_query.answer.assert_called_once()

@pytest.mark.asyncio
async def test_start_generation(mock_callback_query):
    """Тест начала генерации"""
    await start_generation(mock_callback_query)
    mock_callback_query.message.edit_text.assert_called_once()
    mock_callback_query.answer.assert_called_once()

@pytest.mark.asyncio
async def test_generate_image_with_prompt(mock_message):
    """Тест генерации изображения по промпту"""
    with patch('src.handlers.generation_handlers.Text2ImageAPI') as MockAPI:
        mock_api = MockAPI.return_value
        mock_api.generate = AsyncMock(return_value='test-uuid')
        mock_api.check_generation = AsyncMock(return_value=None)
        
        await generate_image_with_prompt(mock_message, "test prompt")
        
        mock_api.generate.assert_called_once()
        assert mock_message.answer.call_count >= 1
