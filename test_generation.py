import unittest
from unittest.mock import AsyncMock, patch
from main import generate_image_with_prompt, check_generation_status, remove_background

class TestImageGeneration(unittest.IsolatedAsyncioTestCase):
    async def test_generate_image_with_prompt(self):
        # Mocking API response
        with patch('main.Text2ImageAPI') as MockAPI:
            mock_api = MockAPI.return_value
            mock_api.generate = AsyncMock(return_value='fake-uuid')
            mock_api.get_model = AsyncMock(return_value=[{"id": 4, "name": "Kandinsky", "version": 3.1, "type": "TEXT2IMAGE"}])

            # Mocking message
            mock_message = AsyncMock()
            mock_message.from_user.id = 12345
            mock_message.text = "Test prompt"

            # Call the function
            await generate_image_with_prompt(mock_message, "Test prompt")

            # Assertions
            mock_api.generate.assert_called_once()
            mock_api.get_model.assert_called_once()

    async def test_check_generation_status(self):
        # Mocking API response
        with patch('main.Text2ImageAPI') as MockAPI:
            mock_api = MockAPI.return_value
            mock_api.check_generation = AsyncMock(side_effect=["INITIAL", "DONE"])

            # Mocking message
            mock_message = AsyncMock()
            mock_message.edit_text = AsyncMock()

            # Call the function
            await check_generation_status(mock_api, 'fake-uuid', mock_message, 12345)

            # Assertions
            self.assertEqual(mock_api.check_generation.call_count, 2)
            mock_message.edit_text.assert_called_with("Изображение успешно сгенерировано повторно.", reply_markup=unittest.mock.ANY)

    async def test_generate_image_with_prompt_error(self):
        # Тестирование обработки ошибок при генерации
        with patch('main.Text2ImageAPI') as MockAPI:
            mock_api = MockAPI.return_value
            mock_api.generate = AsyncMock(side_effect=Exception("API Error"))
            mock_api.get_model = AsyncMock(return_value=[{"id": 4, "name": "Kandinsky", "version": 3.1, "type": "TEXT2IMAGE"}])

            # Мокаем сообщение
            mock_message = AsyncMock()
            mock_message.from_user.id = 12345
            mock_message.text = "Test prompt"
            mock_message.answer = AsyncMock()

            # Вызываем функцию
            await generate_image_with_prompt(mock_message, "Test prompt")

            # Проверяем, что ошибка обработана
            mock_message.answer.assert_called_with("❌ Ошибка при генерации: API Error")

    async def test_remove_background(self):
        # Тестирование функционала удаления фона
        with patch('main.remove') as mock_remove, \
             patch('PIL.Image.open') as mock_image_open, \
             patch('io.BytesIO') as mock_bytesio:
            
            # Мокаем необходимые объекты
            mock_image = AsyncMock()
            mock_image_open.return_value = mock_image
            mock_remove.return_value = mock_image
            mock_bytesio.return_value = b"fake_image_data"

            # Мокаем сообщение и фото
            mock_message = AsyncMock()
            mock_photo = AsyncMock()
            mock_photo.download = AsyncMock(return_value=b"fake_photo_data")
            mock_message.photo = [mock_photo]

            # Вызываем функцию
            result = await remove_background(mock_message)

            # Проверяем результат
            self.assertIsNotNone(result)
            mock_remove.assert_called_once()

if __name__ == '__main__':
    unittest.main()
