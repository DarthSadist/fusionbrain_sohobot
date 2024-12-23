import aiohttp
import json
import logging
from typing import Dict, List, Optional, Union, Any

logger = logging.getLogger(__name__)

class Text2ImageAPI:
    """Класс для работы с API генерации изображений"""
    
    def __init__(self, api_key: str, secret_key: str):
        """
        Инициализация клиента API
        
        Args:
            api_key: API ключ
            secret_key: Секретный ключ
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = "https://api-key.fusionbrain.ai"
        self.headers = {
            "Content-Type": "application/json",
            "X-Key": api_key,
            "X-Secret": secret_key
        }
    
    async def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Union[Dict, List, Any]:
        """
        Выполнение запроса к API
        
        Args:
            method: HTTP метод
            endpoint: Конечная точка API
            data: Данные для отправки
            
        Returns:
            Union[Dict, List, Any]: Ответ от API
            
        Raises:
            Exception: При ошибке запроса
        """
        url = f"{self.base_url}/{endpoint}"
        
        try:
            async with aiohttp.ClientSession(headers=self.headers) as session:
                async with session.request(method, url, json=data) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        raise Exception(f"API request failed: {error_text}")
                        
        except aiohttp.ClientError as e:
            logger.error(f"Network error: {str(e)}")
            raise Exception(f"Network error: {str(e)}")
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response: {str(e)}")
            raise Exception("Invalid response from API")
            
        except Exception as e:
            logger.error(f"API request error: {str(e)}")
            raise
    
    async def get_model(self) -> List[Dict]:
        """
        Получение списка доступных моделей
        
        Returns:
            List[Dict]: Список моделей
            
        Raises:
            Exception: При ошибке запроса
        """
        return await self._make_request("GET", "key/api/v1/models")
    
    async def generate(self, prompt: str, model_id: int, width: int = 1024, height: int = 1024) -> str:
        """
        Запуск генерации изображения
        
        Args:
            prompt: Текстовое описание
            model_id: ID модели
            width: Ширина изображения
            height: Высота изображения
            
        Returns:
            str: UUID задачи генерации
            
        Raises:
            Exception: При ошибке запроса
        """
        data = {
            "type": "GENERATE",
            "numImages": 1,
            "width": width,
            "height": height,
            "generateParams": {
                "query": prompt
            }
        }
        
        response = await self._make_request("POST", f"key/api/v1/text2image/run/{model_id}", data)
        return response.get("uuid")
    
    async def check_generation(self, uuid: str) -> Union[List[str], Dict[str, Any]]:
        """
        Проверка статуса генерации
        
        Args:
            uuid: UUID задачи
            
        Returns:
            Union[List[str], Dict[str, Any]]: Результат генерации или статус
            
        Raises:
            Exception: При ошибке запроса
        """
        return await self._make_request("GET", f"key/api/v1/text2image/status/{uuid}")
