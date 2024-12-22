import logging
import aiohttp
from typing import List, Dict, Any, Optional

class CensorshipError(Exception):
    pass

class Text2ImageAPI:
    MAX_PROMPT_LENGTH = 500

    def __init__(self, api_key: str, secret_key: str):
        self.URL = 'https://api-key.fusionbrain.ai'
        self.api_key = api_key
        self.secret_key = secret_key
        self.logger = logging.getLogger(__name__)

    async def _make_request(self, method: str, url: str, **kwargs) -> Dict[str, Any]:
        """Выполняет запрос к API с правильной авторизацией"""
        headers = {
            "X-API-KEY": self.api_key,
            "X-SECRET-KEY": self.secret_key
        }
        if 'headers' in kwargs:
            kwargs['headers'].update(headers)
        else:
            kwargs['headers'] = headers

        async with aiohttp.ClientSession() as session:
            async with session.request(method, url, **kwargs) as response:
                if response.status == 451:
                    raise CensorshipError("Контент не прошел модерацию")
                response.raise_for_status()
                return await response.json()

    def _prepare_prompt(self, prompt: str) -> str:
        """Подготовка промпта: обрезка до максимальной длины"""
        return prompt[:self.MAX_PROMPT_LENGTH]

    async def get_model(self) -> List[Dict[str, Any]]:
        """Получение списка доступных моделей"""
        url = f"{self.URL}/key/api/v1/models"
        return await self._make_request('GET', url)

    async def generate(self, prompt: str, model_id: int, width: int = 1024, height: int = 1024) -> str:
        """Запуск генерации изображения"""
        url = f"{self.URL}/key/api/v1/text2image/run"
        data = {
            "type": "GENERATE",
            "width": width,
            "height": height,
            "generateParams": {
                "query": self._prepare_prompt(prompt)
            }
        }
        response = await self._make_request('POST', url, json=data)
        return response['uuid']

    async def check_generation(self, uuid: str) -> Optional[bytes]:
        """Проверка статуса генерации"""
        url = f"{self.URL}/key/api/v1/text2image/status/{uuid}"
        response = await self._make_request('GET', url)
        
        if response['status'] == 'DONE':
            if 'images' in response and response['images']:
                return response['images'][0]
        return None
