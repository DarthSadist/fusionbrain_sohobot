import logging
import aiohttp
from typing import Dict, Any, Optional

class Text2ImageAPI:
    """API клиент для работы с FusionBrain."""
    
    MAX_PROMPT_LENGTH = 500

    def __init__(self, api_key: str, secret_key: str):
        self.URL = 'https://api-key.fusionbrain.ai'
        self.api_key = api_key
        self.secret_key = secret_key
        self.logger = logging.getLogger(__name__)
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            await self._session.close()
            self._session = None

    async def _make_request(self, method: str, url: str, **kwargs) -> Dict[str, Any]:
        """Выполняет запрос к API с правильной авторизацией."""
        if not self._session:
            self._session = aiohttp.ClientSession()

        headers = {
            "X-API-KEY": self.api_key,
            "X-SECRET-KEY": self.secret_key,
        }
        kwargs['headers'] = headers

        try:
            async with self._session.request(method, url, **kwargs) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            self.logger.error(f"API request failed: {str(e)}", extra={
                'operation': 'api_request',
                'url': url,
                'method': method
            })
            raise

    def _prepare_prompt(self, prompt: str) -> str:
        """Подготовка промпта: обрезка до максимальной длины."""
        return prompt[:self.MAX_PROMPT_LENGTH]

    async def get_model(self) -> Dict[str, Any]:
        """Получение списка доступных моделей."""
        url = f"{self.URL}/key/api/v1/models"
        return await self._make_request('GET', url)

    async def generate(self, prompt: str, model_id: int, width: int = 1024, height: int = 1024) -> Dict[str, Any]:
        """Запуск генерации изображения."""
        url = f"{self.URL}/key/api/v1/text2image/run"
        prepared_prompt = self._prepare_prompt(prompt)
        
        data = {
            "type": "GENERATE",
            "numImages": 1,
            "width": width,
            "height": height,
            "generateParams": {
                "query": prepared_prompt,
                "negative_prompt": "",
                "model_id": model_id,
            }
        }
        
        return await self._make_request('POST', url, json=data)

    async def check_generation(self, uuid: str) -> Dict[str, Any]:
        """Проверка статуса генерации."""
        url = f"{self.URL}/key/api/v1/text2image/status/{uuid}"
        return await self._make_request('GET', url)
