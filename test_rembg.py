from rembg import remove
from PIL import Image

# Создаем тестовое изображение
img = Image.new('RGB', (100, 100), color='red')
img.save('test.png')

# Пробуем использовать rembg
input_image = Image.open('test.png')
output = remove(input_image)
output.save('output.png')
