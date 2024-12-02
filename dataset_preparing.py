import os
from PIL import Image, ImageFilter
import random
from tqdm import tqdm

# Путь к оригинальным изображениям
DATASET_PATH = "img_align_celeba"
OUTPUT_PATH = "prepared_dataset"
os.makedirs(OUTPUT_PATH, exist_ok=True)


# Функция для добавления повреждений
def add_defects(image: Image.Image):
    # Преобразование в черно-белое
    # if random.random() > 0.5:
    #     image = image.convert("L").convert("RGB")

    # Добавление размытия
    # if random.random() > 0.5:
    image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(2, 5)))

    # Добавление шума
    # if random.random() > 0.4:
    #     noise = Image.effect_noise(image.size, random.randint(20, 50))
    #     image = Image.blend(image, noise.convert("RGB"), alpha=0.5)

    return image


# Подготовка датасета
def prepare_dataset(dataset_path, output_path, image_size=256):
    image_files = os.listdir(dataset_path)
    for image_file in tqdm(image_files, desc="Preparing dataset"):
        try:
            # Загрузка изображения
            image_path = os.path.join(dataset_path, image_file)
            original = Image.open(image_path).convert("RGB").resize((image_size, image_size))

            # Создание поврежденного изображения
            damaged = add_defects(original)  # Правая половина

            # Создание парного изображения
            combined = Image.new("RGB", (image_size * 2, image_size))
            combined.paste(damaged, (0, 0))  # Поврежденное слева
            combined.paste(original, (image_size, 0))  # Оригинал справа

            # Сохранение
            combined.save(os.path.join(output_path, image_file))
        except Exception as e:
            print(f"Ошибка с файлом {image_file}: {e}")


# Запуск подготовки
prepare_dataset(DATASET_PATH, OUTPUT_PATH)