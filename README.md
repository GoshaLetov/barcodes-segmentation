## Сегментация штрих-кодов

### Данные

Включает 540 снимков штрих-кодов с разметкой bounding box`ов.
Скачать данные и их разметку можно по ссылке с [Яндекс.Диск](https://disk.yandex.ru/d/pRFNuxLQUZcDDg). Либо при помощи команды - будем описано ниже.

Структура папки с данными должна быть следующей:
```
data:
  images - Снимки штрих-кодов:
     image1.jpg
     image2.jpg
  annotations.tsv - Файл с разметкой снимков        
```

### Подготовка проекта 
1. Создание и активация окружения
```
python3.10 -m venv /path/to/new/virtual/environment
```
```
source /path/to/new/virtual/environment/bin/activate
```

2. Установка пакетов

В активированном окружении:
```
pip install -U pip 
pip install -r requirements.txt
```

3. Подготовка данных

В активированном окружении:
```
mkdir experiments
make prepare_linux
make download_train_data
```

4. Настройка ClearML

Вводим команду, далее следуем инструкциям:
```
clearml-init
```

5. Конфигурация

Для изменения конфигурации обучения необходимо поменять параметры файла [config.yaml](configs/config.yaml).

6. Запуск обучения
Для запуска обучения запускаем скрипт.
```
python train.py
```
Перед запуском необходимо изменить путь до папки с данными в [файле](src/constants.py).
По умолчанию используется конфиг под названием `config.yaml`. Для сохранения конфигурации необходимо изменить название.

6. Сохранение весов модели

Изменить путь до чекпоинта модели и путь до сконвертированной модели.
```
PYTHONPATH=. src/convert.py path/to/model.ckpt path/to/model.onnx
dvc add path/to/model.onnx
dvc push path/to/model.onnx.dvc
```

### Использование готовой модели

Результаты экспериментов хранятся в папке [experiments](experiments):

1. Загрузить результаты экспериментов можно при помощи команды:
```
dvc pull path/to/model.onnx.dvc
```

2. Использовать модель на конкретном примере можно при помощи команды:
```
PYTHONPATH=. python src/postprocess.py path/to/model.onnx path/to/image
```
