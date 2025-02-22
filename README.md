# Модель прогнозирования добычи нефти для скважин

## Обзор
Этот проект реализует модель глубокого обучения для прогнозирования объемов добычи нефти из скважин, используя комбинацию временных и пространственных признаков. Модель использует как архитектуру трансформеров, так и графовые нейронные сети для захвата сложных взаимосвязей между скважинами и их производственными паттернами.

## Особенности
- Прогнозирование временных рядов для нескольких скважин
- Моделирование пространственных взаимосвязей между скважинами
- Обработка временных данных на основе трансформеров
- Графовая нейронная сеть для пространственных зависимостей
- Поддержка прогнозирования давления и объемов добычи
- Возможности предварительной обработки и масштабирования данных

## Модели
- `epoch=485-step=1458.ckpt` - для модифицированных данных
- `epoch=477-step=1434.ckpt` - для исходных данных
### Входные данные
Данные хранятся в формате CSV и находятся в папке `gdm`(4 реализции Brugge). Например, файл `gdm/FY-SF-KM-1-1.csv` содержит данные о скважинах. Всего 4 реализацции разных Brugge.

### Результаты обработки
Результаты обработки данных для нейронной сети (НС) сохраняются в подготовленных CSV-файлах, которые также находятся в папке `train`. Скрипт `parser.py` отвечает за загрузку и предварительную обработку данных, включая преобразование единиц измерения и объединение различных источников данных.
После выполнения нейронной сети результаты сохраняются в CSV-файлы, которые содержат предсказанные и фактические значения для каждой скважины. Эти файлы сохраняются в папке, соответствующей названию модели (например, FY-SF-KM-1-1), и имеют формат well_{well_id}_predictions.csv, где {well_id} — это идентификатор скважины.
### Примеры данных
1. **CSV файл**: `gdm/FY-SF-KM-1-1.csv`
   ```csv
   Объект	Шаг	Дата	Дней	"Дебит нефти, ст.м3/сут"	"Приёмистость воды, ст.м3/сут"	"Обводнённость, доля"	"Забойное давление, Бара"	"WBP9, Бара"
   I1	2	02.01.2009	1.0000	+0.00000000e+00	+0.00000000e+00	+0.00000000e+00	+0.00000000e+00	+0.00000000e+00
   I1	3	03.01.2009	2.0000	+0.00000000e+00	+0.00000000e+00	+0.00000000e+00	+0.00000000e+00	+0.00000000e+00
   ...
   ```

2. **GRDECL файл**: `gdm/FY-SF-KM-1-1.grdecl`
   ```grdecl
   -- Format      : Generic ECLIPSE style (ASCII) grid geometry and properties (*.GRDECL)
   -- Exported by : Petrel 2007.1.1 Schlumberger
   -- User name   : kgeel
   -- Date        : Monday, March 03 2008 18:01:40
   -- Project     : Brugge Field realisations 080303 v14 [2007].pet
   -- Grid        : Part of 60K Realisations Grid

   -- Property name in Petrel : Facies 
   FACIES  
   0 0 0 0 
   0 0 0 0 
   ...
   ```
## Требования
Основные зависимости:
- torch>=2.5.1
- torch-geometric>=2.6.1
- pytorch-lightning>=2.4.0
- pandas>=2.2.3
- numpy>=2.1.3
- matplotlib>=3.9.2
- tsl>=2.0

Для полного списка зависимостей смотрите `requirements.txt`.

## Установка

1. Создайте виртуальное окружение:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Для Windows: venv\Scripts\activate
   ```

2. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```

## Структура проекта
- `train.py` - Основной скрипт для обучения и определения модели. Используйте его для тренировки модели на подготовленных данных.
- `evaluate.py` - Скрипт для оценки модели и визуализации результатов. Используйте его для получения прогнозов и построения графиков.
- `parser.py` - Скрипт для предварительной обработки данных. Используйте его для загрузки и подготовки данных перед обучением.
- `mod.py` - Скрипт для модификации и предварительной обработки данных. Используйте его для обнуления значений давления и дебита в определенных временных интервалах.
- `evaluate_1_task.py` - Неудачная реализация изменения забойного давления и дебита.
## Использование

1. Подготовка данных:
   ```bash
   python parser.py
   ```
   Этот скрипт загружает данные из CSV-файлов, данные преобразуются в нужный формат и сохраняются в папке traim
    заменить в строчке ``` gdm_name  = 'FY-SF-KP-7-33'```


2. Модификация данных:
   ```bash
   python mod.py
   ```
   Этот скрипт обнуляет значения давления и дебита для определенных скважин в заданных временных интервалах. Входные данные берутся из подготовленного CSV-файла, и результаты сохраняются обратно в тот же файл.
    заменить в строчке ``` gdm_name = 'FY-SF-KM-1-1'```


3. Обучение модели:
   ```bash
   python train.py
   ```
   Этот скрипт запускает процесс обучения модели на подготовленных данных. Входные данные берутся из CSV-файла, созданного на предыдущем шаге.
   заменить в строчке ``` gdm_name = 'FY-SF-KM-1-1'```

4. Оценка модели:
   ```bash
   python evaluate.py
   ```
   Этот скрипт загружает обученную модель и визуализирует результаты прогнозирования. 
    заменить в строках ```gdm_name = 'FY-SF-KM-1-1', model = 'logs/epoch=477-step=1434.ckpt'```
    так же необходимо создать папку с именнм реализации, куда будут сохраняться графики

## ВНИМАНИЕ
Если меняете реализацию в python evaluate.py, то обязательно заменить и в python train.py, так как оттуда берется построение pipeline 

## Архитектура модели
Модель сочетает в себе два основных компонента:
1. Обработка временных данных на основе трансформеров.
2. Графовая нейронная сеть для пространственных взаимосвязей.

Ключевые параметры:
- Входное окно: 3 временных шага
- Горизонт прогнозирования: 12 временных шагов
- Размер скрытого слоя: 32
- Количество слоев трансформера: 2

## Формат данных
Входные данные должны быть в формате CSV с следующими колонками:
- Название скважины
- Дата
- Объемы добычи
- Измерения давления
- Координаты скважины (X, Y)

## Лицензия
MIT

## Авторы
Карина Янышевская

## Благодарности
- Библиотека TSL (Torch Spatiotemporal)
- PyTorch Geometric
