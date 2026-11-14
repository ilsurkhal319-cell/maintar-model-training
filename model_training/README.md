# Продолжить обучение MaintAR (5 классов)

Это детектор YOLOv8s: он отмечает дефекты рамками. Сохранённая расширенная модель уже начала обучение от нашей базовой модели на 2 класса. Для продолжения используйте `handoff.py resume`. Точный набор Colab: **21 387 train + 2 139 val** (негативные изображения включены).

## Быстрый запуск на компьютере с NVIDIA GPU

Из корня репозитория (Python 3.12 или 3.13):

```bash
git pull
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r model_training/requirements.txt
python -c "import torch; print(torch.cuda.is_available())"
python model_training/handoff.py prepare
python model_training/handoff.py resume
```

Команда проверки GPU должна вывести `True`. Если выводит `False`, проверьте драйвер и установку PyTorch с CUDA для вашей системы; дообучать весь набор на CPU долго. В Windows активируйте окружение через `.venv\Scripts\activate`, а при проблемах с загрузчиками добавьте `--workers 0`.

`prepare` скрыто запросит **ваш Roboflow Private API Key**, скачает 12 выбранных экспортов и восстановит исходную разметку и разбиение train/val из Colab. Ключ не сохраняется. Права аккаунта должны позволять экспорт всех 12 проектов. Загруженные источники кешируются; при сетевой ошибке повторите команду. Если данные уже есть локально:

```bash
python model_training/handoff.py prepare --offline --raw model_training/data/raw
```

Изображения не включены в Git. Для скачивания и подготовки освободите несколько десятков ГБ. При изменении файлов экспорта проверка SHA256 остановит подготовку, чтобы случайно не подменить данные. Не редактируйте изображения в подготовленном каталоге: при возможности они связаны с кешем жёсткими ссылками.

## Веса и результаты

Файлы в `model_training/handoff/`:

- `basic/weights/best.pt` — готовая базовая модель: коррозия + трещины, 30 эпох; mAP50 **0.76743**, mAP50–95 **0.53366** на её валидационной части.
- `expanded/weights/last.pt` — расширенная модель после 2 полных эпох, со состоянием оптимизатора. `resume` продолжает с **3-й эпохи**, исходная цель — 30 эпох всего.
- `expanded/weights/best.pt` — лучшие веса расширенного запуска на момент остановки. Это ранний результат, модель ещё не дообучена.
- `expanded/results.csv`, `expanded/args.yaml` — журнал и исходные настройки запуска.
- `expanded_dataset.json.gz` — точные имена, SHA256 изображений, рамки и разбиение данных из того же Colab.
- `sources.json` — закреплённые проекты и версии экспортов.

После запуска результат сохраняется в `model_training/runs/expanded_resume/weights/`. Исходные веса в `handoff/` сохраняются. Для второго продолжения укажите новый каталог и свежий `last.pt`:

```bash
python model_training/handoff.py resume --weights model_training/runs/expanded_resume/weights/last.pt --run model_training/runs/expanded_resume_2
```

`resume` восстанавливает оптимизатор и расписание обучения. Число эпох берётся из checkpoint. Для нового цикла с новым числом эпох используйте `finetune`: это старт с обученных весов и новым оптимизатором.

```bash
python model_training/handoff.py finetune --epochs 30
```

Для расширения именно готовой базовой модели на 2 класса в новую модель на 5:

```bash
python model_training/handoff.py finetune --weights model_training/handoff/basic/weights/best.pt --run model_training/runs/expanded_from_basic --epochs 30
```

При нехватке видеопамяти добавьте `--batch 8` (при необходимости `--batch 4`). На T4 использовались 640 px, batch 16 и 2 workers. Перед запуском можно проверить параметры без обучения: `python model_training/handoff.py resume --dry-run --device cpu`.

## В Google Colab

Выберите Runtime → Change runtime type → T4 GPU. Затем выполните ячейки по очереди:

```python
!git clone https://github.com/kayeros/MaintAR.git /content/MaintAR
%cd /content/MaintAR
%pip install -q -r model_training/requirements.txt
!nvidia-smi
```

```python
from getpass import getpass
import os, subprocess, sys
os.environ['ROBOFLOW_API_KEY'] = getpass('Roboflow Private API Key: ')
try:
    subprocess.run([sys.executable, 'model_training/handoff.py', 'prepare'], check=True)
finally:
    os.environ.pop('ROBOFLOW_API_KEY', None)
```

```python
!python model_training/handoff.py resume
```

Скачивание идёт напрямую в Colab. Загружать датасет с ноутбука через `files.upload()` не нужно. После обучения сохраните `best.pt`, `last.pt`, `args.yaml`, `results.csv` и манифест запуска: `/content` временный и исчезает при сбросе runtime.

## Классы и ограничения

| ID | Класс | Значение |
|---|---|---|
| 0 | surface_corrosion | Видимая коррозия |
| 1 | pipe_crack | Видимая трещина трубы |
| 2 | pipe_leak | Видимая утечка на трубе |
| 3 | cable_insulation_damage | Повреждение изоляции кабеля |
| 4 | fastener_issue | Визуальная проблема крепежа |

Порядок фиксирован. Утечка на фото не доказывает неисправность конкретного уплотнения, а вид болта не определяет момент затяжки. Наборы просмотрены выборочно; полная ручная проверка всей разметки не выполнена. Для реального качества нужны независимые фото оборудования из кейса. mAP — метрика поиска рамок, а не процент правильных диагнозов. Метрики 2 и 5 классов напрямую не сравниваются.

Рабочая модель приложения имеет старые 9 классов. Эти 5-классовые веса — кандидат для обучения; скрипт не публикует их в приложение автоматически. Подключение требует согласовать названия классов и проверить независимые фото.

## Что отправлять в Git

Весь пакет находится в одной директории `model_training/`: код, README, requirements и checkpoints. Каждый checkpoint меньше лимита GitHub 100 MiB; для регулярных последующих версий весов лучше отдельное хранилище или Git LFS. Не включать `data/`, `runs/`, `.venv/`, `.env` и API-ключи. Соблюдайте лицензии источников (проекты и версии — в `sources.json`; актуальные условия проверяйте на странице каждого проекта Roboflow Universe).
