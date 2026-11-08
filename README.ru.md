# personal-threats

[English](README.md) | [Русский](README.ru.md)

Python-библиотека для локальной обработки диалогов и анализа персональных
угроз. CLI позволяет импортировать сообщения, управлять историей и запускать
инференс.

**Обученная модель будет добавлена позже.** Импорт сообщений, история SQLite
и очереди обработки уже работают без модели. Для оценки угроз требуется
совместимый комплект файлов модели.

## Возможности

- Импорт JSONL-файлов и JSON-экспорта Telegram Desktop.
- Хранение истории диалогов, версий сообщений и отметок удаления в SQLite.
- Продолжение очереди после перезапуска и пересчёт результатов после правок.
- Потоковая обработка с ограниченным контекстом и перекрывающимися окнами;
  длина сообщения — до 65 536 кодовых точек Unicode.
- Подключение ONNX-моделей для инференса на CPU или NVIDIA CUDA.
- Python API и CLI для получения результатов, вывод и логи в JSON.

## Установка

Требуются Linux, CPython 3.14+ и SQLite 3.37+. Из корня репозитория:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps .
```

`requirements.txt` содержит закреплённые зависимости CPU-инференса,
тестирования и сборки. Для CUDA используется отдельное окружение
с совместимыми библиотеками NVIDIA и установка `'.[cuda]'`.
Минимальная установка библиотеки — `python -m pip install .`.

## Использование в Python

Хранение сообщений и просмотр статистики доступны до подключения модели:

```python
from personal_threats import HistoryStore, Message

with HistoryStore("var/history.sqlite3") as history:
    history.ingest(
        Message("dialog-demo", "1", "user-a", "Hello. Привет.")
    )
    print(history.statistics())
```

После добавления модели она загружается через `ThreatAnalyzer.from_bundle()`,
а результаты анализа возвращаются методами `process_message()` или
`process_messages()`. Эти методы являются генераторами: для завершения
обработки необходимо перебрать их результаты.
Полный пример приведён в [examples/analyze.py](examples/analyze.py).

## CLI

```bash
personal-threats --db var/history.sqlite3 init
personal-threats --db var/history.sqlite3 import-jsonl examples/messages.jsonl
personal-threats --db var/history.sqlite3 stats
```

Обязательные поля JSONL: `dialog_id`, `message_id`, `sender_id` и `text`.
Папки JSON-экспорта Telegram импортируются командой `import-telegram`.
Список команд доступен через `personal-threats --help`.

## Подключение модели

Совместимый комплект файлов ONNX-модели размещается в `models/` или другой
локальной директории. Путь передаётся в `ThreatAnalyzer.from_bundle()` либо
через параметр CLI `--model`. Файлы проверяются по манифесту; модель
загружается с диска без автоматического скачивания.

Интерфейс модели предусматривает оценку наличия угрозы и шесть категорий:
физический вред, принуждение, преследование, доксинг, репутационный
и имущественный вред. Формат комплекта файлов и профили срабатываний описаны
в [models/README.ru.md](models/README.ru.md).

## Структура проекта

- `src/personal_threats/` — библиотека и CLI.
- `tests/` — тесты компонентов и их взаимодействия.
- `examples/` — примеры использования, резервного копирования и диагностики.
- `models/` — инструкции по добавлению модели.

## Тесты

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

## Лицензия

[Apache-2.0](LICENSE). Отдельно поставляемые модели сохраняют свои лицензии.
