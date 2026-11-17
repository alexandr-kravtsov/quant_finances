# models/registry.py
import json
import joblib
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger("model_registry")

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def save_model(model, ticker: str, metadata: Dict[str, Any]) -> Path:
    """
    Сохраняет модель и метаданные в реестр.
    
    Параметры:
        model: обученная модель (любой sklearn-совместимый объект)
        ticker: тикер
        metadata: словарь с метаданными (должен содержать 'model_type')
    
    Возвращает:
        Path: путь к сохранённой модели
    """
    model_type = metadata.get('model_type', 'unknown')
    
    # Определяем следующую версию
    ticker_dir = MODELS_DIR / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)
    
    # Ищем существующие версии этого типа
    existing = list(ticker_dir.glob(f"{model_type}_v*.pkl"))
    if existing:
        versions = []
        for p in existing:
            try:
                # Извлекаем номер версии из имени файла
                version_str = p.stem.split('_v')[-1]
                versions.append(int(version_str))
            except:
                pass
        next_version = max(versions) + 1 if versions else 1
    else:
        next_version = 1
    
    # Имена файлов
    model_filename = f"{model_type}_v{next_version}.pkl"
    meta_filename = f"{model_type}_v{next_version}_meta.json"
    
    model_path = ticker_dir / model_filename
    meta_path = ticker_dir / meta_filename
    
    # Сохраняем модель
    joblib.dump(model, model_path)
    
    # Добавляем версию и дату в метаданные
    metadata['version'] = next_version
    metadata['model_type'] = model_type
    metadata['ticker'] = ticker
    metadata['created_at'] = datetime.now().isoformat()
    metadata['status'] = 'PAPER'  # по умолчанию — не активна
    
    # Сохраняем метаданные
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # Обновляем общий metadata.json (хранит последнюю активную модель)
    main_meta_path = ticker_dir / "metadata.json"
    if main_meta_path.exists():
        with open(main_meta_path, 'r') as f:
            main_meta = json.load(f)
    else:
        main_meta = {}
    
    # Если это первая модель, делаем её ACTIVE автоматически
    if not main_meta.get('active'):
        metadata['status'] = 'ACTIVE'
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        main_meta['active'] = {
            'model_type': model_type,
            'version': next_version
        }
        with open(main_meta_path, 'w', encoding='utf-8') as f:
            json.dump(main_meta, f, indent=2, ensure_ascii=False)
    else:
        # Обновляем main_meta, но не перезаписываем active
        with open(main_meta_path, 'w', encoding='utf-8') as f:
            json.dump(main_meta, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Модель сохранена: {model_path} (версия {next_version})")
    return model_path


def load_model(ticker: str, model_type: str, version: int):
    """
    Загружает конкретную версию модели.
    """
    model_path = MODELS_DIR / ticker / f"{model_type}_v{version}.pkl"
    if not model_path.exists():
        logger.error(f"Модель не найдена: {model_path}")
        return None
    return joblib.load(model_path)


def get_active_model(ticker: str):
    """
    Возвращает активную модель и её метаданные для тикера.
    """
    meta = get_model_info(ticker)
    if meta is None:
        return None, None
    model = load_model(ticker, meta['model_type'], meta['version'])
    return model, meta


def get_model_info(ticker: str) -> Optional[Dict]:
    """
    Возвращает метаданные активной модели для тикера.
    Если активной нет, возвращает None.
    """
    ticker_dir = MODELS_DIR / ticker
    if not ticker_dir.exists():
        return None
    
    main_meta_path = ticker_dir / "metadata.json"
    if not main_meta_path.exists():
        return None
    
    with open(main_meta_path, 'r') as f:
        main_meta = json.load(f)
    
    active = main_meta.get('active')
    if not active:
        return None
    
    model_type = active.get('model_type')
    version = active.get('version')
    if not model_type or not version:
        return None
    
    # Загружаем полные метаданные этой версии
    meta_path = ticker_dir / f"{model_type}_v{version}_meta.json"
    if not meta_path.exists():
        return None
    
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    return meta


def list_models(ticker: str) -> List[Dict]:
    """
    Возвращает список всех версий моделей для тикера с метаданными.
    """
    ticker_dir = MODELS_DIR / ticker
    if not ticker_dir.exists():
        return []
    
    versions = []
    for meta_path in ticker_dir.glob("*_meta.json"):
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            versions.append(meta)
        except:
            continue
    
    # Сортируем по версии
    versions.sort(key=lambda x: x.get('version', 0))
    return versions


def activate_model(ticker: str, model_type: str, version: int) -> bool:
    """
    Активирует указанную версию модели (переводит её в статус ACTIVE).
    Все остальные модели для этого тикера становятся PAPER.
    """
    ticker_dir = MODELS_DIR / ticker
    if not ticker_dir.exists():
        logger.error(f"Тикер {ticker} не найден")
        return False
    
    # Проверяем, что модель существует
    model_path = ticker_dir / f"{model_type}_v{version}.pkl"
    if not model_path.exists():
        logger.error(f"Модель {model_type}_v{version} не найдена")
        return False
    
    # Обновляем метаданные всех моделей: делаем их PAPER
    for meta_path in ticker_dir.glob("*_meta.json"):
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            if meta.get('model_type') == model_type and meta.get('version') == version:
                meta['status'] = 'ACTIVE'
            else:
                meta['status'] = 'PAPER'
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
        except:
            continue
    
    # Обновляем основной metadata.json
    main_meta_path = ticker_dir / "metadata.json"
    main_meta = {}
    if main_meta_path.exists():
        with open(main_meta_path, 'r') as f:
            main_meta = json.load(f)
    
    main_meta['active'] = {
        'model_type': model_type,
        'version': version
    }
    with open(main_meta_path, 'w', encoding='utf-8') as f:
        json.dump(main_meta, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Модель {model_type}_v{version} активирована для {ticker}")
    return True


def deactivate_model(ticker: str, model_type: str, version: int) -> bool:
    """
    Деактивирует модель (переводит в статус RETIRED).
    Если это была активная модель, активная удаляется из metadata.json.
    """
    ticker_dir = MODELS_DIR / ticker
    if not ticker_dir.exists():
        return False
    
    # Обновляем метаданные модели
    meta_path = ticker_dir / f"{model_type}_v{version}_meta.json"
    if not meta_path.exists():
        return False
    
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    meta['status'] = 'RETIRED'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    
    # Если это была активная, удаляем из main_meta
    main_meta_path = ticker_dir / "metadata.json"
    if main_meta_path.exists():
        with open(main_meta_path, 'r') as f:
            main_meta = json.load(f)
        if main_meta.get('active', {}).get('model_type') == model_type and \
           main_meta.get('active', {}).get('version') == version:
            del main_meta['active']
            with open(main_meta_path, 'w', encoding='utf-8') as f:
                json.dump(main_meta, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Модель {model_type}_v{version} деактивирована")
    return True