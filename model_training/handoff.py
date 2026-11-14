"""Rebuild the exact Colab dataset and continue the saved five-class run."""
from __future__ import annotations
import argparse
import getpass
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import zipfile

HERE = Path(__file__).resolve().parent
ASSETS = HERE / 'handoff'
DATA = HERE / 'data' / 'expanded_handoff'


def read_manifest():
    with gzip.open(ASSETS / 'expanded_dataset.json.gz', 'rt') as f:
        return json.load(f)


def download_source(source, raw, key):
    import requests
    project, version = source['project'], source['version']
    dest = raw / (project.replace('/', '__') + f'__v{version}')
    if (dest / '.download_complete').exists() or (dest / 'data.yaml').exists():
        return
    # Keep credentials out of exception messages and console output.
    try:
        link = None
        for _ in range(120):
            response = requests.get(f'https://api.roboflow.com/{project}/{version}/yolov8',
                                    params={'api_key': key}, timeout=60)
            if response.status_code not in (200, 202):
                raise ValueError('export unavailable')
            link = (response.json().get('export') or {}).get('link')
            if link:
                break
            time.sleep(5)
        if not link:
            raise ValueError('export timeout')
        raw.mkdir(parents=True, exist_ok=True)
        archive = raw / (dest.name + '.zip')
        with requests.get(link, stream=True, timeout=180) as response:
            response.raise_for_status()
            with archive.open('wb') as f:
                for chunk in response.iter_content(1024 * 1024):
                    f.write(chunk)
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                if not (dest / name).resolve().is_relative_to(dest.resolve()):
                    raise ValueError('invalid archive path')
            z.extractall(dest)
        (dest / '.download_complete').touch()
        archive.unlink()
    except Exception:
        raise RuntimeError(f'Не удалось скачать {project} v{version}; проверьте доступ к проекту и сеть. Повторите команду.') from None


def prepare(args):
    import yaml
    meta = read_manifest()
    sources = json.loads((ASSETS / 'sources.json').read_text())
    raw = args.raw.resolve()
    missing = [s for s in sources if not (raw / (s['project'].replace('/', '__') + f"__v{s['version']}") / 'data.yaml').exists()]
    key = (os.environ.get('ROBOFLOW_API_KEY') or getpass.getpass('Roboflow Private API Key (скрыто): ')) if missing and not args.offline else ''
    if not args.offline:
        for i, s in enumerate(sources, 1):
            print(f"[{i}/{len(sources)}] {s['project']}", flush=True)
            download_source(s, raw, key)
    key = ''
    wanted = {r['sha256'] for r in meta['manifest']}
    found = {}
    for source in sources:
        folder = raw / (source['project'].replace('/', '__') + f"__v{source['version']}")
        if not folder.exists():
            folder = raw / source['project'].replace('/', '__')
        for p in folder.rglob('*'):
            if p.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'} or 'images' not in p.parts:
                continue
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            if digest in wanted:
                found[digest] = p
    absent = wanted - found.keys()
    if absent:
        raise SystemExit(f'Не найдено {len(absent)} изображений с нужным SHA256. Датасет не готов; обучение не запущено. Проверьте версии экспортов.')
    out = args.data.resolve()
    if out.exists():
        raise SystemExit('Каталог уже существует: укажите новый --data. Существующие данные сохранены.')
    destination = out
    out = destination.with_name(destination.name + '.preparing')
    if out.exists():
        raise SystemExit(f'Есть незавершённая подготовка: {out}. Укажите новый --data или проверьте этот каталог.')
    for r in meta['manifest']:
        if r['split'] not in {'train', 'val'} or Path(r['file']).name != r['file']:
            raise ValueError('Invalid manifest path')
        image = out / 'images' / r['split'] / r['file']
        label = out / 'labels' / r['split'] / (Path(r['file']).stem + '.txt')
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(found[r['sha256']], image)
        except OSError:
            shutil.copyfile(found[r['sha256']], image)
        label.write_text('\n'.join(r['labels']) + ('\n' if r['labels'] else ''))
    import yaml
    (out / 'data.yaml').write_text(yaml.safe_dump(dict(path=str(destination), train='images/train', val='images/val', names=meta['classes'], nc=len(meta['classes']))))
    (out / 'stats.json').write_text(json.dumps(meta, ensure_ascii=False))
    out.rename(destination)
    print('Готово:', destination / 'data.yaml', meta['images'])


def train(args):
    import yaml
    from ultralytics import YOLO
    data = args.data.resolve() / 'data.yaml'
    cfg = yaml.safe_load(data.read_text())
    expected = read_manifest()['classes']
    if cfg['names'] != expected:
        raise SystemExit('Неверные классы или порядок классов в data.yaml')
    checkpoint = args.weights.resolve()
    if not checkpoint.is_file():
        raise SystemExit(f'Нет весов: {checkpoint}. Сначала получите файлы handoff.')
    # Only load trusted project checkpoints: torch .pt files can contain pickle code.
    model = YOLO(str(checkpoint))
    run = args.run.resolve()
    if run.exists():
        raise SystemExit('Каталог результата уже существует; задайте новый --run.')
    options = dict(data=str(data), device=args.device, batch=args.batch, workers=args.workers,
                   save_dir=str(run), plots=True)
    if args.command == 'resume':
        if model.ckpt.get('epoch', -1) < 0 or model.ckpt.get('optimizer') is None:
            raise SystemExit('Веса не содержат состояния продолжения. Используйте finetune с best.pt.')
        if list(model.names.values()) != expected:
            raise SystemExit('Resume требует те же 5 классов. Для расширения используйте finetune.')
        print('Продолжение с эпохи', model.ckpt['epoch'] + 2, 'до исходного числа эпох:', model.ckpt['train_args']['epochs'])
        options['resume'] = str(checkpoint)
    else:
        options.update(epochs=args.epochs, imgsz=640, project=str(run.parent), name=run.name,
                       patience=12, cos_lr=True, close_mosaic=5, seed=0)
    if args.dry_run:
        print(json.dumps(options, ensure_ascii=False, indent=2))
        return
    # Write provenance before training so it survives an interruption.
    def record(trainer):
        dest = Path(trainer.save_dir)
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ASSETS / 'expanded_dataset.json.gz', dest / 'expanded_dataset.json.gz')
        (dest / 'handoff_run.json').write_text(json.dumps(dict(checkpoint=str(checkpoint),
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(), mode=args.command,
            data=str(data), classes=expected), ensure_ascii=False, indent=2))
    model.add_callback('on_pretrain_routine_start', record)
    model.train(**options)
    print('Веса:', Path(model.trainer.save_dir) / 'weights')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    subs = ap.add_subparsers(dest='command', required=True)
    p = subs.add_parser('prepare', help='Скачать 12 источников и восстановить точное разбиение Colab')
    p.add_argument('--raw', type=Path, default=HERE / 'data' / 'raw')
    p.add_argument('--data', type=Path, default=DATA)
    p.add_argument('--offline', action='store_true')
    for mode in ('resume', 'finetune'):
        p = subs.add_parser(mode)
        p.add_argument('--weights', type=Path, default=ASSETS / 'expanded' / 'weights' / ('last.pt' if mode == 'resume' else 'best.pt'))
        p.add_argument('--data', type=Path, default=DATA)
        p.add_argument('--run', type=Path, default=HERE / 'runs' / ('expanded_' + mode))
        p.add_argument('--device', default='0')
        p.add_argument('--batch', type=int, default=16)
        p.add_argument('--workers', type=int, default=2)
        p.add_argument('--epochs', type=int, default=30, help='Только finetune: число новых эпох')
        p.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    prepare(args) if args.command == 'prepare' else train(args)

if __name__ == '__main__':
    main()
