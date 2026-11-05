"""기존 숫자 나이 데이터와 추가 ZIP을 합칩니다. 원본과 기존 결과는 보관합니다."""
from pathlib import Path, PurePosixPath
from zipfile import ZipFile
from io import BytesIO
from collections import Counter
import csv
import hashlib
import json
import random
import struct
from PIL import Image

PROJECT = Path(__file__).resolve().parent
OLD = PROJECT / 'data/age_regression'
DATA = PROJECT / 'data/age_merged'
RESULTS = PROJECT / 'results/merged'
NEW_ZIP = Path(r'C:\Users\sangn\Downloads\archive (1).zip')


def prepare():
    (DATA / 'images').mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(OLD / 'labels.csv', encoding='utf-8') as f:
        old_rows = list(csv.DictReader(f))
    old_split = json.loads((PROJECT / 'results/age_split.json').read_text())
    old_val = set(old_split['validation'])
    report = {'old_input': len(old_rows), 'new_zip': str(NEW_ZIP), 'invalid': [],
              'conflicts': [], 'duplicate_extra_files': 0}
    groups = {}

    def add(raw, age, source, old_index=None):
        try:
            with Image.open(BytesIO(raw)) as opened:
                extension = '.png' if opened.format == 'PNG' else '.jpg'
                image = opened.convert('RGB')
                # 파일 압축 형식 대신 이미지 크기와 RGB 픽셀로 중복을 비교합니다.
                digest = hashlib.sha256(struct.pack('>II', *image.size) + image.tobytes()).hexdigest()
        except (OSError, ValueError) as e:
            report['invalid'].append({'source': source, 'reason': str(e)})
            return
        entry = {'age': age, 'source': source, 'old_index': old_index}
        if digest not in groups:
            filename = 'images/' + digest + extension
            (DATA / filename).write_bytes(raw)
            groups[digest] = {'file': filename, 'entries': []}
        groups[digest]['entries'].append(entry)

    for index, row in enumerate(old_rows):
        add((OLD / row['file']).read_bytes(), int(float(row['age'])),
            'original:' + row['source'], index)
    with ZipFile(NEW_ZIP) as archive:
        files = [entry for entry in archive.infolist() if not entry.is_dir()]
        report['new_input'] = len(files)
        for number, entry in enumerate(files, 1):
            # 추가 ZIP은 파일명이 아니라 부모 폴더 이름이 나이입니다.
            label = PurePosixPath(entry.filename).parent.name
            if not label.isdigit() or not 0 <= int(label) <= 116:
                raise ValueError(f'나이 폴더를 확인하세요: {entry.filename}')
            add(archive.read(entry), int(label), 'additional:' + entry.filename)
            if number % 3000 == 0:
                print(f'추가 데이터 확인: {number}/{len(files)}', flush=True)

    rows, new_only = [], []
    for digest, group in groups.items():
        entries = group['entries']
        if len({entry['age'] for entry in entries}) > 1:
            report['conflicts'].append({'pixel_hash': digest, 'entries': entries})
            continue
        report['duplicate_extra_files'] += len(entries)-1
        old_indices = [e['old_index'] for e in entries if e['old_index'] is not None]
        # 같은 이미지가 과거 훈련/검증 양쪽에 있으면 검증 쪽에만 남깁니다.
        split = 'validation' if any(i in old_val for i in old_indices) else ('train' if old_indices else '')
        row = {'file': group['file'], 'age': entries[0]['age'], 'source': entries[0]['source'],
               'pixel_hash': digest, 'old_indices': json.dumps(old_indices), 'split': split,
               'sources': json.dumps([e['source'] for e in entries], ensure_ascii=False)}
        rows.append(row)
        if not old_indices:
            new_only.append(row)

    # 새로운 고유 사진만 8:2로 나누고 기존 검증 사진은 검증으로 유지합니다.
    random.Random(42).shuffle(new_only)
    new_val_count = int(len(new_only)*0.2)
    for index, row in enumerate(new_only):
        row['split'] = 'validation' if index < new_val_count else 'train'
    with open(DATA / 'labels.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['file', 'age', 'source', 'pixel_hash', 'old_indices', 'split', 'sources'])
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter(row['split'] for row in rows)
    report.update({'usable_images': len(rows), 'train_count': counts['train'],
                   'validation_count': counts['validation'], 'new_unique_images': len(new_only),
                   'retained_old_validation_images': sum(bool(json.loads(r['old_indices'])) and r['split']=='validation' for r in rows),
                   'retained_old_images': sum(bool(json.loads(r['old_indices'])) for r in rows),
                   'conflicting_file_count': sum(len(g['entries']) for g in report['conflicts']),
                   'age_min': min(r['age'] for r in rows), 'age_max': max(r['age'] for r in rows),
                   'age_counts': dict(sorted(Counter(r['age'] for r in rows).items()))})
    assert len(rows)+report['duplicate_extra_files']+report['conflicting_file_count']+len(report['invalid']) == len(old_rows)+report['new_input']
    (RESULTS / 'data_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ['age_counts','invalid','conflicts']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    prepare()
