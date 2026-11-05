"""파일명 맨 앞의 나이를 읽어 숫자 나이 예측용 이미지와 CSV를 만듭니다."""
from pathlib import Path, PurePosixPath
from zipfile import ZipFile
from io import BytesIO
from collections import Counter
import csv
import hashlib
import json
import re
from PIL import Image

PROJECT = Path(__file__).resolve().parent.parent
ARCHIVE = Path(r'C:\Users\sangn\Downloads\archive.zip')
DATA = PROJECT / 'data' / 'age_regression'


def prepare():
    images = DATA / 'images'
    images.mkdir(parents=True, exist_ok=True)
    report = {'source_folder': 'faces_02/part3', 'invalid': [], 'conflicts': [], 'duplicates': []}
    grouped = {}
    with ZipFile(ARCHIVE) as archive:
        for entry in archive.infolist():
            if entry.is_dir() or not entry.filename.startswith('faces_02/part3/'):
                continue
            # 예: 25_0_0_201701....jpg → 첫 숫자인 25만 나이 정답으로 사용합니다.
            name = PurePosixPath(entry.filename).name
            match = re.match(r'^(\d{1,3})_', name)
            if not match:
                raise ValueError(f'나이 라벨을 읽을 수 없습니다: {name}')
            age = int(match.group(1))
            if not 0 <= age <= 116:
                raise ValueError(f'나이 라벨 범위 확인 필요: {name}')
            raw = archive.read(entry)
            digest = hashlib.sha256(raw).hexdigest()
            grouped.setdefault(digest, []).append({'source': entry.filename, 'age': age, 'raw': raw})

    rows = []
    for digest, entries in grouped.items():
        if len({entry['age'] for entry in entries}) != 1:
            report['conflicts'].extend(entry['source'] for entry in entries)
            continue
        first = entries[0]
        report['duplicates'].extend(entry['source'] for entry in entries[1:])
        try:
            with Image.open(BytesIO(first['raw'])) as image:
                image.convert('RGB').load()
        except (OSError, ValueError) as error:
            report['invalid'].append({'source': first['source'], 'error': str(error)})
            continue
        # 확장자가 .pg 등으로 잘못 붙은 파일도 실제 이미지가 정상일 때만 사용합니다.
        filename = digest + '.jpg'
        (images / filename).write_bytes(first['raw'])
        rows.append({'file': 'images/' + filename, 'age': first['age'], 'source': first['source']})
    with open(DATA / 'labels.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['file', 'age', 'source'])
        writer.writeheader()
        writer.writerows(rows)
    ages = [row['age'] for row in rows]
    report.update({'raw_files': sum(len(v) for v in grouped.values()), 'usable_images': len(rows),
                   'age_min': min(ages), 'age_max': max(ages),
                   'age_counts': dict(sorted(Counter(ages).items()))})
    (PROJECT / 'results').mkdir(exist_ok=True)
    (PROJECT / 'results/age_data_report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print({k: v for k, v in report.items() if k not in ['age_counts', 'invalid', 'duplicates', 'conflicts']})
    print('중복:', len(report['duplicates']), '정답 충돌:', len(report['conflicts']), '손상:', len(report['invalid']))


if __name__ == '__main__':
    prepare()
