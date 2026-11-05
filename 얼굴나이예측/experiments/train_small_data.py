# %% [markdown]
# # 데이터 수 비교: EfficientNetV2-S, 기존 사진의 훈련 2,510장만 사용
# 기존 3,137장은 훈련 2,510장 + 기존 검증 627장입니다.
# 모델 구조와 학습법을 유지하고, 훈련 사진 수만 10,190장 → 2,510장으로 줄입니다.
# 공정하게 비교하려고 검증 사진 2,546장은 통합 실험과 똑같이 유지합니다.
# 추가 사진은 훈련에 사용하지 않습니다. 2,546장은 별도 최종 테스트가 아닌 모델 선택용 검증입니다.

# %%
# 0. 필요한 모듈
from pathlib import Path
import csv
import json
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt

project = Path(__file__).resolve().parent.parent if '__file__' in globals() else Path.cwd()
data_path = project / 'data/age_merged'
results = project / 'results/data_size_comparison/efficientnet_v2_s_small'
results.mkdir(parents=True, exist_ok=True)
torch.manual_seed(42)
random.seed(42)
# 작은 이미지 전처리에 지나치게 많은 CPU 스레드를 쓰지 않도록 제한합니다.
torch.set_num_threads(4)

# %%
# 1. 이미지 전처리. 웹 예측에서도 똑같이 사용해야 합니다.
transform = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# %%
# 2. 이번에는 폴더명이 아닌 CSV의 나이 숫자가 정답입니다.
# 그래서 ImageFolder 대신 Dataset의 __len__, __getitem__을 직접 만듭니다.
class AgeDataset(Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        with open(root / 'labels.csv', encoding='utf-8') as file:
            self.rows = list(csv.DictReader(file))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row['file']) as image:
            image = self.transform(image.convert('RGB'))
        # 분류의 정수 번호가 아니라 소수 계산이 가능한 float 정답입니다.
        age = torch.tensor(float(row['age']), dtype=torch.float32)
        return image, age

dataset = AgeDataset(data_path, transform)
ages = [float(row['age']) for row in dataset.rows]
print('이미지:', len(dataset), '| 나이 라벨 범위:', min(ages), '~', max(ages))

# %%
# 3. 기존 사진 중 훈련으로 정해둔 2,510장만 사용합니다.
# old_indices가 있으면 기존 3,137장에 포함됐던 사진입니다.
train_indices = [i for i, row in enumerate(dataset.rows)
                 if row['split'] == 'train' and json.loads(row['old_indices'])]
val_indices = [i for i, row in enumerate(dataset.rows) if row['split'] == 'validation']
assert len(train_indices) == 2510 and len(val_indices) == 2546
original_split = json.loads((project/'results/age_split.json').read_text(encoding='utf-8'))
old_train_ids = {j for i in train_indices for j in json.loads(dataset.rows[i]['old_indices'])}
assert old_train_ids == set(original_split['train'])
# 나이 배열은 원래 행 번호로 조회할 수 있도록 통합 목록 순서를 유지합니다.
ages = [float(row['age']) for row in dataset.rows]
train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
train_hashes = {dataset.rows[i]['pixel_hash'] for i in train_indices}
val_hashes = {dataset.rows[i]['pixel_hash'] for i in val_indices}
assert train_hashes.isdisjoint(val_hashes), '훈련/검증에 같은 픽셀이 중복됨'
split = {'train': train_indices, 'validation': val_indices}
(results / 'age_split.json').write_text(json.dumps(split, indent=2), encoding='utf-8')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('훈련:', len(train_dataset), '검증:', len(val_dataset))
print('장치:', torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU')
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0,
                          pin_memory=device.type == 'cuda')
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0,
                        pin_memory=device.type == 'cuda')

# %%
# EfficientNet은 fc 대신 classifier[-1], layer4 대신 마지막 두 특징 모듈을 사용합니다.
# 4. 기존 가중치 고정 → 새 FC층으로 교체 (이번에는 출력 1개)
model = models.efficientnet_v2_s(weights='DEFAULT')
for param in model.parameters():
    param.requires_grad = False
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
model = model.to(device)
print('새 출력층:', model.classifier[-1])

# %%
# 5. 회귀: 실제 나이와 예측 나이의 차이를 줄입니다.
# 학습할 때만 나이를 100으로 나누어 수치 범위를 줄이고, 예측 후에는 100을 곱합니다.
target_scale = 100.0
criterion = nn.L1Loss()
optimizer = optim.Adam(model.classifier[-1].parameters(), lr=0.001)
num_epochs = 10
history = {'train_mae': [], 'val_mae': []}
best_mae = float('inf')
started = time.time()

# %%
# 6. 학습
for epoch in range(num_epochs):
    if epoch == 2:
        # 처음 2회는 새 FC층만 학습합니다.
        # 이후에는 얼굴 나이 특징을 더 배우도록 마지막 특징 모듈 features[6], features[7]도 조금 수정합니다.
        # 이 과정을 미세조정(fine-tuning)이라고 합니다. 앞쪽 블록은 계속 고정합니다.
        for param in model.features[-2:].parameters():
            param.requires_grad = True
        optimizer = optim.Adam([
            {'params': model.features[-2:].parameters(), 'lr': 0.0001},
            {'params': model.classifier[-1].parameters(), 'lr': 0.001}
        ])
        print('3번째 epoch부터 마지막 특징 블록 + FC층 미세조정', flush=True)

    # eval은 기울기 계산을 끄지 않습니다. 고정된 BatchNorm 통계를 유지합니다.
    model.eval()
    model.classifier[-1].train()
    train_error = 0.0
    for step, (images, labels) in enumerate(train_loader, 1):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images).squeeze(1)  # [배치, 1] → [배치]
        loss = criterion(outputs, labels / target_scale)
        loss.backward()
        optimizer.step()
        predicted_age = (outputs.detach() * target_scale).clamp(0, 116)
        train_error += (predicted_age - labels).abs().sum().item()

    # 검증에는 가중치 수정이 없습니다. 오차 단위는 '세'입니다.
    model.eval()
    val_error = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            predictions = (model(images.to(device)).squeeze(1) * target_scale).clamp(0, 116)
            val_error += (predictions - labels.to(device)).abs().sum().item()
    history['train_mae'].append(train_error / len(train_dataset))
    history['val_mae'].append(val_error / len(val_dataset))
    mae = history['val_mae'][-1]
    print(f"[{epoch+1}/{num_epochs}] 훈련 MAE {history['train_mae'][-1]:.2f}세 | "
          f'검증 MAE {mae:.2f}세', flush=True)
    if mae < best_mae:
        best_mae = mae
        checkpoint = {'model_state_dict': model.state_dict(), 'architecture': 'efficientnet_v2_s',
                      'task': 'age_regression', 'output_dim': 1, 'target_scale': target_scale,
                      'epoch': epoch+1, 'validation_mae': mae,
                      'label_age_min': min(ages), 'label_age_max': max(ages),
                      'dataset': 'original_train_common_validation', 'strategy': 'early8', 'finetune_start_epoch': 3,
                      'training_source_total': 3137, 'validation_protocol': 'same_2546_as_large_model',
                      'total_parameters': sum(p.numel() for p in model.parameters()),
                      'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad), 'train_count': len(train_dataset), 'validation_count': len(val_dataset)}
        # 웹에서 모델을 읽는 순간과 겹치지 않도록 완성한 파일을 한 번에 교체합니다.
        temp_path = results / 'age_regression_best.tmp'
        torch.save(checkpoint, temp_path)
        temp_path.replace(results / 'age_regression_best.pth')
    (results / 'age_history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
print(f'최고 검증 MAE: {best_mae:.2f}세 / 학습 시간: {(time.time()-started)/60:.1f}분')

# %%
# 7. 저장한 최고 모델 평가 및 기준 모델과 비교
saved = torch.load(results / 'age_regression_best.pth', map_location=device, weights_only=True)
model.load_state_dict(saved['model_state_dict'])
model.eval()
actuals, predictions = [], []
with torch.no_grad():
    for images, labels in val_loader:
        guessed = (model(images.to(device)).squeeze(1) * target_scale).clamp(0, 116).cpu().tolist()
        predictions.extend(guessed)
        actuals.extend(labels.tolist())
errors = [abs(a-p) for a, p in zip(actuals, predictions)]
train_median = torch.tensor([ages[i] for i in train_dataset.indices]).median().item()
baseline = sum(abs(a-train_median) for a in actuals) / len(actuals)
metrics = {'validation_mae': sum(errors)/len(errors),
           'median_absolute_error': torch.tensor(errors).median().item(),
           'baseline_mae': baseline, 'baseline_constant_age': train_median,
           'within_5_years': sum(e <= 5 for e in errors)/len(errors),
           'validation_count': len(actuals), 'train_count': len(train_dataset),
           'saved_epoch': saved['epoch']}
(results / 'age_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
with open(results / 'age_validation_predictions.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=['file', 'label_age', 'predicted_age', 'absolute_error'])
    writer.writeheader()
    for i, actual, prediction, error in zip(val_dataset.indices, actuals, predictions, errors):
        writer.writerow({'file': dataset.rows[i]['file'], 'label_age': actual,
                         'predicted_age': prediction, 'absolute_error': error})
print('무조건 훈련 나이 중앙값만 출력하는 기준 MAE:', round(baseline, 2), '세')
print('±5세 이내 예측 비율:', f"{metrics['within_5_years']:.2%}")

# %%
# 8. 학습 그래프와 검증 예측 산점도 저장
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(range(1, num_epochs+1), history['train_mae'], label='Train')
axes[0].plot(range(1, num_epochs+1), history['val_mae'], label='Validation')
axes[0].set(xlabel='Epoch', ylabel='MAE (years)', title='Lower is better')
axes[0].legend()
axes[1].scatter(actuals, predictions, s=9, alpha=0.45)
axes[1].plot([0, 116], [0, 116], '--', color='gray')
axes[1].set(xlabel='Label age', ylabel='Predicted age', title='Validation predictions')
plt.tight_layout()
plt.savefig(results / 'age_training_history.png')
plt.show()
plt.close(fig)

# %%
# 9. 검증 사진을 앞에서부터 12장 선택합니다. 잘 맞힌 사진만 골라내지 않습니다.
examples = [{'file': dataset.rows[i]['file'], 'age': float(dataset.rows[i]['age'])}
            for i in val_dataset.indices[:12]]
(results / 'age_examples.json').write_text(json.dumps(examples, indent=2), encoding='utf-8')


# %%
# 학습과 검증이 모두 끝났다는 표시입니다.
(results / 'ready.json').write_text(json.dumps({'architecture':saved['architecture'],'validation_mae':saved['validation_mae']},indent=2),encoding='utf-8')
