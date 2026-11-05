# %% [markdown]
# # 기존 + 추가 데이터를 합친 ResNet18 나이 회귀
# 기존 파일명 나이 + 추가 ZIP의 나이 폴더를 함께 사용합니다. 모델과 학습 방법은 유지합니다.

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
results = project / 'results/merged'
results.mkdir(exist_ok=True)
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
# 3. 통합 준비 단계에서 정한 분할을 그대로 사용합니다.
# 기존 검증 사진은 훈련에 넣지 않고, 추가된 고유 사진만 8:2로 나눴습니다.
train_indices = [i for i, row in enumerate(dataset.rows) if row['split'] == 'train']
val_indices = [i for i, row in enumerate(dataset.rows) if row['split'] == 'validation']
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
# 4. 기존 가중치 고정 → 새 FC층으로 교체 (이번에는 출력 1개)
model = models.resnet18(weights='DEFAULT')
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, 1)
model = model.to(device)
print('새 FC층:', model.fc)

# %%
# 5. 회귀: 실제 나이와 예측 나이의 차이를 줄입니다.
# 학습할 때만 나이를 100으로 나누어 수치 범위를 줄이고, 예측 후에는 100을 곱합니다.
target_scale = 100.0
criterion = nn.L1Loss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
num_epochs = 10
history = {'train_mae': [], 'val_mae': []}
best_mae = float('inf')
started = time.time()

# %%
# 6. 학습
for epoch in range(num_epochs):
    if epoch == 2:
        # 처음 2회는 새 FC층만 학습합니다.
        # 이후에는 얼굴 나이 특징을 더 배우도록 마지막 특징 블록(layer4)도 조금 수정합니다.
        # 이 과정을 미세조정(fine-tuning)이라고 합니다. 앞쪽 블록은 계속 고정합니다.
        for param in model.layer4.parameters():
            param.requires_grad = True
        optimizer = optim.Adam([
            {'params': model.layer4.parameters(), 'lr': 0.0001},
            {'params': model.fc.parameters(), 'lr': 0.001}
        ])
        print('3번째 epoch부터 마지막 특징 블록 + FC층 미세조정', flush=True)

    # eval은 기울기 계산을 끄지 않습니다. 고정된 BatchNorm 통계를 유지합니다.
    model.eval()
    model.fc.train()
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
        checkpoint = {'model_state_dict': model.state_dict(), 'architecture': 'resnet18',
                      'task': 'age_regression', 'output_dim': 1, 'target_scale': target_scale,
                      'epoch': epoch+1, 'validation_mae': mae,
                      'label_age_min': min(ages), 'label_age_max': max(ages),
                      'dataset': 'merged', 'train_count': len(train_dataset), 'validation_count': len(val_dataset)}
        # 웹에서 모델을 읽는 순간과 겹치지 않도록 완성한 파일을 한 번에 교체합니다.
        temp_path = results / 'age_regression_resnet18_best.tmp'
        torch.save(checkpoint, temp_path)
        temp_path.replace(results / 'age_regression_resnet18_best.pth')
    (results / 'age_history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
print(f'최고 검증 MAE: {best_mae:.2f}세 / 학습 시간: {(time.time()-started)/60:.1f}분')

# %%
# 7. 저장한 최고 모델 평가 및 기준 모델과 비교
saved = torch.load(results / 'age_regression_resnet18_best.pth', map_location=device, weights_only=True)
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
# 10. 데이터 추가 전후를 같은 기존 검증 사진에서 비교합니다.
# 전체 통합 검증 지표와는 별도로 계산합니다. 새 모델만 다른 사진으로 평가하지 않습니다.
reference_indices = [i for i in val_indices if json.loads(dataset.rows[i]['old_indices'])]
reference_loader = DataLoader(Subset(dataset, reference_indices), batch_size=32, shuffle=False)
old_checkpoint = torch.load(project / 'results/age_regression_resnet18_best.pth',
                            map_location='cpu', weights_only=True)
old_model = models.resnet18(weights=None)
old_model.fc = nn.Linear(old_model.fc.in_features, 1)
old_model.load_state_dict(old_checkpoint['model_state_dict'])
old_model = old_model.to(device)
old_model.eval()
comparison = {'same_validation_count': len(reference_indices)}
for name, candidate in [('before', old_model), ('after', model)]:
    total_error, within_5 = 0.0, 0
    with torch.no_grad():
        for images, labels in reference_loader:
            guessed = (candidate(images.to(device)).squeeze(1)*100).clamp(0, 116).cpu()
            error = (guessed-labels).abs()
            total_error += error.sum().item()
            within_5 += (error <= 5).sum().item()
    comparison[name] = {'mae': total_error / len(reference_indices),
                        'within_5_years': within_5 / len(reference_indices)}
(results / 'comparison.json').write_text(json.dumps(comparison, indent=2), encoding='utf-8')
print('같은 기존 검증 사진에서 추가 전후 비교:', comparison)
