"""숫자 나이 회귀 모델의 로딩과 예측. 웹에서 학습 코드는 실행하지 않습니다."""
from pathlib import Path
from time import perf_counter
import torch
import torch.nn as nn
from torchvision import transforms, models

WEB_PATH = Path(__file__).resolve().parent
# 학습 자료와 결과는 옆의 학습 폴더에서 읽습니다.
TRAINING_PATH = WEB_PATH.parent / '얼굴나이예측'
MODEL_PATH = TRAINING_PATH / 'results/age_regression_resnet18_best.pth'
DATA_PATH = TRAINING_PATH / 'data/age_regression'
transform = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_model(model_path=MODEL_PATH):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    saved = torch.load(model_path, map_location='cpu', weights_only=True)
    if saved['task'] != 'age_regression':
        raise ValueError('숫자 나이 회귀 모델이 필요합니다.')
    architecture = saved['architecture']
    if architecture not in {'resnet18','resnet152','efficientnet_v2_s'}:
        raise ValueError('지원하지 않는 모델 구조입니다.')
    model = getattr(models, architecture)(weights=None)
    if architecture.startswith('resnet'):
        model.fc = nn.Linear(model.fc.in_features, 1)
    else:
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
    model.load_state_dict(saved['model_state_dict'])
    model = model.to(device)
    model.eval()
    names = {'resnet18':'ResNet18','resnet152':'ResNet152','efficientnet_v2_s':'EfficientNetV2-S'}
    info = {'model_name': names[architecture] + ' · 숫자 나이 예측', 'architecture': architecture,
            'task': saved['task'], 'output_dim': saved['output_dim'], 'file': Path(model_path).name,
            'dataset': saved.get('dataset', 'original'),
            'train_count': saved.get('train_count'), 'validation_count': saved.get('validation_count'),
            'saved_epoch': saved['epoch'], 'validation_mae': saved['validation_mae'],
            'target_scale': saved['target_scale'],
            'label_age_min': saved['label_age_min'], 'label_age_max': saved['label_age_max'],
            'device': str(device),
            'device_name': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}
    # 기존 저장본은 3번째 회차부터 미세조정한 모델입니다.
    info['strategy'] = saved.get('strategy', 'early8')
    info['finetune_start_epoch'] = saved.get('finetune_start_epoch', 3)
    start_epoch = info['finetune_start_epoch']
    info['actual_finetune_epochs'] = max(0, saved['epoch']-start_epoch+1) if start_epoch else 0
    info['total_parameters'] = sum(p.numel() for p in model.parameters())
    info['trainable_parameters'] = saved.get('trainable_parameters',
        513 if info['actual_finetune_epochs']==0 else 8394241)
    return model, device, info


def predict(image, model, device, target_scale):
    # 학습 때와 동일한 전처리 후 배치 차원을 추가합니다.
    x = transform(image.convert('RGB')).unsqueeze(0).to(device)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    started = perf_counter()
    with torch.no_grad():
        # 분류가 아니므로 softmax를 쓰지 않습니다. 학습 때 나눈 100을 다시 곱합니다.
        raw_age = (model(x).squeeze() * target_scale).item()
    predicted_age = min(116.0, max(0.0, raw_age))  # 학습 검증 때와 같은 범위 제한
    return predicted_age, (perf_counter() - started) * 1000
