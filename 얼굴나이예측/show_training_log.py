"""저장된 실제 학습 기록을 읽어 화면에 표시함. 재학습·모델 수정은 하지 않음."""
from pathlib import Path
import json
import tkinter as tk
from tkinter.scrolledtext import ScrolledText

root_path=Path(__file__).resolve().parent
saved_path=root_path/'results/architectures/efficientnet_v2_s'
history=json.loads((saved_path/'age_history.json').read_text(encoding='utf-8'))
metrics=json.loads((saved_path/'age_metrics.json').read_text(encoding='utf-8'))
lines=['EfficientNetV2-S 실제 저장 학습 로그',
       '훈련 10,190장 / 검증 2,546장 / 총 10에포크',
       '1~2회: FC만 학습 / 3~10회: 마지막 특징 모듈 2개 + FC',
       '저장 기록 조회 화면입니다. 새로운 학습을 실행하는 화면이 아닙니다.','']
for epoch,(train,val) in enumerate(zip(history['train_mae'],history['val_mae']),1):
    suffix='  ← 최고 저장 모델' if epoch==metrics['saved_epoch'] else ''
    lines.append(f'[{epoch:2d}/10] 훈련 MAE {train:5.2f}세 | 검증 MAE {val:5.2f}세{suffix}')
lines += ['',f"최고 검증 MAE: {metrics['validation_mae']:.2f}세 / 저장 회차: {metrics['saved_epoch']}회차",'',
          '원본 학습 코드: train.py',
          '원본 기록: results/architectures/efficientnet_v2_s/age_history.json']
text='\n'.join(lines)
output=saved_path/'training_log.txt'
output.write_text(text,encoding='utf-8-sig')
print(text)
window=tk.Tk();window.title('FaceAge · 실제 저장 학습 로그 확인')
window.geometry('1000x650+60+70')
box=ScrolledText(window,wrap='word',font=('맑은 고딕',12),padx=18,pady=18)
box.pack(fill='both',expand=True);box.insert('1.0',text)
box.tag_configure('best',foreground='#175E3A',font=('맑은 고딕',12,'bold'))
for i,line in enumerate(lines,1):
    if '최고 저장 모델' in line or '최고 검증 MAE:' in line:box.tag_add('best',f'{i}.0',f'{i}.end')
box.configure(state='disabled')
window.mainloop()
