"""사진별 예측 기록에서 회귀 오차와 세 가지 정답 비율을 계산합니다."""
import csv
import math
from pathlib import Path
from functools import lru_cache


def rounded_age(value):
    # 숫자 나이는 0 이상입니다. 엑셀 ROUND와 같이 0.5에서 올림합니다.
    return math.floor(float(value)+0.5)


@lru_cache(maxsize=24)
def _read_metrics(filename,modified):
    with open(filename,encoding='utf-8-sig') as file:rows=list(csv.DictReader(file))
    actual=[float(r['label_age']) for r in rows]
    predicted=[float(r['predicted_age']) for r in rows]
    errors=[p-y for p,y in zip(predicted,actual)]
    count=len(rows);mse=sum(e*e for e in errors)/count
    mean=sum(actual)/count;variance=sum((y-mean)**2 for y in actual)
    exact=sum(rounded_age(p)==y for p,y in zip(predicted,actual))
    within5=sum(abs(e)<=5 for e in errors);within10=sum(abs(e)<=10 for e in errors)
    return {'count':count,'mae':sum(abs(e) for e in errors)/count,'mse':mse,'rmse':math.sqrt(mse),
      'r2':1-sum(e*e for e in errors)/variance if variance else None,
      'exact_count':exact,'exact_accuracy':exact/count,
      'within5_count':within5,'within_5_years':within5/count,
      'within10_count':within10,'within_10_years':within10/count}


def metrics_for_csv(filename):
    path=Path(filename)
    return _read_metrics(str(path),path.stat().st_mtime_ns)
