# Streaming Algorithm Experiment (Bloom Filter & Count-Min Sketch)

대용량 데이터 스트림에서 전체 데이터를 저장하지 않고 근사 계산을 수행하는
스트리밍 알고리즘 2종을 직접 구현하고 정확도·메모리·시간 트레이드오프를 분석.

## 파일
- `gen_data.py`     : 200,000건 클릭스트림 데이터 생성 (Zipf 분포)
- `algorithms.py`   : Bloom Filter, Count-Min Sketch 순수 Python 구현
- `run_experiment.py`: Ground truth 계산, 파라미터 스윕, 정확도/메모리/시간 측정, 그림 생성
- `build_report.py` : 결과 PDF 보고서 생성

## 실행
```
python gen_data.py
python run_experiment.py
python build_report.py
```
