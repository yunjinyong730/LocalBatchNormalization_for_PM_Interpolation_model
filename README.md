# 🗺️ 미세먼지 보간 모델을 위한 지역배치 정규화 기법 도입
## Local Batch Normalization Technique for Improving Air Pollution Interpolation Models

### **저자 및 소속**

- **윤진용**, 안석호, 서영덕(인하대학교 컴퓨터공학과 / 전기컴퓨터공학과)

### **연구 목적**

- 미세먼지 측정소 부족 문제를 해결하기 위한 보간(interpolation) 모델 개발
- 기존 모델에서 고려하지 못한 공간적 안정성을 강화하기 위해 지역 배치 정규화(Batch Normalization, BN)를 도입.

### **핵심 내용**

1. **배치 정규화 기법의 활용**
    - **Global BN**: 전체 측정소 데이터를 기반으로 정규화.
    - **Local BN**: 각 측정소별 데이터로 정규화, 공간적 특성 반영.
    - 시간적 데이터 일관성을 유지하면서 공간적 안정성을 확보.
2. **실험 데이터**
    - **지역**: Antwerp 지역의 32개 미세먼지 측정소.
    - **데이터 종류**: PM2.5 센서 데이터를 사용 (분 단위 측정).
    - 결측치는 IDW (Inverse Distance Weighting) 기법으로 보완.
3. **모델 구조 및 실험**
    - ConvLSTM 및 CNN 모델에 Global BN과 Local BN을 적용하여 성능 비교.
    - 모델 평가 지표로 spRMSE 사용.
4. **결과**
    - Local BN > Global BN > Baseline 순으로 보간 정확도 향상.
    - 특정 시점에서 Local BN을 적용한 보간값은 실제 측정값과 근소한 차이만 존재.

### **결론**

- 배치 정규화를 통해 미세먼지 보간 모델의 성능 향상 가능.
- 특히, Local BN 방식은 공간적 특성을 반영하여 기존 모델 대비 더 높은 정밀도를 제공.

### **키워드**

- Air Pollution, Interpolation, ConvLSTM, Batch Normalization, PM2.5

### **참고 문헌**

1. V. D. Le et al., "Spatiotemporal Deep Learning Model for CityWide Air Pollution Interpolation and Prediction", IEEE BIGCOMP, 2020.
2. D. Wong et al., "Comparison of spatial interpolation methods for the estimation of air quality data", Nature Journal of Exposure Science & Environmental Epidemiology, 2004.
3. S. H. Kim et al., "Explainable AI-driven high-fidelity IAQ prediction model for subway stations", Building and Environment, 2024.
4. C. S. Laurent et al., "Batch Normalized Recurrent Neural Networks", IEEE ICASSP, 2015

...

### **그림 및 표 요약**

<img width="701" alt="그림1" src="https://github.com/user-attachments/assets/4babeb0b-2cda-47ae-8954-789b09a8678b">

- **Fig 1**: 모델 구조도 - 기존 Global BN과 Local BN 적용 차이를 시각적으로 설명.

<img width="770" alt="표1" src="https://github.com/user-attachments/assets/07007553-5a04-42cb-964d-a42a889caf09">

- **표 1**: 모델별 보간 성능 비교 - Local BN 모델이 가장 높은 정확도를 기록.

