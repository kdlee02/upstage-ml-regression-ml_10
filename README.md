# House Price Prediction
## Team Five Guys

| <img src="https://github.com/ohseungtae.png" width="120"/> | <img src="https://github.com/JBreals.png" width="120"/> | <img src="https://github.com/kdlee02.png" width="120"/> | <img src="https://github.com/hwang1999.png" width="120"/> | 
| :--------------------------------------------------------: | :------------------------------------------------------: | :------------------------------------------------------: | :--------------------------------------------------------: | 
| [오승태](https://github.com/ohseungtae)                   | [김정빈](https://github.com/JBreals)                   | [이경도](https://github.com/kdlee02)                   | [황준엽](https://github.com/hwang1999)                   |
| 팀장, EDA, 결측치처리, 모델링                                   | EDA, 결측치처리, 모델링                                | EDA, 결측치처리, 이상치처리                                   | EDA, 이상치처리, 파생변수 생성                                 |

## 0. Overview
### Environment
- GPU 3080, Visual Studio Code

### Requirements

- 아래 라이브러리들이 필요합니다. 설치를 위해 `requirements.txt`를 사용할 수 있습니다:
pip install -r requirements.txt


## 1. Competiton Info

### Overview

- 주어진 데이터를 활용하여 서울의 아파트 실거래가를 효과적으로 예측하는 모델을 개발하는 대회

### Timeline

- 2025.05.01 10:00 ~ 2025.05.15 19:00

## 2. Components

### Directory

e.g.
```
├── code
│   ├── EDA
│   │   └── [OST] EDA.ipynb
│   │   └── [KJB] EDA.ipynb
│   │   └── [HJY] EDA.ipynb
│   │   └── [LKD] EDA.ipynb
|   ├── data preprocessing
│   │   └── 결측치처리
│   │   └── 이상치처리
│   │   └── 파생변수생성
│   │   └── 변수 인코딩
│   │   └── encoding.py
|   ├── modeling
│   │   └── [HJY] Lightgbm.ipynb
│   │   └── [HJY] Randomforest.ipynb
│   │   └── [OST] xgboost.ipynb
│   │   └── [OST] catboost.ipynb
│   │   └── encoding.py
...
├── docs
│   ├── pdf
│   │   └── (Template) [패스트캠퍼스] Upstage AI Lab 1기_그룹 스터디 .pptx
└── data
└── requirements.txt 
```

## 3. Data descrption

### Dataset overview

- 국토교통부에서 제공하는 아파트 실거래가 데이터
- 서울시에서 제공하는 지하철역과 버스정류장 데이터
- 평가 데이터

### EDA

- 타겟 변수, 그외 변수들에 대해 시각화 후 데이터의 특성을 파악
- 상관관계를 확인하여 변수간의 관계 파악
- 결측치 및 이상치를 시각화하여 데이터 전처리 인사이트 도출
- 도메인 지식을 조사하여 파생변수 인사이트 도출

### Data Processing

- 외부 데이터를 통한 결측치 처리
- 40% 이상 급락, 급상승 값 이상치 처리
- 도메인 지식을 이용한 다양한 파생변수 생성
- unique값이 많은 변수 타겟 인코딩 그외 라벨 인코딩
- 모델링시 사용되는 '구'별 구분을 통해 연속형 변수 결측치 선형으로 채움

## 4. Modeling

### Model descrition

- Xgboost, Lightgbm, RandomForest

### Modeling Process

- 구별로 주택 가격이 높은구, 중간구, 낮은구로 나눠서 각각 모델링
- 데이터의 왜곡 해결을 위해 '동'변수를 기준으로 stratifiedKfold로 fold를 나눔
- 각각 개별 모델 성능과 3개의 모델 성능을 앙상블한 모델로 리더보드 제출

## 5. Result

### Leader Board

![image](https://github.com/user-attachments/assets/e6d402d9-6955-42e7-b7cd-287556dff0e4)

- 5등(RMSE : 11757.4924)

![image](https://github.com/user-attachments/assets/716ace1d-731a-4380-9aad-91f4f7695778)

- 비공식 2등(RMSE : 10847.5206) mid submission때 성능이 낮아서 사용하지 못했으나 실제 성능은 좋아서 아쉬웠음.
   
## etc


### 사용한 외부 데이터
- https://www.data.go.kr/data/15058453/openapi.do#/
- https://www.data.go.kr/data/15049335/fileData.do

