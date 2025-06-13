import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

# 계약년월 formatting
def str_to_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    # 문자열에서 특수문자 제거('-')
    df[col] = df[col].astype(str).str.replace('-', '')
    # 문자열을 datetime으로 변환
    df[col] = pd.to_datetime(df[col], format='%Y%m')
    df[col] = df[col].dt.strftime('%Y-%m')
    return df

# target encoding 함수
def gu_target_encoding(df_train: pd.DataFrame, df_test: pd.DataFrame, target_col: str='target', encode_col: str = '구', 
                   alpha: float = 0.5, global_mean: float = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_encoded_train = df_train.copy()
    df_encoded_test = df_test.copy()
    
    # 전체 평균 계산
    if global_mean is None:
        global_mean = df_train[target_col].mean()
    
    # 카테고리별 통계 계산
    category_stats = df_train.groupby(encode_col).agg({
        target_col: ['count', 'mean']
    }).reset_index()
    
    category_stats.columns = [encode_col, 'count', 'mean']
    
    # # 스무딩 적용
    # category_stats['smoothed_mean'] = (
    #     (category_stats['count'] * category_stats['mean'] + alpha * global_mean) / 
    #     (category_stats['count'] + alpha)
    # )
    
    # 인코딩 적용
    encoding_map = dict(zip(category_stats[encode_col], category_stats['mean']))
    
    # 학습 데이터에 적용
    df_encoded_train[f'{encode_col}_target'] = df_encoded_train[encode_col].map(encoding_map)
    
    # 테스트 데이터에 적용 (학습 데이터에 없는 카테고리는 전체 평균 사용)
    df_encoded_test[f'{encode_col}_target'] = df_encoded_test[encode_col].map(encoding_map)
    df_encoded_test[f'{encode_col}_target'].fillna(global_mean, inplace=True)
    
    return df_encoded_train, df_encoded_test

# 동 target encoding 함수
def dong_target_encoding(df_train: pd.DataFrame, df_test: pd.DataFrame, target_col: str = 'target') -> tuple[pd.DataFrame, pd.DataFrame]:
    df_encoded_train = df_train.copy()
    df_encoded_test = df_test.copy()

    #동 target encoding은 구 평균 집값에 대한 비율로 encoding
    df_gu_dong = df_encoded_train.groupby(['구', '동'])[target_col].mean().reset_index()
    df_gu = df_encoded_train.groupby('구')[target_col].mean().reset_index()
    df_gu.columns = ['구', '구_target']
    df_gu_dong = df_gu_dong.merge(df_gu, on='구', how='left')
    df_gu_dong['동_target'] = df_gu_dong[target_col] / df_gu_dong['구_target']
    

    encoding_map = dict(zip(df_gu_dong['동'], df_gu_dong['동_target']))

    df_encoded_train['동_target'] = df_encoded_train['동'].map(encoding_map)
    df_encoded_test['동_target'] = df_encoded_test['동'].map(encoding_map)

    # 테스트 데이터 결측치 채울 때 구 평균 사용
    df_encoded_test['동_target'].fillna(1, inplace=True)

    return df_encoded_train, df_encoded_test

# 아파트명 target encoding 함수
def apartment_target_encoding(df_train: pd.DataFrame, df_test: pd.DataFrame, target_col: str = 'target', alpha: float = 0.5) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_encoded_train = df_train.copy()
    df_encoded_test = df_test.copy()

    # 아파트명 면적당가격
    df_cost_area = df_encoded_train.groupby(['아파트명', '구', '동'])[['전용면적', target_col]].agg(['count', 'mean']).reset_index()
    df_cost_area.columns = ['아파트명', '구', '동', '_', 'area_mean', 'count', 'target_mean']
    df_cost_area['면적당가격'] = df_cost_area['target_mean'] / df_cost_area['area_mean']

    # 구 면적당가격
    df_gu = df_encoded_train.groupby('구')[['전용면적', target_col]].mean().reset_index()
    df_gu['면적당가격'] = df_gu[target_col] / df_gu['전용면적']

    # 구 면적당가격 매핑
    gu_encoding_map = dict(zip(df_gu['구'], df_gu['면적당가격']))
    df_cost_area['구_면적당가격'] = df_cost_area['구'].map(gu_encoding_map)

    # 스무딩 적용
    df_cost_area['면적당가격_smoothed'] = (
        (df_cost_area['면적당가격'] * df_cost_area['count'] + alpha * df_cost_area['구_면적당가격']) / 
        (df_cost_area['count'] + alpha)
    )
    

    # 동 면적당가격 
    df_dong = df_encoded_train.groupby('동')[['전용면적', target_col]].mean().reset_index()
    df_dong['면적당가격'] = df_dong[target_col] / df_dong['전용면적']

    # 동 면적당가격 매핑
    dong_encoding_map = dict(zip(df_dong['동'], df_dong['면적당가격']))
    df_cost_area['동_면적당가격'] = df_cost_area['동'].map(dong_encoding_map)

    # 테스트 데이터 결측치 채울 때 동 면적당가격 혹은 구 면적당가격으로 채우기
    df_encoded_test['동_면적당가격'] = df_encoded_test['동'].map(dong_encoding_map)
    df_encoded_test['구_면적당가격'] = df_encoded_test['구'].map(gu_encoding_map)

    # 아파트명 면적당가격 매핑
    df_encoding_map = dict(zip(df_cost_area['아파트명'], df_cost_area['면적당가격_smoothed']))
    df_encoded_train['아파트명_면적당가격'] = df_encoded_train['아파트명'].map(df_encoding_map)
    df_encoded_test['아파트명_면적당가격'] = df_encoded_test['아파트명'].map(df_encoding_map)
    df_encoded_test['아파트명_면적당가격'].fillna(df_encoded_test['동_면적당가격'], inplace=True)
    df_encoded_test['아파트명_면적당가격'].fillna(df_encoded_test['구_면적당가격'], inplace=True)
    df_encoded_test.drop(columns=['동_면적당가격', '구_면적당가격'], inplace=True)

    df_encoded_train.drop(columns=['아파트명', '구', '동', '시군구'], inplace=True)
    df_encoded_test.drop(columns=['아파트명', '구', '동', '시군구'], inplace=True)
    return df_encoded_train, df_encoded_test
    
# 라벨 인코딩 함수
def label_encoding(df_train: pd.DataFrame, df_test: pd.DataFrame, excp: list = []) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    dt_train = df_train.copy()
    dt_test = df_test.copy()
    categorical_features = dt_train.select_dtypes(include=[np.object]).columns
    categorical_features = categorical_features.drop(excp)

    # 각 변수에 대한 LabelEncoder를 저장할 딕셔너리
    label_encoders = {}

    # Implement Label Encoding
    for col in tqdm( categorical_features ):
        # 모든 고유값 수집 (학습 + 테스트)

        # 모든 고유값 수집 (학습 + 테스트)
        all_unique_values = pd.concat([dt_train[col], dt_test[col]]).unique()
        
        # 레이블 인코더 생성 및 학습
        lbl = LabelEncoder()
        lbl.fit(all_unique_values.astype(str))  # 문자열로 변환
        
        # 변환
        dt_train[col] = lbl.transform(dt_train[col].astype(str))
        dt_test[col] = lbl.transform(dt_test[col].astype(str))
        
        label_encoders[col] = lbl
    return dt_train, dt_test, label_encoders

# 계약년월 추세선대로 encoding 하는 함수
def contract_encoding(df_train:pd.DataFrame, df_test:pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
  df_train_copy = df_train.copy()
  df_test_copy = df_test.copy()

  df_train_copy = str_to_datetime(df_train_copy, '계약년월')
  df_test_copy = str_to_datetime(df_test_copy, '계약년월')

  df_time_check = df_train_copy.groupby('계약년월')['target'].mean().reset_index()
  df_time_check.sort_values(by='계약년월', inplace=True)
  df_time_check['계약년월'] = pd.to_datetime(df_time_check['계약년월']).dt.strftime('%Y-%m')
  
  # 계약년월을 datetime으로 변환
  df_time_check['계약년월'] = pd.to_datetime(df_time_check['계약년월'])
  df_time_check = df_time_check.sort_values('계약년월').reset_index(drop=True)
  df_time_check['연도'] = df_time_check['계약년월'].dt.year

  # 2017년 기준으로 데이터 나누기
  df_before_2017 = df_time_check[df_time_check['연도'] < 2017].copy()
  df_after_2017 = df_time_check[df_time_check['연도'] >= 2017].copy()

  # 인코딩 맵 저장용 dict
  encoding_map = {}


  # 인코딩 값 생성
  def trend_encode(df, label):
      if len(df) < 2:
          return None
      n = len(df)
      y_start = df['target'].values[:3].mean()
      x = np.arange(n).reshape(-1, 1)
      y = df['target'].values.reshape(-1, 1)
      model = LinearRegression().fit(x, y)
      y_end = model.predict([[n - 1]])[0][0]
      y_trend = np.linspace(y_start, y_end, n)

      # 인코딩 맵에 저장
      for i, dt in enumerate(df['계약년월']):
          ym_str = dt.strftime('%Y-%m')
          encoding_map[ym_str] = y_trend[i]

      return model, y_start, n

  # 각 구간별 추세선 및 인코딩
  model_before, _, _ = trend_encode(df_before_2017, '2017년 이전 추세')
  model_after, y_start_after, n_after = trend_encode(df_after_2017, '2017년 이후 추세')

  # === 3개월 미래 예측 ===
  future_months = 3
  last_date = df_after_2017['계약년월'].max()

  for i in range(1, future_months + 1):
      future_date = last_date + pd.DateOffset(months=i)
      x_future = np.array([[n_after - 1 + i]])
      y_future = model_after.predict(x_future)[0][0]
      encoding_map[future_date.strftime('%Y-%m')] = y_future

  df_train_copy['계약년월'] = df_train_copy['계약년월'].map(encoding_map)
  df_test_copy['계약년월'] = df_test_copy['계약년월'].map(encoding_map)

  return df_train_copy, df_test_copy

