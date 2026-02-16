# 인간 라벨링을 위한 데이터 전처리

import pandas as pd
import math

# training_data.csv 파일 읽기
df = pd.read_csv('training_data.csv')

# 파일명 마지막 글자로 gender 열 생성 ('F' 또는 'M')
df['gender'] = df['filename'].str[-1]

# 원하는 총 샘플 개수 (10% 기준)
desired_female_total = 1880
desired_male_total = 595

# cluster_label 별 샘플 개수 결정 (균등 분배)
female_label0 = math.floor(desired_female_total / 2)
female_label1 = desired_female_total - female_label0
male_label0 = math.floor(desired_male_total / 2)
male_label1 = desired_male_total - male_label0

# 성별에 따라 데이터 분리
female_df = df[df['gender'] == 'F']
male_df = df[df['gender'] == 'M']

# 각 성별 내에서 cluster_label별로 샘플링  
sampled_f0 = female_df[female_df['cluster_label'] == 0].sample(n=female_label0, random_state=42)
sampled_f1 = female_df[female_df['cluster_label'] == 1].sample(n=female_label1, random_state=42)
sampled_m0 = male_df[male_df['cluster_label'] == 0].sample(n=male_label0, random_state=42)
sampled_m1 = male_df[male_df['cluster_label'] == 1].sample(n=male_label1, random_state=42)

# 샘플링 결과를 합치고 filename 기준 오름차순 정렬
result = pd.concat([sampled_f0, sampled_f1, sampled_m0, sampled_m1])
result = result.sort_values(by='filename')

# 샘플링 결과를 합치고 filename 기준 오름차순 정렬
result = pd.concat([sampled_f0, sampled_f1, sampled_m0, sampled_m1])
result = result.sort_values(by='filename')

# 각각의 데이터 개수 프린트
print("여성 cluster_label 0:", len(sampled_f0))
print("여성 cluster_label 1:", len(sampled_f1))
print("남성 cluster_label 0:", len(sampled_m0))
print("남성 cluster_label 1:", len(sampled_m1))
print("전체 샘플 개수:", len(result))


result.to_csv('training_data_filtered.csv', index=False)

print("Filtered data saved as training_data_filtered.csv")