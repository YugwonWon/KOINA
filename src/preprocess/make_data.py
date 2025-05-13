import pandas as pd

# 학습 데이터 생성을 위한 전처리
# 데이터 로드
female_data = pd.read_csv("clustered_features_Female_k2.csv")
male_data = pd.read_csv("clustered_features_Male_k2.csv")

# 여성 데이터 레이블 반전 (0 ↔ 1)
female_data["cluster_label"] = female_data["cluster_label"].apply(lambda x: 1 if x == 0 else 0)

# 남녀 데이터 합치기
all_data = pd.concat([female_data, male_data], ignore_index=True)

# 저장
all_data.to_csv("training_data.csv", index=False)
print("학습 데이터를 training_data.csv로 저장 완료!")
