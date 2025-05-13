import pandas as pd
import numpy as np

# CSV 파일 로드
df = pd.read_csv("aligner_time_difference_ms.csv")

# === 이상치 제거 (IQR 기반) ===
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)  # 1분위수 (25%)
    Q3 = df[column].quantile(0.75)  # 3분위수 (75%)
    IQR = Q3 - Q1  # IQR 계산
    lower_bound = Q1 - 1.5 * IQR  # 하한값
    upper_bound = Q3 + 1.5 * IQR  # 상한값
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# 이상치 제거 적용
df_filtered = remove_outliers(df, "시작 시간 차이 (ms)")
df_filtered = remove_outliers(df_filtered, "종료 시간 차이 (ms)")

# === 50ms 단위로 500ms까지만 정확도 계산 ===
tolerance_levels = list(range(50, 501, 50))  # 50ms ~ 500ms, 50ms 단위 증가

# 정확도 계산용 데이터 저장
accuracy_data = {
    "Category": ["Start time alignment", "End time alignment"]
}

# 정확도 계산 (누적된 비율 적용)
for i, tol in enumerate(tolerance_levels):
    if i == 0:
        # 첫 번째 구간은 독립적인 비율
        accuracy_data[f"<{tol}ms"] = [
            (df_filtered["시작 시간 차이 (ms)"] < tol).mean(),
            (df_filtered["종료 시간 차이 (ms)"] < tol).mean()
        ]
    else:
        # 이전 값과 비교하여 최대값 유지 (누적 정확도)
        accuracy_data[f"<{tol}ms"] = [
            max(accuracy_data[f"<{tolerance_levels[i-1]}ms"][0], (df_filtered["시작 시간 차이 (ms)"] < tol).mean()),
            max(accuracy_data[f"<{tolerance_levels[i-1]}ms"][1], (df_filtered["종료 시간 차이 (ms)"] < tol).mean())
        ]

# 정확도 테이블 생성
accuracy_df = pd.DataFrame(accuracy_data)

# 시작/종료 시간 차이의 평균 및 표준편차 계산 (이상치 제거 후)
stats = {
    "총 샘플 개수 (이상치 제거 후)": len(df_filtered),
    "최대 시작 시간 차이 (ms) (이상치 제거 후)": df_filtered["시작 시간 차이 (ms)"].max(),
    "최대 종료 시간 차이 (ms) (이상치 제거 후)": df_filtered["종료 시간 차이 (ms)"].max(),
    "시작 시간 차이 평균 (ms) (이상치 제거 후)": df_filtered["시작 시간 차이 (ms)"].mean(),
    "시작 시간 차이 표준편차 (ms) (이상치 제거 후)": df_filtered["시작 시간 차이 (ms)"].std(),
    "종료 시간 차이 평균 (ms) (이상치 제거 후)": df_filtered["종료 시간 차이 (ms)"].mean(),
    "종료 시간 차이 표준편차 (ms) (이상치 제거 후)": df_filtered["종료 시간 차이 (ms)"].std(),
}

# 통계 데이터 출력
print("\n==== 통계 분석 결과 (이상치 제거 후) ====")
for key, value in stats.items():
    print(f"{key}: {value:.4f}")

# 정확도 테이블 출력 (500ms까지만 표시)
print("\n==== 정확도 테이블 (500ms까지만, 이상치 제거 후) ====")
print(accuracy_df.to_string(index=False))  # 모든 열을 생략 없이 출력

# CSV 파일 저장
accuracy_df.to_csv("aligner_accuracy_table_500ms_filtered.csv", index=False)

print("\n==== CSV 저장 완료 (500ms까지만, 이상치 제거 후) ====")
