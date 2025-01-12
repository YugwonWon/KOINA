import os
import numpy as np
import pandas as pd
import pickle
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from textgrid import TextGrid
from tqdm import tqdm

# 데이터 경로
DATA_DIR = "/data1/users/yugwon/SDRW"
OUTPUT_DIR = "out/fig/"
PKL_DIR = "data/pkl/1000"

# 출력 폴더 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PKL_DIR, exist_ok=True)

# TCoG와 Points(pct) 데이터를 저장할 리스트
points_pct_data = []
tcog_data = []

# TextGrid 파일 읽기 함수
def extract_textgrid_data(filepath):
    tg = TextGrid()
    tg.read(filepath)

    # Points(pct) 티어 추출
    points_tier = next((tier for tier in tg.tiers if tier.name == "Points(pct)"), None)
    if points_tier:
        # 90 ~ 100% 구간의 (time, F0)만 수집
        points = [(point.time, float(point.mark)) 
                  for point in points_tier.points 
                  if 90 <= point.time <= 100 and point.mark not in ("", "NaN")]
        if len(points) > 2:  # 최소 3개의 포인트가 있어야 "시계열"로서 활용 가능
            # time 기준으로 정렬
            points_sorted = sorted(points, key=lambda x: x[0])
            points_pct_data.append(points_sorted)
            
            # TCoG 티어 추출
            tcog_tier = next((tier for tier in tg.tiers if tier.name == "TCoG"), None)
            if tcog_tier and len(tcog_tier.points) > 0:
                tcog_time = tcog_tier.points[0].time  # TCoG 시간 값
                tcog_data.append(tcog_time)
            else:
                # TCoG가 없는 경우를 위해 빈 값(또는 np.nan)으로 채울 수도 있음
                tcog_data.append(np.nan)
            return True
        

# 지정된 폴더 내의 모든 TextGrid 파일 처리
def process_textgrid_files(directory):
    valid_files = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".TextGrid"):
                filepath = os.path.join(root, file)
                is_detect = extract_textgrid_data(filepath)
                if is_detect:
                    valid_files += 1
                
    print(f"valid_files: {valid_files}")
# 데이터 저장 및 불러오기 함수
def save_data_to_pkl(filepath, data):
    with open(filepath, "wb") as f:
        pickle.dump(data, f)

def load_data_from_pkl(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)

def load_or_process_textgrid_data():
    global points_pct_data, tcog_data  # 전역 변수 갱신 명시

    points_pct_path = os.path.join(PKL_DIR, "points_pct_data.pkl")
    tcog_path = os.path.join(PKL_DIR, "tcog_data.pkl")

    if os.path.exists(points_pct_path) and os.path.exists(tcog_path):
        print("Loading data from .pkl files...")
        points_pct_data = load_data_from_pkl(points_pct_path)
        tcog_data = load_data_from_pkl(tcog_path)
    else:
        print("Processing TextGrid files...")
        process_textgrid_files(DATA_DIR)
        save_data_to_pkl(points_pct_path, points_pct_data)
        save_data_to_pkl(tcog_path, tcog_data)

    return points_pct_data, tcog_data

# 여기서부터는 '대표 특징(feature)'을 추출해 K-means 클러스터링하는 로직

# 데이터 로드
points_pct_data, tcog_data = load_or_process_textgrid_data()

# 만약 points_pct_data와 tcog_data의 길이가 다르다면, 
# 빈 데이터 처리를 어떻게 할지 결정해야 함.
num_items = min(len(points_pct_data), len(tcog_data))

# ------------------------------------------------------------
# (1) 각 아이템별로 대표 특징을 추출
# ------------------------------------------------------------
features = []
valid_indices = []  # 실제로 유효한 (포인트 >= 2개) 아이템의 인덱스를 기록

for i in range(num_items):
    item_points = points_pct_data[i]
    t_cog = tcog_data[i]

    if len(item_points) < 2:
        # 포인트가 너무 적으면 skip (또는 NaN으로 채움)
        continue

    # 시간/주파수 분리
    times, f0s = zip(*item_points)
    times = np.array(times)
    f0s = np.array(f0s)

    # 대표 특징 계산
    start_time = times[0]
    end_time   = times[-1]
    start_f0   = f0s[0]
    end_f0     = f0s[-1]
    mean_f0    = np.mean(f0s)
    max_f0     = np.max(f0s)
    min_f0     = np.min(f0s)

    # 전체 구간 기울기
    # (end_f0 - start_f0) / (end_time - start_time)가 0으로 나눠질 가능성도 체크
    time_diff = (end_time - start_time)
    if abs(time_diff) < 1e-9:
        slope = 0.0
    else:
        slope = (end_f0 - start_f0) / time_diff

    # TCoG도 feature에 포함 (없으면 NaN일 수도 있음)
    if pd.isna(t_cog):
        t_cog = -1  # 예: 결측값이면 -1 처리 등

    # 하나의 7차원 벡터로 만든다
    feature_vec = [start_f0, end_f0, mean_f0, max_f0, min_f0, slope, t_cog]
    features.append(feature_vec)
    valid_indices.append(i)

# 넘파이 배열로 변환
features_array = np.array(features)

print(f"Number of valid items for clustering: {len(features_array)}")
if len(features_array) == 0:
    print("No valid items found. Check your data filtering conditions!")
    exit()

# ------------------------------------------------------------
# (2) 군집 수 탐색 (silhouette analysis 등)
# ------------------------------------------------------------
max_clusters = 8
cluster_range = range(2, max_clusters + 1)
inertia_values = []
silhouette_scores = []

for n_clusters in cluster_range:
    kmeans_tmp = KMeans(n_clusters=n_clusters, random_state=42)
    labels_tmp = kmeans_tmp.fit_predict(features_array)
    inertia_values.append(kmeans_tmp.inertia_)

    if len(set(labels_tmp)) > 1:  # silhouette_score는 클러스터가 2개 이상이어야 가능
        sil_score = silhouette_score(features_array, labels_tmp)
    else:
        sil_score = -1
    silhouette_scores.append(sil_score)

# Elbow & Silhouette 시각화
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(cluster_range, inertia_values, marker="o", label="Inertia")
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(cluster_range, silhouette_scores, marker="o", color="orange", label="Silhouette Score")
plt.title("Silhouette Analysis")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "elbow_silhouette_features.png"))
plt.close()

# silhouette score가 최대가 되는 k값 선택
optimal_clusters = cluster_range[np.argmax(silhouette_scores)]
print(f"Optimal number of clusters based on silhouette: {optimal_clusters}")

# ------------------------------------------------------------
# (3) 최종 K-means 클러스터링
# ------------------------------------------------------------
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
labels = kmeans.fit_predict(features_array)

# labels는 features_array 순서(= valid_indices 순서)에 대응
# 즉, 실제로는 valid_indices[i] 번째 아이템에 해당한다.

# ------------------------------------------------------------
# (4) 아이템별로 선을 연결하여 시각화 (클러스터별 색상, 평균 TCoG 범례 표시)
# ------------------------------------------------------------
plt.figure(figsize=(10, 6))
colors = plt.cm.get_cmap("tab10", optimal_clusters)

for cluster_id in range(optimal_clusters):
    # 같은 클러스터에 속한 아이템들의 'features_array 내 index'
    cluster_item_indices = [i for i, lbl in enumerate(labels) if lbl == cluster_id]

    # 군집별 평균 TCoG 계산
    # features_array[:, 6]에 TCoG가 있으므로, 이 클러스터 멤버들의 TCoG만 추출
    cluster_tcog_vals = features_array[cluster_item_indices, 6]
    mean_tcog = np.mean(cluster_tcog_vals) if len(cluster_tcog_vals) > 0 else np.nan

    # 실제 points_pct_data 인덱스 (valid_indices에서 역매핑)
    actual_item_indices = [valid_indices[idx] for idx in cluster_item_indices]

    # 군집 아이템들을 반복하면서 선 그리기
    for item_idx in actual_item_indices:
        item_points = points_pct_data[item_idx]
        if len(item_points) < 2:
            continue
        times, f0s = zip(*item_points)
        plt.plot(times, f0s, color=colors(cluster_id), alpha=0.4)

    # 군집 별 범례 표시 (한 번만 표시) - Mean TCoG 값까지 포함
    plt.plot([], [], color=colors(cluster_id), 
             label=f"Cluster {cluster_id} (Mean TCoG={mean_tcog:.2f})")

plt.title("K-means Clustering (Representative Features) - Item-wise Lines")
plt.xlabel("Time (%)")
plt.xlim(90, 100)
plt.ylabel("F0 (Hz)")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "kmeans_clustering_features_itemwise.png"))
plt.close()

print("Clustering and plotting complete!")
