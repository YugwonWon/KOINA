import os
import numpy as np
import pandas as pd
import pickle
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from textgrid import TextGrid
from tqdm import tqdm


def extract_textgrid_data(filepath):
    """
    파일 경로 끝에 _F 또는 _M 이 성별을 의미하므로,
    성별을 파악하고 해당 성별 리스트에 데이터를 저장.
    """
    # 성별 파악
    if "_F.TextGrid" in filepath:
        gender = "F"
    elif "_M.TextGrid" in filepath:
        gender = "M"
    else:
        # 만약 성별이 명시되지 않았으면 스킵하거나, 로깅하고 리턴
        # 여기서는 스킵 처리
        return False

    tg = TextGrid()
    tg.read(filepath)

    # Points(pct) 티어 추출
    points_tier = next((tier for tier in tg.tiers if tier.name == "Points(pct)"), None)
    if points_tier:
        # 90 ~ 100% 구간의 (time, F0)만 수집
        points = [(point.time, float(point.mark))
                  for point in points_tier.points
                  if 90 <= point.time <= 100 and point.mark not in ("", "NaN")]

        # 최소 3개의 포인트가 있어야 시계열적 활용 가능
        if len(points) > 2:
            points_sorted = sorted(points, key=lambda x: x[0])

            # TCoG 티어 추출
            tcog_tier = next((tier for tier in tg.tiers if tier.name == "TCoG"), None)
            if tcog_tier and len(tcog_tier.points) > 0:
                tcog_time = tcog_tier.points[0].time
            else:
                tcog_time = np.nan  # 없으면 NaN 처리

            # 성별 리스트에 저장
            if gender == "F":
                points_pct_data_F.append(points_sorted)
                tcog_data_F.append(tcog_time)
            else:  # gender == "M"
                points_pct_data_M.append(points_sorted)
                tcog_data_M.append(tcog_time)

            return True
    return False

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
    """
    남녀 데이터를 각각 로드/처리
    """
    global points_pct_data_F, tcog_data_F
    global points_pct_data_M, tcog_data_M

    # 파일 경로
    ppf_path = os.path.join(PKL_DIR, "points_pct_data_F.pkl")
    tcf_path = os.path.join(PKL_DIR, "tcog_data_F.pkl")
    ppm_path = os.path.join(PKL_DIR, "points_pct_data_M.pkl")
    tcm_path = os.path.join(PKL_DIR, "tcog_data_M.pkl")

    # 남녀 파일 존재 여부 확인
    if (os.path.exists(ppf_path) and os.path.exists(tcf_path)
        and os.path.exists(ppm_path) and os.path.exists(tcm_path)):
        print("Loading data from .pkl files...")
        points_pct_data_F = load_data_from_pkl(ppf_path)
        tcog_data_F       = load_data_from_pkl(tcf_path)
        points_pct_data_M = load_data_from_pkl(ppm_path)
        tcog_data_M       = load_data_from_pkl(tcm_path)
    else:
        print("Processing TextGrid files...")
        process_textgrid_files(DATA_DIR)
        # 저장
        save_data_to_pkl(ppf_path, points_pct_data_F)
        save_data_to_pkl(tcf_path, tcog_data_F)
        save_data_to_pkl(ppm_path, points_pct_data_M)
        save_data_to_pkl(tcm_path, tcog_data_M)

    return points_pct_data_F, tcog_data_F, points_pct_data_M, tcog_data_M


def compute_features(points_pct_data, tcog_data):
    """
    남/녀 공통으로 사용될 특징 벡터 계산 로직.
    points_pct_data: 리스트[ (time, F0), (time, F0), ... ] * n
    tcog_data:       리스트[ TCoG, TCoG, ... ] (길이 동일)
    """
    features = []
    valid_indices = []
    
    num_items = min(len(points_pct_data), len(tcog_data))
    
    for i in range(num_items):
        item_points = points_pct_data[i]
        t_cog = tcog_data[i]

        if len(item_points) < 2:
            continue

        times, f0s = zip(*item_points)
        times = np.array(times)
        f0s = np.array(f0s)

        start_time = times[0]
        end_time   = times[-1]
        start_f0   = f0s[0]
        end_f0     = f0s[-1]
        mean_f0    = np.mean(f0s)
        max_f0     = np.max(f0s)
        min_f0     = np.min(f0s)
        
        # 기울기
        time_diff = (end_time - start_time)
        if abs(time_diff) < 1e-9:
            slope = 0.0
        else:
            slope = (end_f0 - start_f0) / time_diff

        if pd.isna(t_cog):
            t_cog = -1  # 혹은 NaN 대체값

        feature_vec = [start_f0, end_f0, mean_f0, max_f0, min_f0, slope, t_cog]
        features.append(feature_vec)
        valid_indices.append(i)

    return np.array(features), valid_indices


def run_kmeans(features_array, valid_indices, points_pct_data, title_suffix):
    """
    K-Means 실행 및 결과 그래프 출력
    """
    if len(features_array) == 0:
        print(f"[{title_suffix}] No valid items found. Check your data!")
        return

    max_clusters = 8
    cluster_range = range(2, max_clusters+1)
    inertia_values = []
    silhouette_scores = []

    for n_clusters in cluster_range:
        kmeans_tmp = KMeans(n_clusters=n_clusters, random_state=42)
        labels_tmp = kmeans_tmp.fit_predict(features_array)
        inertia_values.append(kmeans_tmp.inertia_)
        
        if len(set(labels_tmp)) > 1:
            sil_score = silhouette_score(features_array, labels_tmp)
        else:
            sil_score = -1
        silhouette_scores.append(sil_score)

    # Elbow & Silhouette 그래프
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(cluster_range, inertia_values, marker="o", label="Inertia")
    plt.title(f"Elbow Method [{title_suffix}]")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(cluster_range, silhouette_scores, marker="o", color="orange", label="Silhouette Score")
    plt.title(f"Silhouette Analysis [{title_suffix}]")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"elbow_silhouette_{title_suffix}.png"))
    plt.close()

    optimal_clusters = cluster_range[np.argmax(silhouette_scores)]
    print(f"[{title_suffix}] Optimal clusters: {optimal_clusters}")

    # 최종 K-Means
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    labels = kmeans.fit_predict(features_array)

    # 시각화
    plt.figure(figsize=(10, 6))
    colors = plt.cm.get_cmap("tab10", optimal_clusters)

    for cluster_id in range(optimal_clusters):
        cluster_item_indices = [i for i, lbl in enumerate(labels) if lbl == cluster_id]
        cluster_tcog_vals = features_array[cluster_item_indices, 6]
        mean_tcog = np.mean(cluster_tcog_vals) if len(cluster_tcog_vals) > 0 else np.nan

        actual_item_indices = [valid_indices[idx] for idx in cluster_item_indices]

        for item_idx in actual_item_indices:
            item_points = points_pct_data[item_idx]
            if len(item_points) < 2:
                continue
            times, f0s = zip(*item_points)
            plt.plot(times, f0s, color=colors(cluster_id), alpha=0.4)
        
        plt.plot([], [], color=colors(cluster_id), 
                 label=f"Cluster {cluster_id} (Mean TCoG={mean_tcog:.2f})")

    plt.title(f"K-means Clustering [{title_suffix}]")
    plt.xlabel("Time (%)")
    plt.xlim(90, 100)
    plt.ylabel("F0 (Hz)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, f"kmeans_{title_suffix}.png"))
    plt.close()

    print(f"[{title_suffix}] Clustering and plotting done!\n")
    
    if optimal_clusters == 2:
        # 2개 클러스터 centroid
        centroids = kmeans.cluster_centers_  # shape = (2, n_features)
        
        # centroid 차이(절댓값)
        diff = np.abs(centroids[0] - centroids[1])  # shape=(n_features,)

        # 차이가 큰 순서대로 정렬
        sorted_idx = np.argsort(diff)[::-1]
        sorted_diff = diff[sorted_idx]
        sorted_feature_names = [feature_names[i] for i in sorted_idx]

        # 바 차트로 저장
        plt.figure(figsize=(8, 6))
        plt.bar(range(len(sorted_diff)), sorted_diff, tick_label=sorted_feature_names)
        plt.title(f"Feature Differences (2 Clusters) - {title_suffix}")
        plt.ylabel("Absolute difference in centroid")
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, f"centroid_diff_features_{title_suffix}.png")
        plt.savefig(output_path)
        plt.close()

        print(f"[{title_suffix}] Saved centroid difference chart: {output_path}")

if __name__ == '__main__':
            
    feature_names = [
        "start_f0", 
        "end_f0", 
        "mean_f0", 
        "max_f0", 
        "min_f0", 
        "slope", 
        "TCoG"
    ]

    # 데이터 경로
    DATA_DIR = "/data1/users/yugwon/SDRW2"
    OUTPUT_DIR = "out/fig/"
    PKL_DIR = "data/pkl/1000-mf"


    # 출력 폴더 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PKL_DIR, exist_ok=True)

    # 남녀별 TCoG와 Points(pct) 데이터를 저장할 리스트(혹은 dict)
    points_pct_data_F = []
    tcog_data_F = []

    points_pct_data_M = []
    tcog_data_M = []
    
    # ----------------------------- #
    # 여기서부터 K-means 로직 부분
    # ----------------------------- #

    points_pct_data_F, tcog_data_F, points_pct_data_M, tcog_data_M = load_or_process_textgrid_data()

    # ----------------------
    # 실제 실행부
    # ----------------------

    # 1) 여성(F) 데이터
    features_F, valid_indices_F = compute_features(points_pct_data_F, tcog_data_F)
    run_kmeans(features_F, valid_indices_F, points_pct_data_F, "Female")

    # 2) 남성(M) 데이터
    features_M, valid_indices_M = compute_features(points_pct_data_M, tcog_data_M)
    run_kmeans(features_M, valid_indices_M, points_pct_data_M, "Male")
