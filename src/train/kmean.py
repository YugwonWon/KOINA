import os
import numpy as np
import pandas as pd
import pickle
import argparse

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from textgrid import TextGrid
from tqdm import tqdm

# 전역 경로 설정
DATA_DIR = "/data1/users/yugwon/SDRW2"
OUTPUT_DIR = "out/fig/"
PKL_DIR = "data/pkl/1000-mf"

# 남녀별 TCoG와 Points(pct) 데이터를 저장할 리스트
points_pct_data_F = []
tcog_data_F = []

points_pct_data_M = []
tcog_data_M = []

# 특성 이름 정의 (7차원)
feature_names = [
    "start_f0", 
    "end_f0", 
    "mean_f0", 
    "max_f0", 
    "min_f0", 
    "slope", 
    "TCoG"
]

# -----------------------------
# (A) 텍스트그리드 처리 부분
# -----------------------------
def extract_textgrid_data(filepath):
    """
    파일 경로 끝에 _F 또는 _M 이 성별을 의미하므로,
    성별을 파악하고 해당 성별 리스트에 데이터를 저장.
    """
    if "_F.TextGrid" in filepath:
        gender = "F"
    elif "_M.TextGrid" in filepath:
        gender = "M"
    else:
        return False

    tg = TextGrid()
    tg.read(filepath)

    # Points(pct) 티어 추출
    points_tier = next((tier for tier in tg.tiers if tier.name == "Points(pct)"), None)
    if points_tier:
        # 80 ~ 100% 구간의 (time, F0)만 수집 (사용자 요청으로 80으로 변경)
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
                tcog_time = np.nan  # 없으면 NaN

            if gender == "F":
                points_pct_data_F.append(points_sorted)
                tcog_data_F.append(tcog_time)
            else:  # M
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

    ppf_path = os.path.join(PKL_DIR, "points_pct_data_F.pkl")
    tcf_path = os.path.join(PKL_DIR, "tcog_data_F.pkl")
    ppm_path = os.path.join(PKL_DIR, "points_pct_data_M.pkl")
    tcm_path = os.path.join(PKL_DIR, "tcog_data_M.pkl")

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
        os.makedirs(PKL_DIR, exist_ok=True)
        save_data_to_pkl(ppf_path, points_pct_data_F)
        save_data_to_pkl(tcf_path, tcog_data_F)
        save_data_to_pkl(ppm_path, points_pct_data_M)
        save_data_to_pkl(tcm_path, tcog_data_M)

    return points_pct_data_F, tcog_data_F, points_pct_data_M, tcog_data_M

# -----------------------------
# (B) feature 벡터 계산
# -----------------------------
def compute_features(points_pct_data, tcog_data):
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
            t_cog = -1  # NaN 대체

        feature_vec = [start_f0, end_f0, mean_f0, max_f0, min_f0, slope, t_cog]
        features.append(feature_vec)
        valid_indices.append(i)

    return np.array(features), valid_indices

# -----------------------------
# (C) K-Means + 결과 시각화
# -----------------------------
def plot_elbow_silhouette(features_array, title_suffix):
    """
    최적 cluster 수 찾기 (Elbow, Silhouette, Calinski-Harabasz Score를 한 번에 시각화)
    """
    max_clusters = 8
    cluster_range = range(2, max_clusters+1)
    inertia_values = []
    silhouette_scores = []
    ch_scores = []

    for n_clusters in cluster_range:
        kmeans_tmp = KMeans(n_clusters=n_clusters, random_state=42)
        labels_tmp = kmeans_tmp.fit_predict(features_array)

        # Inertia (Elbow용)
        inertia_values.append(kmeans_tmp.inertia_)

        # Silhouette Score (클러스터가 1개면 계산 불가하므로 분기)
        if len(set(labels_tmp)) > 1:
            sil_score = silhouette_score(features_array, labels_tmp)
        else:
            sil_score = -1
        silhouette_scores.append(sil_score)

        # Calinski-Harabasz Score
        # (CH는 클러스터가 1개 이상이면 계산 가능하나 2개 이상이 일반적)
        ch_score = calinski_harabasz_score(features_array, labels_tmp)
        ch_scores.append(ch_score)

    # figure를 (1행 X 3열) 서브플롯 형태로 만듦
    fig = plt.figure(figsize=(18, 5))

    # (1) Elbow Method
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(cluster_range, inertia_values, marker="o", label="Inertia")
    ax1.set_title(f"Elbow Method [{title_suffix}]")
    ax1.set_xlabel("Number of Clusters")
    ax1.set_ylabel("Inertia")
    ax1.grid(True)
    ax1.legend()

    # (2) Silhouette Score
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(cluster_range, silhouette_scores, marker="o", color="orange", label="Silhouette")
    ax2.set_title(f"Silhouette Analysis [{title_suffix}]")
    ax2.set_xlabel("Number of Clusters")
    ax2.set_ylabel("Silhouette Score")
    ax2.grid(True)
    ax2.legend()

    # (3) Calinski-Harabasz Score
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(cluster_range, ch_scores, marker="o", color="green", label="CH Score")
    ax3.set_title(f"Calinski-Harabasz [{title_suffix}]")
    ax3.set_xlabel("Number of Clusters")
    ax3.set_ylabel("CH Score")
    ax3.grid(True)
    ax3.legend()

    fig.tight_layout()

    fname = f"elbow_silhouette_ch_{title_suffix}.png"
    plt.savefig(os.path.join(OUTPUT_DIR, fname))
    plt.close()

    # silhouette_scores 중 최대값을 주는 cluster 수를 "최적"으로 계속 사용
    optimal_clusters = cluster_range[np.argmax(silhouette_scores)]
    print(f"[{title_suffix}] Optimal clusters by silhouette: {optimal_clusters}")
    return optimal_clusters

def run_kmeans_and_visualize(features_array, valid_indices, points_pct_data, title_suffix,
                             use_pca=False):
    """최종 K-Means 및 결과 시각화 + PCA 옵션"""
    if len(features_array) == 0:
        print(f"[{title_suffix}] No valid items found. Check your data!")
        return

    # 1) elbow & silhouette -> best_k 찾기
    optimal_clusters = plot_elbow_silhouette(features_array, title_suffix)
    print(f"[{title_suffix}] -> Best K from silhouette: {optimal_clusters}")

    # 2) 보고 싶은 클러스터 수 목록 (예: 2, 3, 4)
    cluster_candidates = [2, 3, 4]

    # 중복 방지를 위해, silhouette 분석 결과도 포함할 수 있음
    if optimal_clusters not in cluster_candidates:
        cluster_candidates.append(optimal_clusters)
    cluster_candidates = sorted(set(cluster_candidates))  # 중복 제거 및 정렬

    # 3) 지정된 k값 각각에 대해 K-Means & 시각화
    for n_clusters in cluster_candidates:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(features_array)

        # (A) 군집별 time-F0 라인 그래프
        plt.figure(figsize=(10, 6))
        colors = plt.cm.get_cmap("tab10", n_clusters)

        for cluster_id in range(n_clusters):
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

        plt.title(f"K={n_clusters} Clusters [{title_suffix}]")
        plt.xlabel("Time (%)")
        plt.xlim(90, 100)
        plt.ylabel("F0 (Hz)")
        plt.legend()
        plt.grid(True)
        savepath = os.path.join(OUTPUT_DIR, f"kmeans_{title_suffix}_k{n_clusters}.png")
        plt.savefig(savepath)
        plt.close()

        print(f"[{title_suffix}] Clustering with k={n_clusters} done. Saved: {savepath}")

        # (B) 2클러스터 시 centroid 차이
        if n_clusters == 2:
            centroids = kmeans.cluster_centers_
            diff = np.abs(centroids[0] - centroids[1])
            sorted_idx = np.argsort(diff)[::-1]
            sorted_diff = diff[sorted_idx]
            sorted_feature_names = [feature_names[i] for i in sorted_idx]

            plt.figure(figsize=(8, 6))
            plt.bar(range(len(sorted_diff)), sorted_diff, tick_label=sorted_feature_names)
            plt.title(f"Feature Differences (2 Clusters) - {title_suffix}")
            plt.ylabel("Absolute difference in centroid")
            plt.tight_layout()
            output_path = os.path.join(OUTPUT_DIR, f"centroid_diff_features_{title_suffix}.png")
            plt.savefig(output_path)
            plt.close()

            print(f"[{title_suffix}] (k=2) centroid difference chart: {output_path}")

        # (C) k>2면 pairwise centroid 차이
        elif n_clusters > 2:
            centroids = kmeans.cluster_centers_
            Kc = n_clusters
            diffs_accum = np.zeros(features_array.shape[1])
            pair_count = 0
            for i in range(Kc):
                for j in range(i+1, Kc):
                    diff_ij = np.abs(centroids[i] - centroids[j])
                    diffs_accum += diff_ij
                    pair_count += 1

            if pair_count > 0:
                mean_diff = diffs_accum / pair_count
                sorted_idx = np.argsort(mean_diff)[::-1]
                sorted_diff = mean_diff[sorted_idx]
                sorted_feature_names = [feature_names[i] for i in sorted_idx]

                plt.figure(figsize=(8, 6))
                plt.bar(range(len(sorted_diff)), sorted_diff, tick_label=sorted_feature_names)
                plt.title(f"Avg Pairwise Centroid Diff (K={Kc}) - {title_suffix}")
                plt.ylabel("Mean abs difference")
                plt.tight_layout()
                output_path = os.path.join(OUTPUT_DIR, f"avg_pairwise_centroid_k{Kc}_{title_suffix}.png")
                plt.savefig(output_path)
                plt.close()

                print(f"[{title_suffix}] (k={Kc}) pairwise centroid difference chart: {output_path}")

        # (D) PCA 시각화
        if use_pca:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(features_array)
            plt.figure(figsize=(8, 6))
            for cluster_id in range(n_clusters):
                cluster_indices = (labels == cluster_id)
                plt.scatter(X_pca[cluster_indices, 0], X_pca[cluster_indices, 1],
                            alpha=0.6, label=f"Cluster {cluster_id}")

            plt.title(f"PCA (2D) Visualization - K={n_clusters}, {title_suffix}")
            plt.xlabel(f"PC1 (var: {pca.explained_variance_ratio_[0]*100:.1f}%)")
            plt.ylabel(f"PC2 (var: {pca.explained_variance_ratio_[1]*100:.1f}%)")
            plt.legend()
            plt.grid(True)
            pcafile = os.path.join(OUTPUT_DIR, f"pca_{title_suffix}_k{n_clusters}.png")
            plt.savefig(pcafile)
            plt.close()

            print(f"[{title_suffix}] (k={n_clusters}) PCA scatter plot: {pcafile}")

# -----------------------------
# (D) main
# -----------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_scaling", action="store_true", 
                        help="Apply Z-score standardization for features.", default=True)
    parser.add_argument("--use_pca", action="store_true", 
                        help="Visualize PCA 2D scatter with cluster labels.", default=True)
    args = parser.parse_args()

    # 출력 폴더 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PKL_DIR, exist_ok=True)

    # 1) 로딩 or 처리
    points_pct_data_F, tcog_data_F, points_pct_data_M, tcog_data_M = load_or_process_textgrid_data()

    # 2) 여성(F) 처리
    features_F, valid_indices_F = compute_features(points_pct_data_F, tcog_data_F)
    if len(features_F) > 0 and args.use_scaling:
        # Z-score
        scaler_F = StandardScaler()
        features_F = scaler_F.fit_transform(features_F)
    run_kmeans_and_visualize(features_F, valid_indices_F, points_pct_data_F, 
                             "Female", use_pca=args.use_pca)

    # 3) 남성(M) 처리
    features_M, valid_indices_M = compute_features(points_pct_data_M, tcog_data_M)
    if len(features_M) > 0 and args.use_scaling:
        # Z-score
        scaler_M = StandardScaler()
        features_M = scaler_M.fit_transform(features_M)
    run_kmeans_and_visualize(features_M, valid_indices_M, points_pct_data_M, 
                             "Male", use_pca=args.use_pca)
