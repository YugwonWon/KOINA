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
PKL_DIR = "data/pkl/1000-mf-90-100"

# 남녀별 TCoG와 Points(pct) 데이터를 저장할 리스트
points_pct_data_F = []
tcog_data_F = []
base_names_F = []  

points_pct_data_M = []
tcog_data_M = []
base_names_M = []  

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
        # 90 ~ 100% 구간의 (time, F0)만 수집
        points = [(point.time, float(point.mark))
                  for point in points_tier.points
                  if 90 <= point.time <= 100 and point.mark not in ("", "NaN")]

        if len(points) >= 2:
            points_sorted = sorted(points, key=lambda x: x[0])

            # TCoG 티어
            tcog_tier = next((tier for tier in tg.tiers if tier.name == "TCoG"), None)
            if tcog_tier and len(tcog_tier.points) > 0:
                tcog_time = tcog_tier.points[0].time
            else:
                tcog_time = np.nan
            
            base_name = os.path.basename(filepath).replace(".TextGrid", "")
            
            if gender == "F":
                points_pct_data_F.append(points_sorted)
                tcog_data_F.append(tcog_time)
                base_names_F.append(base_name)  # ✅ 파일명 저장 추가
            else:
                points_pct_data_M.append(points_sorted)
                tcog_data_M.append(tcog_time)
                base_names_M.append(base_name) 
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
    global points_pct_data_F, tcog_data_F, base_names_F
    global points_pct_data_M, tcog_data_M, base_names_M  # ✅ 전역 변수 선언 추가

    ppf_path = os.path.join(PKL_DIR, "points_pct_data_F.pkl")
    tcf_path = os.path.join(PKL_DIR, "tcog_data_F.pkl")
    bnf_path = os.path.join(PKL_DIR, "base_names_F.pkl")  # ✅ 파일명 저장

    ppm_path = os.path.join(PKL_DIR, "points_pct_data_M.pkl")
    tcm_path = os.path.join(PKL_DIR, "tcog_data_M.pkl")
    bnm_path = os.path.join(PKL_DIR, "base_names_M.pkl")  # ✅ 파일명 저장

    if all(os.path.exists(path) for path in [ppf_path, tcf_path, bnf_path, ppm_path, tcm_path, bnm_path]):
        print("Loading data from .pkl files...")
        points_pct_data_F = pickle.load(open(ppf_path, "rb"))
        tcog_data_F = pickle.load(open(tcf_path, "rb"))
        base_names_F = pickle.load(open(bnf_path, "rb"))  # ✅ 파일명도 로드

        points_pct_data_M = pickle.load(open(ppm_path, "rb"))
        tcog_data_M = pickle.load(open(tcm_path, "rb"))
        base_names_M = pickle.load(open(bnm_path, "rb"))  # ✅ 파일명도 로드
    else:
        print("Processing TextGrid files...")
        process_textgrid_files(DATA_DIR)
        os.makedirs(PKL_DIR, exist_ok=True)
        pickle.dump(points_pct_data_F, open(ppf_path, "wb"))
        pickle.dump(tcog_data_F, open(tcf_path, "wb"))
        pickle.dump(base_names_F, open(bnf_path, "wb"))  # ✅ 파일명 저장 추가

        pickle.dump(points_pct_data_M, open(ppm_path, "wb"))
        pickle.dump(tcog_data_M, open(tcm_path, "wb"))
        pickle.dump(base_names_M, open(bnm_path, "wb"))  # ✅ 파일명 저장 추가

    return points_pct_data_F, tcog_data_F, base_names_F, points_pct_data_M, tcog_data_M, base_names_M

# -----------------------------
# (B) feature 벡터 계산
# -----------------------------
def compute_features(points_pct_data, tcog_data, base_names):
    """특성 벡터 생성"""
    features, valid_indices, valid_names = [], [], []
    
    for i in range(min(len(points_pct_data), len(tcog_data), len(base_names))):
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
            t_cog = -1

        feature_vec = [start_f0, end_f0, mean_f0, max_f0, min_f0, slope, t_cog]
        features.append(feature_vec)
        valid_indices.append(i)
        valid_names.append(base_names[i])

    return np.array(features), valid_indices, valid_names

# -----------------------------
# (E) 통계량 저장 함수
# -----------------------------
def save_basic_stats_to_csv(features_array, feature_names, gender_label):
    """
    features_array: shape (N, 7)
    feature_names : list of 7 feature name strings
    gender_label  : "Female" or "Male" (used in filename)
    """
    if len(features_array) == 0:
        print(f"No features for {gender_label}, skip stats CSV.")
        return

    df = pd.DataFrame(features_array, columns=feature_names)
    # 원하는 통계량: support(count), mean, std, min, median(50%), max
    # => df.describe()로 대부분 얻을 수 있고, rename/reorder 가능
    describe_df = df.describe(percentiles=[0.5])  # 기본으로 count, mean, std, min, 50%, max
    # .describe()의 50% -> median
    # columns: feature_names
    # rows: count, mean, std, min, 50%, max

    # 추가로 rename index "50%" -> "median", 그리고 reorder
    # reindex -> ["count", "mean", "std", "min", "median", "max"]
    describe_df = describe_df.rename(index={"50%":"median"})
    # describe_df가 (8행 x 7열). 8행: [count, mean, std, min, 25%, median, 75%, max]
    # 25%, 75%는 필요 없다면 제거
    describe_df = describe_df.drop(["25%", "75%"], errors="ignore")

    # 재정렬
    # 남은 건 ["count", "mean", "std", "min", "median", "max"]
    final_index_order = ["count", "mean", "std", "min", "median", "max"]
    describe_df = describe_df.reindex(final_index_order)

    # 저장
    csv_path = os.path.join(OUTPUT_DIR, f"basic_stats_{gender_label}.csv")
    describe_df.to_csv(csv_path)
    print(f"Saved basic stats CSV: {csv_path}")

# -----------------------------
# (C) K-Means + 결과 시각화
# -----------------------------
def plot_elbow_silhouette(features_array, title_suffix):
    max_clusters = 8
    cluster_range = range(2, max_clusters + 1)
    inertia_values = []
    silhouette_scores = []
    ch_scores = []

    for n_clusters in cluster_range:
        kmeans_tmp = KMeans(n_clusters=n_clusters, random_state=42)
        labels_tmp = kmeans_tmp.fit_predict(features_array)

        inertia_values.append(kmeans_tmp.inertia_)

        if len(set(labels_tmp)) > 1:
            sil_score = silhouette_score(features_array, labels_tmp)
        else:
            sil_score = -1
        silhouette_scores.append(sil_score)

        ch_score = calinski_harabasz_score(features_array, labels_tmp)
        ch_scores.append(ch_score)

    fig = plt.figure(figsize=(18, 5))

    # Elbow Method
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(cluster_range, inertia_values, marker="o", label="Inertia")
    for i, val in enumerate(inertia_values):
        ax1.text(cluster_range[i], val, f"{val:.2f}", ha="center", va="bottom", fontsize=8)  # 레이블 추가
    ax1.set_title(f"Elbow Method [{title_suffix}]")
    ax1.set_xlabel("Number of Clusters")
    ax1.set_ylabel("Inertia")
    ax1.grid(True)
    ax1.legend()

    # Silhouette Analysis
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(cluster_range, silhouette_scores, marker="o", color="orange", label="Silhouette")
    for i, val in enumerate(silhouette_scores):
        ax2.text(cluster_range[i], val, f"{val:.2f}", ha="center", va="bottom", fontsize=8)  # 레이블 추가
    ax2.set_title(f"Silhouette Analysis [{title_suffix}]")
    ax2.set_xlabel("Number of Clusters")
    ax2.set_ylabel("Silhouette Score")
    ax2.grid(True)
    ax2.legend()

    # Calinski-Harabasz Score
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(cluster_range, ch_scores, marker="o", color="green", label="CH Score")
    for i, val in enumerate(ch_scores):
        ax3.text(cluster_range[i], val, f"{val:.2f}", ha="center", va="bottom", fontsize=8)  # 레이블 추가
    ax3.set_title(f"Calinski-Harabasz [{title_suffix}]")
    ax3.set_xlabel("Number of Clusters")
    ax3.set_ylabel("CH Score")
    ax3.grid(True)
    ax3.legend()

    fig.tight_layout()

    fname = f"elbow_silhouette_ch_{title_suffix}.png"
    plt.savefig(os.path.join(OUTPUT_DIR, fname))
    plt.close()

    optimal_clusters = cluster_range[np.argmax(silhouette_scores)]
    print(f"[{title_suffix}] Optimal clusters by silhouette: {optimal_clusters}")
    return optimal_clusters

def save_clustered_features_to_csv(features_array, labels, base_names, gender_label, n_clusters):
    """클러스터 결과를 원본 데이터와 함께 CSV로 저장"""
    if len(features_array) == 0:
        print(f"No features for {gender_label}, skipping CSV output.")
        return
    
    df = pd.DataFrame(features_array, columns=feature_names)
    df["filename"] = base_names  # ✅ 파일명 추가
    df["cluster_label"] = labels

    csv_path = os.path.join(OUTPUT_DIR, f"clustered_features_{gender_label}_k{n_clusters}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved clustered features CSV: {csv_path}")
    
def run_kmeans_and_visualize(features_array, valid_indices, points_pct_data, title_suffix,
                             use_pca=False):
    if len(features_array) == 0:
        print(f"[{title_suffix}] No valid items found. Check your data!")
        return

    optimal_clusters = plot_elbow_silhouette(features_array, title_suffix)
    print(f"[{title_suffix}] -> Best K from silhouette: {optimal_clusters}")

    cluster_candidates = [2, 3, 4]
    if optimal_clusters not in cluster_candidates:
        cluster_candidates.append(optimal_clusters)
    cluster_candidates = sorted(set(cluster_candidates))

    for n_clusters in cluster_candidates:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(features_array)

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

        # K=2 centroid diff
        if n_clusters == 2:
            centroids = kmeans.cluster_centers_
            diff = np.abs(centroids[0] - centroids[1])
            sorted_idx = np.argsort(diff)[::-1]
            sorted_diff = diff[sorted_idx]
            sorted_feature_names = [feature_names[i] for i in sorted_idx]

            plt.figure(figsize=(8, 6))
            bars = plt.bar(range(len(sorted_diff)), sorted_diff, tick_label=sorted_feature_names)

            # 막대 위에 값 레이블 추가
            for bar in bars:
                height = bar.get_height()  # 각 막대의 높이(값)
                plt.text(bar.get_x() + bar.get_width() / 2.0,  # x 위치
                        height,  # y 위치
                        f'{height:.2f}',  # 표시할 텍스트
                        ha='center', va='bottom', fontsize=10)  # 텍스트 정렬과 크기 설정

            plt.title(f"Avg Pairwise Centroid Diff (K=2) - {title_suffix}")
            plt.ylabel("Mean abs difference")
            plt.tight_layout()
            output_path = os.path.join(OUTPUT_DIR, f"centroid_diff_features_{title_suffix}.png")
            plt.savefig(output_path)
            plt.close()

            print(f"[{title_suffix}] (k=2) centroid difference chart: {output_path}")

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
                bars = plt.bar(range(len(sorted_diff)), sorted_diff, tick_label=sorted_feature_names)

                # 막대 위에 값 레이블 추가
                for bar in bars:
                    height = bar.get_height()  # 각 막대의 높이(값)
                    plt.text(bar.get_x() + bar.get_width() / 2.0,  # x 위치
                            height,  # y 위치
                            f'{height:.2f}',  # 표시할 텍스트
                            ha='center', va='bottom', fontsize=10)  # 텍스트 정렬과 크기 설정

                plt.title(f"Avg Pairwise Centroid Diff (K={Kc}) - {title_suffix}")
                plt.ylabel("Mean abs difference")
                plt.tight_layout()
                output_path = os.path.join(OUTPUT_DIR, f"avg_pairwise_centroid_k{Kc}_{title_suffix}.png")
                plt.savefig(output_path)
                plt.close()

                print(f"[{title_suffix}] (k={Kc}) pairwise centroid difference chart: {output_path}")
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
    points_pct_data_F, tcog_data_F, base_names_F, points_pct_data_M, tcog_data_M, base_names_M = load_or_process_textgrid_data()

    # 2) 여성(F) 처리
    features_F, valid_indices_F, valid_names_F= compute_features(points_pct_data_F, tcog_data_F, base_names_F)
    # 통계량 CSV 저장 (스케일링 전)
    save_basic_stats_to_csv(features_F, feature_names, "Female_raw")

    if len(features_F) > 0 and args.use_scaling:
        scaler_F = StandardScaler()
        features_F = scaler_F.fit_transform(features_F)

    # 통계량 CSV 저장 (스케일링 후) - 필요시
    save_basic_stats_to_csv(features_F, feature_names, "Female_scaled")

    run_kmeans_and_visualize(features_F, valid_indices_F, points_pct_data_F, 
                             "Female", use_pca=args.use_pca)

    # 3) 남성(M) 처리
    features_M, valid_indices_M, valid_names_M = compute_features(points_pct_data_M, tcog_data_M, base_names_M)
    # 통계량 CSV 저장 (스케일링 전)
    save_basic_stats_to_csv(features_M, feature_names, "Male_raw")

    if len(features_M) > 0 and args.use_scaling:
        scaler_M = StandardScaler()
        features_M = scaler_M.fit_transform(features_M)

    # 통계량 CSV 저장 (스케일링 후)
    save_basic_stats_to_csv(features_M, feature_names, "Male_scaled")

    run_kmeans_and_visualize(features_M, valid_indices_M, points_pct_data_M, 
                             "Male", use_pca=args.use_pca)
    
    kmeans_F = KMeans(n_clusters=2, random_state=42).fit(features_F)
    kmeans_M = KMeans(n_clusters=2, random_state=42).fit(features_M)

    labels_F_k2, labels_M_k2 = kmeans_F.labels_, kmeans_M.labels_

    save_clustered_features_to_csv(features_F, labels_F_k2, valid_names_F, "Female", 2)
    save_clustered_features_to_csv(features_M, labels_M_k2, valid_names_M, "Male", 2)