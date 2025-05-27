import os
import numpy as np
import pandas as pd
import pickle
import argparse

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from scipy.stats import ttest_ind, f_oneway
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from plotnine import (
    ggplot, aes, geom_point, labs, scale_color_brewer,
    theme_minimal, theme, guides, guide_legend,
    element_text, element_blank, element_line,      # ← element_line 임포트
    facet_wrap
)
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests

import matplotlib.pyplot as plt
from textgrid import TextGrid
from tqdm import tqdm

# 전역 경로 설정
DATA_DIR = "/data1/users/yugwon/SDRW2"
OUTPUT_DIR = "out/fig4/"
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
                base_names_F.append(base_name)  # 파일명 저장 추가
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
    global points_pct_data_M, tcog_data_M, base_names_M  # 전역 변수 선언 추가

    ppf_path = os.path.join(PKL_DIR, "points_pct_data_F.pkl")
    tcf_path = os.path.join(PKL_DIR, "tcog_data_F.pkl")
    bnf_path = os.path.join(PKL_DIR, "base_names_F.pkl")  # 파일명 저장

    ppm_path = os.path.join(PKL_DIR, "points_pct_data_M.pkl")
    tcm_path = os.path.join(PKL_DIR, "tcog_data_M.pkl")
    bnm_path = os.path.join(PKL_DIR, "base_names_M.pkl")  # 파일명 저장

    if all(os.path.exists(path) for path in [ppf_path, tcf_path, bnf_path, ppm_path, tcm_path, bnm_path]):
        print("Loading data from .pkl files...")
        points_pct_data_F = pickle.load(open(ppf_path, "rb"))
        tcog_data_F = pickle.load(open(tcf_path, "rb"))
        base_names_F = pickle.load(open(bnf_path, "rb"))  # 파일명도 로드

        points_pct_data_M = pickle.load(open(ppm_path, "rb"))
        tcog_data_M = pickle.load(open(tcm_path, "rb"))
        base_names_M = pickle.load(open(bnm_path, "rb"))  # 파일명도 로드
    else:
        print("Processing TextGrid files...")
        process_textgrid_files(DATA_DIR)
        os.makedirs(PKL_DIR, exist_ok=True)
        pickle.dump(points_pct_data_F, open(ppf_path, "wb"))
        pickle.dump(tcog_data_F, open(tcf_path, "wb"))
        pickle.dump(base_names_F, open(bnf_path, "wb"))  # 파일명 저장 추가

        pickle.dump(points_pct_data_M, open(ppm_path, "wb"))
        pickle.dump(tcog_data_M, open(tcm_path, "wb"))
        pickle.dump(base_names_M, open(bnm_path, "wb"))  # 파일명 저장 추가

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
    df["filename"] = base_names  # 파일명 추가
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

    cluster_candidates = [2, 3, 4, 5]
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
            plt.ylim(0, 2.5)
            plt.yticks(np.arange(0, 2.6, 0.5))
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
                plt.ylim(0, 2.5)
                plt.yticks(np.arange(0, 2.6, 0.5))
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
                

            plt.title(f"PCA 2D K-Means - K={n_clusters} [{title_suffix}]")
            plt.xlabel(f"PC1 (var: {pca.explained_variance_ratio_[0]*100:.1f}%)")
            plt.ylabel(f"PC2 (var: {pca.explained_variance_ratio_[1]*100:.1f}%)")
            plt.legend()
            plt.grid(True)
            pcafile = os.path.join(OUTPUT_DIR, f"pca_{title_suffix}_k{n_clusters}.png")
            plt.savefig(pcafile)
            plt.close()

            print(f"[{title_suffix}] (k={n_clusters}) PCA scatter plot: {pcafile}")

        if args.use_tsne:
            plot_tsne_plotnine(
                features_array, labels,
                f"K-Means k={n_clusters} [{title_suffix}]",
                OUTPUT_DIR,
                gender=title_suffix,
                model_tag=f"K-Means_k{n_clusters}"
            )
            
# -----------------------------------------------
# GM · BGM 클러스터링 & 시각화 + 통계검정
# -----------------------------------------------
def run_mixture_and_visualize(features_array, valid_indices, points_pct_data,
                              title_suffix,              # "Female" / "Male"
                              model_tag,    # GaussianMixture / BGMM
                              cluster_candidates=(2,3,4,5),
                              use_pca=False):
    """
    K-Means 버전을 그대로 복사해 model_class(means_) 만 바꾼 함수.
    model_tag: "GM" / "BGM" 같이 파일명‧타이틀에 붙일 짧은 문자열
    """
    if len(features_array) == 0:
        print(f"[{title_suffix}-{model_tag}] No valid items found. Check your data!")
        return

    for n_clusters in cluster_candidates:
        # 디폴트 값 사용
        if model_tag == "GM":              
            model = GaussianMixture(
                n_components = n_clusters,
                n_init = 10,
                random_state = 42           
            )
        else:                               
            model = BayesianGaussianMixture(
                n_components = n_clusters,
                n_init = 10,
                random_state = 42
            )
        model.fit(features_array)
        labels = model.predict(features_array)
        centroids = model.means_                        # ← KMeans의 cluster_centers_ 대응

        # -------- ② F0 윤곽 꺾은선 그림 ----------------
        plt.figure(figsize=(10, 6))
        colors = plt.cm.get_cmap("tab10", n_clusters)
        for cluster_id in range(n_clusters):
            cluster_item_indices = np.where(labels == cluster_id)[0]
            cluster_tcog_vals   = features_array[cluster_item_indices, 6]
            mean_tcog = np.mean(cluster_tcog_vals) if len(cluster_tcog_vals) else np.nan

            actual_item_indices = [valid_indices[idx] for idx in cluster_item_indices]
            for item_idx in actual_item_indices:
                item_points = points_pct_data[item_idx]
                if len(item_points) < 2:  # 방어
                    continue
                times, f0s = zip(*item_points)
                plt.plot(times, f0s, color=colors(cluster_id), alpha=0.4)

            plt.plot([], [], color=colors(cluster_id),
                     label=f"Cluster {cluster_id} (Mean TCoG={mean_tcog:.2f})")

        plt.title(f"{model_tag} K={n_clusters} [{title_suffix}]")
        plt.xlabel("Time (%)"); plt.xlim(90, 100)
        plt.ylabel("F0 (Hz)")
        plt.legend(); plt.grid(True)
        savepath = os.path.join(OUTPUT_DIR,
                                f"{model_tag.lower()}_{title_suffix}_k{n_clusters}.png")
        plt.savefig(savepath); plt.close()
        print(f"[{title_suffix}-{model_tag}] k={n_clusters} plot saved → {savepath}")

        # -------- ③ 특성 중요도(centroid diff) ----------
        if n_clusters == 2:
            diff = np.abs(centroids[0] - centroids[1])
        else:
            # 모든 센트로이드 쌍의 평균 절대차
            diffs_accum = np.zeros(features_array.shape[1]); pair_cnt = 0
            for i in range(n_clusters):
                for j in range(i+1, n_clusters):
                    diffs_accum += np.abs(centroids[i] - centroids[j])
                    pair_cnt   += 1
            diff = diffs_accum / pair_cnt

        sorted_idx   = diff.argsort()[::-1]
        sorted_diff  = diff[sorted_idx]
        sorted_names = [feature_names[i] for i in sorted_idx]

        plt.figure(figsize=(8, 6))
        bars = plt.bar(range(len(sorted_diff)), sorted_diff, tick_label=sorted_names)
        for b in bars:
            h = b.get_height()
            plt.text(b.get_x() + b.get_width()/2, h, f"{h:.2f}",
                     ha="center", va="bottom", fontsize=9)
        title_bar = "Centroid Diff" if n_clusters == 2 else "Avg Pairwise Centroid Diff"
        plt.title(f"{title_bar} ({model_tag} K={n_clusters}) - {title_suffix}")
        plt.ylim(0, 2.5); plt.yticks(np.arange(0, 2.6, 0.5)); plt.ylabel("Abs diff")
        plt.tight_layout()
        fname_bar = f"{model_tag.lower()}_centroid_k{n_clusters}_{title_suffix}.png"
        plt.savefig(os.path.join(OUTPUT_DIR, fname_bar)); plt.close()

        # # -------- ④ 통계 검정 --------------------------
        # test_feature_differences(features_array, labels, feature_names,
        #                          f"{model_tag}-{title_suffix}-k{n_clusters}")

        # -------- ⑤ PCA 산점도(선택) --------------------
        if use_pca:
            pca   = PCA(n_components=2)
            X_pca = pca.fit_transform(features_array)
            plt.figure(figsize=(8,6))
            for cid in range(n_clusters):
                mask = labels == cid
                plt.scatter(X_pca[mask,0], X_pca[mask,1],
                            alpha=0.6, label=f"Cl {cid}")
            plt.title(f"PCA 2D – {model_tag} k={n_clusters} [{title_suffix}]")
            plt.xlabel(f"PC1 (var:{pca.explained_variance_ratio_[0]*100:.1f}%)")
            plt.ylabel(f"PC2 (var:{pca.explained_variance_ratio_[1]*100:.1f}%)")
            plt.legend(); plt.grid(True)
            pcapath = os.path.join(OUTPUT_DIR,
                                   f"{model_tag.lower()}_pca_k{n_clusters}_{title_suffix}.png")
            plt.savefig(pcapath); plt.close()
            print(f"[{title_suffix}-{model_tag}] k={n_clusters} PCA saved → {pcapath}")
            
        if args.use_tsne:
            plot_tsne_plotnine(
                features_array, labels,
                f"{model_tag} k={n_clusters} [{title_suffix}]",
                OUTPUT_DIR,
                gender=title_suffix,
                model_tag=f"{model_tag}_k{n_clusters}"
            )

def plot_tsne_plotnine(features, labels,
                       title, save_dir,
                       gender=None, model_tag=None,
                       perplexity=30, n_iter=1000,
                       point_alpha=0.65, point_size=1.6):
    # ── t-SNE 변환 ─────────────────────────────
    tsne = TSNE(n_components=2, perplexity=perplexity,
                n_iter=n_iter, init="pca", random_state=42)
    X_tsne = tsne.fit_transform(features)

    df = pd.DataFrame({
        "tsne1":   X_tsne[:, 0],
        "tsne2":   X_tsne[:, 1],
        "cluster": labels.astype(str)
    })
    if gender:    df["gender"] = gender
    if model_tag: df["model"]  = model_tag

    # ── 기본 플롯 ───────────────────────────────
    p = (
    ggplot(df, aes("tsne1", "tsne2", color="cluster"))
    + geom_point(alpha=0.5, size=1.3, stroke=0)           # ← 덜 떡지게
    + scale_color_brewer(type="qual", palette="Set2")
    + labs(title=title, x="t-SNE-1", y="t-SNE-2")
    + theme_minimal(base_size=11)
    + theme(
        panel_grid_major = element_line(color="#d9d9d9", size=0.25),
        panel_grid_minor = element_blank(),
        plot_title       = element_text(size=14, weight='bold', ha='center'),
        axis_title       = element_text(size=11, weight='bold'),
        # -------- 범례 --------
        legend_position      = "right",       # ← 캔버스 밖 오른쪽
        legend_background    = element_blank(),
        legend_key           = element_blank(),
        legend_title         = element_text(size=9, weight='bold')
    )
    + guides(
        color = guide_legend(
            title="Cluster",
            override_aes={"size":3, "alpha":1, "stroke":0}
            )
        )
    )

    # ── facet이 필요할 때만 추가 ────────────────
    facet_formula = None
    if "model" in df.columns and df["model"].nunique() > 1 and \
       "gender" in df.columns and df["gender"].nunique() > 1:
        facet_formula = "~ gender + model"
    elif "model" in df.columns and df["model"].nunique() > 1:
        facet_formula = "~ model"
    elif "gender" in df.columns and df["gender"].nunique() > 1:
        facet_formula = "~ gender"

    if facet_formula:
        p += facet_wrap(facet_formula)
    else:
        # 단일 facet일 때 strip 제거(제목 중복 방지)
        p += theme(strip_background=element_blank(),
                   strip_text=element_blank())

    # ── 저장 ─────────────────────────────────
    fname = os.path.join(save_dir,
                         f"tsne_{title.replace(' ', '_')}.png")
    p.save(
    fname,
    dpi=300,
    width=7, height=4,             # 폭을 1인치 정도만 넓혀 둠
    units="in",
    bbox_inches='tight'            # ← 범례·제목 잘림 방지
    )
    print(f"[t-SNE] {title} → {fname}")

def plot_tsne_scatter(features, labels, title, save_dir):
    """
    features : (N, D) numpy array  –  스케일링 완료본 사용 권장
    labels   : (N,)   cluster 레이블
    title    : 그래프 제목 (모델·K 포함)
    save_dir : 출력 폴더
    """
    # perplexity는 샘플 수 5~50% 사이에서 실험 (기본 30)
    tsne = TSNE(n_components=2, perplexity=30,
                n_iter=1000, random_state=42, init="pca")  # init="pca" → 수렴 안정
    X_tsne = tsne.fit_transform(features)

    n_clusters = len(np.unique(labels))
    colors = plt.cm.get_cmap("tab10", n_clusters)

    plt.figure(figsize=(8, 6))
    for cid in range(n_clusters):
        mask = labels == cid
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                    alpha=0.6, label=f"Cl {cid}",
                    color=colors(cid))
    plt.title(f"t-SNE 2D – {title}")
    plt.xlabel("t-SNE-1"); plt.ylabel("t-SNE-2")
    plt.legend(); plt.grid(True)
    fname = os.path.join(save_dir,
                         f"tsne_{title.replace(' ', '_')}.png")
    plt.tight_layout(); plt.savefig(fname); plt.close()
    print(f"[t-SNE] {title} plot saved → {fname}")

# -----------------------------
# (D) main
# -----------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_scaling", action="store_true", 
                        help="Apply Z-score standardization for features.", default=True)
    parser.add_argument("--use_pca", action="store_true", 
                        help="Visualize PCA 2D scatter with cluster labels.", default=True)
    parser.add_argument("--use_tsne", action="store_true",
                        help="Visualize t-SNE 2D scatter with cluster labels.", default=True)
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

    run_mixture_and_visualize(features_F, valid_indices_F, points_pct_data_F,
                          "Female", "GM",
                          use_pca=args.use_pca)
    
    run_mixture_and_visualize(features_F, valid_indices_F, points_pct_data_F,
                          "Female", "BGM",
                          use_pca=args.use_pca)
    
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
    
    run_mixture_and_visualize(features_M, valid_indices_M, points_pct_data_M,
                          "Male", "GM",
                          use_pca=args.use_pca)

    run_mixture_and_visualize(features_M, valid_indices_M, points_pct_data_M,
                          "Male", "BGM",
                          use_pca=args.use_pca)
    
    kmeans_F = KMeans(n_clusters=2, random_state=42).fit(features_F)
    kmeans_M = KMeans(n_clusters=2, random_state=42).fit(features_M)

    labels_F_k2, labels_M_k2 = kmeans_F.labels_, kmeans_M.labels_

    save_clustered_features_to_csv(features_F, labels_F_k2, valid_names_F, "Female", 2)
    save_clustered_features_to_csv(features_M, labels_M_k2, valid_names_M, "Male", 2)