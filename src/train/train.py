import os
import pickle
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter

import shap

from torch.utils.data import ConcatDataset, TensorDataset
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

import logging
import matplotlib.pyplot as plt

from imblearn.under_sampling import RandomUnderSampler

import itertools
import seaborn as sns
import matplotlib.colors as mcolors
from scipy import stats
from statsmodels.stats.multitest import multipletests

# 출력 디렉토리 설정
OUTPUT_DIR = "out/models"
LOG_FILE_PATH = "out/logs/train.log"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)

# 로깅 설정
logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("train")
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())  # 콘솔 출력 추가

# 랜덤 시드 고정 (재현 가능)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# TensorBoard 설정
log_dir = os.path.join(OUTPUT_DIR, "tensorboard_" + datetime.now().strftime("%Y%m%d-%H%M%S"))
writer = SummaryWriter(log_dir=log_dir)

# 데이터셋 클래스 정의
class SpeechDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)

        # 입력(X), 출력(Y) 설정
        self.feature_names = ["start_f0", "end_f0", "mean_f0", "max_f0", "min_f0", "slope", "TCoG"]
        self.X = df[self.feature_names].values
        self.y = df["cluster_label"].values  # 0 = 하강, 1 = 상승
        self.filenames = df["filename"].values  # 파일명 추가
        
        # 데이터 정규화 -> load_data()에서 처리
        # self.scaler = StandardScaler()
        # self.X = self.scaler.fit_transform(self.X)
        
        # Tensor 변환
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)

        # 원본 데이터 저장 (층화 추출용)
        self.df = df  

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# MLP 모델 정의
class MLP(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.2):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # 첫 번째 은닉층 이후 dropout
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # 두 번째 은닉층 이후 dropout
            nn.Linear(32, 1)  # 출력 클래스 2개 (하강/상승)
        )
    
    def forward(self, x):
        return self.layers(x)


# 데이터 로드 & 층화 샘플링 적용 + Undersampling (성별 정보 반영)

def load_data_mixed(pseudo_csv, human_csv, batch_size=32):
    # ───────────────────── 데이터프레임 로드 ─────────────────────
    pseudo_df = pd.read_csv(pseudo_csv)
    human_df  = pd.read_csv(human_csv)

    feature_names = ["start_f0","end_f0","mean_f0","max_f0",
                     "min_f0","slope","TCoG"]

    # ───── 1) 인간 라벨 6-2-2 분할 ─────
    gender_h = human_df["filename"].str[-1].values
    y_h      = human_df["cluster_label"].values
    strat_h  = np.array([f"{y}_{g}" for y,g in zip(y_h, gender_h)])

    idx_h = np.arange(len(human_df))
    sss1  = StratifiedShuffleSplit(1, test_size=0.40, random_state=SEED)
    train_h_idx, temp_h_idx = next(sss1.split(idx_h, strat_h))

    strat_temp = strat_h[temp_h_idx]
    sss2  = StratifiedShuffleSplit(1, test_size=0.50, random_state=SEED)
    valid_rel, test_rel = next(sss2.split(temp_h_idx, strat_temp))
    valid_h_idx, test_h_idx = temp_h_idx[valid_rel], temp_h_idx[test_rel]

    # ───── 2) 군집 라벨(90 %)은 train 전용 ─────
    gender_p = pseudo_df["filename"].str[-1].values
    y_p      = pseudo_df["cluster_label"].values
    strat_p  = np.array([f"{y}_{g}" for y,g in zip(y_p, gender_p)])
    idx_p    = np.arange(len(pseudo_df))

    rus = RandomUnderSampler(random_state=SEED)
    train_p_idx, _ = rus.fit_resample(idx_p.reshape(-1,1), strat_p)
    train_p_idx = train_p_idx.flatten()

    X_p = pseudo_df[feature_names].values.astype(np.float32)
    X_h = human_df[feature_names].values.astype(np.float32)
    print(f"mean:{np.mean(X_p)}, std:{np.std(X_p)}")
    
    
    def make_tensor(idx, X, y):
        X = torch.tensor(X[idx])
        y = torch.tensor(y[idx])
        return TensorDataset(X, y)

    train_ds = ConcatDataset([
        make_tensor(train_p_idx, X_p, y_p),   # 군집
        make_tensor(train_h_idx, X_h, y_h)    # 인간(60 %)
    ])
    valid_ds = make_tensor(valid_h_idx, X_h, y_h)
    test_ds  = make_tensor(test_h_idx,  X_h, y_h)

    train_loader = DataLoader(train_ds, batch_size, shuffle=True,  drop_last=False)
    valid_loader = DataLoader(valid_ds, batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size, shuffle=False)

    # ───── 4) 평가용 파일명 리스트 반환 ─────
    valid_fnames = human_df["filename"].values[valid_h_idx]
    test_fnames  = human_df["filename"].values[test_h_idx]

    human_dataset = SpeechDataset(human_csv)   # 파일명·라벨 전체 보존
    
    def print_dist(name, idx_list, y_arr, fnames):
        labels = y_arr[idx_list]
        genders= [fname[-1] for fname in fnames[idx_list]]
        combined = [f"{y}_{g}" for y,g in zip(labels, genders)]
        cnt = Counter(combined)
        total = len(idx_list)
        logger.info(
            f"▶ {name} set: {total} samples — 분포: "
            f"{', '.join(f'{k}:{v}' for k, v in cnt.items())}"
        )

    print_dist("Train(pseudo)", train_p_idx, y_p, pseudo_df["filename"].values)
    print_dist("Train(human)", train_h_idx, y_h,    human_df["filename"].values)
    print_dist("Valid",       valid_h_idx, y_h,    human_df["filename"].values)
    print_dist("Test",        test_h_idx,  y_h,    human_df["filename"].values)

    return (train_loader, valid_loader, test_loader,
            human_dataset, valid_fnames, test_fnames)

# Feature Importance 분석 및 시각화
def plot_feature_importance(model):
    import seaborn as sns
    import matplotlib.colors as mcolors

    feature_weights = model.layers[0].weight.detach().cpu().numpy()
    feature_importance = np.abs(feature_weights).mean(axis=0)  # 평균 절대값 사용
    feature_names = ["start_f0", "end_f0", "mean_f0", "max_f0", "min_f0", "slope", "TCoG"]

    # 바 그래프 (내림차순 + 그라데이션 적용)
    sorted_indices = np.argsort(feature_importance)[::-1]
    sorted_features = [feature_names[i] for i in sorted_indices]
    sorted_importance = feature_importance[sorted_indices]

    plt.figure(figsize=(10, 6))
    cmap = mcolors.LinearSegmentedColormap.from_list("", ["navy", "skyblue"])  # 진한 색 → 옅은 색
    colors = [cmap(i / len(sorted_importance)) for i in range(len(sorted_importance))]
    bars = plt.barh(sorted_features, sorted_importance, color=colors)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importance in MLP Model")

    # 바 위에 값 추가
    for bar, imp in zip(bars, sorted_importance):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, f"{imp:.3f}", va="center")

    plt.gca().invert_yaxis()  # 위에서 아래로 정렬
    plt.tight_layout()
    plt.savefig(os.path.join("out/models", "feature_importance_bar.png"))
    logger.info("📊 Feature importance bar chart saved as 'out/models/feature_importance_bar.png'")

    # 상관관계 플롯
    correlation_matrix = np.corrcoef(feature_weights.T)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        xticklabels=feature_names,
        yticklabels=feature_names,
        cbar_kws={"label": "Correlation"},
        fmt=".2f"
    )
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join("out/models", "feature_correlation_matrix.png"))
    logger.info("📊 Feature correlation matrix saved as 'out/models/feature_correlation_matrix.png'")

    # 레이더 차트 (특성별 평균 중요도)
    normalized_importance = sorted_importance / max(sorted_importance)
    angles = np.linspace(0, 2 * np.pi, len(feature_names), endpoint=False).tolist()
    feature_importance_circle = np.concatenate((normalized_importance, [normalized_importance[0]]))
    angles += angles[:1]

    # 레이더 차트 생성
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    ax.fill(angles, feature_importance_circle, color="skyblue", alpha=0.4)
    ax.plot(angles, feature_importance_circle, color="blue", linewidth=2)

    # 반지름 축 설정 (정규화된 중요도에 대응)
    ax.set_yticks(np.linspace(0, 1, 5))  # 정규화된 범위로 반지름 설정
    ax.set_yticklabels([f"{v:.2f}" for v in np.linspace(0, max(sorted_importance), 5)])  # 원래 중요도 값 표시
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(sorted_features)

    # 제목과 저장
    ax.set_title("Radar Chart of Feature Importance (Normalized)", va="bottom")
    plt.tight_layout()
    plt.savefig(os.path.join("out/models", "feature_importance_radar_normalized.png"))
    logger.info("📊 Feature importance radar chart saved as 'out/models/feature_importance_radar_normalized.png'")
    
    # ─────────── 통계 검정 추가 ───────────
    # 1) neuron × feature 행렬에서 절대값 분포 추출
    W = model.layers[0].weight.detach().cpu().numpy()      # shape (n_neurons, 7)
    absW = np.abs(W)                                       # (n_neurons, 7)
    order = ["end_f0","max_f0","mean_f0","min_f0",
             "start_f0","slope","TCoG"]
    idx   = [feature_names.index(f) for f in order]
    data  = absW[:, idx]                                   # (n_neurons,7)

    # 2) Friedman 검정¹: 7개 그룹 간 분포 차이 확인
    stat, p = stats.friedmanchisquare(*[data[:,i] for i in range(data.shape[1])])
    print(f"▶ Friedman χ² = {stat:.3f}, p = {p:.3e}")

    # 3) 사후검정: Wilcoxon 부호검정 + Holm–Bonferroni 보정²
    pairs   = list(itertools.combinations(range(len(order)), 2))
    pvals   = []
    pairs_nm= []
    for i,j in pairs:
        _, p_ij = stats.wilcoxon(data[:,i], data[:,j])
        pvals.append(p_ij)
        pairs_nm.append(f"{order[i]} vs {order[j]}")
    rej, p_corr, _, _ = multipletests(pvals, alpha=0.05, method="holm")

    # 결과 출력
    for name, p0, p1, r in zip(pairs_nm, pvals, p_corr, rej):
        sig = "유의" if r else "ns"
        print(f"  - {name}: unadj p={p0:.3e}, adj p={p1:.3e} → {sig}")
    
# 학습 과정 시각화 함수
def plot_training_curves(train_losses, valid_losses, train_accuracies, valid_accuracies):
    epochs = range(1, len(train_losses) + 1)

    # Loss 곡선
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, valid_losses, label="Valid Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"))
    logger.info("📈 Loss curve saved as 'loss_curve.png'")

    # Accuracy 곡선
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_accuracies, label="Train Accuracy")
    plt.plot(epochs, valid_accuracies, label="Valid Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_curve.png"))
    logger.info("📈 Accuracy curve saved as 'accuracy_curve.png'")
    
# SHAP 분석 및 시각화 함수 (Checkpointing 추가)
def plot_shap_analysis_combined(model,
                                valid_loader,
                                test_loader,
                                valid_fnames,          # 리스트
                                test_fnames,           # 리스트
                                output_dir="out/models"):

    # 1) 모델 예측 함수
    def model_predict(X):
        with torch.no_grad():
            x = torch.tensor(X, dtype=torch.float32,
                             device=next(model.parameters()).device)
            return model(x).cpu().numpy()

    # 2) Valid·Test 묶어서 행렬/라벨/파일명 만들기
    X_chunks, y_chunks, fname_chunks = [], [], []

    # helper: 로더와 대응하는 파일명 리스트를 함께 다룬다
    for loader, fnames in zip((valid_loader, test_loader),
                              (valid_fnames, test_fnames)):
        ptr = 0
        for X_b, y_b in loader:
            bs = len(y_b)
            X_chunks.append(X_b.numpy())
            y_chunks.append(y_b.numpy())
            fname_chunks.extend(fnames[ptr:ptr+bs])
            ptr += bs

    X_all = np.vstack(X_chunks)           # (N, 7)
    y_all = np.hstack(y_chunks)           # (N,)
    filenames = fname_chunks              # 길이 N

    feature_names = np.array([
        "start_f0", "end_f0", "mean_f0",
        "max_f0", "min_f0", "slope", "TCoG"
    ])

    # 3) SHAP 값 계산 또는 체크포인트 로드
    ckpt = os.path.join(output_dir, "shap_values_combined.pkl")
    if os.path.exists(ckpt):
        with open(ckpt, "rb") as f:
            shap_values = pickle.load(f)
        explainer = None
        base_val  = 0
    else:
        explainer   = shap.KernelExplainer(model_predict, X_all[:100])
        shap_values = explainer.shap_values(X_all)   # (N, 7, 1)  ← 여기!
        if isinstance(shap_values, list):            # 리스트일 때
            shap_values = shap_values[0]             # (N, 7, 1)
        # 새로 추가: 불필요한 3번째 축 제거
        if shap_values.ndim == 3 and shap_values.shape[-1] == 1:
            shap_values = shap_values.squeeze(-1)    # (N, 7)
        base_val = explainer.expected_value
        with open(ckpt, "wb") as f:
            pickle.dump(shap_values, f)

    # 4) Summary plot
    exp = shap.Explanation(
    values        = shap_values,     # (N, 7)
    base_values   = np.repeat(base_val, len(shap_values)),
    data          = X_all,           # (N, 7)
    feature_names = feature_names
    )
    plt.figure(figsize=(8, 6))
    shap.plots.beeswarm(exp, max_display=7, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_summary_all.png"))
    plt.close()

    # 5) 클래스별 waterfall 샘플 50개씩 ---------------------------------
    def save_waterfall(idx, label_str):
        expl = shap.Explanation(
            values        = shap_values[idx],
            base_values   = base_val,
            data          = X_all[idx],
            feature_names = feature_names
        )
        plt.figure(figsize=(6, 4))
        shap.plots.waterfall(expl, max_display=7, show=False)
        plt.title(f"{filenames[idx]} ({label_str})")
        plt.tight_layout()
        fname_png = f"shap_waterfall_{label_str}_{filenames[idx]}.png"
        plt.savefig(os.path.join(output_dir, fname_png))
        plt.close()

    class_0_idx = np.where(y_all == 0)[0][:50]
    class_1_idx = np.where(y_all == 1)[0][:50]

    for i in class_0_idx:
        save_waterfall(i, "downward")
    for i in class_1_idx:
        save_waterfall(i, "upward")
        
    def save_beeswarm(shap_vals_subset, X_subset, idx_subset, label_str):
        """클래스별 beeswarm summary 저장"""
        exp = shap.Explanation(
            values        = shap_vals_subset,                     # (n_k, 7)
            base_values   = np.repeat(base_val, len(idx_subset)), # 길이 n_k
            data          = X_subset,                             # (n_k, 7)
            feature_names = feature_names
        )
        plt.figure(figsize=(8, 6))
        shap.plots.beeswarm(exp, max_display=7, show=False)       # 신 API → spine 버그 없음
        plt.title(f"SHAP Beeswarm ({label_str})")
        plt.tight_layout()
        fname = f"shap_beeswarm_{label_str}.png"
        plt.savefig(os.path.join(output_dir, fname))
        plt.close()

    # ① 클래스별 인덱스
    class_0_idx = np.where(y_all == 0)[0]
    class_1_idx = np.where(y_all == 1)[0]

    # ② 0-클래스 summary
    save_beeswarm(
        shap_vals_subset = shap_values[class_0_idx],
        X_subset         = X_all[class_0_idx],
        idx_subset       = class_0_idx,
        label_str        = "downward"
    )

    # ③ 1-클래스 summary
    save_beeswarm(
        shap_vals_subset = shap_values[class_1_idx],
        X_subset         = X_all[class_1_idx],
        idx_subset       = class_1_idx,
        label_str        = "upward"
    )

    print("SHAP summary & waterfall plots have been saved.")
    
# 모델 학습 함수 (수정: Loss/Accuracy 기록)
def train_model(model, train_loader, valid_loader, criterion, optimizer, device, num_epochs=120, checkpoint_path="out/models/best_checkpoint.pth"):
    best_valid_loss = float("inf")
    train_losses, valid_losses = [], []
    train_accuracies, valid_accuracies = [], []

    for epoch in range(num_epochs):
        model.train()
        total_loss, correct = 0, 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            # output = model(X_batch) shape (batch, 2)
            # loss = criterion(output, y_batch) integer class label
            logits = model(X_batch).squeeze(1)  # shape (batch,)
            loss = criterion(logits, y_batch.float())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            # correct += (output.argmax(1) == y_batch).sum().item()
            probs   = torch.sigmoid(logits)
            preds   = (probs >= 0.5).long()
            correct += (preds == y_batch).sum().item()
        
        avg_train_loss = total_loss / len(train_loader)
        train_acc = correct / len(train_loader.dataset)
        train_losses.append(total_loss / len(train_loader))
        train_accuracies.append(train_acc)
        
        writer.add_scalar("Loss/train", total_loss / len(train_loader), epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        
        model.eval()
        valid_loss, valid_correct = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in valid_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                # 1) 로짓 추출 (shape: [batch,])
                logits = model(X_batch).squeeze(-1)
                
                # 2) loss 계산 (float 타입 레이블)
                loss = criterion(logits, y_batch.float())
                valid_loss += loss.item()
                
                # 3) 확률 변환 & 예측
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).long()
                valid_correct += (preds == y_batch).sum().item()
        

        # 4) 지표 집계
        avg_valid_loss = valid_loss / len(valid_loader)
        valid_acc = valid_correct / len(valid_loader.dataset)
        valid_losses.append(valid_loss / len(valid_loader))
        valid_accuracies.append(valid_acc)

        # 5) TensorBoard 기록
        writer.add_scalar("Loss/valid", valid_loss / len(valid_loader), epoch)
        writer.add_scalar("Accuracy/valid", valid_acc, epoch)

        # 6) 로그 출력 및 체크포인트 저장
        logger.info(
        f"Epoch [{epoch+1}/{num_epochs}] "
        f"Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}, "
        f"Train Acc: {train_acc:.4f}, Valid Acc: {valid_acc:.4f}"
    )
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"✔ Best model saved at epoch {epoch+1}")

    # 학습 곡선 시각화
    plot_training_curves(train_losses, valid_losses, train_accuracies, valid_accuracies)

# 성능 리포트 (Test 결과)
def evaluate_model(model, test_loader, device, dataset):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            # output = model(X_batch)
            # predictions = output.argmax(1).cpu().numpy()
            logits = model(X_batch).squeeze(1) # (batch,)
            probs  = torch.sigmoid(logits)
            predictions = (probs >= 0.5).long().cpu().numpy()
            y_pred.extend(predictions)
            y_true.extend(y_batch.cpu().numpy())

    # Classification Report
    report = classification_report(y_true, y_pred, target_names=["Falling", "Rising"], output_dict=True)
    logger.info(f"Test Report:\n{pd.DataFrame(report).transpose()}")
    report_path = os.path.join(OUTPUT_DIR, "classification_report.csv")
    pd.DataFrame(report).transpose().to_csv(report_path)
    logger.info(f"📊 Classification report saved as '{report_path}'")
    

# 실행
if __name__ == "__main__":
    # 데이터 로드
    train_loader, valid_loader, test_loader, human_ds, \
    valid_fnames, test_fnames = load_data_mixed(
        pseudo_csv="train-data/training_data.csv",
        human_csv="train-data/training_data_human.csv",
        batch_size=32
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MLP(input_dim=7).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, valid_loader,
                criterion, optimizer, device, num_epochs=2000)
    evaluate_model(model, test_loader, device, None) 
    
    
    # # 데이터 로드 (학습 없이 데이터셋만 로드)
    # train_loader, valid_loader, test_loader, human_ds, \
    # valid_fnames, test_fnames = load_data_mixed(
    #     pseudo_csv="train-data/training_data.csv",
    #     human_csv="train-data/training_data_human.csv",
    #     batch_size=32
    # )

    # 디바이스 설정
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 모델 생성 (체크포인트 불러오기 위해 초기화 필요)
    model = MLP(input_dim=7).to(device)

    # 체크포인트 불러오기
    checkpoint_path = "out/models/best_checkpoint.pth"  # 저장된 모델 가중치 경로
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()  # 평가 모드로 설정
        logger.info(f"Model checkpoint loaded from '{checkpoint_path}'")
    else:
        raise FileNotFoundError(f"Error: No checkpoint found at '{checkpoint_path}'")

    # 시각화 함수 실행 (학습 없이 바로 분석 진행)
    plot_feature_importance(model)
    # 모델·체크포인트 로딩 후
    plot_shap_analysis_combined(
        model,
        valid_loader,
        test_loader,
        valid_fnames,
        test_fnames,
        output_dir="out/models"
    )

