import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import shap
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report
import random
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler

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
            nn.Linear(32, 2)  # 출력 클래스 2개 (하강/상승)
        )
    
    def forward(self, x):
        return self.layers(x)


# 데이터 로드 & 층화 샘플링 적용 + Undersampling (성별 정보 반영)
def load_data(csv_file, batch_size=32):
    from torch.utils.data import TensorDataset

    # --- 기존 stratified split & undersampling 로직 그대로 ---
    dataset = SpeechDataset(csv_file)  
    feature_names = dataset.feature_names
    df = dataset.df  # raw pandas.DataFrame

    # 레이블·성별 결합 stratification용
    gender = np.array([fname[-1] for fname in dataset.filenames])
    y_all = dataset.y.numpy()
    strat_labels = np.array([f"{y}_{g}" for y, g in zip(y_all, gender)])

    idx = np.arange(len(dataset))
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=SEED)
    train_idx, temp_idx = next(sss1.split(idx, strat_labels))

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
    temp_strat = strat_labels[temp_idx]
    valid_rel, test_rel = next(sss2.split(temp_idx, temp_strat))
    valid_idx = temp_idx[valid_rel]
    test_idx  = temp_idx[test_rel]

    # Train만 undersampling
    rus = RandomUnderSampler(random_state=SEED)
    train_idx_res, _ = rus.fit_resample(train_idx.reshape(-1,1), strat_labels[train_idx])
    train_idx = train_idx_res.flatten()

    # --- 여기가 핵심: raw feature matrix 꺼내기 ---
    X_all = df[feature_names].values.astype(np.float32)  # shape (N,7)

    # (1) Train으로만 scaler fit
    scaler = StandardScaler().fit(X_all[train_idx])

    # (2) Train/Valid/Test 각각 transform
    X_train = scaler.transform(X_all[train_idx])
    X_valid = scaler.transform(X_all[valid_idx])
    X_test  = scaler.transform(X_all[test_idx])

    y_train = y_all[train_idx]
    y_valid = y_all[valid_idx]
    y_test  = y_all[test_idx]

    # (3) TensorDataset으로 묶기
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    valid_ds = TensorDataset(torch.tensor(X_valid), torch.tensor(y_valid))
    test_ds  = TensorDataset(torch.tensor(X_test),  torch.tensor(y_test))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size)

    # 데이터셋 개수 및 레이블 분포 로깅 (결합 레이블 "y_gender" 기준)
    def log_label_distribution_from_idx(name, idx_list):
        from collections import Counter
        # 원본 dataset에서 각 인덱스에 해당하는 y와 filename[-1]을 뽑아서 결합
        combined = [
            f"{dataset.y[idx].item()}_{dataset.filenames[idx][-1]}"
            for idx in idx_list
        ]
        counts = Counter(combined)
        dist = ", ".join(f"{lab}: {cnt}" for lab, cnt in counts.items())
        logger.info(f"📝 {name} 레이블 분포 (y_gender): {dist}")

    # 언더샘플링 후 train_idx가 재정의 되었으니, 로그는 여기서
    log_label_distribution_from_idx("Train", train_idx)
    log_label_distribution_from_idx("Valid", valid_idx)
    log_label_distribution_from_idx("Test",  test_idx)

    return train_loader, valid_loader, test_loader, dataset

# Feature Importance 분석 및 시각화
def plot_feature_importance(model, dataset):
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
def plot_shap_analysis_combined(model, valid_loader, test_loader, dataset, output_dir="out/models"):

    # 모델 예측 함수 정의
    def model_predict(X):
        model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(next(model.parameters()).device)
            logits = model(X_tensor).cpu().numpy()
            return logits

    # Valid와 Test 데이터 병합
    data_list = []
    y_list = []
    filenames = []

    for loader in [valid_loader, test_loader]:
        for X_batch, y_batch in loader:
            data_list.append(X_batch.numpy())
            y_list.append(y_batch.numpy())
            filenames += [dataset.filenames[idx] for idx in range(len(y_batch))]

    X_combined = np.concatenate(data_list, axis=0)  # Valid + Test 데이터
    y_combined = np.concatenate(y_list, axis=0)  # Valid + Test 라벨
    feature_names = ["start_f0", "end_f0", "mean_f0", "max_f0", "min_f0", "slope", "TCoG"]

    # Checkpoint 파일 경로
    checkpoint_path = os.path.join(output_dir, "shap_values_combined.pkl")

    # SHAP Explainer 생성
    if os.path.exists(checkpoint_path):
        logger.info(f"SHAP values checkpoint found. Loading from {checkpoint_path}.")
        with open(checkpoint_path, "rb") as f:
            shap_values = pickle.load(f)
        explainer = None  # 저장된 SHAP 값 사용 시 explainer는 필요 없음
    else:
        logger.info(f"Calculating SHAP values for combined dataset. This may take some time...")
        explainer = shap.KernelExplainer(model_predict, X_combined)  # 첫 100개 샘플로 배경 데이터 구성
        shap_values = np.array(explainer.shap_values(X_combined))  # SHAP 값 계산

        # 계산 결과 저장
        with open(checkpoint_path, "wb") as f:
            pickle.dump(shap_values, f)
        logger.info(f"SHAP values saved to {checkpoint_path}.")

    # SHAP 값 차원 변환 (필요 시)
    if shap_values.shape[-1] == 2:  # (num_samples, num_features, num_classes)
        shap_values = np.transpose(shap_values, (2, 0, 1))  # (num_classes, num_samples, num_features)

    if shap_values.shape[0] != 2:  # 클래스가 2개가 아니면 에러 발생
        raise ValueError(f"Unexpected shape for SHAP values after transformation: {np.shape(shap_values)}")

    # SHAP Summary Plot (전체 데이터)
    for i, class_shap_values in enumerate(shap_values):
        plt.figure()
        shap.summary_plot(class_shap_values, X_combined, feature_names=feature_names, show=False)
        if i == 0:
            suffix = 'downward'
        else:
            suffix = 'upward'
        plt.title(f"SHAP Summary Plot for Class {i}({suffix})")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"shap_summary_class_{i}({suffix}).png"))
        plt.close()
        logger.info(f"📊 SHAP summary plot saved for Class {i}: 'shap_summary_class_{i}.png'")

    # 클래스별로 샘플 5개씩 선택 (하강: 0, 상승: 1)
    class_0_indices = np.where(y_combined == 0)[0][:100]  # 하강 샘플 5개
    class_1_indices = np.where(y_combined == 1)[0][:100]  # 상승 샘플 5개

    # explainer 값 처리 (SHAP Explainer가 없으면 기본값 설정)
    if explainer is not None:
        expected_value = explainer.expected_value if isinstance(explainer.expected_value, (int, float)) else explainer.expected_value[0]
    else:
        expected_value = 0  # explainer가 없으면 기본값으로 0 사용
    
    for sample_idx in class_0_indices:
        shap.force_plot(
            expected_value,
            shap_values[0][sample_idx],
            feature_names=feature_names,
            matplotlib=True
        )
        filename = filenames[sample_idx]
        
        plt.title(f"SHAP Force Plot for {filename} (Downward Intonation)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"shap_force_downward_{filename}.png"))
        plt.close()
        logger.info(f"📊 SHAP force plot saved for {filename} (Downward Intonation)")

    for sample_idx in class_1_indices:
        shap.force_plot(
            expected_value,
            shap_values[1][sample_idx],
            feature_names=feature_names,
            matplotlib=True
        )
        filename = filenames[sample_idx]
        plt.title(f"SHAP Force Plot for {filename} (Upward Intonation)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"shap_force_upward_{filename}.png"))
        plt.close()
        logger.info(f"📊 SHAP force plot saved for {filename} (Upward Intonation)")

    logger.info("SHAP analysis and visualization completed for combined dataset.")
    
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
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (output.argmax(1) == y_batch).sum().item()

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
                output = model(X_batch)
                loss = criterion(output, y_batch)
                valid_loss += loss.item()
                valid_correct += (output.argmax(1) == y_batch).sum().item()
        
        valid_acc = valid_correct / len(valid_loader.dataset)
        valid_losses.append(valid_loss / len(valid_loader))
        valid_accuracies.append(valid_acc)
        writer.add_scalar("Loss/valid", valid_loss / len(valid_loader), epoch)
        writer.add_scalar("Accuracy/valid", valid_acc, epoch)
        
        logger.info(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {total_loss:.4f}, Valid Loss: {valid_loss:.4f}, Train Acc: {train_acc:.4f}, Valid Acc: {valid_acc:.4f}")

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
            output = model(X_batch)
            predictions = output.argmax(1).cpu().numpy()
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
    train_loader, valid_loader, test_loader, dataset = load_data("training_data.csv")

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = MLP(input_dim=7).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, valid_loader, criterion, optimizer, device, num_epochs=150)
    evaluate_model(model, test_loader, device, dataset)
    
    
    # 데이터 로드 (학습 없이 데이터셋만 로드)
    # train_loader, valid_loader, test_loader, dataset = load_data("training_data.csv")

    # 디바이스 설정
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # 모델 생성 (체크포인트 불러오기 위해 초기화 필요)
    # model = MLP(input_dim=7).to(device)

    # 체크포인트 불러오기
    checkpoint_path = "out/models/best_checkpoint2.pth"  # 저장된 모델 가중치 경로
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()  # 평가 모드로 설정
        logger.info(f"Model checkpoint loaded from '{checkpoint_path}'")
    else:
        raise FileNotFoundError(f"❌ Error: No checkpoint found at '{checkpoint_path}'")

    # 시각화 함수 실행 (학습 없이 바로 분석 진행)
    plot_feature_importance(model, dataset)
    plot_shap_analysis_combined(model, valid_loader, test_loader, dataset)

    