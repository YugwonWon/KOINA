import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
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

# ✅ 출력 디렉토리 설정
OUTPUT_DIR = "out/models"
LOG_FILE_PATH = "out/logs/train.log"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)

# ✅ 로깅 설정
logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("train")
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())  # 콘솔 출력 추가

# ✅ 랜덤 시드 고정 (재현 가능)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ✅ TensorBoard 설정
log_dir = os.path.join(OUTPUT_DIR, "tensorboard_" + datetime.now().strftime("%Y%m%d-%H%M%S"))
writer = SummaryWriter(log_dir=log_dir)

# ✅ 데이터셋 클래스 정의
class SpeechDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)

        # ✅ 입력(X), 출력(Y) 설정
        self.feature_names = ["start_f0", "end_f0", "mean_f0", "max_f0", "min_f0", "slope", "TCoG"]
        self.X = df[self.feature_names].values
        self.y = df["cluster_label"].values  # 0 = 하강, 1 = 상승

        # ✅ 데이터 정규화
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)

        # ✅ Tensor 변환
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)

        # ✅ 원본 데이터 저장 (층화 추출용)
        self.df = df  

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ✅ MLP 모델 정의
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # 출력 클래스 2개 (하강/상승)
        )
    
    def forward(self, x):
        return self.layers(x)

# ✅ 데이터 로드 & 층화 샘플링 적용 + Undersampling
def load_data(csv_file, batch_size=32):
    dataset = SpeechDataset(csv_file)

    # ✅ 층화 샘플링 (Stratified Split)
    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    indices = np.arange(len(dataset))
    y_labels = dataset.y.numpy()

    train_idx, temp_idx = next(stratified_split.split(dataset.X, y_labels))
    valid_test_split = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
    valid_idx, test_idx = next(valid_test_split.split(dataset.X[temp_idx], y_labels[temp_idx]))

    # ✅ Undersampling 적용 (Train 데이터만)
    rus = RandomUnderSampler(random_state=SEED)
    train_idx, _ = rus.fit_resample(train_idx.reshape(-1, 1), y_labels[train_idx])
    train_idx = train_idx.flatten()  # 1차원 배열로 변환

    # ✅ Subset을 이용하여 데이터셋 나누기
    train_dataset = Subset(dataset, train_idx)
    valid_dataset = Subset(dataset, valid_idx)
    test_dataset = Subset(dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    logger.info(f"📝 데이터셋 개수: Train: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}")

    return train_loader, valid_loader, test_loader, dataset

# ✅ Feature Importance 분석 및 시각화
def plot_feature_importance(model, dataset):
    import seaborn as sns
    import matplotlib.colors as mcolors

    feature_weights = model.layers[0].weight.detach().cpu().numpy()
    feature_importance = np.abs(feature_weights).mean(axis=0)  # 평균 절대값 사용
    feature_names = ["start_f0", "end_f0", "mean_f0", "max_f0", "min_f0", "slope", "TCoG"]

    # ✅ 바 그래프 (내림차순 + 그라데이션 적용)
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

    # ✅ 바 위에 값 추가
    for bar, imp in zip(bars, sorted_importance):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, f"{imp:.3f}", va="center")

    plt.gca().invert_yaxis()  # 위에서 아래로 정렬
    plt.tight_layout()
    plt.savefig(os.path.join("out/models", "feature_importance_bar.png"))
    logger.info("📊 Feature importance bar chart saved as 'out/models/feature_importance_bar.png'")

    # ✅ 상관관계 플롯
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

    # ✅ 레이더 차트 (특성별 평균 중요도)
    angles = np.linspace(0, 2 * np.pi, len(feature_names), endpoint=False).tolist()
    feature_importance_circle = np.concatenate((sorted_importance, [sorted_importance[0]]))
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    ax.fill(angles, feature_importance_circle, color="skyblue", alpha=0.4)
    ax.plot(angles, feature_importance_circle, color="blue", linewidth=2)
    ax.set_yticks([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(sorted_features)
    ax.set_title("Radar Chart of Feature Importance", va="bottom")
    plt.tight_layout()
    plt.savefig(os.path.join("out/models", "feature_importance_radar.png"))
    logger.info("📊 Feature importance radar chart saved as 'out/models/feature_importance_radar.png'")
    
# ✅ 학습 과정 시각화 함수
def plot_training_curves(train_losses, valid_losses, train_accuracies, valid_accuracies):
    epochs = range(1, len(train_losses) + 1)

    # ✅ Loss 곡선
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

    # ✅ Accuracy 곡선
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
    
# ✅ 모델 학습 함수 (수정: Loss/Accuracy 기록)
def train_model(model, train_loader, valid_loader, criterion, optimizer, device, num_epochs=50, checkpoint_path="out/models/best_checkpoint.pth"):
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

    # ✅ 학습 곡선 시각화
    plot_training_curves(train_losses, valid_losses, train_accuracies, valid_accuracies)

# ✅ 성능 리포트 (Test 결과)
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

    # ✅ Classification Report
    report = classification_report(y_true, y_pred, target_names=["Falling", "Rising"], output_dict=True)
    logger.info(f"Test Report:\n{pd.DataFrame(report).transpose()}")
    report_path = os.path.join(OUTPUT_DIR, "classification_report.csv")
    pd.DataFrame(report).transpose().to_csv(report_path)
    logger.info(f"📊 Classification report saved as '{report_path}'")
    

# ✅ 실행
if __name__ == "__main__":
    train_loader, valid_loader, test_loader, dataset = load_data("training_data.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_dim=7).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, valid_loader, criterion, optimizer, device, num_epochs=100)
    evaluate_model(model, test_loader, device, dataset)
    plot_feature_importance(model, dataset)
