import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ================= 配置区 =================
FEATURE_DIR = "/data/users/yubo_wang/ESC-50/projected_token" # 刚才保存 .npy 的路径
SAVE_MODEL_PATH = "esc50_projection_probe.pth"
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# =========================================

class ESC50FeatureDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths
        
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        
        # 1. 加载特征 [T, 1280]
        # 注意：我们之前保存的是 float32 的 npy，直接加载即可
        feature = np.load(path)
        
        # 2. Mean Pooling (关键步骤)
        # 将 [T, 1280] -> [1280]
        # 这一步让模型具有时间平移不变性，且统一了维度
        feature_mean = np.mean(feature, axis=0)
        
        # 3. 解析标签
        # ESC-50 文件名格式: {fold}-{clip_id}-{take}-{target}.npy
        # 例如: 1-100032-A-0.npy -> 类别是 0
        filename = os.path.basename(path)
        label_str = filename.split('.')[0].split('-')[-1]
        label = int(label_str)
        
        return torch.FloatTensor(feature_mean), torch.tensor(label, dtype=torch.long)

class SimpleProbe(nn.Module):
    def __init__(self, input_dim=2048, num_classes=50):
        super().__init__()
        # 这是一个最简单的线性探针
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024), # LayerNorm 有助于稳定特征分布
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.fc(x)

def train():
    # 1. 准备数据
    # 获取所有 .npy 文件
    all_files = glob.glob(os.path.join(FEATURE_DIR, "*.npy"))
    if len(all_files) == 0:
        print(f"Error: {FEATURE_DIR} 下没有找到 .npy 文件！")
        return

    print(f"--> Found {len(all_files)} feature files.")
    
    # 划分训练集和测试集 (8:2)
    train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=40)
    
    train_ds = ESC50FeatureDataset(train_files)
    val_ds = ESC50FeatureDataset(val_files)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 2. 初始化模型
    model = SimpleProbe().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"--> Start training Linear Probe on {DEVICE}...")

    # 3. 训练循环
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        # --- Train ---
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for feats, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(feats)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_acc = 100 * correct / total
        
        # --- Validation ---
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for feats, labels in val_loader:
                feats, labels = feats.to(DEVICE), labels.to(DEVICE)
                outputs = model(feats)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        print(f"    Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), SAVE_MODEL_PATH)
            print(f"    --> Best model saved! ({best_acc:.2f}%)")

    print(f"\nTraining Finished. Best Validation Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    train()