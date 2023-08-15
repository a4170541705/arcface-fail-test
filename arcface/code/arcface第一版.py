import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class ArcFace(nn.Module):
    def __init__(self, num_classes, emb_size, margin=0.5):
        super(ArcFace, self).__init__()
        self.num_classes = num_classes
        self.emb_size = emb_size
        self.margin = margin

        # 创建权重参数
        self.weights = nn.Parameter(torch.FloatTensor(num_classes, emb_size))

    def forward(self, embeddings, targets):
        # 归一化特征向量和权重
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weights = F.normalize(self.weights, p=2, dim=1)

        # 计算cosine相似度
        cos_theta = F.linear(embeddings, weights)

        # 获取目标类别的权重
        target_weights = weights[targets]

        # 计算角度余弦值和角度差
        cos_theta_m = torch.cos(self.margin - cos_theta)
        cos_theta = cos_theta - target_weights * (1 - cos_theta_m)

        return cos_theta

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.ToTensor(),          # 将图像转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])

# 加载数据集
train_dataset = torchvision.datasets.CIFAR10(root='D:/學校資料/畢業專題/人臉辨識/face_dataset/AFAD-Full/15/111', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='D:/學校資料/畢業專題/人臉辨識/face_dataset/AFAD-Full/15/111', train=False, transform=transform, download=True)

# 假设您的 train_dataset 是一个 Dataset 对象
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 初始化模型、损失函数和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ArcFace(num_classes=10, emb_size=64).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 训练模型
for epoch in range(10):
    for batch_idx, (data, target) in enumerate(train_dataset):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data.view(data.size(0), -1))
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')

print("Training finished.")

# 保存模型
torch.save(model.state_dict(), 'arcface_model.pth')