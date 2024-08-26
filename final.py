import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 路径
images_dir = r'D:\A桌面杂项\开学前学习\job\images'
labels_dir = r'D:\A桌面杂项\开学前学习\job\labels'


# 数据集类
class NailSegmentationDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.images = os.listdir(images_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.images[idx])
        label_path = os.path.join(self.labels_dir, self.images[idx])

        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert('L')

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        label = torch.tensor(np.array(label), dtype=torch.long)

        return image, label


# 卷积神经网络
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.pool(self.relu(self.conv2(x1)))
        x3 = self.up(self.relu(self.conv3(x2)))
        x4 = self.conv4(x3)
        return self.sigmoid(x4)


# 参数
batch_size = 2  # 减小批量大小
learning_rate = 0.0001  # 减小学习率
num_epochs = 50  # 增加训练次数

# 数据加载器
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

train_dataset = NailSegmentationDataset(images_dir, labels_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 模型、损失函数和优化器
model = SimpleCNN().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练循环
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device).float()  # 将图像和标签移动到GPU

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

# 测试并显示对比图
model.eval()
test_image_path = r'D:\A桌面杂项\开学前学习\job\myhand.jpg'
test_image = Image.open(test_image_path).convert('RGB')
test_image_transformed = transform(test_image).unsqueeze(0).to(device)  # 将测试图像移动到GPU

with torch.no_grad():
    output = model(test_image_transformed)
    output = output.squeeze().cpu().numpy()

# 可视化原始图像与模型输出的对比
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(np.array(test_image))
ax[0].set_title('Original Image')
ax[1].imshow(output, cmap='gray')
ax[1].set_title('Model Output')
plt.show()