MNIST数据集是一个经典的手写数字数据集，包含0到9的手写数字图像。每个图像是28x28像素的灰度图像，总共有60,000张训练图像和10,000张测试图像。

如果需要一个包含更多类别的图像数据集，可以考虑以下几个常用的数据集：

1. **CIFAR-10**：包含10个类别的彩色图像，每个类别有6,000张图像，总共有60,000张32x32像素的彩色图像。
2. **CIFAR-100**：与CIFAR-10类似，但包含100个类别，每个类别有600张图像，总共有60,000张32x32像素的彩色图像。
3. **ImageNet**：一个大规模的图像数据集，包含超过1000个类别的数百万张图像，常用于图像分类和物体检测任务。

### 使用CIFAR-10数据集的示例

如果想使用CIFAR-10数据集，可以按照以下步骤进行修改：

#### 更新数据加载部分
```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 加载CIFAR-10数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
]) # 归一化
train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
```

#### 更新生成器和判别器模型
由于CIFAR-10图像是32x32像素的彩色图像（3个通道），需要相应地调整生成器和判别器模型：

```python
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 3 * 32 * 32),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), 3, 32, 32)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3 * 32 * 32, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), 3 * 32 * 32)
        x = self.model(x)
        return x
```

#### 更新保存图像的部分
由于CIFAR-10图像是彩色图像，需要相应地调整保存图像的代码：

```python
def save_images(epoch):
    z = torch.randn(25, 100, device=device)
    gen_imgs = generator(z)
    gen_imgs = gen_imgs.view(gen_imgs.size(0), 3, 32, 32)
    gen_imgs = gen_imgs.cpu().detach().numpy()

    fig, axs = plt.subplots(5, 5)
    cnt = 0
    for i in range(5):
        for j in range(5):
            axs[i, j].imshow(np.transpose(gen_imgs[cnt], (1, 2, 0)) * 0.5 + 0.5)  # 反归一化
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig(f"{config['path']['images']}cifar10_{epoch}.png")
    plt.close()
```

通过这些修改，可以使用CIFAR-10数据集来训练生成对抗网络（GAN）。可以根据需要调整模型结构和训练参数，以获得更好的生成效果。


## 938

在MNIST数据集中，训练集包含60,000张图像。如果你使用的批次大小（batch size）是64，那么每个epoch中的批次数量可以通过以下公式计算：

\[ \text{批次数量} = \left\lceil \frac{\text{总图像数量}}{\text{批次大小}} \right\rceil \]

对于MNIST数据集：

\[ \text{批次数量} = \left\lceil \frac{60000}{64} \right\rceil = 938 \]

这就是为什么你在日志中看到每个epoch有938个批次。

### 验证批次数量
你可以在代码中添加一些调试信息来验证批次数量的计算是否正确：

```python
def train_GAN(epochs, save_intervals):
    train_loader = get_data_loader(config.batch_size)
    print(f"Total number of batches per epoch: {len(train_loader)}")  # 添加这行代码来打印每个epoch的批次数量
    for epoch in range(epochs):
        for i, (imgs, labels) in enumerate(train_loader):
            # 真实图像
            real_imgs = imgs.to(device)
            labels = labels.to(device)
            valid = torch.ones(imgs.size(0), 1, device=device)
            fake = torch.zeros(imgs.size(0), 1, device=device)

            # 训练生成器
            optimizer_G.zero_grad()
            z = torch.randn(imgs.size(0), 100, device=device)
            gen_imgs = generator(z, labels)
            g_loss = adversarial_loss(discriminator(gen_imgs, labels), valid)
            g_loss.backward()
            optimizer_G.step()

            # 训练判别器
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(real_imgs, labels), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), labels), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            logger.info(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(train_loader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

        # 保存模型和图像
        if should_save(epoch, save_intervals):
            save_images(epoch)
            save_model(epoch)
    # 保存最终模型
    save_images(epochs-1)
    save_model(epochs-1)
```

### 验证数据加载器
你也可以在数据加载器中添加一些调试信息来验证数据集和批次大小：

```python
import os
import torch
from torchvision import datasets, transforms
from src.config import BASE_DIR, config

def get_data_loader(batch_size=64, data_dir=None):
    if data_dir is None:
        data_dir = os.path.join(BASE_DIR, 'data', 'MNIST', 'raw')
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ]) # 归一化

    train_dataset = datasets.MNIST(root=data_dir, train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print(f"Total number of images: {len(train_dataset)}")  # 添加这行代码来打印总图像数量
    print(f"Batch size: {batch_size}")  # 添加这行代码来打印批次大小
    return train_loader
```

通过这些调试信息，你可以验证批次数量的计算是否正确，并确保数据加载器和批次大小设置正确。