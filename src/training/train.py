import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(f"BASE_DIR: {BASE_DIR}")
sys.path.append(BASE_DIR)
from src.utils.training_recorder import create_training_record
from src.config import TRAINING_PARAMETERS, WEIGHTS_DIR
from matplotlib import pyplot as plt
import torch
import torch.optim as optim
from src.models.cGAN import Generator, Discriminator
from src.data.data_loader import get_data_loader
from src.utils.logger import get_logger
from src.utils.training_utils import should_save, save_images, save_model

# 创建训练记录并获取日志文件名
log_file = create_training_record()
# 初始化日志记录器
logger = get_logger('train', log_file)

# 初始化模型和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
noise_dim = 100
label_dim = 10
img_channels = 1
img_width = 28
img_height = 28
generator = Generator(noise_dim, label_dim, img_channels, img_width, img_height).to(device)
discriminator = Discriminator(img_channels, img_width, img_height, label_dim).to(device)
optimizer_G = optim.Adam(generator.parameters(), lr=TRAINING_PARAMETERS['learning_rate'], betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=TRAINING_PARAMETERS['learning_rate'], betas=(0.5, 0.999))
# adversarial_loss(input, target) = -1/n * (target * log(input) + (1 - target) * log(1 - input))
adversarial_loss = torch.nn.BCELoss()


def load_model(generator, discriminator, epoch):
    generator_path = os.path.join(WEIGHTS_DIR, f"generator_{epoch}.pth")
    discriminator_path = os.path.join(WEIGHTS_DIR, f"discriminator_{epoch}.pth")
    if os.path.exists(generator_path) and os.path.exists(discriminator_path):
        generator.load_state_dict(torch.load(generator_path))
        discriminator.load_state_dict(torch.load(discriminator_path))
        print(f"Loaded model from epoch {epoch}")
    else:
        print(f"No saved model found for epoch {epoch}")

def train_GAN(end_epoch, save_intervals, start_epoch=0):
    train_loader = get_data_loader(TRAINING_PARAMETERS['batch_size'])
    for epoch in range(start_epoch+1, end_epoch+1):
        for i, (imgs, labels) in enumerate(train_loader):
            real_imgs = imgs.to(device) # 真实图像
            real_labels = labels.to(device) # 真实标签
            
            batch_size = imgs.size(0)
            real_targets = torch.ones(batch_size, 1, device=device)  # 判别器的目标输出（真实）
            fake_targets = torch.zeros(batch_size, 1, device=device)  # 判别器的目标输出（假）
            
            # 生成随机噪声和假图像
            noise = torch.randn(batch_size, noise_dim, device=device)
            fake_imgs = generator(noise, real_labels) # 生成器接受一个随机噪声向量作为输入，并生成伪造的数据(这里指的是图像, 即 28*28 的数据s)
            
            # 训练判别器
            optimizer_D.zero_grad()
            
            real_preds = discriminator(real_imgs, real_labels) # 判别器对真实图像的判别
            real_loss = adversarial_loss(real_preds , real_targets) # 判别器对真实图像的判别的损失
            
            fake_preds = discriminator(fake_imgs.detach(), real_labels) # 判别器对生成图像的判别
            fake_loss = adversarial_loss(fake_preds, fake_targets) # 判别器对生成图像的判别的损失
            
            d_loss = (real_loss + fake_loss) / 2 # 3. 判别器的损失
            d_loss.backward()   # 4. 反向传播: 计算梯度
            optimizer_D.step() # 5. 更新判别器的权重
            
            # 训练生成器
            optimizer_G.zero_grad()
            
            new_fake_preds = discriminator(fake_imgs, real_labels) # 使用更新权重后的判别器对生成的图像进行判别
            g_loss = adversarial_loss(new_fake_preds, real_targets) # 3. 计算生成器的损失
            g_loss.backward() # 4. 反向传播: 计算梯度
            optimizer_G.step()  # 5. 更新生成器的权重

            logger.info(f"{epoch},{i},{d_loss.item()},{g_loss.item()}")

        # 保存模型和图像
        if should_save(epoch, save_intervals):
            save_images(epoch=epoch, generator=generator, device=device, real_imgs=real_imgs, real_labels=real_labels)
            save_model(epoch=epoch, generator=generator, discriminator=discriminator)
    # 保存最终模型
    save_images(end_epoch, generator, device)
    save_model(end_epoch, generator, discriminator)

if __name__ == "__main__":
    start_epoch = TRAINING_PARAMETERS.get('start_epoch', 0)  # 之前训练的轮数
    # 加载之前的模型
    if start_epoch > 0:
        load_model(generator, discriminator, start_epoch)
    train_GAN(end_epoch=TRAINING_PARAMETERS['end_epoch'], save_intervals=TRAINING_PARAMETERS['save_intervals'], start_epoch=start_epoch)