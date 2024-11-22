import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(f"BASE_DIR: {BASE_DIR}")
sys.path.append(BASE_DIR)
from src.utils.training_recorder import create_training_record
from src.config import LOGS_DIR, config_dict, IMAGES_DIR, WEIGHTS_DIR,TRAINING_PARAMETERS
import os
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
generator = Generator().to(device)
discriminator = Discriminator().to(device)
optimizer_G = optim.Adam(generator.parameters(), lr=TRAINING_PARAMETERS['learning_rate'], betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=TRAINING_PARAMETERS['learning_rate'], betas=(0.5, 0.999))
# adversarial_loss(input, target) = -1/n * (target * log(input) + (1 - target) * log(1 - input))
adversarial_loss = torch.nn.BCELoss()

def train_GAN(epochs, save_intervals):
    train_loader = get_data_loader(TRAINING_PARAMETERS['batch_size'])
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

if __name__ == "__main__":
    train_GAN(epochs=TRAINING_PARAMETERS['epochs'], save_intervals=TRAINING_PARAMETERS['save_intervals'])