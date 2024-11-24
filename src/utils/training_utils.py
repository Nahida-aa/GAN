import os
import torch
from matplotlib import pyplot as plt
from src.config import IMAGES_DIR, WEIGHTS_DIR

def should_save(epoch, save_intervals):
    for interval in save_intervals:
        if interval['start'] <= epoch < interval['end'] and epoch % interval['interval'] == 0:
            return True
    return False

def save_images(epoch, generator, device, real_imgs, real_labels):
    # 生成随机噪声和生成图像
    z = torch.randn(50, 100, device=device)
    labels = torch.cat([torch.full((5,), i, device=device) for i in range(10)])  # 生成0-9的标签，每个标签5个
    gen_imgs = generator(z, labels)
    gen_imgs = gen_imgs.view(gen_imgs.size(0), 1, 28, 28)
    gen_imgs = gen_imgs.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()

    # 选择前50个真实图像和标签
    real_imgs = real_imgs[:50].cpu().detach().numpy()
    real_labels = real_labels[:50].cpu().detach().numpy()

    fig, axs = plt.subplots(10, 10, figsize=(20, 20))
    cnt = 0
    for i in range(10):
        for j in range(5):
            axs[i, j].imshow(real_imgs[cnt, 0, :, :], cmap='gray')
            axs[i, j].set_title(f"Real: {real_labels[cnt]}")
            axs[i, j].axis('off')
            axs[i, j + 5].imshow(gen_imgs[cnt, 0, :, :], cmap='gray')
            axs[i, j + 5].set_title(f"Gen: {labels[cnt]}")
            axs[i, j + 5].axis('off')
            cnt += 1
    fig.savefig(os.path.join(IMAGES_DIR, f"mnist_comparison_{epoch}.png"))
    plt.close()

def save_model(epoch, generator, discriminator):
    torch.save(generator.state_dict(), os.path.join(WEIGHTS_DIR, f"generator_{epoch}.pth"))
    torch.save(discriminator.state_dict(), os.path.join(WEIGHTS_DIR, f"discriminator_{epoch}.pth"))