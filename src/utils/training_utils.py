import os
import torch
from matplotlib import pyplot as plt
from src.config import IMAGES_DIR, WEIGHTS_DIR

def should_save(epoch, save_intervals):
    for interval in save_intervals:
        if interval['start'] <= epoch < interval['end'] and epoch % interval['interval'] == 0:
            return True
    return False

def save_images(epoch, generator, device):
    z = torch.randn(25, 100, device=device)
    labels = torch.randint(0, 10, (25,), device=device)
    gen_imgs = generator(z, labels)
    gen_imgs = gen_imgs.view(gen_imgs.size(0), 1, 28, 28)
    gen_imgs = gen_imgs.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()

    fig, axs = plt.subplots(5, 5, figsize=(10, 10))
    cnt = 0
    for i in range(5):
        for j in range(5):
            axs[i, j].imshow(gen_imgs[cnt, 0, :, :], cmap='gray')
            axs[i, j].set_title(f"Digit: {labels[cnt]}")
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig(os.path.join(IMAGES_DIR, f"mnist_{epoch}.png"))
    plt.close()

def save_model(epoch, generator, discriminator):
    torch.save(generator.state_dict(), os.path.join(WEIGHTS_DIR, f"generator_{epoch}.pth"))
    torch.save(discriminator.state_dict(), os.path.join(WEIGHTS_DIR, f"discriminator_{epoch}.pth"))