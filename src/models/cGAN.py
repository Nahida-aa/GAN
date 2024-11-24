import torch.nn as nn
import torch

class Generator(nn.Module):
    def __init__(self, noise_dim, label_dim, img_channels, img_width, img_height):
        super(Generator, self).__init__()
        self.img_channels = img_channels
        self.img_width = img_width
        self.img_height = img_height
        self.label_emb = nn.Embedding(label_dim, label_dim)
        self.model = nn.Sequential(
            nn.Linear(noise_dim + label_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, img_channels * img_width * img_height),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_embedding = self.label_emb(labels)
        gen_input = torch.cat((noise, label_embedding), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), self.img_channels, self.img_width, self.img_height)
        return img

class Discriminator(nn.Module):
    def __init__(self, img_channels, img_width, img_height, label_dim):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(label_dim, label_dim)
        self.model = nn.Sequential(
            nn.Linear(img_channels * img_width * img_height + label_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        label_embedding = self.label_emb(labels)
        d_in = torch.cat((img.view(img.size(0), -1), label_embedding), -1)
        validity = self.model(d_in)
        return validity