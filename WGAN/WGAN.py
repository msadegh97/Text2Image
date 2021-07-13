import torch
from torch import  nn


# code is based on Pytorch DCGAN Implementation:
class Generator(nn.Module):
    def __init__(self, z_dim = 100, num_channels =3, embed_dim=1024, img_size= 64, proj_ebmed_dim = 256):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.embed_dim = embed_dim
        self.image_size = img_size
        self.num_channels = num_channels
        self.projected_embed_dim = proj_ebmed_dim
        self.latent_dim = self.z_dim + self.projected_embed_dim
        self.ngf = 64

        self._proj_embed_net = nn.Sequential(
            nn.Linear(self.embed_dim, self.projected_embed_dim),
            nn.BatchNorm1d(self.projected_embed_dim),
            nn.LeakyReLU(negative_slope= 0.2, inplace=True)
        )

        self._gen = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.latent_dim, self.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(self.ngf, self.num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, z, embed_vec):

        proj_embed = self._proj_embed_net(embed_vec).unsqueeze(2).unsqueeze(3)
        latent_vector = torch.cat([proj_embed, z], 1)
        return self._gen(latent_vector)


class Critic(nn.Module):
    def __init__(self, num_channel= 3, img_size= 64, proj_embed_dim = 256, embed_dim= 1024):
        super(Critic, self).__init__()
        self.ndf = 64
        self.num_channel = num_channel
        self.image_size = img_size
        self.embed_dim = embed_dim
        self.projected_embed_dim = proj_embed_dim

        self.projection = nn.Sequential(
            nn.Linear(in_features=self.embed_dim, out_features=self.projected_embed_dim),
            nn.BatchNorm1d(num_features=self.projected_embed_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.critic = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.num_channel, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.GroupNorm(1,self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.GroupNorm(1, self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.GroupNorm(1, self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
        )
        self.emb_net = nn.Sequential(
            nn.Conv2d(self.ndf * 8 + self.projected_embed_dim, 1, 4, 1, 0, bias=False),
        )

    def forward(self, input, embed_vec):
        output = self.critic(input)
        projected_embed = self.projection(embed_vec)
        replicated_embed = projected_embed.repeat(4, 4, 1, 1).permute(2, 3, 0, 1)
        hidden_concat = torch.cat([output, replicated_embed], 1)
        output = self.emb_net(hidden_concat)
        output = output.mean(0)

        return output.view(1)
