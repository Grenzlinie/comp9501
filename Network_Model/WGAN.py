import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.opt = opt

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8)) # type: ignore
            layers.append(nn.LeakyReLU(0.2, inplace=True)) # type: ignore
            return layers

        self.model = nn.Sequential(
            *block(opt.n_properties + opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, opt.n_compositions),
            nn.Softmax(dim=1),

        )


    def forward(self, properties, latent_code):
        # Concatenate properties vector and latent code to produce input
        gen_input = torch.cat((properties, latent_code), -1)
        potential_alloy = self.model(gen_input)
        return potential_alloy


class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opt.n_compositions + opt.n_properties, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )


    def forward(self, compositions, properties):
        # Concatenate properties and compositions to produce input
        d_in = torch.cat((compositions, properties), -1)
        validity = self.model(d_in)
        return validity

