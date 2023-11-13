import argparse
import csv
import numpy as np

import torch
import torch.autograd as autograd

from Data_Process_Util.dataloader import create_dataloader
from Network_Model.WGAN import Generator, Discriminator

from draw import plot_losses

# ------------
#  Preparation
# ------------

# set default hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=150, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=120, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=50, help="dimensionality of the latent space")
parser.add_argument("--n_critic", type=int, default=1, help="number of training steps for discriminator per iter")
# parser.add_argument("--sample_interval", type=int, default=1, help="interval between composition samples")
parser.add_argument("--n_properties", type=int, default=1, help="number of properties")
parser.add_argument("--n_compositions", type=int, default=6, help="number of chemical species")
parser.add_argument("--seed", type=int, default=42, help="seed for training")
opt = parser.parse_args()
print(opt)

# Set seed for reproducibility
torch.manual_seed(opt.seed)
np.random.seed(opt.seed)

# Loss weight for gradient penalty
lambda_gp = 10

# Initialize generator and discriminator
generator = Generator(opt)
discriminator = Discriminator(opt)


# Control to use GPU or CPU
cuda = True if torch.cuda.is_available() else False
if cuda:
    generator.cuda()
    discriminator.cuda()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

# Gradient Penalty
def compute_gradient_penalty(D, real_samples, fake_samples, labels):
    """Calculates the gradient penalty loss for WGAN GP.
       Warning: It doesn't compute the gradient w.r.t the labels, only w.r.t
       the interpolated real and fake samples, as in the WGAN GP paper.
    """
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1)))
    labels = LongTensor(labels)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates, labels)
    fake = Tensor(real_samples.shape[0], 1).fill_(1.0)
    fake.requires_grad = False
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )
    # gradients is a tuple and gradients[0].view(xxx) is a tensor with dimension [batch_size, n_compositions]
    gradients = gradients[0].view(gradients[0].size(0), -1)
    # gradients_penalty is a tensor with dimension [1,]
    gradient_penalty = ((gradients.norm(2, dim=1)-1) ** 2).mean()
    return gradient_penalty


# Load data
train_dataloader, valid_dataloader = create_dataloader('./Data_Warehouse/data.xlsx', batch_size = opt.batch_size)


# ----------
#  Training
# ----------

results = []
for epoch in range(opt.n_epochs):
    temp_d_loss = []
    for i, (compositions, properties) in enumerate(train_dataloader):
        batch_size = opt.batch_size

        # Move to GPU if necessary
        real_comps = compositions.type(Tensor)
        labels = properties.type(LongTensor)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise and labels as generator input, dimension is [batch_size, latent_dim]
        latent_code = Tensor(np.random.normal(0, 1, (batch_size, opt.latent_dim)))

        # Generate a batch of images, dimension is [batch_size, n_compositions]
        fake_comps = generator(labels, latent_code)

        # Real images validation, dimesion is [batch_size, 1]
        real_validity = discriminator(real_comps, labels)
        # Fake images validation, dimesion is [batch_size, 1]
        fake_validity = discriminator(fake_comps, labels)

        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(
                            discriminator, real_comps.data, fake_comps.data,
                            labels.data)
        # Adversarial loss
        d_loss = - torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()

        # Train the generator every n_critic steps
        if i % opt.n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            # Generate a batch of compositions
            fake_comps = generator(labels, latent_code)
            # Loss measures generator's ability to fool the discriminator
            # Train on fake samples
            fake_validity = discriminator(fake_comps, labels)
            euclidean_distance = []
            g_loss = - torch.mean(fake_validity)

            g_loss.backward()
            optimizer_G.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f]"
                % (epoch, opt.n_epochs, i, len(train_dataloader), d_loss.item())
            )

            temp_d_loss.append(d_loss.item())

    temp_d_loss = np.mean(temp_d_loss)

    results.append([epoch, temp_d_loss])


# ----------------------
#  Results Visualization
# ----------------------

# store model
torch.save(generator, './Results/generator.pth')
torch.save(discriminator, './Results/discriminator.pth')


# store loss results
with open('./Results/loss_results.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "D Loss"])
    writer.writerows(results)

#draw loss curve
plot_losses()





