import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.datasets import MNIST
import torchvision.transforms as tv_transforms
import torchvision.datasets as tv_datasets
import torchvision.utils as tv_utils

from torch.utils.tensorboard import SummaryWriter

from models_cgan import Generator, Discriminator, ConditionalGenerator, ConditionalDiscriminator

################################################
# Parse parameters
################################################
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default='cgan', help="name of the model")
parser.add_argument("--data_root", type=str, default='../data', help="data folder")
parser.add_argument("--model_root", type=str, default='./models/', help="model folder")
parser.add_argument("--log_root", type=str, default=f'./logs/', help="log folder")
parser.add_argument("--figure_root", type=str, default=f'./figures/', help="log folder")
parser.add_argument("--seed", type=int, default=0, help="rng seed")
parser.add_argument("--image_size", type=int, default=32, help="image size of the inputs")
parser.add_argument("--batch_size", type=int, default=32, help="batch_size")
parser.add_argument("--gpu_id", type=str, default='0', help="gpu id assigned by cluster")
parser.add_argument("--n_workers", type=int, default=8, help="number of CPU workers for processing input data")
parser.add_argument("--learning_rate", type=float, dest='lr', default=0.0002, help="rng seed")
parser.add_argument("--epochs", type=int, dest='n_epochs', default=20, help="number of epochs")
parser.add_argument("--log_interval", type=int, dest='log_interval', default=100,  help="number of batches to record https://github.com/tt6746690/misc_impl/tree/master/cganhistory")
parser.add_argument("--use_sn", dest='use_sn', default=False, action='store_true', help="use spectral normalization")
parser.add_argument("--no-conditional", dest='no_conditional', default=False, action='store_true', help="not using conditional generator/discriminator")

parser.add_argument("--dim_z", type=int, dest='dim_z', default=32,  help="dimension for latent noise vector of G")
parser.add_argument("--tsboard", dest='tsboard', default=False, action='store_true', help="save logs to tensorboard")



args = parser.parse_args()
locals().update(vars(args))


figure_root = os.path.join(figure_root, model_name)
model_root  = os.path.join(model_root,  model_name)
log_root    = os.path.join(log_root,    model_name)

conditional = (not no_conditional)



dim_z = 50
beta1 = 0.5
num_classes = 10

################################################
# Prepare
################################################

os.makedirs(figure_root, exist_ok=True)
os.makedirs(model_root,  exist_ok=True)
os.makedirs(log_root,    exist_ok=True)

writer = SummaryWriter(log_root)
writer.flush()

torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transforms = tv_transforms.Compose([
    tv_transforms.Resize(image_size),
    tv_transforms.ToTensor(),
    tv_transforms.Normalize((0.5,), (0.5,)),
])

train_loader = torch.utils.data.DataLoader(
    ColorMNIST(root=data_root, download=True, train=True, transform=transforms),
    batch_size=batch_size, shuffle=True, num_workers=n_workers, pin_memory=True)

################################################
# model/loss/optimizer
################################################

conv_channels = [256, 256, 128, 64]
conv_upsample = [True, True, True]

enc_channels = [64, 128, 256, 256]
dec_channels = [256, 128, 64, 64]

conv_channels = [im_channels, 64, 128, 256]
conv_dnsample = [True, True, True]

G = ConditionalGenerator(conv_channels, conv_upsample, num_classes=num_classes, dim_z=dim_z, im_channels=im_channels).to(device)
D = OrdinalConditionalDiscriminator(conv_channels, conv_dnsample, num_classes, use_sn=True).to(device)

criterion_D = nn.BCEWithLogitsLoss()

optimizer_G = torch.optim.Adam(G.parameters(), lr, (beta1, beta2))
optimizer_D = torch.optim.Adam(D.parameters(), lr, (beta1, beta2))

fixed_z = torch.randn(100, dim_z).to(device)
fixed_c = torch.arange(10).repeat(10).to(device)

real_label, fake_label = 0, 1


################################################
# Training Loop
################################################


for epoch in range(n_epochs):
    for it, (x_real, c_digit, c_color) in enumerate(train_loader):

        # batch_size for last batch might be different ...
        batch_size = x_real.size(0)
        real_labels = torch.full((batch_size, 1), real_label, device=device)
        fake_labels = torch.full((batch_size, 1), fake_label, device=device)
        
        
        if target_type == 'digit':
            c_real = c_digit
        elif target_type == 'color':
            c_real = bin_index(c_color, num_classes)
        else:
            raise Exception()
        
        
        ##############################################################
        # Update Discriminator
        ##############################################################

        # a minibatch of samples from data distribution
        x_real, c_real = x_real.to(device), c_real.to(device)
        
        y = D(x_real, c_real)
        loss_D_real = criterion_D(y, real_labels)
        
        # a minibatch of samples from the model distribution
        z = torch.randn(batch_size, dim_z).to(device)
        c_fake = torch.empty(batch_size, dtype=torch.long).random_(0, num_classes).to(device)

        x_fake = G(z, c_fake)
        y = D(x_fake, c_fake)
        loss_D_fake = criterion_D(y, fake_labels)
        
        # backprop
        optimizer_D.zero_grad()
        loss_D = loss_D_real + loss_D_fake
        loss_D.backward()
        optimizer_D.step()
        
        ##############################################################
        # Update Generator/Encoder
        ##############################################################
        
        
        # a minibatch of samples from the model distribution
        z = torch.randn(batch_size, dim_z).to(device)
        c_fake = torch.empty(batch_size, dtype=torch.long).random_(0, num_classes).to(device)
        x_fake = G(z, c_fake)
        y = D(x_fake, c_fake)
        loss_G = criterion_D(y, real_labels)
        
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        ##############################################################
        # print
        ##############################################################

        loss_D = loss_D.item()
        loss_G = loss_G.item()
        loss_total = loss_D + loss_G

        global_step = epoch*len(train_loader)+it
        writer.add_scalar('loss/total', loss_total, global_step)
        writer.add_scalar('loss/D', loss_D, global_step)
        writer.add_scalar('loss/G', loss_G, global_step)

        if it % log_interval == log_interval-1:
            print(f'[{epoch+1}/{n_epochs}]\t'
                  f'[{(it+1)*batch_size}/{len(train_loader.dataset)} ({100.*(it+1)/len(train_loader):.0f}%)]\t'
                  f'loss: {loss_total:.4}\t'
                  f'loss_D: {loss_D:.4}\t'
                  f'loss_G: {loss_G:.4}\t')
            
            x_fake = G(fixed_z, fixed_c).detach()
            tv_utils.save_image(x_fake,
                os.path.join(figure_root,
                    f'{model_name}_fake_samples_epoch={epoch}_it={it}.png'), nrow=10, normalize=True)

            writer.add_image('mnist', tv_utils.make_grid(x_fake, nrow=10, normalize=True), global_step)
        