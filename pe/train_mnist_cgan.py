import os
import sys

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST

import torchvision 
import torchvision.transforms as tv_transforms
import torchvision.datasets as tv_datasets
import torchvision.utils as tv_utils

from torch.utils.tensorboard import SummaryWriter

from fid import calculate_activation_statistics
from models import MnistCNN, Discriminator, Generator
from inception import InceptionV3
from datasets import ColorMNIST
from plot_tools import plot_im
from utils import makedirs_exists_ok, seed_rng, set_cuda_visible_devices, load_weights_from_file



def train(
    target_type,
    model_name,
    data_root,
    model_root,
    figure_root,
    log_root,
    seed,
    image_size,
    batch_size,
    gpu_id,
    n_workers,
    load_weights,
    lr,
    beta1,
    beta2,
    n_epochs,
    log_interval,
    n_features,
    G_dim_z,
    G_bottom_width,
    n_classes,
    im_channels,
    dataset_func,
    model_activation,
    loss_type):

    makedirs_exists_ok(data_root)
    makedirs_exists_ok(model_root)
    makedirs_exists_ok(figure_root)
    makedirs_exists_ok(log_root)

    writer = SummaryWriter(log_root)
    writer.flush()

    seed_rng(seed)
    device = set_cuda_visible_devices(gpu_id)

    transforms = tv_transforms.Compose([
        tv_transforms.Resize(image_size),
        tv_transforms.ToTensor(),
        tv_transforms.Normalize((0.5,), (0.5,)),
    ])

    train_loader = torch.utils.data.DataLoader(
        dataset_func(
            root=data_root, download=True, train=True, transform=transforms),
        batch_size=batch_size, shuffle=True, num_workers=n_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        dataset_func(
            root=data_root, download=True, train=False, transform=transforms),
        batch_size=batch_size, shuffle=True, num_workers=n_workers, pin_memory=True)

    G = Generator(n_features, G_dim_z, G_bottom_width,
        num_classes=n_classes, im_channels=im_channels, activation=model_activation).to(device)
    D = Discriminator(n_features,
        num_classes=n_classes, im_channels=im_channels, activation=model_activation).to(device)
        

    optimizer_G = torch.optim.Adam(G.parameters(), lr, (beta1, beta2))
    optimizer_D = torch.optim.Adam(D.parameters(), lr, (beta1, beta2))

    if loss_type == 'hinge':
        # hinge loss
        criterion_G = lambda D_xf, D_xr: -torch.mean(D_xf)
        criterion_D = lambda D_xf, D_xr: \
            torch.mean(torch.relu(1. - D_xf)) + \
            torch.mean(torch.relu(1. + D_xr))
    elif loss_type == 'dcgan':
        criterion_G = lambda D_xf, D_xr: torch.mean(F.softplus(-D_xf))
        criterion_D = lambda D_xf, D_xr: torch.mean(F.softplus(-D_xr)) + torch.mean(F.softplus(D_xf))

    loss_bce = nn.BCELoss()

    def sample_from_G(G, batch_size, dim_z, device, n_classes):
        # noise
        z = torch.empty(batch_size, dim_z, dtype=torch.float32, device=device).normal_()
        # conditioned variable
        c = torch.from_numpy(np.random.randint(low=0, high=n_classes, size=(batch_size,)))
        c = y.type(torch.long).to(device)

        x_fake = G(z, c)
        return x_fake, c

    fixed_z = torch.empty(100, G_dim_z, dtype=torch.float32, device=device).normal_()
    fixed_y = torch.arange(10).repeat(10).type(torch.long).to(device)

    real_label, fake_label = 0, 1
        

    for epoch in range(n_epochs):
        for it, (x, y_digit, y_color) in enumerate(train_loader):
        
            # batch_size for last batch might be different ...
            batch_size = x.size(0)
            real_labels = torch.full((batch_size,), real_label, device=device)
            fake_labels = torch.full((batch_size,), fake_label, device=device)
            
            if loss_type == 'original':
                criterion_G = lambda D_xf, D_xr: loss_bce(F.sigmoid(D_xf), real_labels)
                criterion_D = lambda D_xf, D_xr: loss_bce(F.sigmoid(D_xf), fake_labels) + loss_bce(F.sigmoid(D_xr), real_labels)

            if target_type == 'color':
                x, y = x.to(device), (y_color < 0.5).long().to(device)
            elif target_type == 'digit':
                x, y = x.to(device), y_digit.long().to(device)
            else:
                raise Exception()

            
            # Generator
            
            x_fake, c = sample_from_G(G, batch_size, G_dim_z, device, n_classes)
            D_xf = D(x_fake, c)
            loss_G = criterion_G(D_xf, None)
            
            G.zero_grad()
            loss_G.backward()
            optimizer_G.step()


            # Discriminator
            
            x_fake, c = sample_from_G(G, batch_size, G_dim_z, device, n_classes)
            D_xf = D(x_fake, c)
            D_xr = D(x, y)
            loss_D = criterion_D(D_xf, D_xr)
            
            D.zero_grad()
            loss_D.backward()
            optimizer_D.step()
            

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
                
                x_fake = G(fixed_z, fixed_y).detach()
                tv_utils.save_image(x_fake,
                    os.path.join(figure_root,
                        f'{model_name}_fake_samples_epoch={epoch}_it={it}.png'), nrow=10)

                writer.add_image('mnist', tv_utils.make_grid(x_fake), global_step)
            

        torch.save(G.state_dict(), os.path.join(model_root, f'G_epoch_{epoch}.pt'))
        torch.save(D.state_dict(), os.path.join(model_root, f'D_epoch_{epoch}.pt'))


if __name__ == '__main__':

    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default='cgan', help="name of the model")
    parser.add_argument("--data_root", type=str, default='../data', help="data folder")
    parser.add_argument("--model_root", type=str, default='./models/mnist_cgan', help="model folder")
    parser.add_argument("--figure_root", type=str, default='./figures/mnist_cgan', help="figures folder")
    parser.add_argument("--log_root", type=str, default=f'./logs/mnist_cgan', help="log folder")
    parser.add_argument("--load_weights", type=str, default='', help="optional .pt model file to initialize generator with")
    parser.add_argument("--seed", type=int, default=0, help="rng seed")
    parser.add_argument("--image_size", type=int, default=32, help="image size of the inputs")
    parser.add_argument("--batch_size", type=int, default=32, help="batch_size")
    parser.add_argument("--gpu_id", type=str, default='0', help="gpu id assigned by cluster")
    parser.add_argument("--n_workers", type=int, default=4, help="number of CPU workers for processing input data")
    parser.add_argument("--learning_rate", type=float, dest='lr', default=0.0002, help="rng seed")
    parser.add_argument("--epochs", type=int, dest='n_epochs', default=20, help="number of epochs")
    parser.add_argument("--log_interval", type=int, dest='log_interval', default=100,  help="number of batches to print loss / plot figures")

    parser.add_argument("--beta1", type=float, dest='beta1', default=0, help="ADAM beta1")
    parser.add_argument("--beta2", type=float, dest='beta2', default=0.9, help="ADAM beta2")
    parser.add_argument("--n_features", type=int, dest='n_features', default=32,  help="# features of last conv in G")
    parser.add_argument("--G_dim_z", type=int, dest='G_dim_z', default=32,  help="dimension for latent noise vector of G")
    parser.add_argument("--G_bottom_width", type=int, dest='G_bottom_width', default=4,  help="dimension of feature map before 1st conv in G")
    parser.add_argument("--target_type", type=str, dest='target_type', default='digit', help="in {'digit', 'color'}")
    parser.add_argument("--dataset_func", type=str, dest='dataset_func', default='colorMNIST', help="in {'MNIST', 'colorMNIST'}")
    parser.add_argument("--loss_type", type=str, dest='loss_type', default='original', help="in {'hinge', 'dcgan', 'original'}")


    args = parser.parse_args()

    if args.target_type not in ['digit', 'color']:
        print('target type should be either "digit" or "color"')

    d = vars(args)

    d['model_activation'] = nn.LeakyReLU(0.2)

    if args.dataset_func == 'MNIST':
        d['im_channels'] = 1
        d['dataset_func'] = MNIST
        d['n_classes'] = 10
        d['target_type'] = 'digit'
    elif args.dataset_func == 'colorMNIST':
        d['im_channels'] = 3
        d['dataset_func'] = ColorMNIST
        if args.target_type == 'digit':
            d['n_classes'] = 10
        elif args.target_type == 'color':
            d['n_classes'] = 2
        else:
            raise Exception()


    args.model_name  = f'{args.model_name}_{args.target_type}'
    args.model_root  = os.path.join(args.model_root, args.model_name)
    args.log_root    = os.path.join(args.log_root,   args.model_name)
    args.figure_root = os.path.join(args.figure_root,args.model_name)

    print(d)

    train(**d)