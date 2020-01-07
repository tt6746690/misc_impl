# reference: https://github.com/pytorch/examples/blob/master/dcgan/main.py
#
import os

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision 
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter

class Generator(nn.Module):

    def __init__(self, nz, nf, nc):
        """
            nz      dimension of noise 
            nf      dimension of features in last conv layer
            nc      number of channels in the image

            In DCGAN paper for LSUN dataset, nz=100, nf=128, nc=3
        """
        super(Generator, self).__init__()

        def block(in_channels, out_channels, stride=2, padding=1, batch_norm=True, nonlinearity=nn.ReLU(True)):
            """ stride=1, padding=0: H_out = H_in + 3       # 1 -> 4
                stride=2, padding=1: H_out = 2 * H_in       # doubles
            """
            return [
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=padding, bias=False),
                *( [nn.BatchNorm2d(out_channels)] if batch_norm else [] ),
                nonlinearity,
            ]

        self.model = nn.Sequential(
            # (nz)   x 1 x 1
            *block(nz,   8*nf, stride=1, padding=0),
            # (8*nf) x 4 x 4
            *block(8*nf, 4*nf),
            # (4*nf) x 8 x 8
            *block(4*nf, 2*nf),
            # (2*nf) x 16 x 16
            *block(2*nf,   nf),
            # (nf) x 32 x 32
            *block(nf,     nc, batch_norm=False, nonlinearity=nn.Tanh()),
            # (nc) x 64 x 64
        )

    def forward(self, z):
        """
            z       (N, nz, 1, 1)
                noise vector
            Returns (N, nc, h, w)
                image generated from model distribution
                
        """
        return self.model(z)
    
    
class Discriminator(nn.Module):
    
    def __init__(self, nc, nf):
        """
            nc      number of channels in the image
            nf      dimension of features of first conv layer

            In DCGAN paper for LSUN dataset, nc=3
        """
        super(Discriminator, self).__init__()
        
        def block(in_channels, out_channels,
                  stride=2, padding=1,
                  batch_norm=True,
                  nonlinearity=nn.LeakyReLU(0.2, inplace=True)):
            """ stride=1, padding=0: H_out = H_in - 3              # 4 -> 1
                stride=2, padding=1: H_out = floor((H_in-1)/2 +1)  # roughly halves
            """
            return [
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=padding, bias=False),
                *( [nn.BatchNorm2d(out_channels)] if batch_norm else [] ),
                nonlinearity,
            ]
        
        self.model = nn.Sequential(
            # (nc) x 64 x 64
            *block(nc,     nf, batch_norm=False),
            # (nf) x 32 x 32
            *block(nf,   2*nf),
            # (2*nf) x 16 x 16
            *block(2*nf, 4*nf),
            # (4*nf) x 8 x 8
            *block(4*nf, 8*nf),
            # (8*nf) x 4 x 4
            *block(8*nf, 1, stride=1, padding=0, batch_norm=False, nonlinearity=nn.Sigmoid()),
            # 1 x 1 x 1
        )
        
        
    def forward(self, x):
        """
            x        (N, nc, h, w)
            Returns  (N,)
                classification probability that x comes from data distribution
        """
        x = self.model(x)
        return  x.view(-1, 1).squeeze(1)
        

def weights_initialization(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)



def train(
    nz,
    nfg,
    nfd,
    nc,
    data_root,
    figure_root,
    model_root,
    log_root,
    model_name,
    load_weights_generator,
    load_weights_discriminator,
    image_size,
    batch_size,
    lr,
    beta1,
    n_epochs,
    n_batches_print,
    seed,
    n_workers,
    gpu_id,
):

    os.makedirs(data_root, exist_ok=True)
    os.makedirs(model_root, exist_ok=True)
    os.makedirs(figure_root, exist_ok=True)
    os.makedirs(log_root, exist_ok=True)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    trainset = datasets.MNIST(root=data_root, download=True,
                transform=transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ]))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=n_workers)

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G = Generator(nz, nfg, nc).to(device)
    G.apply(weights_initialization)
    if load_weights_generator != '':
        G.load_state_dict(torch.load(load_weights_generator))
        
    D = Discriminator(nc, nfd).to(device)
    D.apply(weights_initialization)
    if load_weights_discriminator != '':
        D.load_state_dict(torch.load(load_weights_discriminator))

    criterion = nn.BCELoss()

    fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)

    # label flipping helps with training G!
    real_label = 0
    fake_label = 1

    optimizerD = torch.optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = torch.optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))

    writer = SummaryWriter(log_root)
    writer.flush()

    for epoch in range(n_epochs):
        
        for it, (x_real, _) in enumerate(trainloader):
            
            # batch_size for last batch might be different ...
            batch_size = x_real.size(0)
            real_labels = torch.full((batch_size,), real_label, device=device)
            fake_labels = torch.full((batch_size,), fake_label, device=device)
            
            ##############################################################
            # Update Discriminator: Maximize E[log(D(x))] + E[log(1 - D(G(z)))]
            ##############################################################
            
            D.zero_grad()
            
            # a minibatch of samples from data distribution
            x_real = x_real.to(device)
            
            y = D(x_real)
            loss_D_real = criterion(y, real_labels)
            loss_D_real.backward()
            
            D_x = y.mean().item()
            
            # a minibatch of samples from the model distribution
            z = torch.randn(batch_size, nz, 1, 1, device=device)
            
            x_fake = G(z)
            # https://github.com/pytorch/examples/issues/116
            # If we do not detach, then, although x_fake is not needed for gradient update of D,
            #   as a consequence of backward pass which clears all the variables in the graph
            #   graph for G will not be available for gradient update of G
            # Also for performance considerations, detaching x_fake will prevent computing 
            #   gradients for parameters in G
            y = D(x_fake.detach())
            loss_D_fake = criterion(y, fake_labels)
            loss_D_fake.backward()
            
            D_G_z1 = y.mean().item()
            loss_D = loss_D_real + loss_D_fake
            
            optimizerD.step()
            
            ##############################################################
            # Update Generator: Minimize E[log(1 - D(G(z)))] => Maximize E[log(D(G(z))))]
            ##############################################################
            
            G.zero_grad()
            
            y = D(x_fake)
            loss_G = criterion(y, real_labels)
            loss_G.backward()
            
            D_G_z2 = y.mean().item()
            
            optimizerG.step()
            
            ##############################################################
            # write/print
            ##############################################################

            if it % n_batches_print == n_batches_print-1:

                # best loss: -log 4 = -1.38
                # D_x: 1 D_G_z1: 0 D_G_z2: 1
                print(f"[{epoch+1}/{n_epochs}][{it+1}/{len(trainloader)}] loss: {loss_D.item()+loss_G.item():.4} loss_D: {loss_D.item():.4}  loss_G: {loss_G.item():.4} D_x: {D_x:.4} D(G(z1)): {D_G_z1:.4} D(G(z2)): {D_G_z2:.4}" ) 

                x_fake = G(fixed_noise)
                vutils.save_image(x_fake.detach(), os.path.join(figure_root, f'{model_name}_fake_samples_epoch={epoch}_it={it}.png'))


            if it == 0:
                global_step = epoch*len(trainloader)+it
                writer.add_scalar('discriminator/D(x)', D_x, global_step)
                writer.add_scalar('discriminator/D(G(z1))', D_G_z1, global_step)
                writer.add_scalar('discriminator/D(G(z2))', D_G_z2, global_step)
                writer.add_scalar('loss/total', loss_D.item()+loss_G.item(), global_step)
                writer.add_scalar('loss/D', loss_D.item(), global_step)
                writer.add_scalar('loss/G', loss_G.item(), global_step)
                writer.add_scalar('gradient/G_conv_W_first', G.model[0].weight.grad.mean().detach().cpu().item(), global_step)
                writer.add_scalar('gradient/G_conv_W_last', G.model[-2].weight.grad.mean().detach().cpu().item(), global_step)
                writer.add_scalar('gradient/D_conv_W_first', D.model[0].weight.grad.mean().detach().cpu().item(), global_step)
                writer.add_scalar('gradient/D_conv_W_last', D.model[-2].weight.grad.mean().detach().cpu().item(), global_step)
                writer.add_image('mnist', torchvision.utils.make_grid(x_fake), global_step)

        torch.save(G.state_dict(), os.path.join(model_root, f'G_epoch_{epoch}.pt'))
        torch.save(D.state_dict(), os.path.join(model_root, f'D_epoch_{epoch}.pt'))



if __name__ == '__main__':

    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str,
                        default='../data',
                        help="data folder")
    parser.add_argument("--model_root", type=str,
                        default='./models',
                        help="model folder")
    parser.add_argument("--figure_root", type=str,
                        default='./figures',
                        help="figure folder")
    parser.add_argument("--log_root", type=str,
                        default=f'./logs/',
                        help="log folder")
    parser.add_argument("--model_name", type=str,
                        default='dcgan',
                        help="name of the model")
    parser.add_argument("--load_weights_generator", type=str,
                        default='',
                        help="optional .pt model file to initialize generator with")
    parser.add_argument("--load_weights_discriminator", type=str,
                        default='',
                        help="optional .pt model file to initialize discriminator with")
    parser.add_argument("--epochs", type=int, dest='n_epochs',
                        default=50,
                        help="number of epochs")
    parser.add_argument("--print_every", type=int, dest='n_batches_print',
                        default=100,
                        help="number of batches to print loss / plot figures")
    parser.add_argument("--batch_size", type=int,
                        default=64,
                        help="batch_size")
    parser.add_argument("--seed", type=int,
                        default=1,
                        help="rng seed")
    parser.add_argument("--learning_rate", type=float, dest='lr',
                        default=0.0002,
                        help="rng seed")
    parser.add_argument("--n_workers", type=int,
                        default=8,
                        help="number of CPU workers for processing input data")
    parser.add_argument("--gpu_id", type=str,
                        default='0',
                        help="gpu id assigned by cluster")
    parser.add_argument("--beta1", type=float,
                        default=0.5,
                        help="ADAM parameter")
    parser.add_argument("--image_size", type=int,
                        default=64,
                        help="image size of the inputs")
    parser.add_argument("--nfg", type=int,
                        default=64,
                        help=" dimension of features in last conv layer of generator")
    parser.add_argument("--nfd", type=int,
                        default=64,
                        help=" dimension of features in first conv layer of discriminator")
    parser.add_argument("--nz", type=int,
                        default=100,
                        help=" dimension of latent space")
    parser.add_argument("--nc", type=int,
                        default=1,
                        help=" number of channels of input image")

    args = parser.parse_args()
    args.model_name = f'{args.model_name}_{datetime.now().strftime("%Y.%m.%d-%H:%M:%S")}'

    args.model_root = os.path.join(args.model_root, args.model_name)
    args.figure_root = os.path.join(args.figure_root, args.model_name)
    args.log_root = os.path.join(args.log_root, args.model_name)

    train(**vars(args))