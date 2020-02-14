import os

from itertools import chain

import torch
import torch.nn as nn
import torch.distributions as dist
from torch.utils.tensorboard import SummaryWriter

import torchvision 
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils

from distributions import logpdf_gaussian


class Generator(nn.Module):

    def __init__(self, nz, nf, nc, nc_gaussian):
        """
            nz      dimension of noise and latent codes
            nf      dimension of features in last conv layer
            nc      number of channels in the image
        """
        super(Generator, self).__init__()

        def block(in_channels, out_channels,
                stride=2,
                padding=1,
                batch_norm=True,
                nonlinearity=nn.ReLU(True)):
            return [
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=padding, bias=False),
                *( [nn.BatchNorm2d(out_channels)] if batch_norm else [] ),
                nonlinearity]

        self.conv_blocks = nn.Sequential(
            *block(nz+10+nc_gaussian,   4*nf, stride=1, padding=0),
            *block(4*nf, 2*nf),
            *block(2*nf,   nf),
            *block(nf,     nc, batch_norm=False, nonlinearity=nn.Tanh()),
        )

    def forward(self, z, c):
        """
            z       (N, nz, 1, 1) or (N, nz)
                incompressible noise vector
            c       (N, 10+nc_gaussian, 1)
                latent code vector
            Returns (N, nc, 32, 32)
                image generated from model distribution
        """
        if len(z.shape) != 4:
            z = z.view(z.shape[0], z.shape[1], 1, 1)
        if len(c.shape) != 4:
            c = c.view(c.shape[0], c.shape[1], 1, 1)
    
        return self.conv_blocks(torch.cat((z, c), 1))
    

class Discriminator(nn.Module):
    
    def __init__(self, nc, nf, nc_gaussian):
        super(Discriminator, self).__init__()
        
        def block(in_channels, out_channels,
                  stride=2,
                  padding=1,
                  batch_norm=True,
                  nonlinearity=nn.LeakyReLU(0.2, inplace=True)):
            return [
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=padding, bias=False),
                *( [nn.BatchNorm2d(out_channels)] if batch_norm else [] ),
                nonlinearity]
        
        self.conv_blocks = nn.Sequential(
            *block(nc,     nf, batch_norm=False),
            *block(nf,   2*nf),
            *block(2*nf, 4*nf),
        )
        
        n_in_units = 4*nf*16
        
        # p(y|x)
        self.fc_y = nn.Sequential(
            nn.Linear(n_in_units, 1),
            nn.Sigmoid())

        # q(c|x) 
        self.fc_categorical = nn.Sequential(
            nn.Linear(n_in_units, 10),
            nn.LogSoftmax(dim=1))
        self.fc_mu = nn.Sequential(
            nn.Linear(n_in_units, nc_gaussian))
        self.fc_logvariance = nn.Sequential(
            nn.Linear(n_in_units, nc_gaussian))

        
    def forward(self, x):
        """
            x        (N, nc, h, w)
            Returns
                y    (N,)
                    classification probability that x comes from data distribution
                logp (N, 10)
                    log probability for 1D categorical latent code
                mu, variance  (N, 2)
                    parameters for 2D Gaussian latent codes
        """
        h = self.conv_blocks(x)
        h = h.view(h.shape[0], -1)
        
        y = self.fc_y(h).squeeze()
        
        logp = self.fc_categorical(h)
        mu = self.fc_mu(h)
        logvariance = self.fc_logvariance(h)

        return  y, logp, mu, logvariance
    
def weights_initialization(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def filter_module_param(module, substr='', include_params=True):
    if include_params:
        return list([(name, parameter) for name, parameter in module.named_parameters() if substr in name])
    else:
        return list([name for name, parameter in module.named_parameters() if substr in name])


def train(
    nz,
    nfg,
    nfd,
    nc,
    nc_gaussian,
    data_root,
    figure_root,
    model_root,
    log_root,
    model_name,
    load_weights_generator,
    load_weights_discriminator,
    image_size,
    batch_size,
    weight_param,
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
    trainloader = torch.utils.data.DataLoader(trainset,
            batch_size=batch_size, shuffle=True, num_workers=n_workers)

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G = Generator(nz, nfg, nc, nc_gaussian).to(device)
    G.apply(weights_initialization)
    if load_weights_generator != '':
        G.load_state_dict(torch.load(load_weights_generator))

    D = Discriminator(nc, nfd, nc_gaussian).to(device)
    D.apply(weights_initialization)
    if load_weights_discriminator != '':
        D.load_state_dict(torch.load(load_weights_discriminator))
        
    D_loss = nn.BCELoss()
    Q_cat_loss = nn.NLLLoss()

    # flip labels ...
    real_label = 0
    fake_label = 1

    optimizerD = torch.optim.Adam(chain(
        D.conv_blocks.parameters(),
        D.fc_y.parameters()),lr=lr, betas=(beta1, 0.999))
    optimizerG = torch.optim.Adam(
        G.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerQ = torch.optim.Adam(chain(
        D.conv_blocks.parameters(),
        D.fc_categorical.parameters(),
        D.fc_mu.parameters(),
        D.fc_logvariance.parameters()), lr=lr, betas=(beta1, 0.999))

    writer = SummaryWriter(log_root)
    writer.flush()

    pdf_c_cat = dist.Categorical(torch.ones(10)/10)
    pdf_c_gaussian = dist.MultivariateNormal(torch.zeros(2), torch.eye(2))

    def sample_latents(batch_size):
        # sample noise
        z = torch.randn(batch_size, nz, device=device)
        # sample latent codes
        c_cat = pdf_c_cat.sample(sample_shape=(batch_size,)).to(device)
        c_cat_onehot = nn.functional.one_hot(c_cat, 10)
        c_gaussian = pdf_c_gaussian.sample(sample_shape=(batch_size,)).to(device)
        c = torch.cat([c_cat_onehot.float(), c_gaussian], dim=-1)

        return z, c, c_cat, c_gaussian

    fixed_z, fixed_c, _, _ = sample_latents(batch_size)


    for epoch in range(n_epochs):

        for it, (x_real, _) in enumerate(trainloader):

            # batch_size for last batch might be different ...
            batch_size = x_real.size(0)
            real_labels = torch.full((batch_size,), real_label, device=device)
            fake_labels = torch.full((batch_size,), fake_label, device=device)

            ##############################################################
            # Update D: Minimize E[-log(D(x))] + E[-log(1 - D(G(c,z)))] (saturating)
            #                                    # D(x) -> p(1|x), probability of x being real
            #           Minimize E[-log(D(x))] + E[-log(D(G(c,z)))]     (non-saturating)
            #                                    # D(x) -> p(0|x), probability of x being fake
            ##############################################################

            optimizerD.zero_grad()

            # a minibatch of samples from data distribution
            x_real = x_real.to(device)

            y,_,_,_ = D(x_real)
            loss_D_real = D_loss(y, real_labels)
            loss_D_real.backward()

            D_x = y.mean().item()

            z, c, c_cat, c_gaussian = sample_latents(batch_size)
            x_fake = G(z, c)
            
            # https://github.com/pytorch/examples/issues/116
            # If we do not detach, then, although x_fake is not needed for gradient update of D,
            #   as a consequence of backward pass which clears all the variables in the graph
            #   Generator's graph will not be available for gradient update of G
            # Also for performance considerations, detaching x_fake will prevent computing 
            #   gradients for parameters in G
            y,_,_,_ = D(x_fake.detach())
            loss_D_fake = D_loss(y, fake_labels)
            loss_D_fake.backward()

            D_G_z1 = y.mean().item()
            loss_D = loss_D_real + loss_D_fake
            optimizerD.step()

            ##############################################################
            # Update G: Minimize E[-log(D(G(c,z))) - \lambda log Q(G(c,z))]
            #                    # D(x) -> p(0|x), probability of x being fake
            # Update Q: Minimize E[-\lambda log Q(x')]
            ##############################################################

            optimizerG.zero_grad()

            y, logp, mu, logvariance = D(x_fake)
            loss_bce = D_loss(y, real_labels)
            
            optimizerQ.zero_grad()
            
            loss_c_cat_nll = Q_cat_loss(logp, c_cat)
            loss_c_gaussian_nll = -logpdf_gaussian(c_gaussian, mu, logvariance)
            loss_Q = loss_c_cat_nll + loss_c_gaussian_nll

            loss_G = loss_bce + weight_param*loss_Q
            loss_G.backward()
            
            optimizerG.step()
            optimizerQ.step()


            ##############################################################
            # write/print
            ##############################################################
            

            loss_D = loss_D.item()
            loss_G = loss_G.item()
            loss_Q = loss_Q.item()

            loss_total = loss_D_real + loss_bce + weight_param*loss_Q

            if it % n_batches_print == n_batches_print-1:
                print(f'[{epoch+1}/{n_epochs}][{it+1}/{len(trainloader)}]'
                    f'loss: {loss_total:.4}\t'
                    f'loss_D: {loss_D:.4}\t'
                    f'loss_G: {loss_G:.4}\t'
                    f'loss_Q: {loss_Q:.4}')
                x_fake = G(fixed_z, fixed_c)
                vutils.save_image(x_fake.detach(),
                    os.path.join(figure_root,
                        f'{model_name}_fake_samples_epoch={epoch}_it={it}.png'))

            # if it == 0:
            global_step = epoch*len(trainloader)+it
            writer.add_scalar('loss/total', loss_total, global_step)
            writer.add_scalar('loss/D', loss_D, global_step)
            writer.add_scalar('loss/G', loss_G, global_step)
            writer.add_scalar('loss/Q', loss_Q, global_step)
            writer.add_image('mnist', torchvision.utils.make_grid(G(fixed_z, fixed_c).detach()), global_step)

            for module, substr, scalar_name in [
                    (D, 'conv_blocks.0', 'grad/D_conv_weights_first'),
                    (D, 'fc_y', 'grad/D_fc_y'),
                    (G, 'conv_blocks.0', 'grad/G_conv_weights_first'),
                    (G, 'conv_blocks.9', 'grad/G_conv_weights_last'),
                ]:
                _, parameters = filter_module_param(module, substr)[0]
                scalar = torch.mean(parameters).cpu().item()
                writer.add_scalar(scalar_name, scalar, global_step)
        
            
        torch.save(G.state_dict(), os.path.join(model_root, f'G_epoch_{epoch}.pt'))
        torch.save(D.state_dict(), os.path.join(model_root, f'D_epoch_{epoch}.pt'))




if __name__ == '__main__':

    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default='../data', help="data folder")
    parser.add_argument("--model_root", type=str, default='./models', help="model folder")
    parser.add_argument("--figure_root", type=str, default='./figures', help="figure folder")
    parser.add_argument("--log_root", type=str, default=f'./logs/', help="log folder")
    parser.add_argument("--model_name", type=str, default='dcgan', help="name of the model")
    parser.add_argument("--load_weights_generator", type=str, default='', help="optional .pt model file to initialize generator with")
    parser.add_argument("--load_weights_discriminator", type=str, default='', help="optional .pt model file to initialize discriminator with")
    parser.add_argument("--epochs", type=int, dest='n_epochs', default=50, help="number of epochs")
    parser.add_argument("--print_every", type=int, dest='n_batches_print', default=100,  help="number of batches to print loss / plot figures")
    parser.add_argument("--batch_size", type=int, default=64, help="batch_size")
    parser.add_argument("--seed", type=int, default=1, help="rng seed")
    parser.add_argument("--learning_rate", type=float, dest='lr', default=0.0002, help="rng seed")
    parser.add_argument("--n_workers", type=int, default=8, help="number of CPU workers for processing input data")
    parser.add_argument("--gpu_id", type=str, default='0', help="gpu id assigned by cluster")
    parser.add_argument("--beta1", type=float, default=0.5, help="ADAM parameter")
    parser.add_argument("--image_size", type=int, default=32, help="image size of the inputs")
    parser.add_argument("--nfg", type=int, default=64, help=" dimension of features in last conv layer of generator")
    parser.add_argument("--nfd", type=int, default=64, help=" dimension of features in first conv layer of discriminator")
    parser.add_argument("--nz", type=int, default=100, help=" dimension of latent space")
    parser.add_argument("--nc", type=int, default=1, help=" number of channels of input image")
    parser.add_argument("--weight_param", type=float, default=1, help=" lambda for the mutual information regularizer")
    parser.add_argument("--nc_gaussian", type=float, default=2, help=" number of Gaussian latent codes")

    args = parser.parse_args()
    args.model_name = f'{args.model_name}_{datetime.now().strftime("%Y.%m.%d-%H:%M:%S")}'

    args.model_root = os.path.join(args.model_root, args.model_name)
    args.figure_root = os.path.join(args.figure_root, args.model_name)
    args.log_root = os.path.join(args.log_root, args.model_name)

    train(**vars(args))