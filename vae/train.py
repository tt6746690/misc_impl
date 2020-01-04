import os
import itertools
import argparse
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision 
import torchvision
import torchvision.datasets as datasets

import models


def train(
    data_root,
    model_root,
    figure_root,
    model_name,
    load_model_file,
    n_epochs,
    n_batches_print,
    batch_size,
    seed,
    lr, 
    enc_sizes,
    dec_sizes,
    latent_size,
    conditional,
):

    os.makedirs(data_root, exist_ok=True)
    os.makedirs(model_root, exist_ok=True)
    os.makedirs(figure_root, exist_ok=True)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: torch.round(x)),
        torchvision.transforms.Lambda(lambda x: x.view(-1)),
    ])

    trainset = datasets.MNIST(root=data_root, train=True, download=True, transform=transforms)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

    testset = datasets.MNIST(root=data_root, train=False, download=True, transform=transforms)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=8)
    
    conditional_size = 10

    enc_sizes = [*enc_sizes, latent_size]
    dec_sizes = [latent_size, *dec_sizes]

    if conditional:
        enc_sizes[0] += conditional_size
        dec_sizes[0] += conditional_size
        prior_network_sizes = [conditional_size, 5, latent_size]

        model_cls = models.CVAE
        model = model_cls(enc_sizes, dec_sizes, prior_network_sizes)
    else:
        model_cls = models.VAE 
        model = model_cls(enc_sizes, dec_sizes)

    if load_model_file is not '':
        model.load_state_dict(torch.load(load_model_file))
        model.to(device)
    else:

        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in tqdm(range(n_epochs), desc='epochs'):

            plot_title = f'latent_space_{model_name}_epochs={epoch}'
            plot_latent_space(trainset, model, 
                plot_title=plot_title,
                saveas=os.path.join(figure_root, f'{plot_title}.png'),
                conditional=conditional)

            plot_title = f'latent_sample_decoded_{model_name}_epochs={epoch}'
            plot_sample_generative(model,
                plot_title=plot_title,
                saveas=os.path.join(figure_root,  f'{plot_title}.png'),
                conditional=conditional)

            for c in range(10):
                plot_title = f'decode_along_a_lattice_{model_name}_c={c}_epochs={epoch}'
                plot_decode_along_a_lattice(model,
                    plot_title = plot_title,
                    saveas = os.path.join(figure_root, f'{plot_title}.png'),
                    conditional = conditional,
                    conditioned_class_label = c)

            running_loss = 0
            for it, (x, c) in enumerate(trainloader):

                x, c = x.to(device), c.to(device)
                c = nn.functional.one_hot(c, conditional_size).float()

                if conditional:
                    out = model(x, c)
                else:
                    out = model(x)

                loss = -model_cls.variational_objective(x, *out)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss
                if it % n_batches_print == n_batches_print-1:
                    print(f'[{epoch+1} {it+1}] loss: {running_loss/n_batches_print}')
                    running_loss = 0.0

        model_filename = os.path.join(model_root, f'{model_name}.pt')
        print(f'Finished Training, saving model to {model_filename}')
        torch.save(model.state_dict(), model_filename)

    
    def model_performance(model, dataloader):
        elbo_sum = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            if conditional:
                out = model(x, nn.functional.one_hot(y, conditional_size).float())
            else:
                out = vae(x)
            elbo_sum += model_cls.variational_objective(x, *out)
        return elbo_sum/len(dataloader)


    print(f"Training set ELBO = {model_performance(model, trainloader)}")
    print(f"Test set ELBO     = {model_performance(model, testloader)}")



def plot_decode_along_a_lattice(
    model,
    conditional = False,
    conditioned_class_label = 0,
    plot_title = 'decode along a lattice',
    n_samples_per_axis = 20,
    axis_range = (-5, 5),
    saveas = None,
    device = torch.device('cuda')
):

    X, Y = np.meshgrid(np.linspace(*axis_range, n_samples_per_axis), np.linspace(*axis_range, n_samples_per_axis))
    z = np.hstack([X.reshape(-1,1), Y.reshape(-1,1)])
    z = torch.from_numpy(z).float().to(device)

    if conditional:
        c = torch.full((z.shape[0],), conditioned_class_label, dtype=torch.int64)
        c = nn.functional.one_hot(c, 10).float()
        c = c.to(device)

        zc = model.combine_latent_and_conditional(z, c)
        y = model.decoder(zc)
    else:
        y = model.decoder(z)
        
    ims = torchvision.utils.make_grid(y.view(-1,1,28,28), padding=1, nrow=n_samples_per_axis)
    ims = np.transpose(ims.cpu().detach().numpy(), (1,2,0))
    plt.title(plot_title)
    plt.imshow(ims)
    plt.axis('off')

    if saveas:
        plt.savefig(saveas)
        plt.clf()
        plt.close('all')
    else:
        plt.show()



def plot_sample_generative(
        model,
        latent_size=2,
        plot_title='sample from latent space',
        conditional=False,
        saveas=None,
        device=torch.device('cuda'),
    ):
    """ Sample from prior
        Decode samples to pixel-wise bernouli in image space
    """

    n_z = 10

    if conditional:
        
        ys = []
        
        for c in range(10):
            c = nn.functional.one_hot(torch.full((n_z,), c, dtype=torch.int64), 10).float()
            c = c.to(device)
            mu, logvariance = model.prior_network(c)
            
            z = models.sample_gaussian(mu, logvariance)
            zc = model.combine_latent_and_conditional(z, c)
            y = model.decoder(zc)
            
            ys.append(y)
        
        ims = torch.cat([im.view(-1,1,28,28) for im in ys], dim=0)
        ims = torchvision.utils.make_grid(ims, padding=1, nrow=n_z)
    else:
        mu = torch.zeros(n_z, latent_size, device=device)
        logvariance = torch.zeros(n_z, latent_size, device=device)

        z = models.sample_gaussian(mu, logvariance)
        y = model.decoder(z)
        xt = models.sample_bernoulli(y)

        ims = torch.cat([im.view(-1,1,28,28) for im in [y, xt]], dim=0)
        ims = torchvision.utils.make_grid(ims, padding=1, nrow=5)
        
    ims = np.transpose(ims.cpu().detach().numpy(), (1,2,0))
    plt.title(plot_title)
    plt.imshow(ims)
    plt.axis('off')

    if saveas:
        plt.savefig(saveas)
        plt.clf()
        plt.close('all')
    else:
        plt.show()


def plot_latent_space(
        dataset, 
        model,
        plot_title='latent space visualization',
        conditional=False,
        saveas=None,
        n_samples_per_class=100,
        device=torch.device('cuda'),
    ):
    """ latent space visualization 
            encode samples and visualize distribution in latent space
    """

    viridis = plt.cm.get_cmap('viridis', 10)

    for c in range(10):

        x = itertools.islice(filter(lambda x: x[1] == c, dataset), n_samples_per_class)
        x = torch.stack(list(zip(*x))[0], 0)
        x = x.to(device)

        if not conditional:
            mu, _ = model.encoder(x)
        else:
            batch_c = nn.functional.one_hot(torch.tensor(c), 10).float()
            batch_c = batch_c.unsqueeze(0).repeat(n_samples_per_class,1)
            batch_c = batch_c.to(device)
            
            xc = model.combine_output_and_conditional(x, batch_c)
            mu, _ = model.encoder(xc)
        
        
        mu = mu.cpu().detach().numpy()

        color = np.ones(n_samples_per_class) * c
        color = color.astype(np.uint8)

        plt.scatter(mu[:,0],mu[:,1],c=np.tile(np.array(viridis(c/10)),(n_samples_per_class,1)),alpha=0.5,label=f'{c}')

    plt.title(plot_title)
    
    if not conditional:
        lim_range = (-6,6)
    else:
        lim_range = (-15,15)
    
    plt.xlim(lim_range)
    plt.ylim(lim_range)
    plt.xlabel('z1')
    plt.ylabel('z2')
    plt.legend()

    if saveas:
        plt.savefig(saveas)
        plt.clf()
        plt.close('all')
    else:
        plt.show()

    return plt


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str,
                        default='../data',
                        help="data folder")
    parser.add_argument("--model_root", type=str,
                        default='./model',
                        help="model folder")
    parser.add_argument("--figure_root", type=str,
                        default='./figure',
                        help="figure folder")
    parser.add_argument("--model_name", type=str,
                        default='',
                        help="name of the model")
    parser.add_argument("--load_model_file", type=str,
                        default='',
                        help="optional .pt model file to initialize with")
    parser.add_argument("--epochs", type=int, dest='n_epochs',
                        default=200,
                        help="number of iterations")
    parser.add_argument("--print_every", type=int, dest='n_batches_print',
                        default=100,
                        help="number of batches to print loss / plot figures")
    parser.add_argument("--batch_size", type=int,
                        default=100,
                        help="batch_size")
    parser.add_argument("--seed", type=int,
                        default=0,
                        help="rng seed")
    parser.add_argument("--learning_rate", type=float, dest='lr',
                        default=0.001,
                        help="rng seed")
    parser.add_argument("--encoder_layer_sizes", type=list, dest='enc_sizes',
                        default=[784, 500],
                        help="encoder layer sizes, excluding size for output layer")
    parser.add_argument("--decoder_layer_sizes", type=list, dest='dec_sizes',
                        default=[500, 784],
                        help="decoder layer sizes, excluding size for input layer")
    parser.add_argument("--latent_size", type=int,
                        default=2,
                        help="latent dimension size")
    parser.add_argument("--conditional", action='store_true',
                        default=False,
                        help="use conditional VAE")

    args = parser.parse_args()

    if args.model_name == '':
        if args.conditional:
            args.model_name = 'cvae'
        else: 
            args.model_name = 'vae'

    train(**vars(args))