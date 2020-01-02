import math
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torch.distributions as dist
import matplotlib.pyplot as plt
import data


def batch_indices(iter):
    idx = iter % num_batches
    return slice(idx * batch_size, (idx+1) * batch_size)


# sampler from Diagonal Gaussian x~N(μ,σ^2 I)
#
def sample_gaussian(mu,logvariance):
    # http://blog.shakirm.com/2015/10/machine-learning-trick-of-the-day-4-reparameterisation-tricks/
    #
    epsilon = torch.randn_like(logvariance)
    return mu + torch.exp(0.5*logvariance)*epsilon

# sampler from Bernoulli
#
def sample_bernoulli(p):
    return (torch.rand_like(p) < p).float()

# log-pdf of x under Diagonal Gaussian N(x|μ,σ^2 I)
#
# x           (batch_size, n_x)
# mu          (batch_size, n_x)
# log_sigma2  (batch_size, n_x)
#
def logpdf_gaussian(x,mu,logvariance):
    # batch dot product: https://github.com/pytorch/pytorch/issues/18027
    #
    # overflow problem fix: put 1/sigma^2 \circ (x-mu) first, ....
    #
    return (-mu.shape[-1]/2)*math.log(2*math.pi) - \
        (1/2)*torch.sum(logvariance,dim=1) - \
        (1/2)*torch.sum((1/torch.exp(logvariance))*(x-mu)*(x-mu),-1)

# log-pdf of x under Bernoulli 
#
# x    (batch_size, n_x)
# p    (batch_size, n_x)
#
def logpdf_bernoulli(x,p):
    return torch.sum(dist.bernoulli.Bernoulli(probs=p).log_prob(x),dim=-1)


class Encoder(nn.Module):
    
    def __init__(self,n_x,n_hidden,n_z):
        super(Encoder, self).__init__()
        
        self.fc1= nn.Linear(n_x,n_hidden)
        self.fc2_mu = nn.Linear(n_hidden,n_z)
        self.fc2_logvariance = nn.Linear(n_hidden,n_z)
        
    def forward(self,x):
        
        h = torch.tanh(self.fc1(x))
        mu = self.fc2_mu(h)
        logvariance = self.fc2_logvariance(h)
        
        return mu, logvariance
    

class StochasticLayer(nn.Module):
    
    def __init__(self):
        super(StochasticLayer, self).__init__()
        pass
    
    def forward(self,mu,logvariance):
        z = sample_gaussian(mu,logvariance)
        return z
    
class Decoder(nn.Module):
    
    def __init__(self,n_x,n_hidden,n_z):
        super(Decoder, self).__init__()
        
        self.fc1 = nn.Linear(n_z,n_hidden)
        self.fc2 = nn.Linear(n_hidden,n_x)
        
    def forward(self, z):
        h = torch.tanh(self.fc1(z))
        y = torch.sigmoid(self.fc2(h))
        
        return y



def variational_objective(x,mu,logvariance,z,y):

    # log_q(z|x) logprobability of z under approximate posterior N(μ,σ^2)

    log_approxposterior_prob = logpdf_gaussian(z,mu,logvariance)

    # log_p_z(z) log probability of z under prior

    log_prior_prob = logpdf_gaussian(z,torch.zeros_like(z),torch.log(torch.ones_like(z)))

    # log_p(x|z) - conditional probability of data given latents.

    log_likelihood_prob = logpdf_bernoulli(x,y)

    # Monte Carlo Estimator of mean ELBO with Reparameterization over M minibatch samples.
    # This is the average ELBO over the minibatch
    # Unlike the paper, do not use the closed form KL between two gaussians,
    # Following eq (2), use the above quantities to estimate ELBO as discussed in lecture

    # number of samples = 1
    elbo = torch.mean(-log_approxposterior_prob + log_likelihood_prob + log_prior_prob)

    return elbo

if __name__ == '__main__':

    
    # Load MNIST and Set Up Data

    N_data, train_images, train_labels, test_images, test_labels = data.load_mnist()

    train_images = np.round(train_images[0:10000])
    train_labels = train_labels[0:10000]
    test_images = np.round(test_images[0:10000])
    test_labels = test_labels[0:10000]

    train_images = torch.from_numpy(train_images).float()
    train_labels = torch.from_numpy(train_labels).float()
    test_images = torch.from_numpy(test_images).float()
    test_labels = torch.from_numpy(test_labels).float()
    
    # Load Saved Model Parameters (if you've already trained)

    trained = True
    n_x = 28*28
    n_hidden = 500
    n_z = 2

    encoder = Encoder(n_x,n_hidden,n_z)
    stochasticlayer = StochasticLayer()
    decoder = Decoder(n_x,n_hidden,n_z)

    if trained:
        device = torch.device("cpu")
        encoder.load_state_dict(torch.load('encoder.pt'))
        decoder.load_state_dict(torch.load('decoder.pt'))
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    encoder.to(device)
    stochasticlayer.to(device)
    decoder.to(device)

    print(f'{encoder}\n{stochasticlayer}\n{decoder}')

    # Set up ADAM optimizer

    if not trained:

        torch.manual_seed(0)
        optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()))

        n_epochs = 200
        batch_size = 100
        num_batches = int(np.ceil(len(train_images) / batch_size))
        n_batches_print = 100

        for epoch in range(n_epochs):

            running_loss = 0.0
            for it in range(num_batches):
                
                iter_images = train_images[batch_indices(it)].to(device)
                iter_labels = train_labels[batch_indices(it)].to(device)

                optimizer.zero_grad()
                
                mu, logvariance = encoder(iter_images)
                z = stochasticlayer(mu, logvariance)
                y = decoder(z)

                loss = -variational_objective(iter_images,mu,logvariance,z,y)
                loss.backward()
            
                optimizer.step()
                
                running_loss += loss
                
                if it % n_batches_print == n_batches_print-1:    # print every 200 mini-batches
                    print(f'[{epoch+1} {it+1}] loss: {running_loss/n_batches_print}')
                    running_loss = 0.0

        print('Finished Training')

        # Save Optimized Model Parameters
        torch.save(encoder.state_dict(), "./encoder.pt")
        torch.save(decoder.state_dict(), "./decoder.pt")


    # ELBO on training set

    mu, logvariance = encoder(train_images.to(device))
    z = stochasticlayer(mu, logvariance)
    y = decoder(z)

    elbo_training = variational_objective(train_images.to(device),mu,logvariance,z,y)
    print(f"Training set ELBO = {elbo_training}")

    # ELBO on test set

    mu, logvariance = encoder(test_images.to(device))
    z = stochasticlayer(mu, logvariance)
    y = decoder(z)

    elbo_testing = variational_objective(test_images.to(device),mu,logvariance,z,y)
    print(f"Training set ELBO = {elbo_testing}")