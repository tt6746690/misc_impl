import math

import torch
import torch.nn as nn
import torch.distributions as dist


"""
sampler from Diagonal Gaussian x~N(μ,σ^2 I)
      http://blog.shakirm.com/2015/10/machine-learning-trick-of-the-day-4-reparameterisation-tricks/
"""
def sample_gaussian(mu, logvariance):
    epsilon = torch.randn_like(logvariance)
    return torch.exp(0.5 * logvariance) * epsilon + mu

"""
sampler from Bernoulli
"""
def sample_bernoulli(p):
    return (torch.rand_like(p) < p).float()


"""
log-pdf of x under Diagonal Gaussian N(x|μ,σ^2 I)

    x           (batch_size, n_x)
    mu          (batch_size, n_x)
    log_sigma2  (batch_size, n_x)
"""
def logpdf_gaussian(x, mu, logvariance):
    """
    batch dot product: https://github.com/pytorch/pytorch/issues/18027
    
    overflow problem fix: put 1/sigma^2 \circ (x-mu) first, ....
    """
    
    return (-mu.shape[-1]/2)*math.log(2*math.pi) - \
        (1/2)*torch.sum(logvariance,dim=1) - \
        (1/2)*torch.sum((1/torch.exp(logvariance))*(x-mu)*(x-mu),-1)

"""
log-pdf of x under Bernoulli 

    x    (batch_size, n_x)
    p    (batch_size, n_x)
"""
def logpdf_bernoulli(x, p):
    return torch.sum(dist.bernoulli.Bernoulli(probs=p).log_prob(x),dim=-1)


"""
encoder = Encoder([784, 500, 2])
"""
class Encoder(nn.Module):

    def __init__(self, layer_sizes):
        super(Encoder, self).__init__()

        self.mlp = nn.Sequential(*[
            mlp_block(in_size, out_size) 
                for in_size, out_size in zip(layer_sizes[:-2], layer_sizes[1:-1])
        ])

        self.fc_mu = nn.Linear(layer_sizes[-2], layer_sizes[-1])
        self.fc_logvariance = nn.Linear(layer_sizes[-2], layer_sizes[-1])
        
    def forward(self, x):

        h = self.mlp(x)
        mu = self.fc_mu(h)
        logvariance = self.fc_logvariance(h)
        
        return mu, logvariance

"""
decoder = Decoder([2, 500, 784])
"""
class Decoder(nn.Module):
    
    def __init__(self, layer_sizes):
        super(Decoder, self).__init__()

        self.mlp = nn.Sequential(*[
            mlp_block(in_size, out_size)
                for in_size, out_size in zip(layer_sizes[:-2], layer_sizes[1:-1])
        ])

        self.last_layer = nn.Sequential(
            nn.Linear(layer_sizes[-2], layer_sizes[-1]),
            nn.Sigmoid()
        )
        
    def forward(self, z):
        h = self.mlp(z)
        y = self.last_layer(h)

        return y
        

class StochasticLayer(nn.Module):
    
    def __init__(self):
        super(StochasticLayer, self).__init__()
        pass
    
    def forward(self, mu, logvariance):
        z = sample_gaussian(mu, logvariance)
        return z


class VAE(nn.Module):

    def __init__(self, enc_sizes, dec_sizes):
        super(VAE, self).__init__()

        self.encoder = Encoder(enc_sizes)
        self.stochasticlayer = StochasticLayer()
        self.decoder = Decoder(dec_sizes)

    def forward(self, x):

        mu, logvariance = self.encoder(x)
        z = self.stochasticlayer(mu, logvariance)
        y = self.decoder(z)

        return mu, logvariance, z, y


    @staticmethod
    def variational_objective(x, mu, logvariance, z, y):

        # log_q(z|x) logprobability of z under approximate posterior N(μ,σ^2)

        log_approxposterior_prob = logpdf_gaussian(z, mu, logvariance)

        # log_p_z(z) log probability of z under prior

        log_prior_prob = logpdf_gaussian(z, torch.zeros_like(z), torch.log(torch.ones_like(z)))

        # log_p(x|z) - conditional probability of data given latents.

        log_likelihood_prob = logpdf_bernoulli(x, y)

        """
        Monte Carlo Estimator of mean ELBO with Reparameterization over M minibatch samples.
        This is the average ELBO over the minibatch
        Unlike the paper, do not use the closed form KL between two gaussians,
        Following eq (2), use the above quantities to estimate ELBO as discussed in lecture
        """

        # number of samples = 1
        elbo = torch.mean(-log_approxposterior_prob + log_likelihood_prob + log_prior_prob)

        return elbo

class CVAE(nn.Module):

    def __init__(self, enc_sizes, dec_sizes, prior_network_sizes):
        super(CVAE, self).__init__()

        self.prior_network = Encoder(prior_network_sizes)
        self.encoder = Encoder(enc_sizes)
        self.stochasticlayer = StochasticLayer()
        self.decoder = Decoder(dec_sizes)

    def combine_output_and_conditional(self, x, c):
        return torch.cat((x, c), 1)

    def combine_latent_and_conditional(self, z, c):
        return torch.cat((z, c), 1)

    def forward(self, x, c):
        """
            x       data (output) variable
            c       conditioned (input) variable
        """

        xc = self.combine_output_and_conditional(x, c)

        rec_statistics = self.encoder(xc)
        z = self.stochasticlayer(*rec_statistics)

        pri_statistics = self.prior_network(c)

        zc = self.combine_latent_and_conditional(z, c)
        y = self.decoder(zc)

        return rec_statistics, z, pri_statistics, y


    @staticmethod
    def variational_objective(x, rec_statistics, z, pri_statistics, y):

        log_approxposterior_prob = logpdf_gaussian(z, *rec_statistics)
        log_prior_prob = logpdf_gaussian(z, *pri_statistics)
        log_likelihood_prob = logpdf_bernoulli(x, y)

        elbo = torch.mean(-log_approxposterior_prob + log_likelihood_prob + log_prior_prob)

        return elbo

    def prediction(self, x, c):
        """ predict y given x conditioned on c
                z* = E[z|x]
                y* = argmax_y p(y|x,z*)
        """

        z_star, _ = self.prior_network(c)

        zc = self.combine_latent_and_conditional(z_star, c)

        y_star = self.decoder(zc)

        return y_star

    def conditional_likelihood(self, x, c, S):
        """ Computes conditional likelihood
                p(x|c) = 1/S \sum_s p(x|c,z^s) where z^s ~ p(z|c)
        """

        c = c.unsqueeze(0).repeat(S,1)

        pri_statistics = self.prior_network(c)
        z = self.stochasticlayer(*pri_statistics)

        zc = self.combine_latent_and_conditional(z, c)
        y = self.decoder(zc)

        return torch.mean(torch.exp(logpdf_bernoulli(x, y)))



def mlp_block(in_size, out_size):
    return nn.Sequential(
        nn.Linear(in_size, out_size),
        nn.Tanh()
    )