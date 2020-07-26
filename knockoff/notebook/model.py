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

def logpdf_bernoulli(x, p, reduction='sum'):
    """ Evaluate log p(x|p) where x~Bern(p)
        
            Equivalent to nn.BCELoss()
            
            ```
            p = torch.randn(3, requires_grad=True)
            x = torch.tensor([1,0,1])
            x_onehot = torch.nn.functional.one_hot(x,2)
            x = x.float()
            p = torch.nn.Sigmoid()(p)
            assert(-torch.nn.BCELoss()(p, x) == logpdf_bernoulli(x, p, reduction='mean'))
            ```
        
        
        x    (batch_size,)
             batch of bernoulli samples
        p    (batch_size, n_classes)
             batch of class probabilities
    """
    log_probs = dist.bernoulli.Bernoulli(probs=p).log_prob(x)
    if reduction == 'mean':
        return torch.mean(log_probs,dim=-1)
    elif reduction == 'sum':
        return torch.sum(log_probs,dim=-1)
    else:
        raise Exception


def logpdf_categorical(x, p, reduction='mean'):
    """ Evaluates log p(x|p) where x~Cat({p_k})
            Equivalent to nn.NLLLoss(),
            
            ```
            p = torch.randn(3, 5, requires_grad=True)
            x = torch.tensor([1,0,4])
            x_onehot = torch.nn.functional.one_hot(x,5)
            
            logp = torch.nn.LogSoftmax(dim=1)(p)
            p = torch.nn.Softmax(dim=1)(p)
            
            log_likelihood = -torch.nn.NLLLoss()(logp, x)
            logpdf_cat = logpdf_categorical(x, p, reduction='mean')
            
            assert(log_likelihood == logpdf_cat)
            ```
        x    (batch_size,)
             batch of categorical samples 
        p    (batch_size, n_classes)
             batch of class probabilities
    """
    log_probs = dist.categorical.Categorical(probs=p).log_prob(x)
    if reduction == 'mean':
        return torch.mean(log_probs,dim=-1)
    elif reduction == 'sum':
        return torch.sum(log_probs,dim=-1)
    else:
        raise Exception


class Encoder(nn.Module):
    """
    encoder = Encoder([784, 500, 2])
    """

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


class Decoder(nn.Module):
    """
    decoder = Decoder([2, 500, 784])
    """
        
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


class LogisticRegression(nn.Module):
    """ 
        # computes softmax and then the cross entropy
        criterion = torch.nn.CrossEntropyLoss() 

        # training loop
            optimizer.zero_grad()
            Yhat = model(X)
            loss = criterion(Yhat, X)
            loss.backward()
            optimizer.step()
    """
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs


class VAEWithLinearDecoder(nn.Module):

    def __init__(self, enc_sizes, output_dim):
        super(VAEWithLinearDecoder, self).__init__()

        self.encoder = Encoder(enc_sizes)
        self.stochasticlayer = StochasticLayer()
        self.decoder = LogisticRegression(enc_sizes[-1],output_dim)

    def forward(self, x):

        mu, logvariance = self.encoder(x)
        z = self.stochasticlayer(mu, logvariance)
        y = self.decoder(z)

        return mu, logvariance, z, y


def mlp_block(in_size, out_size):
    return nn.Sequential(
        nn.Linear(in_size, out_size),
        nn.Tanh()
    )