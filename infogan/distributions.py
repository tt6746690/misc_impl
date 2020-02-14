import math

import torch
import torch.distributions as dist


def logpdf_gaussian(x, mu, logvariance, reduction='mean'):
    """ Evaluate log Normal(x;μ,exp.(σ^2))
            i.e. log-pdf of x under spherical Gaussian N(x|μ,σ^2 I)

        x           (batch_size, n_x)
        mu          (batch_size, n_x)
        log_sigma2  (batch_size, n_x)

    batch dot product: https://github.com/pytorch/pytorch/issues/18027
    
    overflow problem fix: put 1/sigma^2 \circ (x-mu) first, ....
    """
    log_probs = (-mu.shape[-1]/2)*math.log(2*math.pi) - \
        (1/2)*torch.sum(logvariance,dim=1) - \
        (1/2)*torch.sum((1/torch.exp(logvariance))*(x-mu)*(x-mu),-1)
    if reduction == 'mean':
        return torch.mean(log_probs,dim=-1)
    elif reduction == 'sum':
        return torch.sum(log_probs,dim=-1)
    else:
        raise Exception


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

