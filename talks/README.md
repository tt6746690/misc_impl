

+ unsupervised learning NeurIPS 2018 https://www.youtube.com/watch?v=rjZCjosEFpI
    + idea
        + learn useful things for many different tasks
    + density modeling
        + curse of dimensionality: operating on input data (instead of targets, which has a lot less information)
        + not bits created equal: log likelihoods depends on low level details rather than semantics (image content)
        + unclear how to exploit latent representation for future tasks 
    + autoregressive models
        + use chain rule, address curse of dimensionality by some Markov assumption
        + advantage
            + simple: just pick an ordering!
            + easy to sample
            + best log-likelihood for common data
        + disadvantange
            + expensive (can parallel for training/ but must be serial for testing)
            + teacher forcing
                + could generate long sequences well (good models)
                + but a myopic in terms of representation, i.e. focus on the low-level bits, not high level semantics
        + examples
            + WaveNet
                + long span of time, coherent high level structure!
            + PixelRNN, Conditional PixelRNN
                + conditioning on label generate much better images!
            + subsample pixel networks 
                + change order of prediction 
            + generating sequences with RNN
                + each time step where to write modelled as mixture distribution
    + representation learning
        + learn an internal language for network 
            + describe the data well (i.e. maximize mutual information between z and x, autoencoder does this by minimizing recon)
            + re-use description for plan,reason,generalization
        + comparison 
            + task driven representation: limited by requirements of the tasks
            + unsupervised representation: should be more general!
        + contrastive predictive coding


+ MIT 6.S191 deep generative modeling https://www.youtube.com/watch?v=JVb54xhEw6Y
    + application 
        + pretty pictures
        + language translation via conditional generative models
        + generation for simulation: sim2real, make simulation more realistic 
    + autoregressive model 
        + natural ordering to take samples
    + latent variable model 
        + VAE
        + GAN


+ Generative Adversarial Networks NIPS 2016 https://www.youtube.com/watch?v=HGYYEUSm-0Q
    + why study generative models
        + simulate possible futures by generation
        + realistic generation tasks
            + next frame prediction 
                + MSE -> blurry 
                + adversarial -> sharp
            + single-image super-resolution 
    + taxonomy (max-likelihood)
        + explicit density (explicit function for p(x|theta))
            + tractable density
                + fully visible belief nets: PixelRNN/PixelCNN/WaveNet
                    + O(n) sample generation
                    + generation not from latent code
                + change of variable models: nonlinear ICA 1996/ realNVP
                    + transformation must be invertible
                    + latent dimension must match visible dimension
            + approximate density (for intractable density)
                + variational (VAE)
                    + not asymptotically consistent unless q is perfect
                    + samples tend to have lower quality
                + Markov chain (Boltzmann machine)
                    + scalability problem, only works on MNIST
        + implicit density (draw samples)
            + Markov chain: generative stochastic network
            + direct: GAN
    + GAN design 
        + use a latent code
        + asymptotically consistent (unlike variational methods)
        + no Markov chain needed (for training/sampling; holding back RBM)
    + GAN framework
        + G: sampling from noise x = G(z), z ~ p_z
            + tries to make D(G(z)) ~= 1, (i.e. fools the discriminator)
        + D
            + tries to make D(G(z)) ~= 0 (i.e. correctly identify G(z) is from noise)
            + tries to make D(x) ~= 1, x ~ p_x (i.e. correctly identify x from data distribution)
    + generator
        + differentiable 
        + no invertibility requirements (vs. nonlinear ICA)
        + trainable for any size of z
    + training
        + sample on two minibatches, training samples, generated samples
        + update parameter alternating way
    + minimax game
        + D: tries to be a good classifier of where x comes from (p_x or p_model_x)
        + G: minimizes log-probability of discriminator being correct
            + problem: discriminator too powerful, gradient for loss of G -> 0
        + equilibrium is saddle point of the discriminator loss !
        + resembles Jensen-Shannon divergence
    + non-saturating game
        + D: same classification loss
        + G: maximize log-probability of disciminator being mistaken
            + generator can still learn, even when discriminator successfully reject all generator samples
    + what is optimal discriminator function 
        + set derivative of loss w.r.t. D(x) to zero
        + optimal D(x) is a ratio p_x / (p_x + p_model_x)
        + estimate this ratio using supervised learning is a key approximation 
    + DCGAN
        + scale to larger images
        + use batch norm
        + strided deconvolution/upsampling
        + latent space arithmetic !
    + is divergence important ?
        + max likelihood: minimize forward KL(p||q)
            + important to put mass everywhere p is nonzero
        + minimize reverse KL(q||p)
            + important to not put mass on where p is zero
        + both divergence with GAN gets sharp samples both ways
        + the approximation strategy of supervised learning for estimating ratios mattered
        + variational bound makes images blurry!
    + compare to noise contrastive estimation (NCE) / MLE
        + NCE
            + perform discrimination between observed data and generated noise (similar to GAN!)
            + goal is to learn the parameter for model p_x (vs. GAN learns p_x|z)
            + generator is fixed (vs. GAN is learnt) 
                + computational savings
            + optimize by running GD (same with GAN)
        + MLE use a similar loss !
            + need both sample and computing density
        + GAN
            + do not need to compute density, just need to compute D
                + computational savings
    + tips
        + labels improve subjective sample quality (learn label conditional model)
        + one-sided label smoothing (good smoothing)
        + use batch norm
        + balancing G and D
            + usually discriminator wins, that is OK
            + run D more often than G, mixed results. Mixed results
            + do not try to limit D to aoivd making it too smart 
                + use non-saturating cost and label smoothing
    + research frontiers
        + non-convergence problem 
            + 
    + questions
        + how to sample data x ~ p_x ?
            + uniform sampling most straight-forward
                + however most samples are wasted when trying to train the generator
            + importance sampling
                + pick harder samples to better train the generator