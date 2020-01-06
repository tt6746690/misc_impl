

# Talks

- [Talks](#talks)
  - [Unsupervised Deep Learning [Neurips 2018 tutorial]](#unsupervised-deep-learning-neurips-2018-tutorial)
  - [deep generative modeling [MIT 6.S191]](#deep-generative-modeling-mit-6s191)
  - [Generative Adversarial Networks [Ian Goodfellow NIPS 2016 tutorial]](#generative-adversarial-networks-ian-goodfellow-nips-2016-tutorial)
  - [From system 1 deep learning to system 2 deep learning [Yoshia Bengio Neurips 2019 Keynote]](#from-system-1-deep-learning-to-system-2-deep-learning-yoshia-bengio-neurips-2019-keynote)
  - [Variational Bayes and Beyond: Bayesian inference for large [Tamara Brokerick ICML 2018 tutorial]](#variational-bayes-and-beyond-bayesian-inference-for-large-tamara-brokerick-icml-2018-tutorial)
  - [Interpretable Comparison of Distributions and Models [Neurips 2020 tutorial, Arthur Gretton, Dougal Sutherland, Wittawat Jitkrittum]](#interpretable-comparison-of-distributions-and-models-neurips-2020-tutorial-arthur-gretton-dougal-sutherland-wittawat-jitkrittum)




## Unsupervised Deep Learning [[Neurips 2018 tutorial]](https://www.youtube.com/watch?v=rjZCjosEFpI)


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


## deep generative modeling [[MIT 6.S191]](https://www.youtube.com/watch?v=JVb54xhEw6Y)


+ application 
    + pretty pictures
    + language translation via conditional generative models
    + generation for simulation: sim2real, make simulation more realistic 
+ autoregressive model 
    + natural ordering to take samples
+ latent variable model 
    + VAE
    + GAN
        + optimal discriminator (nonparametric)
            + generator minimizes Jensen-Shannon divergence !
        + non-saturating gradient could still misbehave in presence of good discriminator
        + vs. MLE
            + MLE: need to allocate density to nearby (unnatural) images, sampling might introduce blurring images
            + GAN: learns a sharp boundary
        + biGAN
            + able to do posterior inference p_z|x



## Generative Adversarial Networks [[Ian Goodfellow NIPS 2016 tutorial]](https://www.youtube.com/watch?v=HGYYEUSm-0Q)

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
        + game solving algorithms may not approach equilibirum at all!
        + non-convexity 
            + G,D are non-convexity parametric functions, not densities, problem with convergence
        + oscillation
            + can can train for a very long time, generating many different categories of samples, without clearly generating better samples
        + mode collapse
            + low output diversity: G makes similar images over and over again 
                + somewhat explains why conditional generation works pretty well
            + problem with simultaneous gradient descent to train G and D
            + reverse KL loss does not explain mode collapse! (still get mode collapse for forward KL)
            + unrolled GAN 
    + evaluation 
        + models with good likelhood can produce bad samples (VAE)
        + models with good samples can have bad likelihoods
        + models sometimes cannot evaluate likelihood (GAN)
        + not sure how to quantify how good the samples are
    + discrete outputs 
        + G must be differentiable, cannot different if output is discrete 
        + workaround
            + REINFORCE
            + Gumbel-softmax
            + concrete distributions
    + learning interpretable latent codes / controlling the generation process
    + domain-adversarial learning for domain adaptation 
    + robust optimization 
    + board games
+ conclusion 
    + GAN are generative models that use supervised learning to approximate an intractable cost function 
    + finding Nash equilibria in high-dimensional, continuous, non-convex games
+ questions
    + how to sample data x ~ p_x ?
        + uniform sampling most straight-forward
            + however most samples are wasted when trying to train the generator
        + importance sampling
            + pick harder samples to better train the generator



## From system 1 deep learning to system 2 deep learning [[Yoshia Bengio Neurips 2019 Keynote]](https://www.youtube.com/watch?v=FtUbMG3rlFs)


+ future 
    + handle change in distribution generalization 
    + reinforcement learning


## Variational Bayes and Beyond: Bayesian inference for large [[Tamara Brokerick ICML 2018 tutorial]](https://www.youtube.com/watch?v=Moo4-KR5qNg)



+ bayesian inference 
    + analysis goal
        + point estimates
        + coherent uncertainties
    + interpretability, complex, modular, export information
    + software: https://en.wikipedia.org/wiki/Stan_(software)
    + challenges
      + fast (compute, user), reliable inference
+ variational bayes
    + can be very fast! works on large data, large dimension
    + eg
        + text data
        + structure in network/graph
+ bayesian inference
    + build a model: choose prior & likelihood (like a generative model)
    + compute the posterior (hard because no closed form)   
        + compute evidence (denominator) requires integration over high-dim space
        + MCMC goldstandard (accurate but slow)
    + report a summary: posterior mean and (co)variances (hard)
+ approximate bayesian inference (compute evidence)
    + optimization
    + approximate posterior q* over a set of NICE approximate distributions
    + want to find q* closest in distance (KL(q|p)) to p(theta|y)
    + assumptions choices
        + use approximate family distributino
        + optimization: finds q* by minimizing some function/distance 
        + variational bayes: use KL as distance
        + mean field: factorizing assumption for q
        + optimization algorithm
+ Variational Bayes
    + KL divergence
        + good practical performance (point estimates + prediction)
        + fast streaming distributed
        + argmin KL  <=>  argmax ELBO(q)
        + forward KL, some terms are not computable
    + mean field
        + q factorizes coordinate-wise; often exponential family
    + optimization 
        + coordinate descent
        + stochastic variational inference (SVI)
        + automatic differentiation variational inference (ADVI)
+ midge wing length 
    + a simple example where we know the exact answer, can evaluate approximate BI
+ microcredit experiment
+ uncertainty 
    + if p multivariate Gaussian with large correlation
    + MFVB q 
        + underestimates variance
        + no covariance estimates (since RV factors)
+ do not know exact solution, what can we do ?
    + diagnostics
        + KL vs ELBO
    + nicer set of distributions; alternative divergences
    + thoretical guarantees on finite-data quality



## Interpretable Comparison of Distributions and Models [[Neurips 2020 tutorial, Arthur Gretton, Dougal Sutherland, Wittawat Jitkrittum]](https://slideslive.com/38921490/interpretable-comparison-of-distributions-and-models)

s
+ divergence measures
    + (P-Q) integral probability metrics (IPM)
        + Wasserstein metrics
        + maximum mean discrepancy
    + (P/Q) phi/f - divergence
+ IPM
    + find well behaved (smooth) function to maximize difference in expectation
    + intuition: if points are clustered separately, they are probably from different distribution (easier to find such f which maxmizes IPM)
+ MMD