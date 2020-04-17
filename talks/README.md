

# Talks

- [Talks](#talks)
  - [Unsupervised Deep Learning (Neurips 2018 tutorial)](#unsupervised-deep-learning-neurips-2018-tutorial)
  - [deep generative modeling (MIT 6.S191)](#deep-generative-modeling-mit-6s191)
  - [Generative Adversarial Networks (NIPS 2016 tutorial Ian Goodfellow)](#generative-adversarial-networks-nips-2016-tutorial-ian-goodfellow)
  - [From system 1 deep learning to system 2 deep learning (Neurips 2019 Keynote Yoshia Bengio)](#from-system-1-deep-learning-to-system-2-deep-learning-neurips-2019-keynote-yoshia-bengio)
  - [Variational Bayes and Beyond: Bayesian inference for large (ICML 2018 tutorial Tamara Brokerick)](#variational-bayes-and-beyond-bayesian-inference-for-large-icml-2018-tutorial-tamara-brokerick)
  - [Interpretable Comparison of Distributions and Models (Neurips 2020 tutorial, Arthur Gretton, Dougal Sutherland, Wittawat Jitkrittum)](#interpretable-comparison-of-distributions-and-models-neurips-2020-tutorial-arthur-gretton-dougal-sutherland-wittawat-jitkrittum)
  - [Causal Inference and Stable Learning (ICML 2019 tutorial, Peng Cui, Tong Zhang)](#causal-inference-and-stable-learning-icml-2019-tutorial-peng-cui-tong-zhang)
  - [Post-selection Inference for Forward Stepwise Regression, Lasso and other procedures (NIPS 2015 talk, Robert Tibshirani)](#post-selection-inference-for-forward-stepwise-regression-lasso-and-other-procedures-nips-2015-talk-robert-tibshirani)
  - [Knockoffs: using ML for finite-sample controlled variable selection in nonparametric models (video)](#knockoffs-using-ml-for-finite-sample-controlled-variable-selection-in-nonparametric-models-video)
  - [Hopfield Nets](#hopfield-nets)


## Unsupervised Deep Learning [(Neurips 2018 tutorial)](https://www.youtube.com/watch?v=rjZCjosEFpI)


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


## deep generative modeling [(MIT 6.S191)](https://www.youtube.com/watch?v=JVb54xhEw6Y)


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



## Generative Adversarial Networks [(NIPS 2016 tutorial Ian Goodfellow)](https://www.youtube.com/watch?v=HGYYEUSm-0Q)

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



## From system 1 deep learning to system 2 deep learning [(Neurips 2019 Keynote Yoshia Bengio)](https://www.youtube.com/watch?v=FtUbMG3rlFs)


+ future 
    + handle change in distribution generalization 
    + reinforcement learning


## Variational Bayes and Beyond: Bayesian inference for large [(ICML 2018 tutorial Tamara Brokerick)](https://www.youtube.com/watch?v=Moo4-KR5qNg)



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



## Interpretable Comparison of Distributions and Models [(Neurips 2020 tutorial, Arthur Gretton, Dougal Sutherland, Wittawat Jitkrittum)](https://slideslive.com/38921490/interpretable-comparison-of-distributions-and-models)


+ divergence measures
    + (P-Q) integral probability metrics (IPM)
        + Wasserstein metrics
        + maximum mean discrepancy
    + (P/Q) phi/f - divergence
+ IPM
    + find well behaved (smooth) function to maximize difference in expectation
    + intuition: if points are clustered separately, they are probably from different distribution (easier to find such f which maxmizes IPM)
+ MMD



## Causal Inference and Stable Learning [(ICML 2019 tutorial, Peng Cui, Tong Zhang)](https://slideslive.com/38917403/causal-inference-and-stable-learning)


+ motivation 
    + risk-sensitive areas: performance driven -> risk sensitive
+ the problems
    + explainability
        + human in the loop.
        + the model should be understandable to humans so that human can cooperate with algorithm
    + stability
        + no performance guarantee when training/test data distribution are different
        + this is in fact what needs to be addressed, instead of interpretability
            + pacemaker for chest x-ray with disease, the model learns something different. This is due to training/test data distribution different
            + think about linear models
+ where did explanability problem come from: __correlation__
    + correlation is basics of ML, but is not unexplainable
        + unstable  
            + e.g. classifier of dog detects grass as highly indicative of dog in training datset, since grass is correlated with dog. But this correlation is not generalizable to test dataset where there are lots of examples where there is a dog but not grass.... So correlation is unstable
+ not the fault of correlation, but the way we use it!
    + 3 sources of correlation
        + causation
            + causal mechanism
            + T -> Y
                + summer -> ice cream sales
            + stable and explainable !
        + confounding
            + T <- X -> Y
                + T,Y have spurious correlation
                + X is income, T is product offer, Y is accepted
                + T,Y independent conditioned on X, so they are not causal
            + the model ignores X
            + unstable and not explainable
        + sample selection bias
            + T -> S <- Y
                + grass -> sample selection <- dog, i.e. training dataset selects samples with T and Y. But there is no causal relationship between them
                + T,Y independent conditioned on S, so they are not causal
            + unstable and not explainable
+ causality
    + T causes Y <=> changing T leads to change in Y, while keeping everything else constant
    + benefits
        + more explainable and more stable
    + the gap 
        + how to evaluate the outcome of a causal model
        + in the wild
            + high-dimensional
            + highly noisy
            + little prior knowledge (model specification, confounding structures)
        + target problems
            + understanding (causality) vs. prediction (learning)
            + depth vs. scale and performance
+ structural causal model 
    + use graphical model to describe causal mechanisms
    + back door criterion to identify which variable to control to identify causal effect
    + causal estimation with do calculus
+ discover causal structure
    + by definition from some expert, but not generally feasible
    + approaches
        + constraint-based: conditional independence
            + a generative model with strong expressive power, but induces complexity
            + combinatorial problem
        + potential outcome framework (https://en.wikipedia.org/wiki/Rubin_causal_model)
            + simpler
            + suppose all confounders of T are known a priori, and observed
            + computational complexity affordable
            + a discriminative way to estimate treatment's partial effect on outcome
                + estimate Y from T, given all confounder, i.e. children of T and Y
+ causal effect estimation (potential outcome framework)
    + Average causal effect of treatment (ATE)
        + ATE = E[Y(T=1) - Y(T=0)] 
    + counterfactual problem
        + for each person, observe Y_t=1 or Y_t=0; but different groups (t=0,t=1) something else are not constant
    + ideal solution: counterfactual world
        + everything is same, except for T=1 or T=0
        + randomized experiments are gold standards
            + cost
            + unethical
            + unrealistic
                + cannot control data generation process
    + in reality: use observational data only   
        + we cannot estimate ATE by comparing average outcome between two groups with observational data only, since X might not be same due to confounding effects
        + need to balance confounders' effect in T=0 and T=1 group
+ confounder balancing
    + matching
        + large scale data
        + can identify pairs of T=0 and T=1 units whose confounders X are similar or identical to each other
            + distance(X_i,X_j) < epsilon 
        + smalller epsilon, less bias but higher variance
            + OK for low dimensional settings, but in high-dim there will be fewer matches
    + propensity score based methods  
        + high dimensional problem
            + matching -> propensity score estimation
        + e(X) = P(T=1|X)
            + need to be estimated
                + supervised learning (logistic regression)
        + propensity score is sufficient to control/summarize information of confounders
        + match by distance in e(X)
            + distance(X_i,X_j) = |e(X_i) - e(X_j|
    + inverse of propensity weighting (IPW)
        + weight samples by inverse of propensity score
        + induces distribution bias on confounders X, to make them similar
            + i.e. pseudo-population where confounders are the same between treated and control groups
        + problem
            + requires model specification (i.e. classification model) for propensity score
            + high variance when e is close to 0 or 1
    + directly confounder balancing
        + motivation 
          +  non-parametric solution (no model specification)
            + distribution of all moments of variables uniquely determines their distritutions
        + methods
            + learning sample weights by directly balancing confounders' moments in the two groups
            + also minimize entropy of weights 
        + problem
            + need to know all confounders a priori or regard all variables as confounders 
+ differentiated confounder balancing
    + motivation 
        + identify confounder variables
        + different confounder makes different confounding bias
    + method
        + simultaneously learn confounder wieghts (identify which variables are confounders, and its contributions) and sample weights (for confounder balancing)
+ future directions
    + need to address
        + binary -> continuous variable 
        + single -> group of variables



## Post-selection Inference for Forward Stepwise Regression, Lasso and other procedures [(NIPS 2015 talk, Robert Tibshirani)](https://www.youtube.com/watch?v=RKQJEvc02hc)

+ post-selection inference
    + collect data -> select model -> test hypothesis
        + model selection dependent on data
    + p-value, confidence interval used in classical inference applied to post selection inference are not valid anymore
+ polyhedral lemma
    + LAR, Lasso with fixed lambda, forward stepwise selection emits polyhedral constraint set
    + can compute p-value, CI exactly in closed form



## Knockoffs: using ML for finite-sample controlled variable selection in nonparametric models [(video)](http://www.birs.ca/events/2018/5-day-workshops/18w5054/videos/watch/201801171110-Janson.html)

+ pretty clear talk !




## [Hopfield Nets](https://www.youtube.com/watch?v=IP3W7cI01VY&list=PLiPvV5TNogxKKwvKb1RKwkq2hm7ZvpHz0&index=11)


+ energy based model 
+ hopfield nets
    + binary threhsold units with recurrent connections between them
    + behavior: oscillation, converge, chaotic
    + if connections are symmetric, there is a global energy function
        + each binary configuration of whole network has an energy
+ energy function 
    + global energy is sum of local contribution 
        + connection weight and binary states of two neurons
        + `E = -sum_i s_i b_i - sum_i<j s_i s_j w_ij`
            + `s_i` state of `i`-th neuron 
            + `b_i,` `w_ij` are weights
        + energy gap (local change affects global energy)
            + `deltaE_i = b_i + sum_j s_j w_ij`
+ settling to (local) energy minimum
    + start from random state
    + sequential update (update 1 unit)
        + update each unit to whichever of its two states gives the lowest global energy 
        + needs to be sequential, because otherwise energy has the potential to go up
+ hopfield (1982) idea
    + (content addressable memory) memories could be energy minima of a neural net
        + an item can be accessed by just knowing part of its content
        + robust against hardware damage
        + analogy: reconstruct a dinosaur from a few bones
