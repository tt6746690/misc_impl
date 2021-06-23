import ml_collections
import copy

import jax
from gpax import *


def get_config_base():

    config = ml_collections.ConfigDict()

    ## Likelihood
    # {LikMulticlassDirichlet, LikMulticlassSoftmax, LikMultipleNormalKron}
    config.lik_type = 'LikMulticlassDirichlet' 

    config.α_ϵ = 1.
    config.α_δ = 10.
    config.n_mc_samples = 20

    config.image_shape = (14, 14, 1)
    config.patch_shape = (10, 10)

    ## Kernel
    config.output_dim = 2

    # {None, 'CNNMnist'}
    config.patch_encoder = 'CNNMnist'

    # Place inducing inputs over space of patches
    #     Usually set to True, for computational efficiency
    config.inducing_patch = True
    config.n_inducing = 20

    # SpatialTransform
    # {'transl', 'trans+isot_scal'}
    config.T_type = 'transl'
    # Use `kl` in `CovConvolutional`
    config.use_loc_kernel = True

    return config


def get_config_rectangles():
    
    config_base = copy.deepcopy(get_config_base())
    config = ml_collections.ConfigDict(config_base)
    
    config.image_shape = (14, 14, 1)
    config.patch_shape = (3, 3)
    
    return config


def get_config_mnist():
    
    config_base = copy.deepcopy(get_config_base())
    config = ml_collections.ConfigDict(config_base)
    
    config.image_shape = (28, 28, 1)
    config.patch_shape = (10, 10)

    config.output_dim = 3
    config.n_inducing = 20
    
    return config



def get_model_cls(key, config, X):

    if config.lik_type == 'LikMulticlassDirichlet':
        init_val_m = gamma_to_lognormal(np.array([1.]))[0]
    else:
        init_val_m = np.array([0.5])

    mean_fn_cls = partial(MeanConstant, output_dim=config.output_dim,
                                        init_val_m=init_val_m,
                                        flat=False)

    if  config.lik_type == 'LikMulticlassDirichlet':
        lik_cls = partial(LikMulticlassDirichlet, output_dim=config.output_dim,
                                                  init_val_α_ϵ=config.α_ϵ,
                                                  init_val_α_δ=config.α_δ,
                                                  n_mc_samples=config.n_mc_samples)
    elif config.lik_type == 'LikMulticlassSoftmax':
        lik_cls = partial(LikMulticlassSoftmax, output_dim=config.output_dim,
                                                n_mc_samples=config.n_mc_samples)
    else:
        lik_cls = partial(LikMultipleNormalKron, output_dim=config.output_dim)


    if config.use_loc_kernel:
        kl_cls = partial(CovSE, output_scaling=False)
    else:
        kl_cls = partial(CovConstant, output_scaling=False)


    available_encoders = CovPatchEncoder.get_available_encoders()
    if config.patch_encoder in available_encoders:
        g_cls = available_encoders[config.patch_encoder]
        kg_cls = partial(CovPatchEncoder, encoder=config.patch_encoder,
                                          XL_init_fn=partial(g_cls.get_XL,
                                                             image_shape=config.image_shape),
                                          kp_cls=CovSE,
                                          kl_cls=kl_cls)
    else:
        g_cls = LayerIdentity
        kg_cls = partial(CovPatch, image_shape=config.image_shape,
                                   patch_shape=config.patch_shape,
                                   kp_cls=CovSE,
                                   kl_cls=kl_cls)

    kx_cls = partial(CovConvolutional, kg_cls=kg_cls,
                                       inducing_patch=config.inducing_patch)
    k_cls  = partial(CovMultipleOutputIndependent, output_dim=config.output_dim,
                                                   k_cls=kx_cls,
                                                   g_cls=g_cls)

    if config.T_type == '':
        Xu_initial = get_init_patches(
            key, X, config.n_inducing, config.image_shape, config.patch_shape)
        transform_cls = LayerIdentity
    else:
        Xu_ind = random.randint(key, (config.n_inducing,), 0, len(X))
        Xu_initial = np.take(X, Xu_ind, axis=0).reshape((-1, *config.image_shape))
        scal = np.array(config.patch_shape)/np.array(config.image_shape[:2])
        bound_init_fn = partial(spatial_transform_bound_init_fn, in_shape=config.image_shape,
                                                                 out_shape=config.patch_shape)
        transform_cls = partial(SpatialTransform, shape=config.patch_shape,
                                                  n_transforms=config.n_inducing, 
                                                  T_type=config.T_type,
                                                  T_init_fn=None,
                                                  A_init_val=trans2x3_from_scal_transl(scal,(0,0)),
                                                  output_transform=config.use_loc_kernel,
                                                  bound_init_fn=bound_init_fn)

    inducing_loc_cls = partial(InducingLocations,
                               shape=Xu_initial.shape,
                               init_fn=lambda k,s: Xu_initial,
                               transform_cls=transform_cls)
    
    model_cls = partial(SVGP, mean_fn_cls=mean_fn_cls,
                              k_cls=k_cls,
                              lik_cls=lik_cls,
                              inducing_loc_cls=inducing_loc_cls,
                              n_data=len(X),
                              output_dim=config.output_dim)
    
    return model_cls, k_cls, lik_cls, inducing_loc_cls, transform_cls