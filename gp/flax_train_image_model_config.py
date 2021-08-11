import ml_collections

def get_config_base():

    config = ml_collections.ConfigDict()

    # runtime 
    config.gpu_id = '0'
    
    # dataset 
    config.dataset = 'cifar10'
    
    # model
    config.model = 'CNNMnist'
    
    # opitmization 
    config.learning_rate = .03
    config.batch_size = 64
    config.n_epochs = 2
    
    return config


def get_config_mnist():
    
    config = get_config_base()
    config.image_shape = (28, 28, 1)
    config.dataset = 'mnist'
    config.num_classes = 10
    
    return config


def get_config_cifar10():
    
    config = get_config_base()
    config.image_shape = (32, 32, 3)
    config.dataset = 'cifar10'
    config.num_classes = 10
    
    return config


def get_config(config_string='mnist'):
    if config_string == 'mnist':
        return get_config_mnist()
    elif config_string == 'cifar10':
        return get_config_cifar10()
    else:
        raise ValueError(f'{config_string} not valid')