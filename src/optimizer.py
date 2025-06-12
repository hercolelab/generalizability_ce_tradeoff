def get_optimizer(params, config):
    name = config.pop("name")

    
    if name.lower() == "adam":
        from torch.optim import Adam
        
        optim = Adam(params=params, **config)
        
        return optim
    
    elif name.lower() == "sgd":
        from torch.optim import SGD
        
        optim = SGD(params=params, **config)
        
        return optim