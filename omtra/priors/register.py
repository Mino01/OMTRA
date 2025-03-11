

train_prior_register = {}
inference_prior_register = {}

def register_train_prior(name: str):
    """
    Decorator to register training time prior function with a given name.
    :param name: A unique name for the prior.
    """
    def decorator(fn):
        fn.name = name  # Attach the key to the class.
        train_prior_register[name] = fn
        return fn
    return decorator

def register_inference_prior(name: str):
    """
    Decorator to register an inference time prior function with a given name.
    :param name: A unique name for the task.
    """
    def decorator(fn):
        fn.name = name  # Attach the key to the class.
        inference_prior_register[name] = fn
        return fn
    return decorator

# import priors, this triggers the registration of the priors
import omtra.priors.priors