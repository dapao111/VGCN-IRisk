from collections import OrderedDict


class Config(object):
    def __init__(self, config_dict: dict):
        for k, v in config_dict.items():
            setattr(self, k, v)


def get_config(**kwargs):
    '''
    Create a Config object for configuration from input
    '''
    config = OrderedDict(
        [
            ('lr', 1e-5),
            ('beta', (0.9, 0.999)),
            ('alpha', 0.9),
            ('weight_decay', 0.0001),
            ('warmup_step', 100),
            ('eps', 1e-6),
            ('with_cuda', True),
            ('log_freq', 10),
            ('eval_freq', 20),
            ('hidden_size', 256),
            ("decrease_steps", 1000),
            ('eval', False),
            ('amp', False),
            ("gradient_accumulation_steps", 1),
            ("max_grad_norm", 1.0),
            ("eval", False),
            ("save_freq", None),
            ("loss", "bce"),
            ("hidden_dropout_prob", 0.1),
            ("num_labels", 19)
        ]
    )

    if kwargs is not None:
        for key in config.keys():
            if key in kwargs.keys():
                config[key] = kwargs.pop(key)

    return Config(config)
