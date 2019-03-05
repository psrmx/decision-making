from snep.configuration import config

if config['type'] == 'brian':
    from .default_brian import *
elif config['type'] == 'none':
    from .default_empty import *
else:
    raise Exception('Unknown simulation type. Check your configuration.')
