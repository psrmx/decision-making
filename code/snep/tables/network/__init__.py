from snep.configuration import config

if config['type'] == 'brian':
    from .network_brian import *
elif config['type'] == 'none':
    from .network_empty import *
else:
    raise Exception('Unknown simulation type. Check your configuration.')
