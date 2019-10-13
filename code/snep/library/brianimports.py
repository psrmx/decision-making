try:
    from brian2.units import *
    from brian2 import NetworkOperation, NeuronGroup, TimedArray, SpikeMonitor
    from brian2 import PopulationRateMonitor, StateMonitor, Clock

    def unit_eval(value, units):
        return value if isinstance(value, str) else value * eval(units)

except ImportError:
    from snep.configuration import config
    import warnings

    class NetworkOperation:
        def __init__(self, *args, **kwargs):
            pass

    class Monitor:
        def __init__(self, *args, **kwargs):
            pass

    class NeuronGroup:
        def __init__(self, *args, **kwargs):
            pass

    class Connection:
        def __init__(self, *args, **kwargs):
            pass

    class TimedArray:
        def __init__(self, *args, **kwargs):
            pass

    class SpikeMonitor:
        def __init__(self, *args, **kwargs):
            pass

    class PopulationRateMonitor:
        def __init__(self, *args, **kwargs):
            pass

    class StateMonitor:
        def __init__(self, *args, **kwargs):
            pass

    class Clock:
        def __init__(self, *args, **kwargs):
            pass

    def set_group_var_by_array(*args, **kwargs):
        return args, kwargs

    def unit_eval(value, units):
        return value

    if config['type'] == 'brian':
        warnings.warn('Brian was not found, not all units will be applied correctly.')
