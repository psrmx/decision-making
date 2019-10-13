from .network_base import *


class NetworkTables(NetworkTablesReaderBase, NetworkTablesBase):
    # __metaclass__ = LockAllFunctionsMeta
    '''
    A class that knows how to create a subtree of the hdf5 file defined by ExperimentTables. That
    subtree specifies everything about the network and simulation that is part of the experiment.
    '''
    def initialize(self, h5f, parentgroup):
        '''
        Given a parent group this function creates all the necessary default groups and
        tables that this class is responsible for.
        '''
        NetworkTablesBase.initialize(self, h5f, parentgroup)

    def as_dictionary(self, paramspace_pt, brian=True):
        '''
        This returns the entire table of parameters, but with any value
        defined in the paramspace_pt overwritten.
        '''
        from six import iteritems
        params = self.get_general_params(brian)
        if brian:
            paramspace_pt = {k: v.quantity for k, v in iteritems(paramspace_pt)}
        update_params_at_point(params, paramspace_pt)
        return params

