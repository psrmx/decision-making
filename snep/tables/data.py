from __future__ import print_function
import tables
from six import iteritems
from scipy import sparse
from scipy.sparse import csr_matrix
import numpy as np
from snep.utils import csr_make_ints
from collections import namedtuple

mmap_array = namedtuple('mmap_array', ['identifier', 'dtype', 'shape', 'filename', 'T'])


class DataHandler(object):
    def __init__(self, h5f, log_info, log_err):
        self.h5f = h5f
        self.filters = tables.Filters(complevel=5, complib='zlib')
        self.log_info = log_info
        self.log_err = log_err

    def __del__(self):
        if self.h5f and self.h5f.isopen:
            self.h5f.close()

    def read_data_root(self):
        return self._read_node(self.h5f.root, None)

    def store_data_root(self, all_data):
        return self.store_data(self.h5f.root, all_data)

    @staticmethod
    def _maps_int_to_ndarray(data):
        """
        Checks whether data should be stored as a VLArray.
        :param data: dict
                If this is a mapping from integers to ndarrays, then it will be stored as a VLArray
        :return:
        """
        if len(data) == 0:
            return False
        return all([isinstance(k, (int, np.integer)) and isinstance(v, np.ndarray)
                    for k, v in iteritems(data)])

    def store_data(self, group, all_data, overwrite=False):
        # If overwrite is enabled, we want to provide a list
        # of keys that should be deleted. This means any key
        # that maps to a non-dictionary (e.g. an array), or a
        # dictionary that stores a VLArray.
        if overwrite:
            to_delete = [k for k, v in iteritems(all_data)
                         if not isinstance(v, dict) or DataHandler._maps_int_to_ndarray(v)]
            for node in group._f_iter_nodes():
                if node._v_name in to_delete:
                    self.log_info('!!! OVERWRITING ' + node._v_name, self.h5f)
                    node._f_remove(recursive=True)

        for name, value in iteritems(all_data):
            if isinstance(value, dict):
                if DataHandler._maps_int_to_ndarray(value):
                    self._create_vlarray(group, name, value)
                else:
                    subgroup = self._single_get_or_create_group(group, name)
                    self.store_data(subgroup, value, overwrite)
            elif sparse.issparse(value):
                self._store_sparse(group, name, value)
            elif isinstance(value, np.ndarray):
                self._create_carray(group, name, value)
            # elif isinstance(value, tables.carray.CArray):
            #     self._create_carray(group, name, value)
            elif isinstance(value, mmap_array):
                try:
                    mmap = np.memmap(value.filename, value.dtype, 'r', shape=value.shape)
                    if value.T:
                        mmap = mmap.T
                    self._create_carray(group, name, mmap, True)
                except FileNotFoundError as e:
                    self.log_err(self.h5f, e)
                except OSError as e:
                    self.log_err(self.h5f, e)
            elif isinstance(value, bytes):
                self.h5f.create_array(group, name, value)
            elif isinstance(value, str):
                self.h5f.create_array(group, name, value.encode())
            else:
                self.log_info('UNKNOWN TYPE IN DATA {} {}'.format(name, type(value)), self.h5f)

    def _single_get_or_create_group(self, parent, name):
        """
        It's necessary to have both this function and the below because if
        we combine them, the todelete list would not work correctly since
        names would have to be unique across all layers of the hierarchy.
        """
        try:
            group = parent._f_get_child(name)
        except tables.NoSuchNodeError:
            group = self.h5f.create_group(parent, name)
        # else:
        #     if overwrite:
        #         ident = '/'.join((parent._v_name, name))
        #         self.log_info('!!! OVERWRITING group '+ident)
        #         for node in group._f_iter_nodes():
        #             if node._v_name in todelete:
        #                 node._f_remove(recursive=True)
        return group

    def _nested_get_or_create_groups(self, parent, path):
        for name in path.split('/'):
            try:
                parent = parent._f_get_child(name)
            except tables.NoSuchNodeError:
                parent = self.h5f.create_group(parent, name)
        return parent

    def _store_sparse(self, group, name, arr):
        if not sparse.isspmatrix_csr(arr):
            arr = arr.tocsr()

        csr_group = self.h5f.create_group(group, name)
        csr_group._v_attrs.issparse = True

        if arr is not None and arr.nnz > 0:
            indptr, indices = csr_make_ints(arr.indptr, arr.indices)
            self.h5f.create_array(csr_group, 'data',   arr.data)
            self.h5f.create_array(csr_group, 'indptr', indptr)
            self.h5f.create_array(csr_group, 'indices', indices)
            self.h5f.create_array(csr_group, 'shape',  arr.shape)
        self.h5f.flush()

    def _create_carray(self, group, name, data, mmap=False):
        try:
            atom = tables.Atom.from_dtype(data.dtype)
            _d = self.h5f.create_carray(group, name, atom, data.shape, filters=self.filters)
            if mmap:
                _d._v_attrs.was_mmap = True

            _d[...] = data[...]
            self.h5f.flush()
        except Exception as e:
            self.log_err(self.h5f, 'EXCEPTION: {} {} {}'.format(name, np.ndim(data), e))

    def _create_vlarray(self, group, name, data):
        assert len(data), 'vlarray must have at least one element'
        dtype = np.float32
        for v in data.values():
            dtype = v.dtype

        atom = tables.Atom.from_dtype(dtype)
        _d = self.h5f.create_vlarray(group, name, atom, filters=self.filters)
        for i in data.keys():
            _d.append(data[i])
        self.h5f.flush()

    def _read_node(self, node, key):
        """
        :param node:
        :param key: Tuple. Numpy-style fancy index for an array. e.g. (Ellipse, slice(3, -1, 2))
        :return:
        """
        if isinstance(node, tables.Group):
            data = self._read_group(node)
        elif isinstance(node, tables.VLArray):
            data = self._read_VLArray(node)
        else:  # for tables.CArray and tables.Array
            if key is not None:
                data = node.__getitem__(key)
            else:
                try:
                    if 'was_mmap' in node._v_attrs._f_list():
                        data = node.read()
                    else:
                        data = node.read()
                except ValueError as e:
                    print("ERROR reading node {}, {}".format(node, str(e)))
                    data = np.zeros(1)
        return data

    def _read_group(self, group):
        if 'issparse' in group._v_attrs._f_list():
            data = self._read_sparse(group)
        else:
            data = {node._v_name: self._read_node(node, None)
                    for node in group._f_iter_nodes()}
        return data

    @staticmethod
    def _read_VLArray(vlarray, asdictionary=True):
        vla = vlarray.read()
        if asdictionary:
            data = {i: values for i, values in enumerate(vla)}
        else:
            data = [(i, v) for i, values in enumerate(vla) for v in values]
        return data

    @staticmethod
    def _read_sparse(group):
        data = group.data.read()
        indices = group.indices.read()
        indptr = group.indptr.read()
        shape = group.shape.read()
        indptr, indices = csr_make_ints(indptr, indices)
        arr = csr_matrix((data, indices, indptr), shape=shape)
        return arr


def open_data_file(filename, mode='a'):
    class DataHandlerContext(object):
        def __init__(self, filename, mode):
            self.filename = filename
            self.handler = None
            self.mode = mode

        def __enter__(self):
            h5f = tables.open_file(self.filename, mode=self.mode)
            self.handler = DataHandler(h5f, print, print)
            return self.handler

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.handler.h5f.close()

    return DataHandlerContext(filename, mode)


