from tables import IsDescription, Int32Col, Float64Col, StringCol, BoolCol

name_size = 64
coord_size = 256
eqs_size = 512
units_size = 32

class Quantity(IsDescription):
    value = Float64Col()
    units = StringCol(units_size)

class VariantType(IsDescription):
    # vartype can be one of: strcode, float, integer, array, table
    vartype = StringCol(name_size)
    # array, strcode, and table need to store a string in varstring
    #   - strcode is the case where executable code is specified in place of a value
    #   - table stores the name of the referenced table group
    #   - array stores the name of the referenced array
    varstring = StringCol(eqs_size)
    # Both arrays and single values have units
    varint = Int32Col()
    varflt = Float64Col()
    units = StringCol(units_size)

class NamedVariantType(IsDescription):
    name = StringCol(name_size)
    vartype = VariantType()

class TimedVariantType(IsDescription):
    namedvartype = NamedVariantType()
    dt = Quantity()

class NeuronGroup(IsDescription):
    N = Int32Col()
    name = StringCol(name_size)
    model = StringCol(eqs_size)
    method = StringCol(name_size)
    threshold = StringCol(name_size)
    reset = StringCol(name_size)
    refractory = StringCol(name_size)

class Synapses(IsDescription):
    name = StringCol(name_size)
    source = StringCol(name_size)
    target = StringCol(name_size)
    model = StringCol(eqs_size)
    method = StringCol(name_size)
    pre = StringCol(eqs_size)
    post = StringCol(eqs_size)
    connect = StringCol(name_size)

class Subgroup(IsDescription):
    super = StringCol(name_size)
    name = StringCol(name_size)
    start = Int32Col()
    size = Int32Col()

# class Connection(IsDescription):
#     name = StringCol(name_size)
#     popname_pre = StringCol(name_size)
#     popname_post = StringCol(name_size)
#     synapse = StringCol(name_size)
#     connectivity = VariantType()
#     delays = VariantType()

class LinkedRanges(IsDescription):
    coord_a = StringCol(coord_size)
    coord_b = StringCol(coord_size)

class ParamSpaceCoordinate(IsDescription):
    coord = StringCol(coord_size)
    units = StringCol(units_size)
    column = Int32Col()
    alias = BoolCol()

class AliasedParameters(IsDescription):
    name = StringCol(name_size)
    issparse = BoolCol()
