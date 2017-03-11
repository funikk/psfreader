from enum import IntEnum
import numpy as np


class TypeId(IntEnum):
    INT8 = 0x01
    STRING = 0x02
    ARRAY = 0x03
    INT32 = 0x05
    DOUBLE = 0x0b
    COMPLEX_DOUBLE = 0x0c
    STRUCT = 0x10
    TUPLE = 0x12 # type listのconsの意味？
    
def typetype_to_dtype(t):
    if t == TypeId.INT8 :
        return np.int8
    elif t == TypeId.INT32:
        return np.int32
    elif t == TypeId.DOUBLE:
        return np.float
    elif t == TypeId.COMPLEX_DOUBLE:
        return np.complex_
    else:
        raise ValueError('Cannot to be a element of array: Type ' + str(TypeId(t)))

def typeid_to_size(t):
    if t == TypeId.INT8 :
        return 4
    elif t == TypeId.INT32:
        return 4
    elif t == TypeId.DOUBLE:
        return 8
    elif t == TypeId.COMPLEX_DOUBLE:
        return 16
    else:
        raise ValueError('Cannot to be a element of array: Type ' + str(TypeId(t)))

class SectionId(IntEnum):
    HEADER = 0
    TYPE = 1
    SWEEP = 2
    TRACE = 3
    VALUE = 4

class ChunkId(IntEnum):
    MAJOR_SECTION = 0x15
    MINOR_SECTION = 0x16

class PropertyTypeId(IntEnum):
    STRING = 0x21
    INT = 0x22
    DOUBLE = 0x23

class ElementId(IntEnum):
    DATA = 0x10
    GROUP = 0x11
    ZEROPAD = 0x14

class SectionInfo:
    def __init__(self, offset, size):
        self.offset = offset
        self.size = size

    def __repr__(self):
        return 'SectionInfo(offset: ' + repr(self.offset) + ', size: ' + repr(self.size) + ')'

class PSF_Property:
    def __init__(self):
        self.name =''
        self.type = 0
        self.value = 0

    def __str__(self):
        return 'PSF_Property(name: ' + str(self.name) + ', value: ' + str(self.value) + ')'

    def __repr__(self):
        return 'PSF_Property(name: ' + str(self.name) + ', value: ' + str(self.value) + ')'

    def read(self, psffile):
        p_type = psffile.read_uint32()
        self.type = p_type
        
        if p_type == PropertyTypeId.STRING:
            self.name = psffile.read_str()
            self.value = psffile.read_str()
            return True
        elif p_type == PropertyTypeId.INT:
            self.name = psffile.read_str()
            self.value = psffile.read_int32()
            return True
        elif p_type == PropertyTypeId.DOUBLE:
            self.name = psffile.read_str()
            self.value = psffile.read_double()
            return True
        else:
            # raise ValueError('Unexpected Property type number: ' + str(p_type))
            psffile.unread_uint32()
            return False
    
    @staticmethod
    def read_dictionary(psffile):
        properties = dict()
        while True:
            p = PSF_Property()
            if p.read(psffile):
                properties[p.name] = p
            else:
                break

        return properties
 
class PSF_Type:
    def __init__(self):
        self.id = 0
        self.name = ''
        self.arry_type = 0
        self.data_type = 0
        self.typelist = list()
        self.prop = None

    def read(self, psffile, typemap):
        code = psffile.read_uint32()
        if code != ElementId.DATA:
            # raise ValueError('Unexpected Type Type: ' + str(code))
            return False

        self.id = psffile.read_uint32()
        self.name = psffile.read_str()
        self.arry_type = psffile.read_uint32()
        self.data_type = psffile.read_uint32()

        if self.data_type == TypeId.STRUCT:
            self.read_type_list(psffile, typemap)

        self.prop = PSF_Property.read_dictionary(psffile)

        typemap[self.id] = self
        return True

    def read_type_list(self, psffile, typemap):
        while True:
            code = psffile.read_uint32()
            if code != TypeId.TUPLE:
                psffile.unread_uint32()
                break
            
            typedef = PSF_Type()
            valid = typedef.read(psffile, typemap)
            if valid:
                self.typelist.append(typedef)
                typemap[typedef.id] = typedef
            else:
                break

class PSF_Variable:
    def __init__(self):
        self.id = 0
        self.name = ''
        self.type_id = 0
        self.prop = None
        self.is_group = False
    
    def read(self, psffile):
        code = psffile.read_uint32()
        if code != ElementId.DATA:
            psffile.unread_uint32()
            return False
        
        self.id = psffile.read_uint32()
        self.name = psffile.read_str()
        self.type_id = psffile.read_uint32()
        self.prop = PSF_Property.read_dictionary(psffile)
        
        return True

    def __repr__(self):
        return 'Var(id: ' + repr(self.id) + ' name: ' + self.name + ', type_id:' + repr(self.type_id) + ', ' + repr(self.prop) + ')'

    def to_array(self, npoints, psffile):
        psf_type = psffile.types[self.type_id].data_type
        dtype = typetype_to_dtype(psf_type)
        return np.empty(npoints, dtype=dtype)

    def read_data(self, array, i, psffile):
        array[i] = psffile.read_data(psffile.types[self.type_id].data_type)

    def read_data_win(self, array, start, size, psffile):
        psffile.read_data_win(array, start, size, psffile.types[self.type_id].data_type)

    def flatten_value(self, a):
        return [(self, a)] 

class PSF_Group:
    def __init__(self):
        self.id = 0
        self.name = ''
        self.vars = None
        self.is_group = True

    def __repr__(self):
        return 'Group(id: ' + repr(self.id) + ', names: ' + self.name + ', vars: ' + repr(self.vars) +')'

    def read(self, psffile):
        code = psffile.read_uint32()
        if code != ElementId.GROUP:
            psffile.unread_uint32()
            return False
        
        self.id = psffile.read_uint32()
        self.name = psffile.read_str()
        length = psffile.read_uint32()

        self.vars = list()
        for i in range(length):
            v = PSF_Variable()
            if v.read(psffile):
                self.vars.append(v)
            else:
                raise ValueError('Group length is '+ str(length) + ', but actually ' + str(i))
        
        return True

    def to_array(self, npoints, psffile):
        return [(x, x.to_array(npoints, psffile)) for x in self.vars]

    def read_data(self, array, i, psffile):
        for (v, ary) in array:
            v.read_data(ary, i, psffile)

    def read_data_win(self, array, start, size, psffile):
        for (v, ary) in array:
            v.read_data_win(ary, start, size, psffile)

    def flatten_value(self, a):
        res = list()
        for (v, ary) in a:
            res.extend(v.flatten_value(ary))
        return res
