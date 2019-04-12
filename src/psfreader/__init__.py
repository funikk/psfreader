import struct
import io
from psfreader.psfdata import *


class PSFReaderError(ValueError):
    pass


class PSFFile:
    def __init__(self, filename):
        self.filename = filename
        self.fp = open(filename, 'rb')

        self.sections = dict()
        self.types = dict()
        self.properties = dict()
        self.sweep_vars = list()
        self.traces = list()
        self.sweep_value = None
        self.sweep_value_w_var = None
        self.value = None
        self.variables = None
        self.read_points = 0
        self.fp.seek(0, io.SEEK_END)
        self.fsize = self.fp.tell()
        self.fp.seek(0, io.SEEK_SET)
        
        try:
            x = self.read_uint32()
            #if x != 0x0400:
            #    raise ValueError('This file is not a PSF format.')
            self.fp.seek(-12, io.SEEK_END)
            b = self.fp.read(8)
            if b != b'Clarissa':
                raise PSFReaderError('This file is not a PSF format.')
            self.fp.seek(0, io.SEEK_SET)
        except Exception:
            self.has_footer = False
        else:
            self.has_footer = True

    def close(self):
        self.fp.close()

    def read_uint8(self):
        return self.fp.read(1)

    def read_int32(self):
        '''32bit整数を読み込み
        
        uintとintを区別していないためread_uint32()と同等
        '''
        #'''32bit整数を読み込み(リトルエンディアン)'''
        #data = self.fp.read(4)
        #return struct.unpack('<I', data)[0]
        return self.read_uint32()

    def read_uint32(self):
        data = self.fp.read(4)
        return struct.unpack('>I', data)[0]

    def read_double(self):
        data = self.fp.read(8)
        return struct.unpack('>d', data)[0]

    def read_str(self):
        length = self.read_uint32()
        extras = ((length + 3) & ~0x03) - length  # 4byte単位に切り上げたときのパディング
        data = self.fp.read(length)
        self.fp.read(extras)
        return data.decode()

    def unread_uint32(self):
        self.fp.seek(-4, io.SEEK_CUR)

    def read_data(self, t):
        if t == TypeId.INT8:
            return self.read_int32()
        elif t == TypeId.INT32:
            return self.read_int32()
        elif t == TypeId.DOUBLE:
            return self.read_double()
        elif t == TypeId.COMPLEX_DOUBLE:
            r = self.read_double()
            i = self.read_double()
            return complex(r, i)

    def read_data_win(self, array, start, size, t):
        if t == TypeId.INT8:
            for i in range(start, start + size):
                array[i] = self.read_int32()
        elif t == TypeId.INT32:
            for i in range(start, start + size):
                array[i] = self.read_int32()
        elif t == TypeId.DOUBLE:
            length = 8 * size
            dt = np.dtype('>d')
            barray = self.fp.read(length)
            data = np.frombuffer(barray, dtype=dt)
            array[start:start + size] = data
            #for i in range(start, start+size):
            #    array[i] = self.read_double()
        elif t == TypeId.COMPLEX_DOUBLE:
            for i in range(start, start + size):
                re = self.read_double()
                im = self.read_double()
                array[i] = complex(re, im)

    def read_file(self, header_only=False):
        '''
        PSFの全体を読み込み，内部形式に変換する
        '''

        self.completed = True
        if self.has_footer:
            size = self.fsize
            self.fp.seek(self.fsize - 4, io.SEEK_SET)
            datasize = self.read_uint32()

            num_section = (size - datasize - 12) // 8  # //は整数上の除算(端数切り捨て)
            # 12は文字列'Clarissa'の分？
            last_offset = 0
            last_section_num = -1

            toc = size - 12 - num_section * 8  # セクション情報の頭の位置

            sections = dict()
            section_id = -1
            for i in range(num_section):
                self.fp.seek(toc + 8 * i)
                section_id = self.read_uint32()
                section_offset = self.read_uint32()

                if i > 1:  # 2つのセクションの位置の差がサイズである
                    sections[last_section_num].size = section_offset - last_offset
                sections[section_id] = SectionInfo(section_offset, 0)

                last_section_num = section_id
                last_offset = section_offset
            sections[last_section_num].size = size - last_offset  # 最後のセクション
            self.sections = sections

            # セクションごとの前処理
            self.fp.seek(4, io.SEEK_SET)
            self.read_properties()

            if SectionId.TYPE in self.sections:
                self.read_section(SectionId.TYPE)

            if SectionId.SWEEP in self.sections:
                self.read_section(SectionId.SWEEP)

            if SectionId.TRACE in self.sections:
                self.read_section(SectionId.TRACE)

            if header_only:
                return

            if SectionId.VALUE in self.sections:
                self.read_section(SectionId.VALUE)
            else:
                self.value = None
        else:
            self.fp.seek(4, io.SEEK_SET)
            self.read_properties()

            while self.read_section():
                pass

    def read_chunk_preamble(self, chunkid):
        c_id = self.read_uint32()
        if c_id != chunkid:
            raise PSFReaderError('Unexpected ChunkId. Expected: ' + repr(chunkid) + ', Actually: ' + hex(c_id))
        return self.read_uint32()

    def read_section(self, section_num=None):
        if section_num is not None:
            endpos = self.read_section_preamble(section_num)
        else:
            self.fp.seek(-4, io.SEEK_CUR)
            section_num = self.read_uint32()
            if section_num not in set(SectionId):
                return False
            endpos = self.read_chunk_preamble(ChunkId.MAJOR_SECTION)

        if section_num == SectionId.TYPE:
            self.read_types()
            res = True
        elif section_num == SectionId.SWEEP:
            self.read_sweep()
            res = True
        elif section_num == SectionId.TRACE:
            self.read_trace()
            res = True
        elif section_num == SectionId.VALUE:
            if len(self.sweep_vars) == 0:
                self.read_non_sweep_value()
            elif len(self.sweep_vars) != 1:
                raise PSFReaderError('Not supported file format: Sweep variables is more than one.')
            else:
                self.read_sweep_value()
            res = False
        else:
            return False

        self.check_section_end(endpos)

        return res

    def read_section_preamble(self, section):
        sectioninfo = self.sections[section]

        self.fp.seek(sectioninfo.offset, io.SEEK_SET)
        return self.read_chunk_preamble(ChunkId.MAJOR_SECTION)

    def read_properties(self):
        endpos = self.read_chunk_preamble(ChunkId.MAJOR_SECTION)
        self.properties = PSF_Property.read_dictionary(self)
        self.check_section_end(endpos)

    def read_types(self):
        self.types = dict()
        end_sub = self.read_chunk_preamble(ChunkId.MINOR_SECTION)
        while self.fp.tell() < end_sub:
            typedef = PSF_Type()
            typedef.read(self, self.types)

        self.read_index(False)

    def read_index(self, is_trace):
        '''
        一般的に必要ないデータらしく省略
        '''
        pass

    def read_sweep(self):
        self.sweep_vars = list()
        while True:
            s_var = PSF_Variable()
            if s_var.read(self):
                self.sweep_vars.append(s_var)
            else:
                break

    def read_trace(self):
        endsub = self.read_chunk_preamble(ChunkId.MINOR_SECTION)

        valid = True
        self.traces = list()
        while valid and self.fp.tell() < endsub:
            group = PSF_Group()
            valid = group.read(self)
            if valid:
                self.traces.append(group)
            else:
                var = PSF_Variable()
                valid = var.read(self)
                if valid:
                    self.traces.append(var)

        self.read_index(True)

    def read_non_sweep_value(self):
        endsub = self.read_chunk_preamble(ChunkId.MINOR_SECTION)

        valid = True
        res = list()
        while valid and self.fp.tell() < endsub:
            code = self.read_uint32()
            if code == 16:
                var_id = self.read_uint32()
                name = self.read_str()
                psf_type_id = self.read_uint32()
                type_id = self.types[psf_type_id].data_type
                data = self.read_data(type_id)
                prop = PSF_Property.read_dictionary(self)

                var = PSF_Variable()
                var.id = var_id
                var.name = name
                var.type_id = id
                var.prop = prop

                res.append((var, data))

        self.value = self.list_to_map(res)
        self.variables = res

        self.read_index(False)

    def read_sweep_value(self):
        npoints = self.properties['PSF sweep points'].value

        sweep_var = self.sweep_vars[0]
        sweep_type = self.types[sweep_var.type_id].data_type

        if 'PSF window size' in self.properties:
            win_size = self.properties['PSF window size'].value
        else:
            win_size = 0

        if win_size > 0:
            self.read_sweep_value_win(win_size, npoints, sweep_type)
        else:
            self.read_sweep_value_non_win(npoints, sweep_type)

    def read_sweep_value_win(self, win_size, npoints, sweep_type):
        t = typeid_to_dtype(sweep_type)
        sweep = np.empty(npoints, dtype=t)
        value_map = self.array_list_from_trace(npoints, self.traces)
        value, arrays = self.flatten_value(value_map)
        sweep_var_size = typeid_to_size(sweep_type)

        read_points = 0
        while read_points < npoints:
            block_id = self.read_uint32()
            if block_id == ElementId.DATA:
                size = self.read_uint32() & 0x0000ffff

                self.read_data_win(sweep, read_points, size, sweep_type)
                skip_size = win_size - sweep_var_size * size
                for (v, array) in value:
                    self.fp.seek(skip_size, io.SEEK_CUR)  # skip dummy value
                    v.read_data_win(array, read_points, size, self)

                read_points += size
            elif block_id == ElementId.ZEROPAD:
                pad_size = self.read_uint32()
                self.fp.seek(pad_size, io.SEEK_CUR)
            else:
                # raise ValueError('Unexpected data id: ' + str(block_id))
                self.completed = False
                break

        self.read_points = read_points
        self.sweep_value = sweep[0:read_points]
        for (_, a) in arrays.items():
            a.resize(read_points, refcheck=False)
        self.value = arrays
        self.variables = value

    def read_sweep_value_non_win(self, npoints, sweep_type):
        t = typeid_to_dtype(sweep_type)
        sweep = np.empty(npoints, dtype=t)
        sweep_var = self.sweep_vars[0]
        value_map = self.array_list_from_trace_group(npoints, self.traces)
        i = -1

        try:
            while self.fp.tell() < self.fsize:
                elemid = self.read_uint32()  # check x == ElementId.DATA
                var_id = self.read_uint32()  # check x == self.sweep_vars[0].id

                if elemid == ElementId.DATA and var_id == sweep_var.id:
                    i += 1
                    sweep[i] = self.read_data(sweep_type)
                elif elemid == ElementId.GROUP:
                    (v, array) = value_map[var_id]
                    v.read_data(array, i, self)
                else:
                    break
        except struct.error:  # to be fixed Exception type
            self.completed = False

        n = i + 1
        self.sweep_value = sweep[:n]
        self.read_points = n
        self.variables, self.value, self.sweep_value_w_var = self.flatten_value_group(value_map, self.sweep_value)

    def array_list_from_trace(self, npoints, trace):
        return [(x, x.to_array(npoints, self)) for x in trace]

    def array_list_from_trace_group(self, npoints, trace):
        return {x.id: (x, x.to_array_group(npoints, self)) for x in trace}

    def check_section_end(self, endpos):
        self.fp.seek(endpos, io.SEEK_SET)

    def flatten_value(self, value_map):
        variables = list()
        arrays = dict()
        for (v, a) in value_map:
            variables.extend(v.flatten_value(a, arrays))
        return variables, arrays

    def flatten_value_group(self, value_map, sweep_data):
        arrays = dict()
        sweeps = dict()
        for (v, a) in value_map.values():
            v.flatten_value_group(a, arrays, sweeps, sweep_data)
        variables = self.trace_to_signal_names()
        return variables, arrays, sweeps

    def list_to_map(self, lst):
        return {v.name: a for (v, a) in lst}

    def trace_to_signal_names(self):
        signals = list()
        for v in self.traces:
            signals.extend(v.to_signal_list())

        return signals

class PSFReader:
    '''
    Parameter-Storage Format Reader for python.
    '''

    def __init__(self, filename, header_only=False):
        self.psf = PSFFile(filename)
        self.psf.read_file(header_only=header_only)

    def get_header_properties(self):
        '''Return a dictionary of properties'''
        return {key: self.psf.properties[key].value for key in self.psf.properties}

    def get_signal_names(self):
        '''Return a list of signal names in this file'''
        return [v.name for (v, _) in self.psf.variables]

    def is_swept(self):
        return not (self.psf.sweep_vars is None)

    def get_nsweep(self):
        '''Return a number of sweep variables
        
        But this library does not support two sweep variables or higher'''
        return self.psf.properties['PSF sweeps'].value

    def get_sweep_param_name(self):
        '''Return a name of the sweep variable
        
        e.g., 'frequency', 'time'
        '''
        return self.psf.sweep_vars[0].name

    def get_sweep_npoints(self):
        '''Return a length of vectors'''
        return self.psf.properties['PSF sweep points'].value

    def get_sweep_values(self):
        '''Return the value of the sweep variable'''
        return self.psf.sweep_value

    def get_signal_types(self, name):
        '''Return the TypeId of the signal'''
        for v in self.psf.sweep_vars:
            if v.name == name:
                return TypeId(self.psf.types[v.type_id].data_type)

        for (v, _) in self.psf.variables:
            if v.name == name:
                return TypeId(self.psf.types[v.type_id].data_type)

        return None

    def get_signal_units(self, name):
        '''Return the Units of the signal'''
        for v in self.psf.sweep_vars:
            if v.name == name:
                prop = v.prop
                if 'units' in prop:
                    return prop['units'].value
                else:
                    return None

        for (v, _) in self.psf.variables:
            if v.name == name:
                prop = self.psf.types[v.type_id].prop
                if 'units' in prop:
                    return prop['units'].value
                else:
                    return None

        return None

    def get_signal(self, name):
        '''Return the signal value and sweep value(scalar or vector)'''
        if name in self.psf.value:
            return self.psf.value[name]
        else:
            return None

    def get_sweep_values_with_var(self, name):
        if self.psf.sweep_value_w_var is not None:
            if name in self.psf.sweep_value_w_var:
                return self.psf.sweep_value_w_var[name]
            else:
                return None
        elif self.psf.sweep_value is not None:
            return self.psf.sweep_value
        else:
            return None

    def get_read_npoints(self):
        ''' Return read sample length'''
        return self.psf.read_points

    def is_wellformed(self):
        return self.psf.has_footer and self.psf.completed
