import struct
import io
import os
import mmap
import numpy as np
from psfreader.psfdata import (TypeId, SectionInfo, SectionId, ChunkId, ElementId,
                               PSF_Property, PSF_Variable, PSF_Type, PSF_Group,
                               typeid_to_dtype, typeid_to_size, get_big_endian_dtype)


class PSFReaderError(ValueError):
    pass


class PSFFile:
    """PSFファイルを読み込んで内部形式に変換するクラス。

    PSFファイルをバイナリで読み込み、セクション、型定義、スイープ情報、
    トレース定義および実データを解析して保持します。
    """

    def __init__(self, filename: str, lazy: bool = False, ignore_footer: bool = False):
        """PSFファイルを開いて初期化する。

        Args:
            filename (str): PSFファイルのパス。
            lazy (bool): 真の場合、データ部(VALUEセクション)を即座に読み込まない。
            ignore_footer (bool): 真の場合、フッター(TOC)を無視してファイルを走査する。
        """
        self.filename = filename
        self.fp = open(filename, 'rb')
        self.lazy = lazy
        self.ignore_footer = ignore_footer
        self.mmap_obj = None
        self.mmap_size = 0

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

        if self.ignore_footer:
            self.has_footer = False
        else:
            try:
                _ = self.read_uint32()
                # if x != 0x0400:
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
        """ファイルを閉じる。

        Returns:
            None
        """
        if hasattr(self, 'mmap_obj') and self.mmap_obj is not None:
            self.mmap_obj.close()
            self.mmap_obj = None
        self.fp.close()

    def read_uint8(self):
        """1バイトのデータを読み込む。

        Returns:
            bytes: 読み込んだ1バイト。
        """
        return self.fp.read(1)

    def read_int32(self):
        """32ビット整数を読み込む。

        uintとintを区別せず、`read_uint32()`と同等の処理を行う。

        Returns:
            int: 読み込んだ32ビット整数。
        """
        return self.read_uint32()

    def read_uint32(self):
        """4バイトの符号なし整数をビッグエンディアンで読み込む。

        Returns:
            int: 読み込んだ32ビット符号なし整数。
        """
        data = self.fp.read(4)
        return struct.unpack('>I', data)[0]

    def read_double(self):
        """8バイトの倍精度浮動小数点数を読み込む。

        Returns:
            float: 読み込んだ倍精度浮動小数点数。
        """
        data = self.fp.read(8)
        return struct.unpack('>d', data)[0]

    def read_float(self):
        """4バイトの単精度浮動小数点数を読み込む。

        Returns:
            float: 読み込んだ単精度浮動小数点数。
        """
        data = self.fp.read(4)
        return struct.unpack('>f', data)[0]

    def read_complex_float(self):
        """8バイトの単精度複素数（実数+虚数）を読み込む。

        Returns:
            complex: 読み込んだ複素数。
        """
        data = self.fp.read(8)
        r, i = struct.unpack('>2f', data)
        return complex(r, i)

    def skip_data(self, t: TypeId):
        """指定された型に従ってデータ1つ分スキップする。

        Args:
            t (TypeId): スキップするデータの型ID。

        Returns:
            None
        """
        self.fp.seek(typeid_to_size(t), io.SEEK_CUR)

    def read_str(self):
        """長さ付き文字列を読み込む。

        Returns:
            str: デコードした文字列。
        """
        length = self.read_uint32()
        extras = ((length + 3) & ~0x03) - length  # 4byte単位に切り上げたときのパディング
        data = self.fp.read(length)
        self.fp.read(extras)
        return data.decode()

    def unread_uint32(self):
        """直前に読み込んだ4バイト分のファイル位置を戻す。

        Returns:
            None
        """
        self.fp.seek(-4, io.SEEK_CUR)

    def read_data(self, t: TypeId):
        """指定された型に従って1つのデータ値を読み込む。

        Args:
            t (TypeId): 読み込むデータの型ID。

        Returns:
            int|float|complex: 読み込んだデータ。
        """
        if t == TypeId.INT8:
            return self.read_int32()
        elif t == TypeId.INT32:
            return self.read_int32()
        elif t == TypeId.FLOAT:
            return self.read_float()
        elif t == TypeId.COMPLEX_FLOAT:
            return self.read_complex_float()
        elif t == TypeId.DOUBLE:
            return self.read_double()
        elif t == TypeId.COMPLEX_DOUBLE:
            r = self.read_double()
            i = self.read_double()
            return complex(r, i)

    def read_data_win(self, array: np.ndarray, start: int, size: int, t: TypeId):
        """窓付きデータを連続して読み込み、配列に格納する。

        Args:
            array (numpy.ndarray): 読み込み先の配列。Noneの場合はスキップする。
            start (int): 配列に書き込む開始位置。
            size (int): 読み込む要素数。
            t (TypeId): 読み込むデータの型ID。

        Returns:
            None
        """
        if t == TypeId.INT8:
            length = 4 * size
            if array is not None:
                dt = np.dtype('>i4')
                barray = self.fp.read(length)
                data = np.frombuffer(barray, dtype=dt)
                array[start:start + size] = data
            else:
                self.fp.seek(length, io.SEEK_CUR)
        elif t == TypeId.INT32:
            length = 4 * size
            if array is not None:
                dt = np.dtype('>i4')
                barray = self.fp.read(length)
                data = np.frombuffer(barray, dtype=dt)
                array[start:start + size] = data
            else:
                self.fp.seek(length, io.SEEK_CUR)
        elif t == TypeId.FLOAT:
            length = 4 * size
            if array is not None:
                dt = np.dtype('>f4')
                barray = self.fp.read(length)
                data = np.frombuffer(barray, dtype=dt)
                array[start:start + size] = data
            else:
                self.fp.seek(length, io.SEEK_CUR)
        elif t == TypeId.COMPLEX_FLOAT:
            length = 8 * size
            if array is not None:
                dt = np.dtype('>c8')
                barray = self.fp.read(length)
                data = np.frombuffer(barray, dtype=dt)
                array[start:start + size] = data
            else:
                self.fp.seek(length, io.SEEK_CUR)
        elif t == TypeId.DOUBLE:
            length = 8 * size
            if array is not None:
                dt = np.dtype('>d')
                barray = self.fp.read(length)
                data = np.frombuffer(barray, dtype=dt)
                array[start:start + size] = data
            else:
                self.fp.seek(length, io.SEEK_CUR)
        elif t == TypeId.COMPLEX_DOUBLE:
            length = 16 * size
            if array is not None:
                dt = np.dtype('>c16')
                barray = self.fp.read(length)
                data = np.frombuffer(barray, dtype=dt)
                array[start:start + size] = data
            else:
                self.fp.seek(length, io.SEEK_CUR)

    def read_file(self, header_only=False):
        """PSFファイル全体を読み込み、内部データ構造に変換する。

        Args:
            header_only (bool): True の場合はヘッダー情報のみ読み込む。

        Returns:
            None
        """

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
        else:
            self.read_sections()

        # セクションごとの読み込み処理
        # HEADER セクションのプロパティ読み込み
        if SectionId.HEADER in self.sections:
            self.fp.seek(self.sections[SectionId.HEADER].offset, io.SEEK_SET)
            self.read_properties()
        else:
            # フォールバック: 通常はオフセット 4 から開始
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
            if self.lazy:
                # 信号名のリストだけは作成しておく
                self.variables = self.trace_to_signal_names()
                self.value = {}
            else:
                self.read_section(SectionId.VALUE)
        else:
            self.value = None

    def read_sections(self):
        """セクションのプリアンブルを走査して、全セクションの情報を取得する。

        フッター（TOC）がない場合でも、ファイルを走査して各セクションの
        オフセットとサイズを特定し、`self.sections` に格納します。
        HEADERセクションはオフセット4から開始すると仮定し、以降のセクションは
        チャンクの4バイト前に格納されているセクションIDを読み取って識別します。

        Returns:
            None
        """
        self.sections = {}

        # 1. HEADER セクション (SectionId.HEADER = 0)
        # 通常、オフセット 4 から MAJOR_SECTION チャンクが開始する
        self.fp.seek(4, io.SEEK_SET)
        try:
            # HEADER セクションのチャンク情報を読み込む
            # read_chunk_preamble は ChunkId と EndPos を読み取り、EndPos を返す
            end_pos = self.read_chunk_preamble(ChunkId.MAJOR_SECTION)
            self.sections[SectionId.HEADER] = SectionInfo(4, end_pos - 4)
            curr_pos = end_pos
        except (PSFReaderError, struct.error):
            # HEADER が読み込めない場合は終了
            return

        # 2. 以降のセクションを走査 (TYPE, SWEEP, TRACE, VALUE など)
        # 各セクションの ID は、そのセクションの MAJOR_SECTION チャンクの 4 バイト前にある
        while curr_pos < self.fsize:
            # セクション ID を読み取るために 4 バイト戻る
            self.fp.seek(curr_pos - 4, io.SEEK_SET)
            try:
                s_id_raw = self.read_uint32()
            except struct.error:
                break

            # 有効な SectionId か確認
            if s_id_raw not in [s.value for s in SectionId]:
                break

            s_id = SectionId(s_id_raw)

            # MAJOR_SECTION チャンクの開始位置へシーク
            self.fp.seek(curr_pos, io.SEEK_SET)
            try:
                next_end_pos = self.read_chunk_preamble(ChunkId.MAJOR_SECTION)
            except (PSFReaderError, struct.error):
                break

            # セクション情報を保存
            self.sections[s_id] = SectionInfo(curr_pos, next_end_pos - curr_pos)

            # 次のセクションへ
            if next_end_pos <= curr_pos:
                # 異常な EndPos による無限ループを防止
                break
            curr_pos = next_end_pos

            if s_id == SectionId.VALUE:
                # VALUE セクション以降は通常スキャン不要
                break

    def read_chunk_preamble(self, chunkid):
        """チャンクの先頭を読み込み、次のセクション終了位置を取得する。

        Args:
            chunkid (ChunkId): 期待するチャンクID。

        Returns:
            int: このチャンクの終了位置。

        Raises:
            PSFReaderError: 予期しないチャンクIDの場合。
        """
        c_id = self.read_uint32()
        if c_id != chunkid:
            raise PSFReaderError('Unexpected ChunkId. Expected: ' + repr(chunkid) + ', Actually: ' + hex(c_id))
        return self.read_uint32()

    def load_signals(self, target_names=None):
        """要求された信号を読み込む。

        Args:
            target_names (list|None): 読み込む信号名のリスト。Noneの場合は全ての信号を読み込む。
        """
        if SectionId.VALUE in self.sections:
            self.read_section(SectionId.VALUE, target_names=target_names)

    def read_section(self, section_num=None, target_names=None):
        """指定したセクションを読み込む。

        Args:
            section_num (int|None): 読み込むセクションID。None の場合は現在位置から自動判定する。
            target_names (list|None): VALUEセクション読み込み時の対象信号名リスト。

        Returns:
            bool: 次のセクション読み込みを続けるべきかどうか。
        """
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
                self.read_sweep_value(target_names=target_names)
            res = False
        else:
            return False

        self.check_section_end(endpos)

        return res

    def read_section_preamble(self, section):
        """指定したセクションの先頭位置に移動し、チャンク先頭を読み込む。

        Args:
            section (int): セクションID。

        Returns:
            int: チャンク終了位置。
        """
        sectioninfo = self.sections[section]

        self.fp.seek(sectioninfo.offset, io.SEEK_SET)
        return self.read_chunk_preamble(ChunkId.MAJOR_SECTION)

    def read_properties(self):
        """HEADER セクション内のプロパティ辞書を読み込む。

        Returns:
            None
        """
        endpos = self.read_chunk_preamble(ChunkId.MAJOR_SECTION)
        self.properties = PSF_Property.read_dictionary(self)
        self.check_section_end(endpos)

    def read_types(self):
        """TYPE セクションを読み込み、型定義を構築する。

        Returns:
            None
        """
        self.types = dict()
        end_sub = self.read_chunk_preamble(ChunkId.MINOR_SECTION)
        while self.fp.tell() < end_sub:
            typedef = PSF_Type()
            typedef.read(self, self.types)

        self.read_index(False)

    def read_index(self, is_trace):
        """インデックス情報を読み飛ばす。

        Args:
            is_trace (bool): 読み飛ばすセクションが TRACE かどうか。

        Returns:
            None
        """
        pass

    def read_sweep(self):
        """SWEEP セクションを読み込み、スイープ変数を構築する。

        Returns:
            None
        """
        self.sweep_vars = list()
        while True:
            s_var = PSF_Variable()
            if s_var.read(self):
                self.sweep_vars.append(s_var)
            else:
                break

    def read_trace(self):
        """TRACE セクションを読み込み、トレースとグループ定義を構築する。

        Returns:
            None
        """
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
        """VALUE セクションの非スイープ値を読み込む。

        Returns:
            None
        """
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

    def read_sweep_value(self, target_names=None):
        """VALUE セクションのスイープデータを読み込む。

        Args:
            target_names (list|None): 読み込む信号名のリスト。
        """
        npoints = self.properties['PSF sweep points'].value

        sweep_var = self.sweep_vars[0]
        sweep_type = self.types[sweep_var.type_id].data_type

        if 'PSF window size' in self.properties:
            win_size = self.properties['PSF window size'].value
        else:
            win_size = 0

        if win_size > 0:
            self.read_sweep_value_win(win_size, npoints, sweep_type, target_names)
        else:
            self.read_sweep_value_non_win(npoints, sweep_type, target_names)

    def read_sweep_value_win(self, win_size, npoints, sweep_type, target_names=None):
        """ウィンドウ付きスイープデータを読み込む。

        Args:
            win_size (int): ウィンドウサイズ（バイト単位）。
            npoints (int): 期待されるスイープ点数。
            sweep_type (TypeId): スイープ変数の型ID。
            target_names (list|None): 読み込む信号名のリスト。
        """
        if target_names is not None:
            targets = set(target_names)
        else:
            targets = None

        t = typeid_to_dtype(sweep_type)
        if self.sweep_value is None:
            self.sweep_value = np.empty(npoints, dtype=t)
            sweep_to_read = self.sweep_value
        else:
            sweep_to_read = None  # すでに読み込み済みならファイル上はスキップする

        value_map = self.array_list_from_trace(npoints, self.traces, targets)
        value, arrays = self.flatten_value(value_map)
        sweep_var_size = typeid_to_size(sweep_type)

        read_points = 0
        while read_points < npoints:
            block_id = self.read_uint32()
            if block_id == ElementId.DATA:
                size = self.read_uint32() & 0x0000ffff

                self.read_data_win(sweep_to_read, read_points, size, sweep_type)
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
        if sweep_to_read is not None:
            self.sweep_value = self.sweep_value[0:read_points]

        for (_, a) in arrays.items():
            if a is not None:
                a.resize(read_points, refcheck=False)

        if self.value is None:
            self.value = {}
        self.value.update(arrays)
        self.variables = value

    def _calculate_binary_layout(self):
        offsets = {}  # {var_name: (offset_bytes, be_dtype, size_bytes)}
        current_offset = 0

        # 1. スイープ変数
        current_offset += 8  # header (elemid + var_id)
        sweep_var = self.sweep_vars[0]
        sweep_type = self.types[sweep_var.type_id].data_type
        be_dtype = get_big_endian_dtype(sweep_type)
        offsets[sweep_var.name] = (current_offset, be_dtype, typeid_to_size(sweep_type))
        current_offset += typeid_to_size(sweep_type)

        # 2. 各トレース/グループ
        for t in self.traces:
            current_offset += 8  # header
            if t.is_group:  # PSF_Group
                for v in t.vars:
                    v_type = self.types[v.type_id].data_type
                    be_dtype = get_big_endian_dtype(v_type)
                    offsets[v.name] = (current_offset, be_dtype, typeid_to_size(v_type))
                    current_offset += typeid_to_size(v_type)
            else:  # PSF_Variable
                t_type = self.types[t.type_id].data_type
                be_dtype = get_big_endian_dtype(t_type)
                offsets[t.name] = (current_offset, be_dtype, typeid_to_size(t_type))
                current_offset += typeid_to_size(t_type)

        block_size = current_offset
        return block_size, offsets

    def read_sweep_value_non_win(self, npoints, sweep_type, target_names=None):
        """ウィンドウなしスイープデータを読み込む。

        Args:
            npoints (int): 期待されるスイープ点数。
            sweep_type (TypeId): スイープ変数の型ID。
            target_names (list|None): 読み込む信号名のリスト。
        """
        # 1. レイアウトの解析
        block_size, offsets = self._calculate_binary_layout()

        # 2. mmap オブジェクトの作成/更新（ファイル伸長に対応）
        current_size = os.path.getsize(self.filename)
        if self.mmap_obj is None or current_size > self.mmap_size:
            if self.mmap_obj is not None:
                self.mmap_obj.close()
            self.mmap_obj = mmap.mmap(self.fp.fileno(), 0, access=mmap.ACCESS_READ)
            self.mmap_size = current_size

        # 3. VALUE セクションのデータ部をマッピング
        start_offset = self.sections[SectionId.VALUE].offset + 8
        raw_view = memoryview(self.mmap_obj)[start_offset:]
        raw_array = np.frombuffer(raw_view, dtype=np.uint8)

        # 現在ロード可能な点数の算出 (書き込み中の途中ファイルにも完璧に対応)
        actual_points = min(npoints, len(raw_array) // block_size)
        self.read_points = actual_points

        # 2D 配列にリシェイプ
        reshaped = raw_array[:actual_points * block_size].reshape(actual_points, block_size)

        # 4. スイープ変数の取得
        sweep_var = self.sweep_vars[0]
        if sweep_var.name in offsets:
            offset, be_dtype, size = offsets[sweep_var.name]
            col_bytes = reshaped[:, offset:offset + size]
            native_dtype = typeid_to_dtype(sweep_type)
            self.sweep_value = col_bytes.copy().view(be_dtype).reshape(-1).astype(native_dtype)

        # 5. 要求されたシグナルの取得 (lazy load 時は対象の1件、通常時は全件)
        if target_names is not None:
            targets = set(target_names)
        else:
            targets = None

        arrays = {}
        sweep_value_w_var = {}

        for t in self.traces:
            if t.is_group:
                for v in t.vars:
                    if targets is None or v.name in targets:
                        if v.name in offsets:
                            offset, be_dtype, size = offsets[v.name]
                            col_bytes = reshaped[:, offset:offset + size]
                            v_type = self.types[v.type_id].data_type
                            native_dtype = typeid_to_dtype(v_type)
                            arrays[v.name] = col_bytes.copy().view(be_dtype).reshape(-1).astype(native_dtype)
                            sweep_value_w_var[v.name] = self.sweep_value
            else:
                if targets is None or t.name in targets:
                    if t.name in offsets:
                        offset, be_dtype, size = offsets[t.name]
                        col_bytes = reshaped[:, offset:offset + size]
                        t_type = self.types[t.type_id].data_type
                        native_dtype = typeid_to_dtype(t_type)
                        arrays[t.name] = col_bytes.copy().view(be_dtype).reshape(-1).astype(native_dtype)
                        sweep_value_w_var[t.name] = self.sweep_value

        variables = self.trace_to_signal_names()

        if self.value is None:
            self.value = {}
        self.value.update(arrays)
        self.variables = variables

        if self.sweep_value_w_var is None:
            self.sweep_value_w_var = {}
        self.sweep_value_w_var.update(sweep_value_w_var)

    def array_list_from_trace(self, npoints, trace, target_names=None):
        """TRACE リストから配列を生成する。

        Args:
            npoints (int): 生成する配列の長さ。
            trace (list): PSF_Variable または PSF_Group のリスト。
            target_names (set|None): 読み込み対象の信号名セット。

        Returns:
            list: (要素, 配列) のタプルリスト。
        """
        return [(x, x.to_array(npoints, self, target_names)) for x in trace]

    def array_list_from_trace_group(self, npoints, trace, target_names=None):
        """TRACE グループから配列を生成する。

        Args:
            npoints (int): 生成する配列の長さ。
            trace (list): PSF_Group または PSF_Variable のリスト。
            target_names (set|None): 読み込み対象の信号名セット。

        Returns:
            dict: 変数ID をキーとした (要素, 配列) の辞書。
        """
        # 非ウィンドウ形式の場合、全てのIDを網羅する必要がある（読み飛ばすため）
        return {x.id: (x, x.to_array_group(npoints, self, target_names)) for x in trace}

    def check_section_end(self, endpos):
        """セクション終端位置にシークする。

        Args:
            endpos (int): 移動先のファイル位置。

        Returns:
            None
        """
        self.fp.seek(endpos, io.SEEK_SET)

    def flatten_value(self, value_map):
        """TRACE 値をフラット化して配列辞書に展開する。

        Args:
            value_map (list): (要素, 配列) のタプルリスト。

        Returns:
            tuple: (variables, arrays) を返す。
        """
        variables = list()
        arrays = dict()
        for (v, a) in value_map:
            variables.extend(v.flatten_value(a, arrays))
        return variables, arrays

    def flatten_value_group(self, value_map, sweep_data):
        """TRACE グループ値をフラット化して辞書に展開する。

        Args:
            value_map (dict): 変数ID をキーとした (要素, 配列) の辞書。
            sweep_data (numpy.ndarray): スイープ値のデータ。

        Returns:
            tuple: (variables, arrays, sweeps) を返す。
        """
        arrays = dict()
        sweeps = dict()
        for (v, a) in value_map.values():
            v.flatten_value_group(a, arrays, sweeps, sweep_data)
        variables = self.trace_to_signal_names()
        return variables, arrays, sweeps

    def list_to_map(self, lst):
        """変数リストを名前をキーとする辞書に変換する。

        Args:
            lst (list): (変数, 値) のタプルリスト。

        Returns:
            dict: 変数名をキーとした辞書。
        """
        return {v.name: a for (v, a) in lst}

    def trace_to_signal_names(self):
        """TRACE 定義から信号名リストを生成する。

        Returns:
            list: 信号名のリスト。
        """
        signals = list()
        for v in self.traces:
            signals.extend(v.to_signal_list())

        return signals


class PSFReader:
    '''
    Parameter-Storage Format Reader for python.
    '''

    def __init__(self, filename, header_only=False, lazy=False, ignore_footer=False):
        self.lazy = lazy
        self.psf = PSFFile(filename, lazy=lazy, ignore_footer=ignore_footer)
        self.psf.read_file(header_only=header_only)

    def load_all_signals(self):
        '''Read all signals into memory'''
        self.psf.load_signals(target_names=None)

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
        if self.psf.sweep_value is None and self.lazy:
            self.psf.load_signals(target_names=[])
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
        if self.psf.value is not None and name in self.psf.value:
            return self.psf.value[name]

        if self.lazy:
            self.psf.load_signals(target_names=[name])
            if self.psf.value is not None and name in self.psf.value:
                return self.psf.value[name]

        return None

    def get_sweep_values_with_var(self, name):
        if self.psf.sweep_value_w_var is not None:
            if name in self.psf.sweep_value_w_var:
                return self.psf.sweep_value_w_var[name]

        # sweep_value_w_varがNoneでも、lazyモードならロードを試みる
        if self.lazy and (self.psf.sweep_value_w_var is None or name not in self.psf.sweep_value_w_var):
            self.psf.load_signals(target_names=[name])
            if self.psf.sweep_value_w_var is not None and name in self.psf.sweep_value_w_var:
                return self.psf.sweep_value_w_var[name]

        if self.psf.sweep_value is not None:
            return self.psf.sweep_value
        elif self.lazy:
            self.psf.load_signals(target_names=[])
            return self.psf.sweep_value
        else:
            return None

    def get_read_npoints(self):
        ''' Return read sample length'''
        return self.psf.read_points

    def is_wellformed(self):
        return self.psf.has_footer and self.psf.completed
