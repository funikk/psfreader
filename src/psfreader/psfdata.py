from __future__ import annotations

from enum import IntEnum
import numpy as np
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from . import PSFFile


class TypeId(IntEnum):
    INT8 = 0x01
    STRING = 0x02
    ARRAY = 0x03
    INT32 = 0x05
    FLOAT = 0x09
    COMPLEX_FLOAT = 0x0a
    DOUBLE = 0x0b
    COMPLEX_DOUBLE = 0x0c
    STRUCT = 0x10
    TUPLE = 0x12  # type list of cons representation


def typeid_to_dtype(t: TypeId):
    """TypeId に対応する NumPy データ型を返す。

    Args:
        t (TypeId): PSF のデータ型 ID。

    Returns:
        type: NumPy のデータ型。

    Raises:
        ValueError: サポートされていない型 ID の場合。
    """
    if t == TypeId.INT8:
        return np.int8
    elif t == TypeId.INT32:
        return np.int32
    elif t == TypeId.FLOAT:
        return np.float32
    elif t == TypeId.COMPLEX_FLOAT:
        return np.complex64
    elif t == TypeId.DOUBLE:
        return float
    elif t == TypeId.COMPLEX_DOUBLE:
        return complex
    else:
        raise ValueError('Cannot to be a element of array: Type ' + str(TypeId(t)))


def typeid_to_size(t: TypeId) -> int:
    """TypeId に対応するデータサイズを返す。

    Args:
        t (TypeId): PSF のデータ型 ID。

    Returns:
        int: データ型のバイトサイズ。

    Raises:
        ValueError: サポートされていない型 ID の場合。
    """
    if t == TypeId.INT8:
        return 4
    elif t == TypeId.INT32:
        return 4
    elif t == TypeId.FLOAT:
        return 4
    elif t == TypeId.COMPLEX_FLOAT:
        return 8
    elif t == TypeId.DOUBLE:
        return 8
    elif t == TypeId.COMPLEX_DOUBLE:
        return 16
    else:
        raise ValueError('Cannot to be a element of array: Type ' + str(TypeId(t)))


def get_big_endian_dtype(t: TypeId):
    """TypeId に対応するビッグエンディアン用の NumPy データ型を返す。

    Args:
        t (TypeId): PSF のデータ型 ID。

    Returns:
        numpy.dtype: ビッグエンディアン of data type.

    Raises:
        ValueError: サポートされていない型 ID の場合。
    """
    if t == TypeId.INT8:
        return np.dtype('>i4')
    elif t == TypeId.INT32:
        return np.dtype('>i4')
    elif t == TypeId.FLOAT:
        return np.dtype('>f4')
    elif t == TypeId.COMPLEX_FLOAT:
        return np.dtype('>c8')
    elif t == TypeId.DOUBLE:
        return np.dtype('>d')
    elif t == TypeId.COMPLEX_DOUBLE:
        return np.dtype('>c16')
    else:
        raise ValueError('Unsupported type ID: ' + str(t))


class SectionId(IntEnum):
    """PSF 内のセクション ID を定義する列挙型。"""

    HEADER = 0
    TYPE = 1
    SWEEP = 2
    TRACE = 3
    VALUE = 4
#    FOOTER = 0x0f


class ChunkId(IntEnum):
    """PSF で使用されるチャンク種別を定義する列挙型。"""

    MAJOR_SECTION = 0x15
    MINOR_SECTION = 0x16


class PropertyTypeId(IntEnum):
    """PSF プロパティのデータ型を定義する列挙型。"""

    STRING = 0x21
    INT = 0x22
    DOUBLE = 0x23


class ElementId(IntEnum):
    """PSF 内の要素 ID を定義する列挙型。"""

    DATA = 0x10
    GROUP = 0x11
    ZEROPAD = 0x14


class SectionInfo:
    """PSF セクションの位置とサイズを表現するクラス。"""

    def __init__(self, offset, size):
        """SectionInfo を初期化する。

        Args:
            offset (int): セクションのファイル内オフセット。
            size (int): セクションのサイズ。
        """
        self.offset = offset
        self.size = size

    def __repr__(self):
        return f"SectionInfo(offset: {self.offset}, size: {self.size})"


class PSF_Property:
    """PSF ファイルのプロパティを表現するクラス。"""

    def __init__(self):
        self.name = ''
        self.type = 0
        self.value = 0

    def __str__(self):
        return f"PSF_Property(name: {self.name}, value: {self.value})"

    def __repr__(self):
        return f"PSF_Property(name: {self.name}, value: {self.value})"

    def read(self, psffile):
        """PSF ファイルからプロパティを読み込む。

        Args:
            psffile (PSFFile): PSFFile インスタンス。

        Returns:
            bool: 有効なプロパティが読み込まれた場合は True、それ以外は False。
        """
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
    def read_dictionary(psffile) -> Dict[str, PSF_Property]:
        """PSF プロパティ辞書を読み込む。

        Args:
            psffile (PSFFile): PSFFile インスタンス。

        Returns:
            dict: プロパティ名をキーとした PSF_Property オブジェクトの辞書。
        """
        properties = dict()
        while True:
            p = PSF_Property()
            if p.read(psffile):
                properties[p.name] = p
            else:
                break

        return properties


class PSF_Type:
    """PSF の型情報を表現するクラス。"""

    def __init__(self):
        self.id: int = 0
        self.name: str = ''
        self.arry_type: int = 0
        self.data_type: TypeId = 0
        self.typelist: List[PSF_Type] = []
        self.prop: PSF_Property = None

    def read(self, psffile: PSFFile, typemap: Dict[int, PSF_Type]) -> bool:
        """PSF ファイルから型定義を読み込む。

        Args:
            psffile (PSFFile): PSFFile インスタンス。
            typemap (dict): 型 ID をキーとする PSF_Type オブジェクトの辞書。

        Returns:
            bool: 型定義が正常に読み込まれた場合は True、それ以外は False。
        """
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

    def read_type_list(self, psffile: PSFFile, typemap: Dict[int, PSF_Type]):
        """STRUCT タイプのフィールド定義リストを読み込む。"""
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
    """PSF の変数定義を表現するクラス。"""

    def __init__(self):
        self.id = 0
        self.name = ''
        self.type_id = 0
        self.prop = None
        self.is_group = False

    def read(self, psffile):
        """PSF ファイルから変数定義を読み込む。

        Args:
            psffile (PSFFile): PSFFile インスタンス。

        Returns:
            bool: 変数が正常に読み込まれた場合は True、それ以外は False。
        """
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
        return f"Var(id: {self.id} name: {self.name}, type_id:{self.type_id}, {self.prop})"

    def to_array(self, npoints: int, psffile: PSFFile, target_names: Optional[set] = None) -> Optional[np.ndarray]:
        """変数の値格納用の NumPy 配列を生成する。

        Args:
            npoints (int): 配列の要素数。
            psffile (PSFFile): PSFFile インスタンス。
            target_names (set, optional): 読み込み対象の信号名セット。Noneなら全て。

        Returns:
            numpy.ndarray|None: 生成した配列。対象外ならNone。
        """
        if target_names is not None and self.name not in target_names:
            return None

        psf_type = psffile.types[self.type_id].data_type
        dtype = typeid_to_dtype(psf_type)
        return np.empty(npoints, dtype=dtype)

    def to_array_group(self, npoints: int, psffile: PSFFile, target_names: Optional[set] = None):
        """グループ変数用の値配列と有効フラグ配列を生成する。

        Args:
            npoints (int): 配列の要素数。
            psffile (PSFFile): PSFFile インスタンス。
            target_names (set, optional): 読み込み対象の信号名セット。Noneなら全て。

        Returns:
            tuple|None: (値配列, 有効フラグ配列)。対象外ならNone。
        """
        if target_names is not None and self.name not in target_names:
            return None

        psf_type = psffile.types[self.type_id].data_type
        dtype = typeid_to_dtype(psf_type)
        return (np.empty(npoints, dtype=dtype), np.zeros(npoints, dtype=bool))

    def read_data(self, array, i, psffile):
        """単一点の値を読み取って配列に格納する。"""
        if array is not None:
            data_array, data_valid = array
            data_array[i] = psffile.read_data(psffile.types[self.type_id].data_type)
            data_valid[i] = True
        else:
            psffile.skip_data(psffile.types[self.type_id].data_type)

    def read_data_win(self, array, start, size, psffile):
        """ウィンドウ形式のデータを読み取って配列に格納する。"""
        psffile.read_data_win(array, start, size, psffile.types[self.type_id].data_type)

    def flatten_value(self, a, arrays):
        """単一変数の値をフラット化して配列辞書に展開する。"""
        if a is not None:
            arrays[self.name] = a
        return [(self, a)]

    def flatten_value_group(self, a, arrays, sweeps, sweep_data):
        """グループ内の単一変数値をフラット化して辞書に展開する。"""
        if a is not None:
            data_array, data_valid = a
            arrays[self.name] = data_array[data_valid]
            sweeps[self.name] = sweep_data[data_valid]

    def to_signal_list(self):
        """信号リスト用のタプルを返す。"""
        return [(self, None)]


class PSF_Group:
    """PSF のグループ変数を表現するクラス。"""

    def __init__(self):
        self.id = 0
        self.name = ''
        self.vars: List[PSF_Variable] = []
        self.is_group = True

    def __repr__(self):
        return f"Group(id: {self.id}, names: {self.name}, vars: {self.vars})"

    def read(self, psffile):
        """PSF ファイルからグループ定義を読み込む。

        Args:
            psffile (PSFFile): PSFFile インスタンス。

        Returns:
            bool: グループ定義が正常に読み込まれた場合は True、それ以外は False。
        """
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
                raise ValueError('Group length is ' + str(length) + ', but actually ' + str(i))

        return True

    def to_array(self, npoints, psffile, target_names=None):
        """グループ内の各変数に対して配列を生成する。"""
        return [(x, x.to_array(npoints, psffile, target_names)) for x in self.vars]

    def to_array_group(self, npoints, psffile, target_names=None):
        """グループ内の各変数に対してグループ配列を生成する。"""
        return [(x, x.to_array_group(npoints, psffile, target_names)) for x in self.vars]

    def read_data(self, array, i, psffile):
        """グループ内の全変数について単一点データを読み込む。"""
        for (v, ary) in array:
            v.read_data(ary, i, psffile)

    def read_data_win(self, array, start, size, psffile):
        """グループ内の全変数についてウィンドウ形式データを読み込む。"""
        for (v, ary) in array:
            v.read_data_win(ary, start, size, psffile)

    def flatten_value(self, a, arrays):
        """グループ内の値をフラット化して配列辞書に展開する。"""
        variables = list()
        for (v, ary) in a:
            variables.extend(v.flatten_value(ary, arrays))
        return variables

    def flatten_value_group(self, a, arrays, sweeps, sweep_data):
        """グループ内の値をスイープデータ付きでフラット化する。"""
        for (v, ary) in a:
            v.flatten_value_group(ary, arrays, sweeps, sweep_data)

    def to_signal_list(self):
        """グループ内の全信号のリストを返す。"""
        signals = list()
        for x in self.vars:
            signals.extend(x.to_signal_list())

        return signals
