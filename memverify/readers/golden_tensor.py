"""Golden tensor loader.

Reads golden.bin + manifest.json and provides tensor element access.
Supports F32, F16, BF16, I32, I16, I8, U8, U1_PACKED dtypes and 1D/2D/3D shapes.
"""

import json
import struct
from pathlib import Path


# struct format char + whether it's a float type
_DTYPE_READERS = {
    32: ("<f", True),
    16: ("<e", True),   # F16 (IEEE754 half)
    8:  ("<b", False),
}


class GoldenTensor:
    def __init__(self, manifest_path: str, tensor_name: str = "D"):
        self._manifest_path = Path(manifest_path)
        with open(self._manifest_path, "r") as f:
            self._manifest = json.load(f)

        # Support both old ("output" key) and new ("tensors" key) manifest formats
        if "output" in self._manifest:
            output = self._manifest["output"]
        elif "tensors" in self._manifest:
            output = self._manifest["tensors"][tensor_name]
        else:
            raise ValueError("manifest must have 'output' or 'tensors' key")

        self._element_bits = output["element_bits"]
        self._element_bytes = max(self._element_bits // 8, 1)
        self._shape = tuple(output["shape"])
        self._ndim = len(self._shape)
        self._layout = output.get("layout", "row_major")
        self._dtype = output.get("dtype", "")

        # total_bytes: prefer explicit, otherwise compute from shape
        self._total_bytes = output.get("total_bytes")
        if self._total_bytes is None:
            self._total_bytes = 1
            for s in self._shape:
                self._total_bytes *= s
            self._total_bytes *= self._element_bytes

        # stride: use explicit or compute row-major last-dim stride
        self._stride_bytes = output.get("stride_bytes")
        if self._stride_bytes is None and self._ndim >= 2:
            self._stride_bytes = self._shape[-1] * self._element_bytes
        elif self._stride_bytes is None:
            self._stride_bytes = self._total_bytes

        golden_path = self._manifest_path.parent / output["path"]
        with open(golden_path, "rb") as f:
            self._data = f.read()

        if len(self._data) != self._total_bytes:
            raise ValueError(
                f"golden.bin size mismatch: expected {self._total_bytes}, "
                f"got {len(self._data)}"
            )

    @property
    def manifest(self) -> dict:
        return self._manifest

    @property
    def shape(self) -> tuple:
        return self._shape

    @property
    def dtype(self) -> str:
        return self._dtype

    @property
    def element_bits(self) -> int:
        return self._element_bits

    @property
    def element_bytes(self) -> int:
        return self._element_bytes

    @property
    def stride_bytes(self) -> int:
        return self._stride_bytes

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    @property
    def ndim(self) -> int:
        return self._ndim

    def element_count(self) -> int:
        n = 1
        for s in self._shape:
            n *= s
        return n

    def _flat_offset(self, flat_idx: int) -> int:
        return flat_idx * self._element_bytes

    def _read_element(self, offset: int):
        if self._element_bits == 32:
            return struct.unpack_from("<f", self._data, offset)[0]
        elif self._element_bits == 16:
            # F16 or BF16
            raw = struct.unpack_from("<H", self._data, offset)[0]
            if self._dtype in ("F16", ""):
                return struct.unpack_from("<e", self._data, offset)[0]
            return raw  # return raw uint16 for BF16 or unknown
        elif self._element_bits == 8:
            if self._dtype in ("U8", "U1_PACKED"):
                return struct.unpack_from("<B", self._data, offset)[0]
            return struct.unpack_from("<b", self._data, offset)[0]
        else:
            raise ValueError(f"unsupported element_bits: {self._element_bits}")

    def flat(self, idx: int):
        """Access element by flat index."""
        return self._read_element(self._flat_offset(idx))

    def __getitem__(self, key):
        if isinstance(key, tuple):
            if len(key) != self._ndim:
                raise IndexError(f"expected {self._ndim} indices, got {len(key)}")
            flat = 0
            stride = 1
            for i in range(self._ndim - 1, -1, -1):
                flat += key[i] * stride
                stride *= self._shape[i]
            return self._read_element(self._flat_offset(flat))
        elif isinstance(key, int):
            if self._ndim == 1:
                return self._read_element(self._flat_offset(key))
            # Return a row accessor for 2D+
            return _GoldenSlice(self, key)
        else:
            raise TypeError(f"unsupported key type: {type(key)}")

    def raw_bytes(self) -> bytes:
        return self._data

    def to_list(self) -> list:
        n = self.element_count()
        return [self._read_element(self._flat_offset(i)) for i in range(n)]


class _GoldenSlice:
    """Accessor for a sub-slice when indexing into a multi-dimensional tensor."""

    def __init__(self, tensor: GoldenTensor, index: int):
        self._tensor = tensor
        self._index = index

    def __getitem__(self, key):
        # Build a full index tuple
        if isinstance(key, tuple):
            return self._tensor[(self._index, *key)]
        return self._tensor[(self._index, key)]

    def __len__(self) -> int:
        if self._tensor.ndim <= 1:
            return 0
        return self._tensor.shape[1]
