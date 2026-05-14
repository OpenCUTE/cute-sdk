"""Golden tensor loader.

Reads golden.bin + manifest.json and provides tensor element access.
"""

import json
import struct
from pathlib import Path


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
            raise ValueError(
                "manifest must have 'output' or 'tensors' key"
            )
        self._element_bits = output["element_bits"]
        self._element_bytes = self._element_bits // 8
        self._shape = tuple(output["shape"])
        self._stride_bytes = output["stride_bytes"]
        self._layout = output.get("layout", "row_major")
        self._total_bytes = output.get(
            "total_bytes",
            (self._shape[0] - 1) * self._stride_bytes + self._shape[1] * self._element_bytes,
        )

        self._rows = self._shape[0]
        self._cols = self._shape[1]

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

    def element_count(self) -> int:
        return self._rows * self._cols

    def _offset(self, row: int, col: int) -> int:
        if self._layout == "row_major":
            return row * self._stride_bytes + col * self._element_bytes
        else:
            return col * self._stride_bytes + row * self._element_bytes

    def _read_element(self, offset: int) -> int:
        if self._element_bits == 32:
            return struct.unpack_from("<i", self._data, offset)[0]
        elif self._element_bits == 16:
            return struct.unpack_from("<h", self._data, offset)[0]
        elif self._element_bits == 8:
            return struct.unpack_from("<b", self._data, offset)[0]
        else:
            raise ValueError(f"unsupported element_bits: {self._element_bits}")

    def __getitem__(self, key) -> int:
        if isinstance(key, tuple):
            row, col = key
            return self._read_element(self._offset(row, col))
        elif isinstance(key, int):
            row = key
            return _GoldenRow(self, row)
        else:
            raise TypeError(f"unsupported key type: {type(key)}")

    def raw_bytes(self) -> bytes:
        return self._data

    def to_list(self) -> list:
        return [
            [self._read_element(self._offset(r, c)) for c in range(self._cols)]
            for r in range(self._rows)
        ]


class _GoldenRow:
    def __init__(self, tensor: GoldenTensor, row: int):
        self._tensor = tensor
        self._row = row

    def __getitem__(self, col: int) -> int:
        return self._tensor._read_element(self._tensor._offset(self._row, col))

    def __len__(self) -> int:
        return self._tensor._cols
