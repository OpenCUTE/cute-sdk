"""CMemoryLoader store trace reader.

Parses Verilator simulation output containing CMemoryLoader_Store write
requests and reconstructs tensor data from the store trace.
"""

import re
import struct
from pathlib import Path

# WriteRequest line format:
# [CMemoryLoader_Store<  CYCLE>]WriteRequest: RequestVirtualAddr= HEX,
#   ..., CurrentStore_BlockTensor_Major_DIM_Iter: DEC,
#   CurrentStore_BlockTensor_Reduce_DIM_Iter: DEC, RequestData:HEXDATA
_WRITE_REQUEST_RE = re.compile(
    r"WriteRequest:\s*"
    r"RequestVirtualAddr=\s*([0-9a-fA-F]+),\s*"
    r"RequestConherent=\s*\d+,\s*"
    r"RequestSourceID=\s*[0-9a-fA-F]+,\s*"
    r"RequestType_isWrite=\s*\d+,\s*"
    r"CurrentStore_BlockTensor_Major_DIM_Iter:\s*([0-9a-fA-F]+),\s*"
    r"CurrentStore_BlockTensor_Reduce_DIM_Iter:\s*([0-9a-fA-F]+),\s*"
    r"RequestData:([0-9a-fA-F]+)"
)

_CML_PREFIX = "[CMemoryLoader_Store<"


def _hex_to_signed32(val: int) -> int:
    if val >= 0x80000000:
        val -= 0x100000000
    return val


def _parse_data_hex(hex_data: str) -> list[int]:
    """Parse RequestData hex into signed int32 elements.

    The hex payload is a sequence of 8-char (4-byte) big-endian words
    stored in reverse element order.  We reverse the word list so that
    element[0] corresponds to the lowest address.
    """
    chunks = [hex_data[i:i + 8] for i in range(0, len(hex_data), 8)]
    chunks.reverse()
    return [_hex_to_signed32(int(c, 16)) for c in chunks]


class CMLStoreTrace:
    """Parses CMemoryLoader store trace and provides tensor reconstruction.

    Accepts both pre-filtered ``CML_Store_trace.out`` files and complete
    Verilator ``.out`` files (irrelevant lines are silently skipped).
    """

    def __init__(self, trace_path: str):
        self._path = Path(trace_path)
        # (addr, elements, major_iter, reduce_iter)
        self._requests: list[tuple[int, list[int], int, int]] = []
        self._byte_map: dict[int, int] = {}
        self._parse()

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse(self):
        with open(self._path, "r") as f:
            for line in f:
                if _CML_PREFIX not in line:
                    continue
                if "WriteRequest:" not in line:
                    continue

                m = _WRITE_REQUEST_RE.search(line)
                if not m:
                    continue

                addr = int(m.group(1), 16)
                major_iter = int(m.group(2), 16)
                reduce_iter = int(m.group(3), 16)
                hex_data = m.group(4).strip()

                elements = _parse_data_hex(hex_data)
                self._requests.append((addr, elements, major_iter, reduce_iter))

                # Byte-level map (later writes overwrite earlier ones)
                for i, elem in enumerate(elements):
                    for j, b in enumerate(struct.pack("<i", elem)):
                        self._byte_map[addr + i * 4 + j] = b

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def request_count(self) -> int:
        return len(self._requests)

    def get_requests(self) -> list[tuple[int, list[int], int, int]]:
        """Return all parsed requests as (addr, elements, major_iter, reduce_iter)."""
        return list(self._requests)

    def get_base_address(self) -> int | None:
        """Return the address of the first WriteRequest, or None if empty."""
        if self._requests:
            return self._requests[0][0]
        return None

    def get_tensor(
        self,
        base_addr: int,
        shape: tuple[int, int],
        stride_bytes: int,
        element_bits: int,
    ) -> list[list[int]]:
        """Reconstruct a 2-D tensor from store trace data.

        Each WriteRequest carries *N* consecutive int32 elements starting at
        its ``RequestVirtualAddr``.  Elements are placed into the output
        tensor according to:

            offset = elem_addr - base_addr
            row    = offset // stride_bytes
            col    = (offset % stride_bytes) // element_bytes

        Args:
            base_addr: Base virtual address of the tensor.
            shape: ``(rows, cols)`` of the output tensor.
            stride_bytes: Number of bytes per row (may include padding).
            element_bits: Bits per element (8, 16, or 32).

        Returns:
            ``shape[0] x shape[1]`` list-of-lists of signed int32 values.
        """
        rows, cols = shape
        element_bytes = element_bits // 8
        tensor = [[0] * cols for _ in range(rows)]

        for addr, elements, _, _ in self._requests:
            for i, val in enumerate(elements):
                elem_addr = addr + i * element_bytes
                offset = elem_addr - base_addr
                if offset < 0:
                    continue
                row = offset // stride_bytes
                col = (offset % stride_bytes) // element_bytes
                if 0 <= row < rows and 0 <= col < cols:
                    tensor[row][col] = val

        return tensor

    def get_data_at(self, addr: int, num_bytes: int) -> bytes:
        """Read *num_bytes* raw bytes starting at *addr*."""
        return bytes(self._byte_map.get(addr + i, 0) for i in range(num_bytes))

    def addresses(self) -> list[int]:
        """Return all WriteRequest base addresses in trace order."""
        return [r[0] for r in self._requests]
