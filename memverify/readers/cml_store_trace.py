"""CMemoryLoader store trace reader.

Parses Verilator simulation output containing CUTETrace CMLStore.storeData
events and reconstructs tensor data from the store trace.  Older
CMemoryLoader_Store debug printf lines are still accepted as a fallback.
"""

import re
import struct
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path


_CUTE_ROOT = Path(__file__).resolve().parents[3]
_TRACE_PYTHON = _CUTE_ROOT / "trace" / "python"
if _TRACE_PYTHON.exists() and str(_TRACE_PYTHON) not in sys.path:
    sys.path.insert(0, str(_TRACE_PYTHON))

try:
    from cutetrace.catalog import load_catalog
    from cutetrace.decoder import Decoder, TraceDecodeError
    from cutetrace.parser import TraceParseError, parse_file
    from cutetrace.rebuilder import MemoryRebuilder
except ImportError:  # pragma: no cover - legacy fallback for external use
    load_catalog = None
    Decoder = None
    TraceDecodeError = None
    TraceParseError = None
    parse_file = None
    MemoryRebuilder = None

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
_STORE_EVENTS = ("CMLStore.storeData", "VectorStore.storeData")
_STORE_DATA_WIDTH_BYTES = 64


@dataclass(frozen=True)
class StoreRecord:
    addr: int
    payload: bytes
    task_count: int | None = None


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


def _payload_to_signed32_elements(payload: bytes) -> list[int]:
    return [
        struct.unpack_from("<i", payload, offset)[0]
        for offset in range(0, len(payload), 4)
    ]


def _read_element(data: bytes, element_bits: int):
    if element_bits == 32:
        return struct.unpack("<i", data)[0]
    if element_bits == 16:
        return struct.unpack("<H", data)[0]
    if element_bits == 8:
        return struct.unpack("<b", data)[0]
    raise ValueError(f"unsupported element_bits: {element_bits}")


def _flatten_shape(shape: tuple[int, ...]) -> tuple[int, int]:
    inner = shape[-1]
    outer = 1
    for dim in shape[:-1]:
        outer *= dim
    return outer, inner


class CMLStoreTrace:
    """Parses CMemoryLoader store trace and provides tensor reconstruction.

    Accepts both pre-filtered ``CML_Store_trace.out`` files and complete
    Verilator ``.out`` files (irrelevant lines are silently skipped).
    """

    def __init__(self, trace_path: str):
        self._path = Path(trace_path)
        # (addr, elements, major_iter, reduce_iter)
        self._requests: list[tuple[int, list[int], int, int]] = []
        self._records: list[StoreRecord] = []
        self._byte_map: dict[int, int] = {}
        self._trace_source = "none"
        self._parse()

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse(self):
        if self._parse_compact_cute_trace() > 0:
            self._trace_source = "cutetrace"
            return
        self._parse_legacy_debug_trace()
        if self._requests:
            self._trace_source = "legacy_printf"

    def _record_payload(
        self,
        addr: int,
        payload: bytes,
        *,
        major_iter: int = 0,
        reduce_iter: int = 0,
        task_count: int | None = None,
    ) -> None:
        elements = _payload_to_signed32_elements(payload)
        self._requests.append((addr, elements, major_iter, reduce_iter))
        self._records.append(StoreRecord(addr=addr, payload=payload, task_count=task_count))

        # Byte-level map (later writes overwrite earlier ones)
        for i, b in enumerate(payload):
            self._byte_map[addr + i] = b

    def _parse_compact_cute_trace(self) -> int:
        if load_catalog is None or Decoder is None or parse_file is None or MemoryRebuilder is None:
            return 0

        catalog_path = _CUTE_ROOT / "trace" / "catalogs" / "cute_trace.json"
        schema_path = _CUTE_ROOT / "configs" / "schemas" / "cute_trace_catalog.schema.json"
        catalog = load_catalog(catalog_path, schema_path=schema_path,
                               validate_schema=False)
        decoder = Decoder(catalog)
        rebuilder = MemoryRebuilder(data_width_bytes=_STORE_DATA_WIDTH_BYTES)
        count = 0

        for raw in parse_file(self._path):
            if TraceParseError is not None and isinstance(raw, TraceParseError):
                continue
            try:
                event = decoder.decode(raw)
            except TraceDecodeError:
                continue

            if event.event not in _STORE_EVENTS:
                continue

            rebuilder.apply_event(event, event_names=_STORE_EVENTS)
            addr = int(event.fields["vaddr"])
            payload = rebuilder.writes[addr]
            self._record_payload(
                addr,
                payload,
                task_count=int(event.fields.get("task_count", 0)),
            )
            count += 1

        return count

    def _parse_legacy_debug_trace(self):
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
                payload = b"".join(struct.pack("<i", elem) for elem in elements)
                self._record_payload(
                    addr,
                    payload,
                    major_iter=major_iter,
                    reduce_iter=reduce_iter,
                )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def request_count(self) -> int:
        return len(self._requests)

    @property
    def trace_source(self) -> str:
        return self._trace_source

    def get_requests(self) -> list[tuple[int, list[int], int, int]]:
        """Return all parsed requests as (addr, elements, major_iter, reduce_iter)."""
        return list(self._requests)

    def get_records(self) -> list[StoreRecord]:
        """Return all parsed store records in trace order."""
        return list(self._records)

    def get_base_address(self) -> int | None:
        """Return the address of the first WriteRequest, or None if empty."""
        if self._requests:
            return self._requests[0][0]
        return None

    def get_tensor_bytes(
        self,
        base_addr: int,
        shape: tuple[int, ...],
        stride_bytes: int,
        element_bits: int,
    ) -> bytes:
        """Reconstruct a tensor from store trace data as flat row-major bytes.

        For tensors with two or more dimensions, ``stride_bytes`` is the
        stride of the last dimension.

            element address = base + outer_index * stride + inner_index * elem_bytes

        Args:
            base_addr: Base virtual address of the tensor.
            shape: Output tensor shape.
            stride_bytes: Number of bytes per row/last-dimension slice.
            element_bits: Bits per element (8, 16, or 32).

        Returns:
            Raw bytes in logical row-major order.
        """
        element_bytes = element_bits // 8
        outer, inner = _flatten_shape(shape)
        data = bytearray()

        for outer_index in range(outer):
            row_base = base_addr + outer_index * stride_bytes
            for inner_index in range(inner):
                elem_addr = row_base + inner_index * element_bytes
                data.extend(self.get_data_at(elem_addr, element_bytes))
        return bytes(data)

    def get_tensor(
        self,
        base_addr: int,
        shape: tuple[int, ...],
        stride_bytes: int,
        element_bits: int,
    ) -> list[int]:
        """Reconstruct a tensor from store trace data as flat element values."""
        payload = self.get_tensor_bytes(base_addr, shape, stride_bytes, element_bits)
        element_bytes = element_bits // 8
        return [
            _read_element(payload[i:i + element_bytes], element_bits)
            for i in range(0, len(payload), element_bytes)
        ]

    def get_tiled_cpu_memcpy_tensor(
        self,
        shape: tuple[int, int],
        stride_bytes: int,
        element_bits: int,
        tile_shape: tuple[int, int],
    ) -> list[list[int]]:
        """Reconstruct output for tiled matmul with CPU memcpy post-op.

        In this mode CML stores each tile to scratch memory, and the CPU
        copies scratch rows to the final output tensor.  The trace observes
        the scratch writes only, so replay the copy using task order:

            task n -> tile (n / tile_cols, n % tile_cols)
        """
        rows, cols = shape
        tile_rows, tile_cols = tile_shape
        element_bytes = element_bits // 8
        if rows % tile_rows != 0 or cols % tile_cols != 0:
            raise ValueError(f"shape {shape} is not divisible by tile shape {tile_shape}")

        row_bytes = tile_cols * element_bytes
        tile_stride_bytes = row_bytes
        tile_bytes = tile_rows * tile_stride_bytes
        tile_grid_cols = cols // tile_cols
        total_tiles = (rows // tile_rows) * tile_grid_cols

        groups = self._group_records_by_task(tile_bytes)
        output: dict[int, int] = {}

        for tile_index, records in enumerate(groups[:total_tiles]):
            tile_i = tile_index // tile_grid_cols
            tile_j = tile_index % tile_grid_cols
            scratch = self._records_to_byte_map(records)
            if not scratch:
                continue
            scratch_base = min(record.addr for record in records)

            for r in range(tile_rows):
                src = scratch_base + r * tile_stride_bytes
                dst = (
                    (tile_i * tile_rows + r) * stride_bytes
                    + tile_j * tile_cols * element_bytes
                )
                for b in range(row_bytes):
                    output[dst + b] = scratch.get(src + b, 0)

        tensor = [[0] * cols for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                offset = r * stride_bytes + c * element_bytes
                elem = bytes(output.get(offset + i, 0) for i in range(element_bytes))
                tensor[r][c] = _read_element(elem, element_bits)
        return tensor

    def get_tiled_cpu_memcpy_tensor_bytes(
        self,
        shape: tuple[int, int],
        stride_bytes: int,
        element_bits: int,
        tile_shape: tuple[int, int],
    ) -> bytes:
        tensor = self.get_tiled_cpu_memcpy_tensor(
            shape,
            stride_bytes,
            element_bits,
            tile_shape,
        )
        element_bytes = element_bits // 8
        data = bytearray()
        for row in tensor:
            for value in row:
                if element_bits == 32:
                    data.extend(int(value).to_bytes(element_bytes, "little", signed=True))
                elif element_bits == 16:
                    data.extend(int(value).to_bytes(element_bytes, "little", signed=False))
                elif element_bits == 8:
                    data.extend(int(value).to_bytes(element_bytes, "little", signed=True))
                else:
                    raise ValueError(f"unsupported element_bits: {element_bits}")
        return bytes(data)

    def _group_records_by_task(self, tile_bytes: int) -> list[list[StoreRecord]]:
        if any(record.task_count is not None for record in self._records):
            groups: OrderedDict[int, list[StoreRecord]] = OrderedDict()
            for record in self._records:
                key = int(record.task_count or 0)
                groups.setdefault(key, []).append(record)
            return list(groups.values())

        groups: list[list[StoreRecord]] = []
        current: list[StoreRecord] = []
        current_bytes = 0
        for record in self._records:
            current.append(record)
            current_bytes += len(record.payload)
            if current_bytes >= tile_bytes:
                groups.append(current)
                current = []
                current_bytes = 0
        if current:
            groups.append(current)
        return groups

    @staticmethod
    def _records_to_byte_map(records: list[StoreRecord]) -> dict[int, int]:
        byte_map: dict[int, int] = {}
        for record in records:
            for i, b in enumerate(record.payload):
                byte_map[record.addr + i] = b
        return byte_map

    def get_data_at(self, addr: int, num_bytes: int) -> bytes:
        """Read *num_bytes* raw bytes starting at *addr*."""
        return bytes(self._byte_map.get(addr + i, 0) for i in range(num_bytes))

    def addresses(self) -> list[int]:
        """Return all WriteRequest base addresses in trace order."""
        return [r[0] for r in self._requests]
