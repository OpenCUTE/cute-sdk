"""Byte-level compare engine for CUTE memverify.

Compares golden tensor data against trace-reconstructed tensor data
and produces a pass/fail report with mismatch details.
"""

import argparse
import math
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path

from memverify.readers.cml_store_trace import CMLStoreTrace
from memverify.readers.golden_tensor import GoldenTensor


@dataclass
class Mismatch:
    byte_index: int
    byte_offset: int
    expected: int
    actual: int

    @property
    def expected_hex(self) -> str:
        return f"0x{self.expected & 0xFFFFFFFF:08x}"

    @property
    def actual_hex(self) -> str:
        return f"0x{self.actual & 0xFFFFFFFF:08x}"


@dataclass
class CompareResult:
    passed: bool
    total_elements: int
    matched_elements: int
    mismatch_count: int
    mismatches: list[Mismatch] = field(default_factory=list)

    def report(self) -> str:
        lines = []
        status = "PASS" if self.passed else "FAIL"
        lines.append(f"[{status}] {self.matched_elements}/{self.total_elements} matched")
        if self.mismatches:
            lines.append(f"  First mismatches (up to 20):")
            for m in self.mismatches:
                lines.append(
                    f"  byte[{m.byte_index}] offset={m.byte_offset} "
                    f"expected={m.expected} ({m.expected_hex}) "
                    f"actual={m.actual} ({m.actual_hex})"
                )
        return "\n".join(lines)


@dataclass
class FloatMismatch:
    element_index: int
    byte_offset: int
    expected: float
    actual: float
    error_percent: float


@dataclass
class FloatToleranceResult:
    passed: bool
    total_elements: int
    matched_elements: int
    max_error_percent: float
    tolerance_percent: float
    mismatches: list[FloatMismatch] = field(default_factory=list)

    def report(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        max_error = _format_percent(self.max_error_percent)
        lines = [
            f"[{status}] float tolerance {self.matched_elements}/{self.total_elements} "
            f"within {self.tolerance_percent:g}% (max_error={max_error})"
        ]
        if self.mismatches:
            lines.append("  First float mismatches (up to 20):")
            for m in self.mismatches:
                lines.append(
                    f"  element[{m.element_index}] offset={m.byte_offset} "
                    f"expected={m.expected:.9g} actual={m.actual:.9g} "
                    f"error={_format_percent(m.error_percent)}"
                )
        return "\n".join(lines)


_MAX_REPORTED_MISMATCHES = 20
_FLOAT_DTYPES = {"F16", "FP16", "F32", "FP32", "BF16"}


def compare(
    golden: GoldenTensor,
    actual: bytes,
) -> CompareResult:
    expected_bytes = golden.raw_bytes()
    total = len(expected_bytes)
    matched = 0
    mismatches: list[Mismatch] = []

    for index in range(total):
        expected = expected_bytes[index]
        actual_val = actual[index] if index < len(actual) else 0
        if expected == actual_val:
            matched += 1
        elif len(mismatches) < _MAX_REPORTED_MISMATCHES:
            mismatches.append(Mismatch(
                byte_index=index,
                byte_offset=index,
                expected=expected,
                actual=actual_val,
            ))

    mismatch_count = total - matched
    return CompareResult(
        passed=mismatch_count == 0,
        total_elements=total,
        matched_elements=matched,
        mismatch_count=mismatch_count,
        mismatches=mismatches,
    )


def is_float_tensor(golden: GoldenTensor) -> bool:
    return golden.dtype.upper() in _FLOAT_DTYPES


def _format_percent(value: float) -> str:
    if math.isinf(value):
        return "inf%"
    if math.isnan(value):
        return "nan%"
    return f"{value:.6g}%"


def _read_padded(data: bytes, offset: int, width: int) -> bytes:
    chunk = data[offset:offset + width]
    if len(chunk) == width:
        return chunk
    return chunk + bytes(width - len(chunk))


def _read_float(data: bytes, offset: int, dtype: str) -> float:
    dtype = dtype.upper()
    if dtype in ("F16", "FP16"):
        return struct.unpack("<e", _read_padded(data, offset, 2))[0]
    if dtype == "BF16":
        raw = struct.unpack("<H", _read_padded(data, offset, 2))[0]
        return struct.unpack("<f", struct.pack("<I", raw << 16))[0]
    if dtype in ("F32", "FP32"):
        return struct.unpack("<f", _read_padded(data, offset, 4))[0]
    raise ValueError(f"unsupported float dtype for tolerance compare: {dtype}")


def _relative_error_percent(expected: float, actual: float) -> float:
    if math.isnan(expected) or math.isnan(actual):
        return 0.0 if math.isnan(expected) and math.isnan(actual) else math.inf
    if math.isinf(expected) or math.isinf(actual):
        return 0.0 if expected == actual else math.inf
    if expected == 0.0:
        return 0.0 if actual == 0.0 else math.inf
    return abs(actual - expected) / abs(expected) * 100.0


def compare_float_tolerance(
    golden: GoldenTensor,
    actual: bytes,
    tolerance_percent: float,
) -> FloatToleranceResult:
    expected_bytes = golden.raw_bytes()
    dtype = golden.dtype.upper()
    element_bytes = golden.element_bytes
    total = golden.element_count()
    matched = 0
    max_error = 0.0
    mismatches: list[FloatMismatch] = []

    for index in range(total):
        offset = index * element_bytes
        expected = _read_float(expected_bytes, offset, dtype)
        actual_val = _read_float(actual, offset, dtype)
        error_percent = _relative_error_percent(expected, actual_val)

        if not math.isnan(error_percent):
            if math.isinf(error_percent) or error_percent > max_error:
                max_error = error_percent

        if error_percent <= tolerance_percent:
            matched += 1
        elif len(mismatches) < _MAX_REPORTED_MISMATCHES:
            mismatches.append(FloatMismatch(
                element_index=index,
                byte_offset=offset,
                expected=expected,
                actual=actual_val,
                error_percent=error_percent,
            ))

    return FloatToleranceResult(
        passed=matched == total,
        total_elements=total,
        matched_elements=matched,
        max_error_percent=max_error,
        tolerance_percent=tolerance_percent,
        mismatches=mismatches,
    )


def report_with_float_tolerance(
    byte_result: CompareResult,
    tolerance_result: FloatToleranceResult,
) -> str:
    if tolerance_result.passed:
        lines = [
            "[PASS] bit-exact failed; float tolerance passed",
            f"  Bit exact: {byte_result.matched_elements}/{byte_result.total_elements} bytes matched",
            f"  Float tolerance: {tolerance_result.matched_elements}/"
            f"{tolerance_result.total_elements} elements within "
            f"{tolerance_result.tolerance_percent:g}% "
            f"(max_error={_format_percent(tolerance_result.max_error_percent)})",
        ]
        if byte_result.mismatches:
            lines.append("  First bit mismatches (up to 20):")
            for m in byte_result.mismatches:
                lines.append(
                    f"  byte[{m.byte_index}] offset={m.byte_offset} "
                    f"expected={m.expected} ({m.expected_hex}) "
                    f"actual={m.actual} ({m.actual_hex})"
                )
        return "\n".join(lines)

    return byte_result.report() + "\n" + tolerance_result.report()


def _parse_tile_shape(value: str) -> tuple[int, int]:
    try:
        rows_text, cols_text = value.lower().split("x", 1)
        return int(rows_text), int(cols_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"tile shape must be ROWSxCOLS, got {value!r}"
        ) from error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare golden tensor against CML store trace"
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to manifest.json (golden data descriptor)",
    )
    parser.add_argument(
        "--trace",
        required=True,
        help="Path to CML store trace (.out) file",
    )
    parser.add_argument(
        "--tensor",
        default="D",
        help="Tensor name in manifest to compare (default: D)",
    )
    parser.add_argument(
        "--base-addr",
        help="Base virtual address (hex) of the output tensor; "
             "if omitted, uses first WriteRequest address from trace",
    )
    parser.add_argument(
        "--layout",
        choices=("direct", "tiled_cpu_memcpy"),
        default="direct",
        help="Trace reconstruction layout (default: direct)",
    )
    parser.add_argument(
        "--tile-shape",
        type=_parse_tile_shape,
        default=(64, 64),
        help="Tile shape for tiled_cpu_memcpy layout, formatted ROWSxCOLS",
    )
    parser.add_argument(
        "--float-tolerance-percent",
        type=float,
        default=0.1,
        help="Fallback relative error tolerance for floating tensors after "
             "bit-exact compare fails, expressed as percent (default: 0.1)",
    )
    args = parser.parse_args(argv)

    if args.float_tolerance_percent < 0:
        parser.error("--float-tolerance-percent must be non-negative")

    golden = GoldenTensor(args.manifest, tensor_name=args.tensor)
    trace = CMLStoreTrace(args.trace)

    if args.layout == "tiled_cpu_memcpy":
        actual = trace.get_tiled_cpu_memcpy_tensor_bytes(
            golden.shape,
            golden.stride_bytes,
            golden.element_bits,
            args.tile_shape,
        )
    else:
        if args.base_addr:
            base = int(args.base_addr, 16)
        else:
            base = trace.get_base_address()
        if base is None:
            print("[ERROR] No WriteRequest found in trace", file=sys.stderr)
            return 1

        actual = trace.get_tensor_bytes(base, golden.shape, golden.stride_bytes, golden.element_bits)
    result = compare(golden, actual)
    if result.passed:
        print(result.report())
        return 0

    if is_float_tensor(golden):
        tolerance_result = compare_float_tolerance(
            golden,
            actual,
            args.float_tolerance_percent,
        )
        print(report_with_float_tolerance(result, tolerance_result))
        return 0 if tolerance_result.passed else 1

    print(result.report())
    return 1


if __name__ == "__main__":
    sys.exit(main())
