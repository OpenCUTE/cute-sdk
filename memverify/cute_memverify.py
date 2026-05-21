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
    ulp_error: int | None = None


@dataclass
class FloatToleranceResult:
    passed: bool
    total_elements: int
    matched_elements: int
    max_error_percent: float
    tolerance_percent: float
    max_ulp_error: int | None = None
    tolerance_ulp: int | None = None
    mismatches: list[FloatMismatch] = field(default_factory=list)

    def report(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        max_error = _format_percent(self.max_error_percent)
        ulp_text = ""
        if self.tolerance_ulp is not None:
            ulp_text = (
                f" or {self.tolerance_ulp} ULP"
                f" (max_ulp={self.max_ulp_error})"
            )
        lines = [
            f"[{status}] float tolerance {self.matched_elements}/{self.total_elements} "
            f"within {self.tolerance_percent:g}%{ulp_text} "
            f"(max_error={max_error})"
        ]
        if self.mismatches:
            lines.append("  First float mismatches (up to 20):")
            for m in self.mismatches:
                ulp_text = (
                    f" ulp={m.ulp_error}" if m.ulp_error is not None else ""
                )
                lines.append(
                    f"  element[{m.element_index}] offset={m.byte_offset} "
                    f"expected={m.expected:.9g} actual={m.actual:.9g} "
                    f"error={_format_percent(m.error_percent)}{ulp_text}"
                )
        return "\n".join(lines)


_MAX_REPORTED_MISMATCHES = 20
_FLOAT_DTYPES = {"F16", "FP16", "F32", "FP32", "BF16"}
_BF16_DEFAULT_TOLERANCE_ULP = 1


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


def _read_raw_float_bits(data: bytes, offset: int, dtype: str) -> tuple[int, int]:
    dtype = dtype.upper()
    if dtype in ("F16", "FP16", "BF16"):
        return struct.unpack("<H", _read_padded(data, offset, 2))[0], 16
    if dtype in ("F32", "FP32"):
        return struct.unpack("<I", _read_padded(data, offset, 4))[0], 32
    raise ValueError(f"unsupported float dtype for ULP compare: {dtype}")


def _ordered_float_bits(raw: int, bits: int) -> int:
    sign_bit = 1 << (bits - 1)
    mask = (1 << bits) - 1
    return (~raw & mask) if (raw & sign_bit) else (raw | sign_bit)


def _ulp_distance(
    expected_bytes: bytes,
    actual: bytes,
    offset: int,
    dtype: str,
) -> int:
    expected_raw, bits = _read_raw_float_bits(expected_bytes, offset, dtype)
    actual_raw, _ = _read_raw_float_bits(actual, offset, dtype)
    return abs(_ordered_float_bits(expected_raw, bits) -
               _ordered_float_bits(actual_raw, bits))


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
    tolerance_ulp: int | None = None,
) -> FloatToleranceResult:
    expected_bytes = golden.raw_bytes()
    dtype = golden.dtype.upper()
    element_bytes = golden.element_bytes
    total = golden.element_count()
    matched = 0
    max_error = 0.0
    max_ulp_error: int | None = None
    mismatches: list[FloatMismatch] = []

    if tolerance_ulp is None and dtype == "BF16":
        tolerance_ulp = _BF16_DEFAULT_TOLERANCE_ULP

    for index in range(total):
        offset = index * element_bytes
        expected = _read_float(expected_bytes, offset, dtype)
        actual_val = _read_float(actual, offset, dtype)
        error_percent = _relative_error_percent(expected, actual_val)
        ulp_error = None
        if tolerance_ulp is not None:
            ulp_error = _ulp_distance(expected_bytes, actual, offset, dtype)
            if max_ulp_error is None or ulp_error > max_ulp_error:
                max_ulp_error = ulp_error

        if not math.isnan(error_percent):
            if math.isinf(error_percent) or error_percent > max_error:
                max_error = error_percent

        within_percent = error_percent <= tolerance_percent
        within_ulp = (
            tolerance_ulp is not None
            and ulp_error is not None
            and ulp_error <= tolerance_ulp
        )
        if within_percent or within_ulp:
            matched += 1
        elif len(mismatches) < _MAX_REPORTED_MISMATCHES:
            mismatches.append(FloatMismatch(
                element_index=index,
                byte_offset=offset,
                expected=expected,
                actual=actual_val,
                error_percent=error_percent,
                ulp_error=ulp_error,
            ))

    return FloatToleranceResult(
        passed=matched == total,
        total_elements=total,
        matched_elements=matched,
        max_error_percent=max_error,
        tolerance_percent=tolerance_percent,
        max_ulp_error=max_ulp_error,
        tolerance_ulp=tolerance_ulp,
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
            f"{tolerance_result.tolerance_percent:g}%"
            f"{_format_ulp_tolerance(tolerance_result)} "
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


def _format_ulp_tolerance(result: FloatToleranceResult) -> str:
    if result.tolerance_ulp is None:
        return " "
    return f" or {result.tolerance_ulp} ULP (max_ulp={result.max_ulp_error})"


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
    parser.add_argument(
        "--float-tolerance-ulp",
        type=int,
        default=None,
        help="Optional ULP tolerance for floating tensors after bit-exact "
             "compare fails. BF16 defaults to 1 ULP when omitted.",
    )
    args = parser.parse_args(argv)

    if args.float_tolerance_percent < 0:
        parser.error("--float-tolerance-percent must be non-negative")
    if args.float_tolerance_ulp is not None and args.float_tolerance_ulp < 0:
        parser.error("--float-tolerance-ulp must be non-negative")

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
            args.float_tolerance_ulp,
        )
        print(report_with_float_tolerance(result, tolerance_result))
        return 0 if tolerance_result.passed else 1

    print(result.report())
    return 1


if __name__ == "__main__":
    sys.exit(main())
