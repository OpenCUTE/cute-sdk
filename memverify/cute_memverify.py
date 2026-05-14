"""Byte-level compare engine for CUTE memverify.

Compares golden tensor data against trace-reconstructed tensor data
and produces a pass/fail report with mismatch details.
"""

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

from memverify.readers.cml_store_trace import CMLStoreTrace
from memverify.readers.golden_tensor import GoldenTensor


@dataclass
class Mismatch:
    row: int
    col: int
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
                    f"  [{m.row}][{m.col}] offset={m.byte_offset} "
                    f"expected={m.expected} ({m.expected_hex}) "
                    f"actual={m.actual} ({m.actual_hex})"
                )
        return "\n".join(lines)


_MAX_REPORTED_MISMATCHES = 20


def compare(
    golden: GoldenTensor,
    actual: list[list[int]],
) -> CompareResult:
    rows, cols = golden.shape
    total = golden.element_count()
    matched = 0
    mismatches: list[Mismatch] = []

    for r in range(rows):
        for c in range(cols):
            expected = golden[r, c]
            actual_val = actual[r][c]
            if expected == actual_val:
                matched += 1
            elif len(mismatches) < _MAX_REPORTED_MISMATCHES:
                mismatches.append(Mismatch(
                    row=r,
                    col=c,
                    byte_offset=golden._offset(r, c),
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
    args = parser.parse_args(argv)

    golden = GoldenTensor(args.manifest, tensor_name=args.tensor)
    trace = CMLStoreTrace(args.trace)

    base = trace.get_base_address()
    if base is None:
        print("[ERROR] No WriteRequest found in trace", file=sys.stderr)
        return 1

    actual = trace.get_tensor(base, golden.shape, golden.stride_bytes, golden.element_bits)
    result = compare(golden, actual)
    print(result.report())
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
