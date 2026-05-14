"""Import golden data from cutetest .h header files.

Parses C header files containing static arrays and extracts all tensors
(input A/B, golden output, scales) into binary files + manifest.json.

Usage:
    python -m memverify.tools.import_header_golden \\
        --header /path/to/matmul_value_mnk_128_128_128_zeroinit.h \\
        --output golden/manual/tensor/matmul_i8_128_128_128_zeroinit/
"""

import argparse
import json
import re
import struct
from datetime import date
from pathlib import Path


def parse_defines(content: str) -> dict:
    """Extract #define constants from header content."""
    defines = {}
    for m in re.finditer(r"#define\s+(\w+)\s+(.+)", content):
        name = m.group(1)
        value = m.group(2).strip()
        try:
            defines[name] = int(value)
        except ValueError:
            defines[name] = value
    return defines


def find_array_declaration(content: str, array_name: str) -> tuple:
    """Find a static array declaration and return (c_type, dims, data_str).

    Returns:
        (c_type, dimensions_list, raw_data_string)
    """
    pattern = (
        rf"static\s+(\w+)\s+{re.escape(array_name)}"
        r"\s*((?:\[\d+\])+)"
        r"\s*__attribute__\s*\(\(aligned\(\d+\)\)\)\s*=\s*"
    )
    m = re.search(pattern, content)
    if not m:
        return None

    c_type = m.group(1)
    dims_str = m.group(2)
    dims = [int(d) for d in re.findall(r"\[(\d+)\]", dims_str)]

    start = m.end()
    brace_count = 0
    end = start
    for i in range(start, len(content)):
        if content[i] == "{":
            brace_count += 1
        elif content[i] == "}":
            brace_count -= 1
            if brace_count == 0:
                end = i
                break

    data_str = content[start:end + 1]
    return c_type, dims, data_str


def parse_array_data(data_str: str) -> list:
    """Parse C array initializer into nested Python list."""
    normalized = data_str.replace("{", "[").replace("}", "]")
    return eval(normalized)


_C_TYPE_TO_STRUCT = {
    "char": "b",
    "int8_t": "b",
    "uint8_t": "B",
    "short": "h",
    "int16_t": "h",
    "uint16_t": "H",
    "int": "i",
    "int32_t": "i",
    "uint32_t": "I",
    "long": "q",
    "int64_t": "q",
    "uint64_t": "Q",
}

_C_TYPE_TO_BITS = {
    "char": 8,
    "int8_t": 8,
    "uint8_t": 8,
    "short": 16,
    "int16_t": 16,
    "uint16_t": 16,
    "int": 32,
    "int32_t": 32,
    "uint32_t": 32,
    "long": 64,
    "int64_t": 64,
    "uint64_t": 64,
}


def flatten(data: list) -> list:
    result = []
    for item in data:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result


def data_to_bytes(data: list, c_type: str) -> bytes:
    fmt_char = _C_TYPE_TO_STRUCT.get(c_type)
    if not fmt_char:
        raise ValueError(f"unsupported C type: {c_type}")
    flat = flatten(data)
    return struct.pack(f"<{len(flat)}{fmt_char}", *flat)


def tensor_descriptor(c_type: str, dims: list, path: str) -> dict:
    element_bits = _C_TYPE_TO_BITS.get(c_type, 32)
    element_bytes = element_bits // 8
    cols = dims[1] if len(dims) > 1 else 1
    stride_bytes = cols * element_bytes
    dtype_map = {8: "I8", 16: "I16", 32: "I32", 64: "I64"}
    return {
        "path": path,
        "element_bits": element_bits,
        "dtype": dtype_map.get(element_bits, "I32"),
        "layout": "row_major",
        "shape": list(dims),
        "stride_bytes": stride_bytes,
    }


def extract_array(content: str, name: str) -> tuple:
    """Extract an array. Returns (c_type, dims, data_list) or None."""
    result = find_array_declaration(content, name)
    if result is None:
        return None
    c_type, dims, data_str = result
    data = parse_array_data(data_str)
    return c_type, dims, data


def import_header(header_path: str, output_dir: str):
    """Import all tensors from .h file to binary files + manifest.json."""
    header = Path(header_path)
    output = Path(output_dir)

    if not header.exists():
        raise FileNotFoundError(f"Header not found: {header}")

    content = header.read_text()
    defines = parse_defines(content)
    output.mkdir(parents=True, exist_ok=True)

    written = {}

    # Extract A
    a_result = extract_array(content, "a")
    if a_result:
        c_type, dims, data = a_result
        raw = data_to_bytes(data, c_type)
        (output / "input_a.bin").write_bytes(raw)
        written["A"] = tensor_descriptor(c_type, dims, "input_a.bin")
        print(f"  input_a.bin: {len(raw)} bytes ({c_type}[{']['.join(str(d) for d in dims)}])")

    # Extract B
    b_result = extract_array(content, "b")
    if b_result:
        c_type, dims, data = b_result
        raw = data_to_bytes(data, c_type)
        (output / "input_b.bin").write_bytes(raw)
        written["B"] = tensor_descriptor(c_type, dims, "input_b.bin")
        print(f"  input_b.bin: {len(raw)} bytes ({c_type}[{']['.join(str(d) for d in dims)}])")

    # Extract golden output (gloden_c is the consistent golden array name)
    golden_result = extract_array(content, "gloden_c")
    if golden_result:
        c_type, dims, data = golden_result
        raw = data_to_bytes(data, c_type)
        (output / "golden.bin").write_bytes(raw)
        written["D"] = tensor_descriptor(c_type, dims, "golden.bin")
        print(f"  golden.bin:  {len(raw)} bytes ({c_type}[{']['.join(str(d) for d in dims)}])")

    # Extract scale A (blockscale dtypes)
    scale_a_result = extract_array(content, "a_scale")
    if scale_a_result:
        c_type, dims, data = scale_a_result
        raw = data_to_bytes(data, c_type)
        (output / "scale_a.bin").write_bytes(raw)
        written["scale_a"] = tensor_descriptor(c_type, dims, "scale_a.bin")
        print(f"  scale_a.bin: {len(raw)} bytes ({c_type}[{']['.join(str(d) for d in dims)}])")

    # Extract scale B (blockscale dtypes)
    scale_b_result = extract_array(content, "b_scale")
    if scale_b_result:
        c_type, dims, data = scale_b_result
        raw = data_to_bytes(data, c_type)
        (output / "scale_b.bin").write_bytes(raw)
        written["scale_b"] = tensor_descriptor(c_type, dims, "scale_b.bin")
        print(f"  scale_b.bin: {len(raw)} bytes ({c_type}[{']['.join(str(d) for d in dims)}])")

    # Build manifest
    case_id = header.stem
    case_id = re.sub(r"^matmul_value_", "", case_id)
    case_id = re.sub(r"^conv_value_", "", case_id)

    manifest = {
        "id": case_id,
        "op": "matmul",
        "tensors": written,
        "blockscale": {
            "scale_a": "scale_a" in written,
            "scale_b": "scale_b" in written,
        } if "scale_a" in written else None,
        "defines": {
            "M": defines.get("APPLICATION_M"),
            "N": defines.get("APPLICATION_N"),
            "K": defines.get("APPLICATION_K"),
            "bias_type": defines.get("BIAS_TYPE"),
            "transpose": defines.get("TRANSPOSE_RESULT"),
            "element_type": defines.get("ELEMENT_TYPE"),
        },
        "generator": {
            "tool": "manual_import",
            "created": date.today().isoformat(),
        },
    }

    manifest_path = output / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  manifest.json written")

    print(f"\nSummary: {header.name} -> {output}/")
    print(f"  M={defines.get('APPLICATION_M')}, N={defines.get('APPLICATION_N')}, "
          f"K={defines.get('APPLICATION_K')}, BIAS_TYPE={defines.get('BIAS_TYPE')}, "
          f"TRANSPOSE={defines.get('TRANSPOSE_RESULT')}")


def main():
    parser = argparse.ArgumentParser(
        description="Import all tensors from cutetest .h header files"
    )
    parser.add_argument(
        "--header", required=True,
        help="Path to the .h header file",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory for binary files and manifest.json",
    )
    args = parser.parse_args()
    import_header(args.header, args.output)


if __name__ == "__main__":
    main()
