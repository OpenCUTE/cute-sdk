// 本程序生成一个小尺寸 GEMM 测试用例。
//
// A、B、scale 和 C 的初值会被写进一个 C 头文件；随后该头文件会被
// cute_Matmul_cmodel_gemm.c 包含，并编译成 RISC-V 程序，在 Chipyard/CUTE
// 仿真中执行。这里生成的 golden 不使用旧 CUTE FPE 的软件实现，而是调用
// 仓库内 SystemVerilog FPE 对应的 C-model（mymac / mymac_scale）。
//
// 除了 golden 的数值来源不同，DMA 装载、Scratchpad、MatrixTE 计算和
// 写回的外层流程刻意保持与原 datatype_mm_test 一致。

// gendata.h 间接提供数据格式描述符、随机数生成器，以及 mymac/mymac_scale。
#include "gendata.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
// 本文件中的辅助函数只在当前编译单元可见，避免与其他测试程序重名。

// 默认从一个 64 x 64 Scratchpad tile 开始。M/N 可由命令行扩展到多个
// 64 x 64 tile；本测试限制 512，覆盖原始 CUTE GEMM 的常用最大方阵规模。
constexpr int kDefaultM = 64;
constexpr int kDefaultN = 64;
constexpr int kMaxMatrixDimension = 512;

// 默认一次 FPE transaction 的 A/B 宽度为 512 bit。实际值由 --fpe-bytes
// 指定：64 Byte 对应 512 bit，32 Byte 对应 256 bit。
constexpr int kDefaultFpeBytes = 64;

// CUTE 的 bias_type：1 表示不从内存累加 C，3 表示每个 D[i][j] 都从
// C[i][j] 开始累加。测试接口使用更直观的 zero/random 名称，避免调用者直接
// 依赖硬件编码。
enum class BiasMode {
  Zero,
  Random,
  Repeat,
};

// 一种 CUTE opcode 在本测试中的完整数据布局描述。
struct TypeSpec {
  int cute_dtype;       // 写入 CUTE 宏指令的 opcode，当前覆盖 0..14 和 19。
  int cmodel_dtype;     // 传给新版 C-model 的 dtype；与 CUTE opcode 不一定相同。
  const char* name;     // 仅用于日志的可读名称。
  int element_bits;     // A/B 的逻辑元素宽度，不含 scale。
  int default_k;        // 该 dtype 的默认逻辑 GEMM K 维度。
  bool has_scale;       // 是否应调用带 scale 的 mymac_scale 接口。
  bool packed_nibble;   // 两个 4-bit 逻辑元素打包在一个字节中。
  bool integer_nibble_payload;  // packed_nibble 时，是否生成 S4/U4 原始编码而非 E2M1 FP4。
};

// 将 CUTE 支持的 opcode 映射为本测试的 C-model 格式和数据布局。
// dtype 7/8、11/12 在新版 C-model 中使用扩展 dtype 编号，故两侧编号不同。
bool get_type_spec(int cute_dtype, TypeSpec* spec) {
  switch (cute_dtype) {
    case 0: *spec = {0, 0, "I8I8I32", 8, 64, false, false, false}; return true;
    case 1: *spec = {1, 1, "F16F16F32", 16, 64, false, false, false}; return true;
    case 2: *spec = {2, 2, "BF16BF16F32", 16, 64, false, false, false}; return true;
    case 3: *spec = {3, 3, "TF32TF32F32", 32, 64, false, false, false}; return true;
    case 4: *spec = {4, 4, "I8U8I32", 8, 64, false, false, false}; return true;
    case 5: *spec = {5, 5, "U8I8I32", 8, 64, false, false, false}; return true;
    case 6: *spec = {6, 6, "U8U8I32", 8, 64, false, false, false}; return true;
    case 7: *spec = {7, 57, "MXFP8E4M3F32", 8, 64, true, false, false}; return true;
    case 8: *spec = {8, 63, "MXFP8E5M2F32", 8, 64, true, false, false}; return true;
    case 9: *spec = {9, 9, "NVFP4F32", 4, 128, true, true, false}; return true;
    case 10: *spec = {10, 10, "MXFP4F32", 4, 128, true, true, false}; return true;
    case 11: *spec = {11, 32, "FP8E4M3F32", 8, 64, false, false, false}; return true;
    case 12: *spec = {12, 38, "FP8E5M2F32", 8, 64, false, false, false}; return true;
    case 13: *spec = {13, 75, "MXFP6E3M2F32", 8, 64, true, false, false}; return true;
    case 14: *spec = {14, 69, "MXFP6E2M3F32", 8, 64, true, false, false}; return true;
    case 15: *spec = {15, 15, "I4I4I32", 4, 128, false, true, true}; return true;
    case 16: *spec = {16, 16, "I4U4I32", 4, 128, false, true, true}; return true;
    case 17: *spec = {17, 17, "U4I4I32", 4, 128, false, true, true}; return true;
    case 18: *spec = {18, 18, "U4U4I32", 4, 128, false, true, true}; return true;
    case 19: *spec = {19, 19, "MXINT4F32", 4, 128, true, true, true}; return true;
    default: return false;
  }
}

bool parse_u32(const char* text, uint32_t* value) {
  // base=0 允许用户输入十进制、0x 前缀十六进制等 C 风格整数。
  char* end = nullptr;
  const unsigned long parsed = std::strtoul(text, &end, 0);
  if (end == text || *end != '\0' || parsed > 0xffffffffUL) {
    return false;
  }
  *value = static_cast<uint32_t>(parsed);
  return true;
}

bool parse_int(const char* text, int* value) {
  // CUTE opcode 是一个小的无符号字段；这里提前限制到 0..255。
  char* end = nullptr;
  const long parsed = std::strtol(text, &end, 0);
  if (end == text || *end != '\0' || parsed < 0 || parsed > 255) {
    return false;
  }
  *value = static_cast<int>(parsed);
  return true;
}

bool parse_fpe_bytes(const char* text, int* value) {
  int parsed = 0;
  if (!parse_int(text, &parsed) || (parsed != 32 && parsed != 64)) {
    return false;
  }
  *value = parsed;
  return true;
}

bool parse_matrix_dimension(const char* text, int* value) {
  // M/N 是软件 GEMM 的全局维度。这里不允许零、负数或超过本回归范围的值，
  // 以免一次误操作生成过大的静态 RISC-V 数据段。
  char* end = nullptr;
  const long parsed = std::strtol(text, &end, 0);
  if (end == text || *end != '\0' || parsed <= 0 ||
      parsed > kMaxMatrixDimension) {
    return false;
  }
  *value = static_cast<int>(parsed);
  return true;
}

bool parse_logical_k(const char* text, int* value) {
  char* end = nullptr;
  const long parsed = std::strtol(text, &end, 0);
  if (end == text || *end != '\0' || parsed <= 0 || parsed > 16384 ||
      parsed % 64 != 0) {
    return false;
  }
  *value = static_cast<int>(parsed);
  return true;
}

bool parse_transpose(const char* text, bool* value) {
  if (std::strcmp(text, "0") == 0) {
    *value = false;
    return true;
  }
  if (std::strcmp(text, "1") == 0) {
    *value = true;
    return true;
  }
  return false;
}

bool parse_positive_dimension(const char* text, int* value) {
  char* end = nullptr;
  const long parsed = std::strtol(text, &end, 0);
  if (end == text || *end != '\0' || parsed <= 0 || parsed > 512) {
    return false;
  }
  *value = static_cast<int>(parsed);
  return true;
}

bool parse_bias_mode(const char* text, BiasMode* value) {
  if (std::strcmp(text, "zero") == 0) {
    *value = BiasMode::Zero;
    return true;
  }
  if (std::strcmp(text, "random") == 0) {
    *value = BiasMode::Random;
    return true;
  }
  if (std::strcmp(text, "repeat") == 0) {
    *value = BiasMode::Repeat;
    return true;
  }
  return false;
}

const char* bias_mode_name(BiasMode mode) {
  switch (mode) {
    case BiasMode::Zero: return "zero";
    case BiasMode::Random: return "random";
    case BiasMode::Repeat: return "repeat";
  }
  return "unknown";
}

int cute_bias_type(BiasMode mode) {
  switch (mode) {
    case BiasMode::Zero: return 1;
    case BiasMode::Repeat: return 2;
    case BiasMode::Random: return 3;
  }
  return 0;
}

int32_t generate_random_c_raw(const MmaDescriptor& descriptor) {
  // 整数 MMA 的 CAdd 是 I32；浮点 MMA 的 CAdd 是 FP32 原始位模式。
  // 两种情况都存储在 int32_t 容器中，并在 mymac/mymac_scale 内按 dtype 解码。
  return descriptor_integer_result(descriptor)
      ? generate_random_s32()
      : static_cast<int32_t>(gen_random_fp(FP32));
}

int scale_group_elements(const TypeSpec& spec) {
  // 此处表达的是一个 8-bit scale 对应的逻辑 A/B 元素数。它决定一条完整
  // K 向量中的 scale 如何切分给各个 256/512-bit FPE transaction。
  switch (spec.cute_dtype) {
    case 9:  return 16;  // NVFP4: UE4M3 scale 对应 16 个 FP4 元素。
    case 7:
    case 8:
    case 10:
    case 13:
    case 14:
    case 19: return 32;  // MXFP8/MXFP6/MXFP4/MXINT4: 每 32 元素一个 scale。
    default: return 0;
  }
}

std::vector<int8_t> pack_scales_for_cute(const std::vector<int8_t>& logical_scales,
                                          int rows, int scale_count, int tile_rows,
                                          int scales_per_tensor_k) {
  // CUTE 以 Tensor_M/Tensor_N 为一个装载 tile。对于 K 超过一个 Tensor_K
  // 段的 scale dtype，DRAM 布局不是“每行全部 scale 连续”，而是：
  //   tile 0 的 K-segment 0 所有行 -> K-segment 1 所有行 -> ... -> tile 1 ...
  // TaskController 的 BlockScale_*_BaseVaddr 正是按此布局跳过 K segment 和
  // M/N tile。C-model golden 保持 logical_scales 的行优先布局；只有写入 DRAM
  // 的数组需要转换。
  const int k_segments = scale_count / scales_per_tensor_k;
  std::vector<int8_t> packed;
  packed.reserve(logical_scales.size());
  for (int tile_start = 0; tile_start < rows; tile_start += tile_rows) {
    const int tile_size = std::min(tile_rows, rows - tile_start);
    for (int k_segment = 0; k_segment < k_segments; ++k_segment) {
      for (int row = 0; row < tile_size; ++row) {
        const int source = (tile_start + row) * scale_count +
                           k_segment * scales_per_tensor_k;
        packed.insert(packed.end(),
                      logical_scales.begin() + source,
                      logical_scales.begin() + source + scales_per_tensor_k);
      }
    }
  }
  return packed;
}

void store_raw(std::vector<uint8_t>* values, int index, int element_bits, uint32_t raw) {
  // values 统一按字节保存所有 dtype。根据元素位宽计算第 index 个元素的
  // 字节偏移，再以 memcpy 保存原始编码，避免 C++ 指针类型转换和对齐问题。
  const size_t offset = static_cast<size_t>(index) * (element_bits / 8);
  std::memcpy(values->data() + offset, &raw, element_bits / 8);
}

uint32_t load_raw(const std::vector<uint8_t>& values, int index, int element_bits) {
  // 与 store_raw 对称：从通用 byte buffer 恢复一个原始编码。
  // 对 8/16-bit 元素，高位保持为 0；调用者会再按对应有符号类型转换。
  uint32_t raw = 0;
  const size_t offset = static_cast<size_t>(index) * (element_bits / 8);
  std::memcpy(&raw, values.data() + offset, element_bits / 8);
  return raw;
}

void unpack_nibbles_for_integer_cmodel(const uint8_t* packed, int element_count,
                                        std::vector<int8_t>* unpacked) {
  // CUTE/RTL 的每个 byte 按 {element[2i+1], element[2i]} 紧密打包；但
  // fmac_generic_integer_mac() 每次从 src[i] 的低 4 bit 读取一个逻辑元素。
  // 此临时数组只服务于 C-model golden 计算，不会写入 RISC-V 程序或 DRAM。
  unpacked->resize(element_count);
  for (int element = 0; element < element_count; ++element) {
    const uint8_t byte = packed[element / 2];
    (*unpacked)[element] = static_cast<int8_t>(
        (byte >> ((element & 1) * 4)) & 0x0f);
  }
}

void write_input_matrix(FILE* output, const char* name,
                        const std::vector<uint8_t>& values, int rows, int logical_k,
                        int element_bits, bool packed_nibble) {
  // 生成 RISC-V 测试程序要包含的二维 C 数组。
  // 对 4-bit 打包格式，C 中每个 int8_t 元素含两个逻辑元素，所以列数减半。
  const int columns = packed_nibble ? logical_k / 2 : logical_k;

  // C 的数组元素类型决定编译器如何计算地址；它必须和 CUTE 的 A/B stride 一致。
  const char* type = element_bits == 32 ? "int32_t" :
                     element_bits == 16 ? "int16_t" : "int8_t";
  std::fprintf(output,
               "static %s %s[%d][%d] __attribute__((aligned(256))) = {\n",
               type, name, rows, columns);
  for (int row = 0; row < rows; ++row) {
    std::fprintf(output, "{");
    for (int column = 0; column < columns; ++column) {
      int32_t printable = 0;
      // 按原始位模式读取，再转换为有符号十进制文本。即使 TF32/FP32 的文本
      // 显示为负数，写入 C 数组后的二进制编码仍保持不变。
      if (element_bits == 32) {
        printable = static_cast<int32_t>(load_raw(values, row * columns + column, 32));
      } else if (element_bits == 16) {
        printable = static_cast<int16_t>(load_raw(values, row * columns + column, 16));
      } else {
        printable = static_cast<int8_t>(values[row * columns + column]);
      }
      std::fprintf(output, "%d,", printable);
    }
    std::fprintf(output, "},\n");
  }
  std::fprintf(output, "};\n");
}

void write_i8_vector(FILE* output, const char* name, const std::vector<int8_t>& values) {
  // AScale/BScale 在软件测试中是一维数组：先放第 0 行/列的全部 scale，
  // 再放第 1 行/列的全部 scale。具体索引规则由调用端和 scale_count 共同决定。
  std::fprintf(output,
               "static int8_t %s[%zu] __attribute__((aligned(256))) = {\n",
               name, values.size());
  for (size_t index = 0; index < values.size(); ++index) {
    std::fprintf(output, "%d,", static_cast<int>(values[index]));
    if ((index + 1) % 32 == 0) {
      std::fprintf(output, "\n");
    }
  }
  std::fprintf(output, "};\n");
}

void write_i32_matrix(FILE* output, const char* name,
                      const std::vector<int32_t>& values,
                      int rows, int columns) {
  // C、D 与 golden 均按 M x N 的 FP32/INT32 原始 bit 模式输出。
  // 使用 int 是为了兼容已有 CUTE 测试程序的数组声明；本平台 int 为 32 bit。
  std::fprintf(output,
               "static int %s[%d][%d] __attribute__((aligned(256))) = {\n",
               name, rows, columns);
  for (int row = 0; row < rows; ++row) {
    std::fprintf(output, "{");
    for (int column = 0; column < columns; ++column) {
      std::fprintf(output, "%d,", values[row * columns + column]);
    }
    std::fprintf(output, "},\n");
  }
  std::fprintf(output, "};\n");
}

}  // namespace

int main(int argc, char** argv) {
  // 默认 seed 固定为 1，使同一命令可重复生成完全相同的随机测试向量。
  const char* output_path = nullptr;
  uint32_t seed = 1;
  int cute_dtype = -1;
  int fpe_bytes = kDefaultFpeBytes;
  int matrix_m = kDefaultM;
  int matrix_n = kDefaultN;
  int logical_k_override = 0;
  int tensor_k_bytes = 64;
  int scale_tile_m = 64;
  int scale_tile_n = 64;
  BiasMode bias_mode = BiasMode::Zero;
  bool transpose_result = false;

  for (int index = 1; index < argc; ++index) {
    // 支持的命令行形式：
    //   --output <生成的头文件路径>
    //   --cute-dtype <CUTE opcode>
    //   --seed <随机种子，可省略>
    //   --fpe-bytes <32|64>，分别对应 256/512-bit FPE
    //   --bias <zero|random|repeat>，分别对应零、完整随机、按行重复 CAdd
    //   --m/--n <1..512>，GEMM 输出矩阵的全局行/列维度
    //   --k <64 的倍数>，覆盖该 dtype 的默认 K，用于大 K-reduction 验证
    //   --transpose <0|1>，控制 D 是否按转置布局写回内存
    //   --tensor-k-bytes <N>，一个 CUTE Tensor_K 段的字节数（默认 64）
    //   --scale-tile-m/n <N>，CUTE A/B scale 装载 tile 高度/宽度（默认 64）
    if (std::strcmp(argv[index], "--output") == 0 && index + 1 < argc) {
      output_path = argv[++index];
    } else if (std::strcmp(argv[index], "--seed") == 0 && index + 1 < argc) {
      if (!parse_u32(argv[++index], &seed)) {
        std::fprintf(stderr, "invalid --seed value\n");
        return 2;
      }
    } else if (std::strcmp(argv[index], "--cute-dtype") == 0 && index + 1 < argc) {
      if (!parse_int(argv[++index], &cute_dtype)) {
        std::fprintf(stderr, "invalid --cute-dtype value\n");
        return 2;
      }
    } else if (std::strcmp(argv[index], "--fpe-bytes") == 0 && index + 1 < argc) {
      if (!parse_fpe_bytes(argv[++index], &fpe_bytes)) {
        std::fprintf(stderr, "--fpe-bytes must be 32 or 64\n");
        return 2;
      }
    } else if (std::strcmp(argv[index], "--m") == 0 && index + 1 < argc) {
      if (!parse_matrix_dimension(argv[++index], &matrix_m)) {
        std::fprintf(stderr, "--m must be in 1..%d\n", kMaxMatrixDimension);
        return 2;
      }
    } else if (std::strcmp(argv[index], "--n") == 0 && index + 1 < argc) {
      if (!parse_matrix_dimension(argv[++index], &matrix_n)) {
        std::fprintf(stderr, "--n must be in 1..%d\n", kMaxMatrixDimension);
        return 2;
      }
    } else if (std::strcmp(argv[index], "--k") == 0 && index + 1 < argc) {
      if (!parse_logical_k(argv[++index], &logical_k_override)) {
        std::fprintf(stderr, "--k must be a positive multiple of 64 up to 16384\n");
        return 2;
      }
    } else if (std::strcmp(argv[index], "--transpose") == 0 && index + 1 < argc) {
      if (!parse_transpose(argv[++index], &transpose_result)) {
        std::fprintf(stderr, "--transpose must be 0 or 1\n");
        return 2;
      }
    } else if (std::strcmp(argv[index], "--tensor-k-bytes") == 0 && index + 1 < argc) {
      if (!parse_positive_dimension(argv[++index], &tensor_k_bytes)) {
        std::fprintf(stderr, "--tensor-k-bytes must be in 1..512\n");
        return 2;
      }
    } else if (std::strcmp(argv[index], "--scale-tile-m") == 0 && index + 1 < argc) {
      if (!parse_positive_dimension(argv[++index], &scale_tile_m)) {
        std::fprintf(stderr, "--scale-tile-m must be in 1..512\n");
        return 2;
      }
    } else if (std::strcmp(argv[index], "--scale-tile-n") == 0 && index + 1 < argc) {
      if (!parse_positive_dimension(argv[++index], &scale_tile_n)) {
        std::fprintf(stderr, "--scale-tile-n must be in 1..512\n");
        return 2;
      }
    } else if (std::strcmp(argv[index], "--bias") == 0 && index + 1 < argc) {
      if (!parse_bias_mode(argv[++index], &bias_mode)) {
        std::fprintf(stderr, "--bias must be zero or random\n");
        return 2;
      }
    } else {
      std::fprintf(stderr,
                   "usage: %s --output PATH --cute-dtype 0..19 [--seed N] "
                   "[--fpe-bytes 32|64] [--bias zero|random|repeat] "
                   "[--m 1..512] [--n 1..512] [--k multiple-of-64] "
                   "[--transpose 0|1] [--tensor-k-bytes N] "
                   "[--scale-tile-m N] [--scale-tile-n N]\n", argv[0]);
      return 2;
    }
  }

  TypeSpec spec{};
  // 先把用户给出的 CUTE opcode 映射成数据格式描述；不支持的 opcode 立即失败，
  // 防止生成的头文件与 CUTE 硬件实际支持的格式不一致。
  if (!output_path || !get_type_spec(cute_dtype, &spec)) {
    std::fprintf(stderr, "only CUTE dtypes 0..19 are currently supported\n");
    return 2;
  }

  const MmaDescriptor* descriptor = mma_get_descriptor(spec.cmodel_dtype);
  // Descriptor 是新版 C-model 中该数据类型的权威描述，包括 A/B 类型、是否
  // 使用 scale 等。此处交叉检查 TypeSpec，避免手工映射遗漏或写错。
  if (!descriptor || descriptor->uses_scale != spec.has_scale) {
    std::fprintf(stderr, "invalid C-model descriptor for CUTE dtype %d\n", cute_dtype);
    return 2;
  }

  const int logical_k = logical_k_override ? logical_k_override : spec.default_k;
  const int scale_group_size = scale_group_elements(spec);
  if (spec.has_scale && (scale_group_size == 0 || logical_k % scale_group_size != 0)) {
    std::fprintf(stderr, "K=%d does not align to the scale group for CUTE dtype %d\n",
                 logical_k, cute_dtype);
    return 2;
  }
  const int scale_count = spec.has_scale ? logical_k / scale_group_size : 0;
  const int tensor_k_elements = tensor_k_bytes * 8 / spec.element_bits;
  const int scales_per_tensor_k = spec.has_scale
      ? tensor_k_elements / scale_group_size : 0;
  if (spec.has_scale &&
      (tensor_k_elements <= 0 || tensor_k_elements % scale_group_size != 0 ||
       logical_k % tensor_k_elements != 0 || matrix_m % scale_tile_m != 0 ||
       matrix_n % scale_tile_n != 0)) {
    std::fprintf(stderr,
                 "scale layout requires K/Tensor_K and M/N tile alignment for CUTE dtype %d\n",
                 cute_dtype);
    return 2;
  }

  // A 的一行和 B 的一列都恰好是一个长度为 K 的 dot-product 输入。
  // 4-bit 打包格式每个元素占 4 bit，故两个元素共用一个字节；其他格式按元素位宽计算。
  const int bytes_per_row = spec.packed_nibble ? logical_k / 2 : logical_k * spec.element_bits / 8;

  // 一行/列必须可切成整数个 FPE 向量。以 256-bit FPE 为例：
  //   INT8, K=64   -> 64 Byte  -> 2 个 transaction；
  //   TF32, K=64   -> 256 Byte -> 8 个 transaction；
  //   NVFP4, K=128 -> 64 Byte  -> 2 个 transaction。
  if (bytes_per_row <= 0 || bytes_per_row % fpe_bytes != 0) {
    std::fprintf(stderr, "invalid vector packing for CUTE dtype %d\n", cute_dtype);
    return 2;
  }

  // 两套随机数工具都会被使用：C-model 的生成器和标准 C rand()。
  // 两者同时复位，确保同一种 seed 生成完全确定的测试用例。
  reset_random_generator(seed);
  std::srand(seed);

  // a/b 使用字节数组统一容纳不同位宽的数据。布局为：
  //   a: [M][bytes_per_row]，每一行是一条 A 向量；
  //   b: [N][bytes_per_row]，每一项是一条 B 向量，即 B 按列存放。
  // 这种布局让计算 D[row][column] 时可直接取得一条 A 行和一条 B 列。
  std::vector<uint8_t> a(matrix_m * bytes_per_row);
  std::vector<uint8_t> b(matrix_n * bytes_per_row);
  std::vector<int8_t> a_scale(matrix_m * scale_count);
  std::vector<int8_t> b_scale(matrix_n * scale_count);
  // golden 存每个 D 元素的 32-bit FP32/INT32 原始结果；c 是提交给 CUTE 的
  // 初始累加矩阵，d 始终是独立的零初始化输出缓冲区。
  std::vector<int32_t> golden(matrix_m * matrix_n);
  const int c_rows = bias_mode == BiasMode::Repeat ? 1 : matrix_m;
  std::vector<int32_t> c(c_rows * matrix_n, 0);
  std::vector<int32_t> d(matrix_m * matrix_n, 0);

  if (bias_mode != BiasMode::Zero) {
    for (int index = 0; index < c_rows * matrix_n; ++index) {
      c[index] = generate_random_c_raw(*descriptor);
    }
  }

  // 逐行生成 A 及其对应 scale。对于带 scale 的格式，scale_count 个 scale
  // 属于同一条 A 行，后续会送给该行上的所有 PE。
  for (int row = 0; row < matrix_m; ++row) {
    uint8_t* const row_data = a.data() + row * bytes_per_row;
    if (spec.packed_nibble) {
      // 每个 byte 内含两个 4-bit 逻辑元素。MXINT4 使用 S4 编码；
      // NVFP4/MXFP4 则使用 E2M1 浮点编码。
      if (spec.integer_nibble_payload) {
        fill_random_s4_payload_bytes(reinterpret_cast<int8_t*>(row_data), bytes_per_row);
      } else {
        fill_random_e2m1_payload_bytes(reinterpret_cast<int8_t*>(row_data), bytes_per_row);
      }
    } else {
      // 非 FP4 格式逐元素写入原始编码。例如 TF32 在生成器中已将低 13 个
      // FP32 尾数字段清零，但它仍占用完整的 32-bit 存储空间。
      for (int element = 0; element < logical_k; ++element) {
        store_raw(&a, row * logical_k + element, spec.element_bits,
                  generate_random_operand_raw(descriptor->a_kind));
      }
    }
    for (int scale = 0; scale < scale_count; ++scale) {
      // NVFP4 使用 UE4M3 scale；其余当前带 scale 格式使用 UE8M0。
      // 没有 scale 的类型 scale_count=0，因此此循环自然不会执行。
      const int format = spec.cmodel_dtype == MMA_DTYPE_NVFP4
                             ? legacy_mxf4_scale_format(*descriptor)
                             : UE8M0;
      a_scale[row * scale_count + scale] =
          static_cast<int8_t>(gen_random_fp(format));
    }
  }

  // B 的存储单位是“列向量”，而非通常 C 语言矩阵的按行 B[row][column] 布局。
  // 这样 b[column] 就是计算 D[*][column] 所需的 K 个 B 元素。
  for (int column = 0; column < matrix_n; ++column) {
    uint8_t* const column_data = b.data() + column * bytes_per_row;
    if (spec.packed_nibble) {
      if (spec.integer_nibble_payload) {
        fill_random_s4_payload_bytes(reinterpret_cast<int8_t*>(column_data), bytes_per_row);
      } else {
        fill_random_e2m1_payload_bytes(reinterpret_cast<int8_t*>(column_data), bytes_per_row);
      }
    } else {
      for (int element = 0; element < logical_k; ++element) {
        store_raw(&b, column * logical_k + element, spec.element_bits,
                  generate_random_operand_raw(descriptor->b_kind));
      }
    }
    for (int scale = 0; scale < scale_count; ++scale) {
      const int format = spec.cmodel_dtype == MMA_DTYPE_NVFP4
                             ? legacy_mxf4_scale_format(*descriptor)
                             : UE8M0;
      b_scale[column * scale_count + scale] =
          static_cast<int8_t>(gen_random_fp(format));
    }
  }

  // 供 CUTE DMA 读取的 scale 采用 tile/K-segment 布局；golden 计算仍从
  // 行/列优先的逻辑数组读取，二者刻意分离以避免改变 C-model 的语义。
  const std::vector<int8_t> a_scale_dram = spec.has_scale
      ? pack_scales_for_cute(a_scale, matrix_m, scale_count, scale_tile_m,
                             scales_per_tensor_k)
      : std::vector<int8_t>();
  const std::vector<int8_t> b_scale_dram = spec.has_scale
      ? pack_scales_for_cute(b_scale, matrix_n, scale_count, scale_tile_n,
                             scales_per_tensor_k)
      : std::vector<int8_t>();

  // 无 scale 的 mymac() 参数使用本 transaction 的逻辑元素数。对于打包的
  // FP4/MXINT4，mymac_scale() 的 size 采用 payload 字节数，因此是 fpe_bytes。
  const int logical_elements_per_fpe =
      spec.packed_nibble ? fpe_bytes * 2 : fpe_bytes * 8 / spec.element_bits;
  const int cmodel_size_per_fpe =
      spec.packed_nibble ? fpe_bytes : logical_elements_per_fpe;

  // 一条完整 K 向量被切成多少个 256/512-bit FPE transaction。
  const int chunks = bytes_per_row / fpe_bytes;
  const int scales_per_fpe = spec.has_scale
      ? logical_elements_per_fpe / scale_group_size : 0;

  if (spec.has_scale &&
      (scale_group_size == 0 || logical_elements_per_fpe % scale_group_size != 0 ||
       scale_count != chunks * scales_per_fpe)) {
    std::fprintf(stderr, "invalid scale layout for CUTE dtype %d\n", cute_dtype);
    return 2;
  }

  // 为 M x N 个输出元素逐一生成 bit-exact golden：
  //   golden[row][column] = A[row][:] dot B[column][:] + C[row][column]
  // zero 模式以 0 为 C 初值，random 模式以 c[row][column] 为 C 初值。对跨
  // chunk 的类型，第一 chunk 消费该初值，之后上一 chunk 的部分和作为下一轮
  // CAdd 输入，最终得到完整 K 归约结果。
  for (int row = 0; row < matrix_m; ++row) {
    for (int column = 0; column < matrix_n; ++column) {
      const int c_index = bias_mode == BiasMode::Repeat
          ? column : row * matrix_n + column;
      int32_t accum = c[c_index];
      for (int chunk = 0; chunk < chunks; ++chunk) {
        // 取当前 K 分段的两个 256/512-bit 输入。A 选当前行，B 选当前列。
        uint8_t* a_vector = a.data() + row * bytes_per_row + chunk * fpe_bytes;
        uint8_t* b_vector = b.data() + column * bytes_per_row + chunk * fpe_bytes;
        if (spec.has_scale) {
          // AScale 随 A 行索引，BScale 随 B 列索引。scale_count 是完整 K 向量
          // 的总数，所以第 chunk 轮必须跳过前面 chunk 已消费的 scale，不能复用
          // 第 0 段。accum 同时作为 CAdd 输入和新的部分和/最终结果接收位置。
          const int scale_offset = chunk * scales_per_fpe;
          accum = mymac_scale(
              a_vector, a_scale.data() + row * scale_count + scale_offset,
              b_vector, b_scale.data() + column * scale_count + scale_offset,
              &accum, cmodel_size_per_fpe, spec.cmodel_dtype);
        } else {
          // 无 scale 的格式直接走通用 C-model FMA。false 表示要求 FP32/INT32
          // 输出，而不是某些 dtype 可选的 FP16 输出。
          if (mma_is_rtl_int4_dtype(spec.cmodel_dtype)) {
            std::vector<int8_t> a_cmodel;
            std::vector<int8_t> b_cmodel;
            unpack_nibbles_for_integer_cmodel(
                a_vector, logical_elements_per_fpe, &a_cmodel);
            unpack_nibbles_for_integer_cmodel(
                b_vector, logical_elements_per_fpe, &b_cmodel);
            accum = mymac(a_cmodel.data(), b_cmodel.data(), &accum,
                          logical_elements_per_fpe,
                          spec.cmodel_dtype, false);
          } else {
            accum = mymac(a_vector, b_vector, &accum, logical_elements_per_fpe,
                          spec.cmodel_dtype, false);
          }
        }
      }
      golden[row * matrix_n + column] = accum;
    }
  }

  // 将输入和 golden 写成 RISC-V 测试程序可直接 #include 的 C 头文件。
  FILE* output = std::fopen(output_path, "w");
  if (!output) {
    std::perror(output_path);
    return 2;
  }
  std::fprintf(output, "// Generated by generate_gemm_golden.cpp. Do not edit.\n");

  // 这些宏由 cute_Matmul_cmodel_gemm.c 用于构造 CUTE 宏指令、设定 DMA
  // stride 和决定是否传入 A/B scale 地址。
  std::fprintf(output, "#define APPLICATION_M %d\n", matrix_m);
  std::fprintf(output, "#define APPLICATION_N %d\n", matrix_n);
  std::fprintf(output, "#define APPLICATION_K %d\n", logical_k);
  std::fprintf(output, "#define ELEMENT_TYPE %d\n", spec.cute_dtype);
  std::fprintf(output, "#define CMODEL_DTYPE %d\n", spec.cmodel_dtype);
  std::fprintf(output, "#define GOLDEN_SEED %u\n", seed);
  std::fprintf(output, "#define FPE_BYTES %d\n", fpe_bytes);
  std::fprintf(output, "#define HAS_SCALE %d\n", spec.has_scale ? 1 : 0);
  std::fprintf(output, "#define SCALE_COUNT %d\n", scale_count);
  std::fprintf(output, "#define BIAS_TYPE %d\n", cute_bias_type(bias_mode));
  std::fprintf(output, "#define STRIDE_A %d\n", bytes_per_row);
  std::fprintf(output, "#define STRIDE_B %d\n", bytes_per_row);
  std::fprintf(output, "#define STRIDE_C %d\n", matrix_n * 4);
  std::fprintf(output, "#define STRIDE_D %d\n", (transpose_result ? matrix_m : matrix_n) * 4);
  std::fprintf(output, "#define TRANSPOSE_RESULT %d\n", transpose_result ? 1 : 0);
  write_input_matrix(output, "a", a, matrix_m, logical_k, spec.element_bits, spec.packed_nibble);
  write_input_matrix(output, "b", b, matrix_n, logical_k, spec.element_bits, spec.packed_nibble);
  if (spec.has_scale) {
    write_i8_vector(output, "a_scale", a_scale_dram);
    write_i8_vector(output, "b_scale", b_scale_dram);
  }
  // gloden_c 的拼写是已有测试程序使用的历史接口名称，不能在这里改成 golden_c，
  // 否则 cute_Matmul_cmodel_gemm.c 将无法找到该数组。
  write_i32_matrix(output, "gloden_c", golden, matrix_m, matrix_n);
  write_i32_matrix(output, "c", c, c_rows, matrix_n);
  write_i32_matrix(output, "d", d,
                   transpose_result ? matrix_n : matrix_m,
                   transpose_result ? matrix_m : matrix_n);
  std::fclose(output);

  std::printf("generated %s: %s, CUTE opcode %d -> C-model dtype %d, "
              "FPE %d bit, M=%d N=%d K=%d, transpose=%d, seed %u\n",
              output_path, spec.name, spec.cute_dtype, spec.cmodel_dtype,
              fpe_bytes * 8, matrix_m, matrix_n, logical_k,
              transpose_result ? 1 : 0, seed);
  std::printf("bias=%s (BIAS_TYPE=%d)\n", bias_mode_name(bias_mode),
              cute_bias_type(bias_mode));
  return 0;
}
