# Phase 1：人工 golden + memverify

> 状态：**待执行**
> 前置依赖：Phase 0（op spec + case manifest 已定义）
> 验收标准：一条命令可以比较 manual golden 和一份 store trace，至少一个 INT8 matmul case 通过验证。

---

## 目标

建立不依赖 cuteqemu / nvwa 的验证闭环：

```text
cutetest .h 文件 → 导入工具 → golden.bin + manifest.json
                                          ↑
实际仿真输出 .out → trace reader → 内存数据  ─→ compare engine → pass/fail
```

---

## 1.1 定义 golden manifest 格式

**状态**：Phase 0 已定义格式，此处直接复用。

golden manifest 只描述 golden 数据的二进制格式：

```json
{
  "id": "matmul_i8_128_128_128_zeroinit",
  "op": "matmul",
  "output": {
    "path": "golden.bin",
    "element_bits": 32,
    "dtype": "I32",
    "layout": "row_major",
    "shape": [128, 128],
    "stride_bytes": 512,
    "total_bytes": 65536
  },
  "generator": {
    "tool": "manual_import",
    "created": "2026-05-14"
  }
}
```

**执行步骤**：

- [ ] 1.1.1 无需额外工作，Phase 0 已完成格式定义

---

## 1.2 实现 golden tensor loader

**目标**：根据 golden manifest 读取 golden.bin，提供统一的 tensor 访问接口。

**输入**：`manifest.json` 路径
**输出**：Python 对象，可按 `[row][col]` 访问元素

**实现**：`memverify/readers/golden_tensor.py`

```python
class GoldenTensor:
    def __init__(self, manifest_path: str):
        # 读取 manifest.json
        # 打开 golden.bin（相对于 manifest.json 的路径）
        # 按 element_bits、layout、shape 解析 raw bytes

    def __getitem__(self, key):
        # 支持 golden[row][col] 访问

    def element_count(self) -> int:
        ...

    def total_bytes(self) -> int:
        ...
```

**关键细节**：
- golden.bin 存储为 raw binary，row_major，little-endian
- `element_bits=32` → 每 4 字节一个 int32 元素
- `stride_bytes` 可能大于 `cols * element_bytes`（有 padding），需正确处理

**执行步骤**：

- [ ] 1.2.1 创建 `memverify/readers/` 目录
- [ ] 1.2.2 实现 `GoldenTensor` 类
- [ ] 1.2.3 对 I8 matmul 128×128 的 golden.bin 做基本读取验证

---

## 1.3 实现旧 .h golden 导入工具

**目标**：将 cutetest 的 `matmul_value_*.h` 中的 golden 数组提取为 `golden.bin`。

**输入**：`.h` 文件路径 + golden 数组名
**输出**：`golden.bin` + `manifest.json`

**实现**：`memverify/tools/import_header_golden.py`

### .h 文件结构分析

cutetest 的 golden .h 文件有统一结构：

```c
// base_test / I8
#define APPLICATION_M 128
#define APPLICATION_N 128
#define APPLICATION_K 128
#define BIAS_TYPE 1
#define TRANSPOSE_RESULT 0
#define STRIDE_C 512
#define STRIDE_D 512

static char a[128][128] __attribute__((aligned(256))) = { ... };
static char b[128][128] __attribute__((aligned(256))) = { ... };
static int d[128][128] __attribute__((aligned(256))) = { ... };    // golden 输出
static int gloden_c[128][128] __attribute__((aligned(256))) = { ... };
static int c[128][128] __attribute__((aligned(256))) = { ... };

// mxfp8e4m3
static int8_t a[64][64] __attribute__((aligned(256))) = { ... };
static int8_t b[64][64] __attribute__((aligned(256))) = { ... };
static int gloden_c[64][64] __attribute__((aligned(256))) = { ... };  // golden 输出
static int d[64][64] __attribute__((aligned(256))) = { ... };
static int8_t a_scale[128] __attribute__((aligned(256))) = { ... };   // scale 数据
static int8_t b_scale[128] __attribute__((aligned(256))) = { ... };   // scale 数据
```

**关键差异**：

| 来源 | golden 数组名 | dtype | 含义 |
|------|--------------|-------|------|
| base_test I8 | `d` | int32 | 输出 D |
| base_test I8 transpose | `d` | int32 | 输出 D |
| mxfp8e4m3 | `gloden_c` | int32 | 输出（命名拼写不一致） |

### 导入工具逻辑

```python
def import_header(header_path, output_dir, array_name=None):
    # 1. 解析 #define 获取 M, N, K, BIAS_TYPE, TRANSPOSE_RESULT, STRIDE_*
    # 2. 定位目标数组（自动检测或由参数指定）
    # 3. 用正则或 C parser 提取数组数据
    # 4. 转为 raw binary (struct.pack)
    # 5. 生成 golden.bin 和 manifest.json
```

**数组提取策略**：
- 用 Python 的 `re` 匹配 `static (type) (name)[dims...] = {data};`
- `data` 部分直接用 `eval()` 转为 Python list（与现有 compare_result.py 策略一致）
- 按 row-major、little-endian 写入 binary

**执行步骤**：

- [ ] 1.3.1 创建 `memverify/tools/` 目录
- [ ] 1.3.2 实现 .h 解析器（提取 #define 和数组数据）
- [ ] 1.3.3 实现 golden.bin 写入（int32 → 4 bytes little-endian）
- [ ] 1.3.4 实现 manifest.json 自动生成（从 #define 推断 shape/dtype）
- [ ] 1.3.5 对 `matmul_value_mnk_128_128_128_zeroinit.h` 运行导入，验证输出正确

---

## 1.4 实现 CMemoryLoader store trace reader

**目标**：解析 Verilator 仿真输出的 store trace，重建实际写入内存的数据。

**输入**：CML_Store_trace.out 文件路径（或完整 Verilator output .out 文件）
**输出**：Python 对象，可按地址或坐标访问实际写入的数据

**实现**：`memverify/readers/cml_store_trace.py`

### Trace 格式分析

CML_Store_trace.out 是从 Verilator 完整 output 中过滤出 `[CMemoryLoader_Store<` 开头的行。

关键的 WriteRequest 行格式：

```
[CMemoryLoader_Store<...>]...WriteRequest: RequestVirtualAddr=0x..., ..., RequestData:<hex_data>
```

字段：
- `RequestVirtualAddr`：写入的虚拟地址（hex）
- `RequestData`：写入的数据（hex string，little-endian）
- `CurrentStore_BlockTensor_Major_DIM_Iter`：主维度迭代索引
- `CurrentStore_BlockTensor_Reduce_DIM_Iter`：缩减维度迭代索引

### 地址映射逻辑

store trace 中的数据不是按 row-major 连续写的，而是按加速器的分块策略写：

```python
# 地址计算（来自 compare_result.py）
addr_golden = tensor_block_baseaddr + major_index * app_stride + reduce_index * d_datatype
# app_stride = application_n * 4 (每行字节数)
# d_datatype = 4 (每个元素 4 字节)
```

### Reader 接口

```python
class CMLStoreTrace:
    def __init__(self, trace_path: str):
        # 解析所有 WriteRequest 行
        # 建立 (addr → data) 映射

    def get_tensor(self, base_addr: int, shape: tuple, stride_bytes: int, element_bits: int) -> list:
        # 从 trace 中的 store 请求重建完整 tensor
        # 按 major_index * stride + reduce_index * element_size 映射到 golden 坐标

    def get_data_at(self, addr: int, num_bytes: int) -> bytes:
        # 读取指定地址的原始数据
```

**执行步骤**：

- [ ] 1.4.1 实现 WriteRequest 行解析器
- [ ] 1.4.2 实现地址→数据映射
- [ ] 1.4.3 实现 tensor 重建（按 store 分块策略映射到 row-major）
- [ ] 1.4.4 支持从完整 .out 文件自动过滤 CML_Store_Store 行

---

## 1.5 实现 byte-level compare engine

**目标**：比较 golden tensor 和 trace 重建 tensor，输出 pass/fail 报告。

**实现**：`memverify/cute_memverify.py`

### Compare 接口

```python
def compare(golden: GoldenTensor, actual, case_manifest: dict) -> CompareResult:
    """
    golden: GoldenTensor 对象（从 golden.bin 加载）
    actual: 从 trace 重建的 tensor（list of lists 或 flat bytes）
    case_manifest: case.json 内容，用于确定比对模式
    """
```

### CompareResult 报告格式

```python
@dataclass
class Mismatch:
    tensor_index: int      # 第几个 tensor（当前只有输出 D）
    row: int
    col: int
    byte_offset: int       # 在 golden.bin 中的偏移
    address: int           # trace 中的虚拟地址（如果有）
    expected: int          # golden 值
    actual: int            # 实际值
    expected_hex: str
    actual_hex: str

@dataclass
class CompareResult:
    passed: bool
    total_elements: int
    matched_elements: int
    mismatch_count: int
    mismatches: list[Mismatch]  # 最多报告前 20 个
```

### 比对模式

Phase 1 只实现 `bit_exact` 模式：逐元素比较，不允许任何 bit 差异。

### CLI 入口

```bash
python -m memverify.cute_memverify \
    --case tests/runtime/runtime_matmul_i8_128_128_128_zeroinit/case.json \
    --trace run/CML_Store_trace.out
```

**执行步骤**：

- [ ] 1.5.1 实现 `compare()` 函数
- [ ] 1.5.2 实现 `CompareResult` 和 `Mismatch` 数据类
- [ ] 1.5.3 实现 CLI 入口（argparse）
- [ ] 1.5.4 输出人类可读的报告（pass/fail + mismatch 列表）

---

## 1.6 转换第一批人工 golden 数据

**目标**：用导入工具将 3 个 smoke case 的 .h golden 转换为 golden.bin + manifest.json。

### 目标结构

```text
golden/manual/tensor/
├── matmul_i8_128_128_128_zeroinit/
│   ├── manifest.json
│   └── golden.bin                    # 128×128×4 = 65,536 bytes
├── matmul_i8_128_128_128_zeroinit_transpose/
│   ├── manifest.json
│   └── golden.bin
└── matmul_mxfp8e4m3_64_64_64_zeroinit/
    ├── manifest.json
    └── golden.bin                    # 64×64×4 = 16,384 bytes
```

### 导入命令

```bash
python -m memverify.tools.import_header_golden \
    --header /root/opencute/CUTE/cutetest/base_test/matmul_value_mnk_128_128_128_zeroinit.h \
    --array d \
    --output golden/manual/tensor/matmul_i8_128_128_128_zeroinit/

python -m memverify.tools.import_header_golden \
    --header /root/opencute/CUTE/cutetest/base_test/matmul_value_mnk_128_128_128_zeroinit_transpose.h \
    --array d \
    --output golden/manual/tensor/matmul_i8_128_128_128_zeroinit_transpose/

python -m memverify.tools.import_header_golden \
    --header /root/opencute/CUTE/cutetest/datatype_mm_test/mxfp8e4m3/matmul_value_mxfp8_mnk_64_64_64_zeroinit.h \
    --array gloden_c \
    --output golden/manual/tensor/matmul_mxfp8e4m3_64_64_64_zeroinit/
```

**执行步骤**：

- [ ] 1.6.1 运行导入工具转换 3 个 .h 文件
- [ ] 1.6.2 验证 golden.bin 大小与 manifest.json 中的 total_bytes 一致
- [ ] 1.6.3 用 golden tensor loader 回读 golden.bin，确认数据与原始 .h 一致

---

## 执行顺序与依赖

```text
1.2 golden tensor loader ─────────────────┐
                                          │
1.3 .h 导入工具 ──> 1.6 转换 golden 数据  ├──> 1.5 compare engine ──> 端到端验证
                                          │
1.4 trace reader ─────────────────────────┘
```

建议执行顺序：
1. 并行开发 **1.2**（golden loader）、**1.3**（导入工具）、**1.4**（trace reader）
2. 三个基础组件就绪后，**1.6** 用导入工具转换 golden 数据
3. 最后 **1.5** 实现 compare engine，做端到端验证

---

## memverify 目录结构

```text
memverify/
├── cute_memverify.py          # compare engine + CLI 入口
├── readers/
│   ├── __init__.py
│   ├── golden_tensor.py       # golden.bin 读取
│   └── cml_store_trace.py     # CML_Store_trace.out 解析
└── tools/
    ├── __init__.py
    └── import_header_golden.py # .h → golden.bin 导入
```

---

## 不在 Phase 1 范围内的工作

- 不做 cuteqemu memory snapshot reader
- 不做 nvwa golden 生成
- 不做浮点 tolerance，只做 bit-exact
- 不做 conv / layer / model 级别的验证
- 不实现 test runner（Phase 4）
- 不写 cutelib 代码
