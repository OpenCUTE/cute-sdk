# Phase 0：迁移基线和资产清单

> 状态：**待执行**
> 前置依赖：无
> 验收标准：cuteisa artifacts 就位；迁移矩阵完成；第一批 smoke cases 的 op spec 和 case manifest 定义清楚。

---

## 0.1 生成 cuteisa artifacts

**目标**：确保 `cute-sdk/cuteisa/cute_isa_v1/` 下的所有 artifacts 与 `configs/cute_isa_versions/cute_isa_v1.yaml` 同步。

**现状盘点**：

| 文件 | 状态 | 大小 | 说明 |
|------|------|------|------|
| `cuteisa/cute_isa_v1/instruction.h` | 已存在 | 34,461 B | C 头文件，指令编码 + funct 宏 |
| `cuteisa/cute_isa_v1/isa.json` | 已存在 | 16,537 B | JSON 格式 ISA 定义，供 cuteqemu 消费 |
| `cuteisa/cute_isa_v1/cute_fpe.h` | 已存在 | 3,054 B | 数据类型 + FPE 相关宏 |
| `cuteisa/cute_isa_v1/isa_summary.md` | 已存在 | 4,771 B | 人类可读 ISA 摘要（从 isa.json 自动生成） |

**ISA 源文件**：`/root/opencute/CUTE/configs/cute_isa_versions/cute_isa_v1.yaml`

**执行步骤**：

- [ ] 0.1.1 对比 `cute_isa_v1.yaml` 的 enums/instructions 与现有 `instruction.h`、`isa.json`，确认一致性
- [ ] 0.1.2 如有差异，重新生成；如无差异，标记为已验证
- [ ] 0.1.3 确认 `cute_fpe.h` 内容覆盖了 yaml 中 `ElementDataType` 全部 13 种数据类型（I8 → FP8E5M2，value 0–12）
- [ ] 0.1.4 记录生成工具链（是手动生成还是有脚本/CI），便于后续 yaml 更新时重新生成

**验收**：
- 4 个文件均存在且与 yaml 源一致
- 生成方式有文档记录

---

## 0.2 建立 cutetest 迁移矩阵

**目标**：以结构化表格形式记录 `cutetest/` 中所有可迁移 case，标注来源、参数、迁移优先级。

### 矩阵结构

每个 case 记录以下字段：

```
case_id | source_path | level | op | dtype | shape | bias_mode | has_golden_h | has_compare_py | has_trace_out | priority
```

### 资产清单

#### base_test

| case_id | source_path | level | op | dtype | shape | bias_mode | golden_h | compare_py | priority |
|---------|-------------|-------|----|-------|-------|-----------|----------|------------|----------|
| `runtime_hello` | `base_test/cutehello.c` | runtime | hello | I8I8I32 | 128×128×128 | ZeroLoad | `matmul_value_mnk_128_128_128_zeroinit.h` | 无 | P0 |
| `matmul_i8_128_128_128_zeroinit` | `base_test/cute_Matmul_mnk_128_128_128_zeroinit.c` | tensor | matmul | I8I8I32 | 128×128×128 | ZeroLoad | `matmul_value_mnk_128_128_128_zeroinit.h` | 无 | P0 |
| `matmul_i8_128_128_128_zeroinit_transpose` | `base_test/cute_Matmul_mnk_128_128_128_zeroinit_transpose.c` | tensor | matmul | I8I8I32 | 128×128×128 | ZeroLoad + Transpose | `matmul_value_mnk_128_128_128_zeroinit_transpose.h` | 无 | P1 |
| `matmul_i8_128_128_128_fullbias` | `base_test/cute_Matmul_mnk_128_128_128_fullbias.c` | tensor | matmul | I8I8I32 | 128×128×128 | FullBias | `matmul_value_mnk_128_128_128_fullbias.h` | 无 | P1 |
| `matmul_i8_128_128_128_rowrepeat` | `base_test/cute_Matmul_mnk_128_128_128_rowreapeat.c` | tensor | matmul | I8I8I32 | 128×128×128 | RowRepeat | `matmul_value_mnk_128_128_128_rowreapeat.h` | 无 | P1 |
| `conv_i8_49_128_128_k1_s1` | `base_test/cute_conv_mnk_49_128_128_k1_s1_oh7.c` | tensor | conv | I8I8I32 | 49×128×128 k=1 s=1 | — | `conv_value_mnk_49_128_128_k1_s7.h` | 无 | P2 |
| `conv_i8_196_256_256_k3_s1` | `base_test/cute_conv_mnk_196_256_256_k3_s1_oh14.c` | tensor | conv | I8I8I32 | 196×256×256 k=3 s=1 | — | `conv_value_mnk_196_256_256_k3_s1_oh14.h` | 无 | P2 |

#### datatype_mm_test

| case_id | source_path | level | op | dtype | shape | bias_mode | golden_h | compare_py | priority |
|---------|-------------|-------|----|-------|-------|-----------|----------|------------|----------|
| `matmul_mxfp8e4m3_64_64_64_zeroinit` | `datatype_mm_test/mxfp8e4m3/cute_Matmul_mxfp8_mnk_64_64_64_zeroinit.c` | tensor | matmul | Mxfp8e4m3F32 | 64×64×64 | ZeroLoad | `matmul_value_mxfp8_mnk_64_64_64_zeroinit.h` | 有 | **P0** |
| `matmul_mxfp8e4m3_64_64_64_fullbias` | `datatype_mm_test/mxfp8e4m3/cute_Matmul_mxfp8_mnk_64_64_64_fullbias.c` | tensor | matmul | Mxfp8e4m3F32 | 64×64×64 | FullBias | `matmul_value_mxfp8_mnk_64_64_64_fullbias.h` | 有 | P1 |
| `matmul_mxfp8e4m3_128_128_128_zeroinit` | `datatype_mm_test/mxfp8e4m3/cute_Matmul_mxfp8_mnk_128_128_128_zeroinit.c` | tensor | matmul | Mxfp8e4m3F32 | 128×128×128 | ZeroLoad | `matmul_value_mxfp8_mnk_128_128_128_zeroinit.h` | 有 | P1 |
| `matmul_mxfp8e4m3_128_128_128_fullbias` | `datatype_mm_test/mxfp8e4m3/cute_Matmul_mxfp8_mnk_128_128_128_fullbias.c` | tensor | matmul | Mxfp8e4m3F32 | 128×128×128 | FullBias | `matmul_value_mxfp8_mnk_128_128_128_fullbias.h` | 有 | P1 |
| `matmul_mxfp8e4m3_256_256_256_zeroinit` | `datatype_mm_test/mxfp8e4m3/` | tensor | matmul | Mxfp8e4m3F32 | 256×256×256 | ZeroLoad | 有 | 有 | P2 |
| `matmul_mxfp8e4m3_256_256_256_fullbias` | `datatype_mm_test/mxfp8e4m3/` | tensor | matmul | Mxfp8e4m3F32 | 256×256×256 | FullBias | 有 | 有 | P2 |
| `matmul_mxfp8e4m3_512_512_512_zeroinit` | `datatype_mm_test/mxfp8e4m3/` | tensor | matmul | Mxfp8e4m3F32 | 512×512×512 | ZeroLoad | 有 | 有 | P2 |
| `matmul_mxfp8e4m3_512_512_512_fullbias` | `datatype_mm_test/mxfp8e4m3/` | tensor | matmul | Mxfp8e4m3F32 | 512×512×512 | FullBias | 有 | 有 | P2 |

**其他 dtype 子目录**（优先级 P2+，Phase 0 仅列出，不展开）：

| 子目录 | dtype | 有 .c | 有 golden .h | 有 compare_result.py |
|--------|-------|-------|-------------|---------------------|
| `fp8e4m3/` | fp8e4m3F32 | 有 | 有 | 有 |
| `fp8e5m2/` | fp8e5m2F32 | 有 | 有 | 有 |
| `mxfp4/` | mxfp4F32 | 有 | 有 | 有 |
| `mxfp8e5m2/` | Mxfp8e5m2F32 | 有 | 有 | 有 |
| `nvfp4/` | nvfp4F32 | 有 | 有 | 有 |
| `mxfp8e4m3_256ReduceWidth/` | Mxfp8e4m3F32 | 有 | 有 | 有 |
| `nvfp4_256reduceWidth/` | nvfp4F32 | 有 | 有 | 有 |

**执行步骤**：

- [ ] 0.2.1 将上述矩阵写入 `plans/cutetest_migration_matrix.md`，每个 dtype 子目录完整展开
- [ ] 0.2.2 标注哪些 case 有 `CML_Store_trace.out`（即已有 Verilator 仿真输出）
- [ ] 0.2.3 标注 gemm_test / resnet50_test / transformer_test 为"后续阶段"，Phase 0 不展开

---

## 0.3 选定第一批 smoke cases

**目标**：从迁移矩阵中挑选 3 个 case，覆盖 runtime、基础 matmul、blockscale dtype 三个层面。

### Case 1: `runtime_hello`

| 字段 | 值 |
|------|-----|
| case_id | `runtime_hello` |
| 来源 | `cutetest/base_test/cutehello.c` |
| 目的 | 验证基础指令路径：config → issue → query → wait |
| level | runtime |
| op | hello (matmul smoke) |
| dtype | `DataTypeI8I8I32` (value=0) |
| shape | M=128, N=128, K=128 |
| bias_mode | `TaskTypeTensorZeroLoad` (value=1) |
| golden 来源 | `matmul_value_mnk_128_128_128_zeroinit.h` 中的 `d[]` 数组 |
| 输出 element_bits | 32 (I32) |
| 输出 layout | row_major |
| 输出 stride_bytes | 128 × 4 = 512 |
| 输出字节数 | 128 × 128 × 4 = 65,536 |
| transpose | 无 |

**特别说明**：`cutehello.c` 实际调用了 `issue_cute_matmul_marco_inst()`，所以它不仅是 runtime smoke，也隐含了 matmul 正确性。但作为 runtime case，我们只关注"指令能发出去、能 wait 完成"，不要求第一版就做 memverify。

### Case 2: `matmul_i8_128_128_128_zeroinit`

| 字段 | 值 |
|------|-----|
| case_id | `matmul_i8_128_128_128_zeroinit` |
| 来源 | `cutetest/base_test/cute_Matmul_mnk_128_128_128_zeroinit.c` |
| 目的 | 基础 INT8 matmul correctness |
| level | tensor |
| op | matmul |
| dtype | `DataTypeI8I8I32` (value=0) |
| shape | M=128, N=128, K=128 |
| bias_mode | `TaskTypeTensorZeroLoad` (value=1) |
| golden 来源 | `matmul_value_mnk_128_128_128_zeroinit.h` 中的 `gloden_d[]` 数组 |
| 输出 element_bits | 32 (I32) |
| 输出 layout | row_major |
| 输出 stride_bytes | 128 × 4 = 512 |
| 输出字节数 | 128 × 128 × 4 = 65,536 |
| transpose | 无 |

**特别说明**：
- `a[]`, `b[]` 是 char 数组（INT8），`d[]` 是 int 数组（I32）
- C 张量为 ZeroLoad（不加载，初始化为 0）
- `.h` 文件中 golden 变量名为 `gloden_d`（原文拼写），迁移时需注意
- `.h` 文件大小 294,317 B

### Case 3: `matmul_mxfp8e4m3_64_64_64_zeroinit`

| 字段 | 值 |
|------|-----|
| case_id | `matmul_mxfp8e4m3_64_64_64_zeroinit` |
| 来源 | `cutetest/datatype_mm_test/mxfp8e4m3/cute_Matmul_mxfp8_mnk_64_64_64_zeroinit.c` |
| 目的 | blockscale dtype (MXFP8) matmul smoke |
| level | tensor |
| op | matmul |
| dtype | `DataTypeMxfp8e4m3F32` (value=7) |
| shape | M=64, N=64, K=64 |
| bias_mode | `TaskTypeTensorZeroLoad` (value=1) |
| golden 来源 | `matmul_value_mxfp8_mnk_64_64_64_zeroinit.h` 中的 `gloden_c[]` 数组 |
| 输出 element_bits | 32 (FP32) |
| 输出 layout | row_major |
| 输出 stride_bytes | 64 × 4 = 256 |
| 输出字节数 | 64 × 64 × 4 = 16,384 |
| transpose | 有（compare_result.py 中 `transpose = True`） |
| 需要 Scale | 是（MXFP8 需要 block scale，CONFIG_SCALE_A / CONFIG_SCALE_B） |

**特别说明**：
- MXFP8 是 blockscale 类型，需要额外配置 Scale A/B 张量
- `.h` 文件中除 `a[]`/`b[]`/`gloden_c[]` 外，还应有 scale 数据
- 选择 64×64×64 而非更大 shape，是因为 golden `.h` 最小（93,059 B），且 `compare_result.py` 默认 `test_id=1` 对应 shape=128，需确认 64 对应 `test_id=0`
- 输出 golden 变量名为 `gloden_c`（注意是 C 不是 D）

**执行步骤**：

- [ ] 0.3.1 确认以上 3 个 case 的 `.c` 和 `.h` 文件均可读、无损坏
- [ ] 0.3.2 对每个 case，确认 golden 数组名、shape 常量、dtype 枚举值与 yaml 定义一致
- [ ] 0.3.3 对 `matmul_mxfp8e4m3_64_64_64_zeroinit`，确认 scale 数据格式和位置

---

## 0.4 定义 op spec 和 case manifest

**目标**：为第一批 smoke cases 定义两层契约——**op spec** 描述 op 的语义和接口，**case manifest** 描述 op 的一次具体实例化。两者分离，op spec 可被多个 case manifest 复用。

### 设计原则

op spec 是独立于实现和测试的公共契约层：

```text
op spec（契约：这个 op 是什么）
  ↑ 实现它        ↑ 验证它        ↑ 实例化它
cutelib/        tests/        case manifest（这组具体参数下应该产出什么）
```

依赖方向：

```text
case manifest → op spec（引用 op 定义，绑定具体参数）
case manifest → golden manifest（引用期望输出数据）
cutelib/      → op spec（实现 op 定义的接口）
tests/        → cutelib/（调用 lib API）
tests/        → case manifest（读取 case 配置）
memverify/    → golden manifest（读取 golden 数据格式）
```

### 文件组织

```text
cute-sdk/
├── ops/                          ← op spec 层：独立于实现和测试的公共契约
│   ├── tensor/
│   │   └── matmul.yaml           ← matmul op 的语义定义
│   ├── layer/
│   │   └── conv2d_layer.yaml     ← layer 级 op（引用 tensor 级 op 组合）
│   └── model/
│       └── ...
├── cutelib/                      ← op spec 的实现
│   ├── runtime/
│   ├── tensor/
│   ├── layer/
│   └── ...
├── tests/                        ← 用 case manifest 驱动测试
│   └── tensor/
│       └── matmul/
│           └── matmul_i8_128_128_128_zeroinit/
│               ├── case.json     ← 引用 op spec + 绑定具体参数 + 引用 golden
│               └── test.c
├── golden/
│   └── manual/
│       └── tensor/
│           └── matmul_i8_128_128_128_zeroinit/
│               ├── manifest.json ← golden 数据的二进制格式描述
│               └── golden.bin
└── memverify/
```

### 三层各管各的

| 层 | 文件 | 职责 |
|----|------|------|
| **op spec** | `ops/tensor/matmul.yaml` | 这个 op 的语义是什么，接受什么输入，产出什么输出，有哪些属性 |
| **case manifest** | `tests/.../case.json` | 用这个 op 跑一组具体参数，golden 在哪 |
| **golden manifest** | `golden/.../manifest.json` | golden 数据的二进制格式描述（shape、dtype、layout、byte order、generator） |

### op spec 格式：`ops/tensor/matmul.yaml`

Phase 0 只需定义第一批 smoke case 涉及的 op。当前只有 matmul。

```yaml
op: matmul
level: tensor
version: 1

description: 矩阵乘法 D = A × B + C

inputs:
  A:
    type: Tensor
    description: 左操作数矩阵
    dtype:
      - I8I8I32        # value 0
      - I8U8I32        # value 4
      - U8I8I32        # value 5
      - U8U8I32        # value 6
      - F16F16F32      # value 1
      - BF16BF16F32    # value 2
      - TF32TF32F32    # value 3
      - Mxfp8e4m3F32   # value 7
      - Mxfp8e5m2F32   # value 8
      - Nvfp4F32       # value 9
      - Mxfp4F32       # value 10
      - Fp8e4m3F32     # value 11
      - Fp8e5m2F32     # value 12
    shape: [M, K]
  B:
    type: Tensor
    description: 右操作数矩阵
    dtype: same_as_A
    shape: [K, N]
  C:
    type: Tensor
    description: 偏置矩阵（bias）
    dtype: I32 | F32  # 由 A/B dtype 决定累加精度
    shape: [M, N]
    bias_mode:
      - ZeroLoad      # value 1，填充零（不加载）
      - RowRepeat     # value 2，重复加载一行（bias 向量广播）
      - FullLoad      # value 3，完整加载所有数据

outputs:
  D:
    type: Tensor
    description: 输出矩阵
    dtype: same_as_C
    shape: [M, N]

attributes:
  transpose:
    type: bool
    default: false
    description: 是否转置输出 D
  blockscale:
    type: BlockScaleConfig | null
    default: null
    description: blockscale dtype 需要额外提供 scale A/B 张量
    fields:
      scale_a:
        type: Tensor
        description: A 的 block scale 数据
      scale_b:
        type: Tensor
        description: B 的 block scale 数据

semantics: |
  D[m][n] = sum_k( A[m][k] * B[k][n] ) + C[m][n]
  当 bias_mode=ZeroLoad 时，C 全为零
  当 blockscale 启用时，乘法按 block-wise quantized scale 计算

constraints:
  - A.dtype == B.dtype
  - A.shape[1] == B.shape[0]  # K 维度必须匹配
  - C.shape == D.shape == [M, N]
  - blockscale 仅适用于 Mxfp8e4m3 / Mxfp8e5m2 / Nvfp4 / Mxfp4 等 block-quantized dtype
  - scale 数据需 64 byte 对齐（来自 cute_isa_v1.yaml software.data_layout）
  - A/B/C/D tensor 数据需 64 byte 对齐

references:
  isa_yaml: configs/cute_isa_versions/cute_isa_v1.yaml
  isa_group: cute
  instructions:
    - CONFIG_TENSOR_A
    - CONFIG_TENSOR_B
    - CONFIG_TENSOR_C
    - CONFIG_TENSOR_D
    - CONFIG_TENSOR_DIM
    - CONFIG_CONV_PARAMS
    - CONFIG_SCALE_A
    - CONFIG_SCALE_B
    - SEND_MACRO_INST
```

### case manifest 格式：`tests/.../case.json`

case manifest 引用 op spec 并绑定具体参数值。

**Case 1: `runtime_hello`**

```json
{
  "id": "runtime_hello",
  "op_ref": "ops/tensor/matmul.yaml",
  "level": "runtime",
  "description": "smoke test: 验证 config → issue → query → wait 基础指令路径",
  "bindings": {
    "A.dtype": "I8I8I32",
    "A.shape": [128, 128],
    "B.shape": [128, 128],
    "C.bias_mode": "ZeroLoad",
    "D.shape": [128, 128],
    "transpose": false,
    "blockscale": null
  },
  "golden": "golden/manual/tensor/matmul_i8_128_128_128_zeroinit/manifest.json",
  "verify": {
    "mode": "bit_exact"
  }
}
```

**Case 2: `matmul_i8_128_128_128_zeroinit`**

```json
{
  "id": "matmul_i8_128_128_128_zeroinit",
  "op_ref": "ops/tensor/matmul.yaml",
  "level": "tensor",
  "description": "INT8 matmul correctness, M=N=K=128, zero init",
  "bindings": {
    "A.dtype": "I8I8I32",
    "A.shape": [128, 128],
    "B.shape": [128, 128],
    "C.bias_mode": "ZeroLoad",
    "D.shape": [128, 128],
    "transpose": false,
    "blockscale": null
  },
  "golden": "golden/manual/tensor/matmul_i8_128_128_128_zeroinit/manifest.json",
  "verify": {
    "mode": "bit_exact"
  }
}
```

**Case 3: `matmul_mxfp8e4m3_64_64_64_zeroinit`**

```json
{
  "id": "matmul_mxfp8e4m3_64_64_64_zeroinit",
  "op_ref": "ops/tensor/matmul.yaml",
  "level": "tensor",
  "description": "MXFP8 blockscale matmul smoke, M=N=K=64, zero init",
  "bindings": {
    "A.dtype": "Mxfp8e4m3F32",
    "A.shape": [64, 64],
    "B.shape": [64, 64],
    "C.bias_mode": "ZeroLoad",
    "D.shape": [64, 64],
    "transpose": true,
    "blockscale": {
      "scale_a": true,
      "scale_b": true
    }
  },
  "golden": "golden/manual/tensor/matmul_mxfp8e4m3_64_64_64_zeroinit/manifest.json",
  "verify": {
    "mode": "bit_exact"
  }
}
```

### golden manifest 格式：`golden/.../manifest.json`

golden manifest 只描述 golden 数据本身的二进制格式，与 op spec 和 case manifest 无依赖。

**Case 2 的 golden manifest 示例**：

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
    "created": "2026-05-13"
  }
}
```

**Case 3 的 golden manifest 示例**：

```json
{
  "id": "matmul_mxfp8e4m3_64_64_64_zeroinit",
  "op": "matmul",
  "output": {
    "path": "golden.bin",
    "element_bits": 32,
    "dtype": "FP32",
    "layout": "row_major",
    "shape": [64, 64],
    "stride_bytes": 256,
    "total_bytes": 16384
  },
  "generator": {
    "tool": "manual_import",
    "created": "2026-05-13"
  }
}
```

### 执行步骤

- [ ] 0.4.1 创建 `ops/tensor/` 目录，写入 `matmul.yaml` op spec
- [ ] 0.4.2 逐一打开 `.h` golden 文件，确认 golden 数组名、元素类型、shape 常量与 op spec 定义一致
- [ ] 0.4.3 逐一打开 `.c` 源文件，确认实际调用的 dtype 枚举值和 bias_mode
- [ ] 0.4.4 确认 MXFP8 case 的 scale 数据位置（在 `.h` 中还是单独文件），更新 op spec 中 blockscale 描述
- [ ] 0.4.5 三个 case manifest 写入 plan 文档作为草案（不创建实际文件，Phase 1 再落地）

---

## 执行顺序与依赖

```text
0.1 cuteisa 验证
 ├──> 0.2 迁移矩阵（可并行）
 └──> 0.3 选定 smoke cases（依赖 0.2 完成后确认范围）
       └──> 0.4 op spec + case manifest 定义（依赖 0.3 确认后展开）
```

建议执行顺序：
1. 先做 **0.1**（验证 cuteisa），同步做 **0.2**（迁移矩阵）
2. 矩阵完成后做 **0.3**（确认 smoke cases 选择）
3. 最后做 **0.4**（写 op spec、确认 golden 数据、写 case manifest 草案）

---

## 不在 Phase 0 范围内的工作

- 不写任何 `cutelib/` 代码
- 不写任何 `tests/` 代码
- 不实现 `memverify/`
- 不做 `.h` → `.bin` 的 golden 转换（只记录格式）
- 不涉及 `gemm_test`、`resnet50_test`、`transformer_test` 的详细展开
- 不涉及 cuteqemu、nvwa
- 不定义 conv2d / layer / model 级别的 op spec

---

## 待确认问题

1. **cuteisa 生成工具链**：`instruction.h`、`isa.json`、`cute_fpe.h` 是手动编写还是有从 `cute_isa_v1.yaml` 自动生成的脚本？
2. **MXFP8 scale 数据**：`matmul_value_mxfp8_mnk_64_64_64_zeroinit.h` 中是否包含 scale A/B 数组？还是 scale 在 `.c` 文件中单独定义？
3. **golden 变量命名**：部分 `.h` 中使用 `gloden_d`（拼写错误）还是 `d`？需确认每个 case 的实际数组名。
4. **compare_result.py 的 shape 映射**：mxfp8e4m3 的 `compare_result.py` 使用 `test_id` 变量控制 shape，需确认 `test_id=0` 对应 64×64×64。
5. **op spec 格式选择**：当前选择 YAML。是否需要同时提供 JSON Schema 供工具校验？
