# cutetest -> cute-sdk 迁移计划

## 目标

把 `cutetest/` 中已经能跑、能验证、能复用的资产，逐步迁移到
`cute-sdk/`，形成一套面向软件开发的 SDK 闭环。

当前阶段不优先开展 cuteqemu 和 nvwa 的设计。第一阶段的核心目标是：

```text
cutetest 资产
  -> cuteisa 统一 ISA artifacts
  -> 人工 golden + memverify
  -> cutelib/runtime
  -> cutelib/tensor
  -> cutelib/layer / model 样板
  -> cuteqemu + nvwa 接入
```

也就是说，先把“人工 golden + memverify + lib 开发”这条主线打通。
cuteqemu 和 nvwa 后续只作为同一套 memverify / test 接口的新数据来源接入，
而不是一开始就成为阻塞项。

## cutetest 现有资产对标

### base_test

当前角色：

- `cuteMarcoinstHelper.h` 中包含最小 CUTE 指令封装。
- 包含 `cutehello`、基础 INT8 matmul 等样例程序。
- 有本地 Makefile 和部分已生成 `.riscv` 样例。

迁移目标：

- `cutelib/runtime/`
- `tests/runtime/`
- 作为 query、wait、macro issue、基础指令封装的 smoke test。

### datatype_mm_test

当前角色：

- 覆盖 FP8 / MXFP / NVFP4 等 datatype 的 matmul 测试。
- 每个目录有独立 `compare_result.py`。
- 有 `matmul_value_*.h` 形式的人工数据。
- 部分目录保留了 `CML_Store_trace.out`。

迁移目标：

- `cutelib/tensor/`
- `tests/tensor/matmul/`
- `golden/manual/tensor/`
- 作为 dtype regression 的第一批来源。

### gemm_test

当前角色：

- 覆盖更大 shape 的 GEMM。
- 包含 transpose、长 K、不同配置等测试变体。
- compare 脚本里有大量 hard-coded shape / path。

迁移目标：

- tensor 级 shape regression。
- 后续性能/profile 的输入材料。
- 第一阶段只迁 correctness，不急着迁 performance 分析。

### resnet50_test

当前角色：

- 包含 layer / model 级 conv 代码。
- 有大量 ResNet layer 参数头文件。
- 混合了 CUTE task 调度和 vector 后处理逻辑。

迁移目标：

- `cutelib/layer/`
- 先选少量 conv layer 样板迁移。
- 不一开始整包迁完整 ResNet50。

### transformer_test

当前角色：

- 包含 BERT / LLAMA 风格的 transformer 测试材料。
- 有 QKV trace 和 compare 脚本。

迁移目标：

- 后续 tensor / layer / model 样板。
- 等 runtime、tensor、conv layer 稳定后再迁。

## 目标目录形态

```text
cute-sdk/
├── cuteisa/
│   └── cute_isa_v1/
│       ├── instruction.h
│       ├── isa.json
│       ├── cute_fpe.h
│       └── isa_summary.md
├── memverify/
│   ├── cute_memverify.py
│   ├── formats/
│   ├── readers/
│   └── README.md
├── cutelib/
│   ├── runtime/
│   ├── tensor/
│   ├── layer/
│   ├── fusion/
│   └── model/
├── tests/
│   ├── runtime/
│   ├── tensor/
│   ├── layer/
│   └── model/
└── golden/
    └── manual/
```

## tests / lib / golden 的关系

这三个对象不要平行推进，而应该由同一条 case-driven 工作流组织。

核心关系：

```text
golden/manual/<case_id>/      # 期望结果：这个 case 应该产出什么
          ▲
          │
          │ memverify compare
          │
tests/<level>/<op>/<case_id>/ # 场景组织：如何构建、运行、验证这个 case
          │
          │ build / run
          ▼
cutelib/<level>/              # 被测对象：SDK 对外提供的可复用 lib API
```

更具体地说：

- `cutelib/` 是产品代码，是未来用户真正 include 和调用的 SDK lib。
- `tests/` 是使用者视角的 case，它只负责组织输入、调用 lib、运行程序和触发验证。
- `golden/` 是期望输出数据，不应该嵌在 test C 文件里，也不应该依赖 lib。
- `memverify/` 是裁判，读取 golden 和实际输出 memory/trace，然后给出 pass/fail。

依赖方向必须保持清楚：

```text
tests  -> cutelib
tests  -> golden manifest
tests  -> memverify
memverify -> golden

cutelib 不依赖 tests
cutelib 不依赖 golden
golden 不依赖 tests
```

这样做的好处是：同一个 `cutelib` API 可以被多个 tests 复用；同一个 golden
可以先由人工生成，后续再由 nvwa 生成；同一个 test case 未来也可以切换
Verilator / FPGA / cuteqemu 作为实际输出来源。

## 主线工作流

每迁移一个 `cutetest` case，都按下面顺序走：

```text
1. 选 case
   从 cutetest 中挑一个最小可代表场景

2. 写 case manifest
   记录 op、shape、dtype、bias mode、layout、旧来源路径

3. 准备 golden
   先从旧 .h 或人工脚本转换成 golden.bin + manifest.json

4. 写/迁移 test driver
   test C 程序只调用 cutelib API，不再手写底层指令序列

5. 实现或补齐 cutelib API
   runtime -> tensor -> layer 逐级补能力

6. build + run
   生成 .riscv，跑 Verilator 或读取已有 trace/output

7. memverify
   用 source reader 解析实际 memory/trace，并与 golden bit-exact 比对

8. 回归固化
   case 进入 tests regression；旧 compare_result.py 不再是主入口
```

第一阶段推荐把每个迁移 case 落成这样的结构：

```text
tests/tensor/matmul/matmul_i8_128_128_128_zeroinit/
├── case.json              # 这个 case 怎么构建、运行、验证
├── test.c                 # 调用 cutelib/tensor API 的测试程序
└── Makefile               # 或由统一 runner 生成/调用

golden/manual/tensor/matmul_i8_128_128_128_zeroinit/
├── manifest.json          # golden 数据格式、shape、dtype、layout
└── golden.bin             # 期望输出 tensor
```

`case.json` 负责把 test 和 golden 关联起来，例如：

```json
{
  "id": "matmul_i8_128_128_128_zeroinit",
  "level": "tensor",
  "op": "matmul",
  "program": "test.c",
  "golden": "../../../golden/manual/tensor/matmul_i8_128_128_128_zeroinit/manifest.json",
  "source": {
    "kind": "cml_store_trace",
    "path": "run/CML_Store_trace.out"
  },
  "verify": {
    "mode": "bit_exact"
  }
}
```

`manifest.json` 只描述 golden 本身，例如：

```json
{
  "id": "matmul_i8_128_128_128_zeroinit",
  "op": "matmul",
  "dtype": "DataTypeI8I8I32",
  "bias_type": "TaskTypeTensorZeroLoad",
  "shape": {"m": 128, "n": 128, "k": 128},
  "output": {
    "path": "golden.bin",
    "element_bits": 32,
    "layout": "row_major",
    "stride_bytes": 512
  },
  "provenance": {
    "source": "cutetest/base_test/matmul_value_mnk_128_128_128_zeroinit.h"
  }
}
```

这条主线的评价标准不是“目录建好了”，而是每个 case 都能回答四个问题：

1. 这个 case 调用了哪个 `cutelib` API？
2. 它的 golden 从哪里来，格式是什么？
3. 实际输出从哪里来，怎么解析？
4. memverify 是否能独立给出 pass/fail？

## golden 和 memverify 策略

第一阶段只要求支持人工 golden。nvwa 后置。

SDK 不应该长期把 `cutetest` 里的 `.h` 静态数组当作唯一 golden 格式。
迁移时可以从旧的 `matmul_value_*.h` 导入，但 SDK 内部建议落成稳定的
manifest + raw tensor 数据：

```text
golden/manual/tensor/<case_id>/
├── manifest.json
├── input.bin          # 可选
├── weight.bin         # 可选
├── bias.bin           # 可选
└── golden.bin
```

`manifest.json` 建议描述：

- case id
- op 类型：matmul / conv / layer / model
- dtype 和 bias mode
- tensor shape
- 输出元素位宽
- 输出 layout
- 输出 stride
- 期望输出字节数
- 来源路径，例如原始 `cutetest/` 文件

memverify 第一版只做 bit-exact memory compare，不做浮点语义 tolerance。
报告至少包含：

- tensor index
- byte offset
- 地址（如果 source 中有）
- expected bytes / value
- actual bytes / value
- 使用的 source reader

## Phase 0：迁移基线和资产清单

目标：

冻结第一批迁移范围，并保持所有 case 都能追溯回 `cutetest/`。

任务：

| Task | 内容 | 输出 |
|------|------|------|
| 0.1 | 从 `configs/cute_isa_versions/*.yaml` 生成 cuteisa artifacts | `cute-sdk/cuteisa/cute_isa_v1/` |
| 0.2 | 建立 cutetest 迁移矩阵 | `plans/cutetest_migration_matrix.md` |
| 0.3 | 选定第一批 smoke cases | runtime hello、INT8 matmul、一个 blockscale dtype matmul |
| 0.4 | 记录旧测试路径、dtype、shape、bias mode、输出 tensor | 每个 case 的 manifest 草案 |

验收：

- `instruction.h`、`isa.json`、`cute_fpe.h` 已生成到 `cuteisa/`。
- 第一批 case 都有明确的 `cutetest/` 来源路径。
- 第一批 case 都明确 dtype、shape、bias mode 和输出 tensor。

建议第一批 case：

| Case | 来源 | 目的 |
|------|------|------|
| `runtime_hello` | `cutetest/base_test/cutehello.c` | 基础指令路径 |
| `matmul_i8_128_128_128_zeroinit` | `cutetest/base_test/cute_Matmul_mnk_128_128_128_zeroinit.c` | 基础 tensor matmul |
| `matmul_mxfp8e4m3_64_64_64_zeroinit` | `cutetest/datatype_mm_test/mxfp8e4m3/` | blockscale dtype smoke |

## Phase 1：人工 golden + memverify

目标：

建立不依赖 cuteqemu / nvwa 的验证闭环。

任务：

| Task | 内容 | 输出 |
|------|------|------|
| 1.1 | 定义人工 golden case 的 `manifest.json` 格式 | `memverify/formats/` 文档或 schema |
| 1.2 | 实现 raw golden tensor loader | `memverify/formats/raw_tensor.py` |
| 1.3 | 实现旧 `matmul_value_*.h` 导入工具 | `memverify/formats/header_import.py` 或工具脚本 |
| 1.4 | 实现 CMemoryLoader store trace reader | `memverify/readers/cml_store_trace.py` |
| 1.5 | 实现 byte-level compare engine | `memverify/cute_memverify.py` |
| 1.6 | 转换第一批人工 golden 数据 | `golden/manual/...` |

验收：

- 一条命令可以比较 manual golden 和一份 store trace。
- 第一批 case 不再依赖旧目录里的 `compare_result.py`。
- mismatch 报告能显示 tensor offset、expected、actual。
- 至少一个 INT8 matmul case 通过验证。

非目标：

- 不做 cuteqemu memory snapshot。
- 不做 nvwa golden 生成。
- 不做浮点 tolerance，只做 bit-exact 输出内存比对。

## Phase 2：cutelib/runtime

目标：

把 `cutetest/base_test/cuteMarcoinstHelper.h` 迁移为正式 runtime API。

任务：

| Task | 内容 | 输出 |
|------|------|------|
| 2.1 | 创建 runtime include 目录 | `cutelib/runtime/include/` |
| 2.2 | 基于 `cuteisa/.../instruction.h` 实现 `cute_runtime.h` | runtime 指令 API |
| 2.3 | 封装 query 指令 | runtime query helpers |
| 2.4 | 封装 tensor config 指令 | config A/B/C/D、scale A/B、dims、conv params |
| 2.5 | 封装 task issue / wait / dequeue | macro task helpers |
| 2.6 | 迁移 `cutehello` 到 runtime API | `tests/runtime/` |

验收：

- `cute_runtime.h` 只依赖 `cuteisa/cute_isa_v1/instruction.h` 和标准 C 头。
- `cutehello` 不再 include `cutetest/base_test/ygjk.h`。
- 旧 helper 的基础能力都有新 runtime API 对应。

API 对照种子：

| 旧 helper | 新 runtime 方向 |
|-----------|-----------------|
| `issue_cute_config_ATensor` | `cute_config_tensor_a` |
| `issue_cute_config_BTensor` | `cute_config_tensor_b` |
| `issue_cute_config_CTensor` | `cute_config_tensor_c` |
| `issue_cute_config_DTensor` | `cute_config_tensor_d` |
| `issue_cute_config_AScale` | `cute_config_scale_a` |
| `issue_cute_config_BScale` | `cute_config_scale_b` |
| `issue_cute_config_MNK_KERNALSTRIDE` | `cute_config_tensor_dim` |
| `issue_cute_config_CONV` | `cute_config_conv_params` |
| `issue_cute_marco_inst` | `cute_send_macro_inst` |
| `cute_marco_inst_fifo_finish_search` | `cute_query_macro_inst_finish` |

## Phase 3：cutelib/tensor

目标：

把 matmul 和基础 conv task 构造迁移成可复用 tensor API。

任务：

| Task | 内容 | 输出 |
|------|------|------|
| 3.1 | 定义 tensor descriptor | `cutelib/tensor/include/cute_tensor.h` |
| 3.2 | 定义 op API | `cutelib/tensor/include/cute_ops.h` |
| 3.3 | 实现 `cute_matmul` | tensor matmul wrapper |
| 3.4 | 实现 `cute_blockscale_matmul` | scale A/B wrapper |
| 3.5 | 实现第一版 `cute_conv2d_basic` | basic conv wrapper |
| 3.6 | 迁移第一批 matmul 测试 | `tests/tensor/matmul/` |

验收：

- 对第一批 baseline case，`cute_matmul()` 发出的指令序列与
  `issue_cute_matmul_marco_inst()` 等价。
- INT8 matmul 通过 memverify。
- 至少一个 blockscale dtype matmul 通过 memverify。
- 测试 C 代码主要调用 SDK API，不再手写每条 config 指令。

初始 tensor descriptor 可以很小：

```c
typedef struct {
    uint64_t base;
    uint64_t stride;
    uint64_t rows;
    uint64_t cols;
    uint64_t dtype;
} cute_tensor_t;
```

后续可以再扩展。第一版不要过度泛化，先让 correctness 闭环跑通。

## Phase 4：SDK test runner 和 case manifest

目标：

用统一 SDK regression 入口替代分散的 Makefile 和 `compare_result.py`。

任务：

| Task | 内容 | 输出 |
|------|------|------|
| 4.1 | 定义 test case manifest | `tests/**/cases/*.json` |
| 4.2 | 增加 `.riscv` 测试构建入口 | shared Makefile 或 Python runner |
| 4.3 | 增加 Verilator / 已有输出日志读取入口 | test runner |
| 4.4 | 将 memverify 参数绑定到每个 case | test runner |
| 4.5 | 逐步迁移 datatype matmul cases | regression set |

验收：

- 一条命令可以 build + verify 一组 tensor regression。
- 新增 case 不需要复制新的 compare 脚本。
- 测试结果统一输出 pass/fail 和 memverify report 路径。

## Phase 5：cutelib/layer

目标：

从 `resnet50_test` 中迁移少量有代表性的 layer API。不要一开始迁完整
ResNet50。

任务：

| Task | 内容 | 输出 |
|------|------|------|
| 5.1 | 创建 `cute_layer.h` | `cutelib/layer/include/` |
| 5.2 | 选择一个 `1x1 s1` conv 样本 | 第一条 conv layer case |
| 5.3 | 选择一个 `3x3 s1` conv 样本 | 第二条 conv layer case |
| 5.4 | layer 实现必须复用 tensor API | 不直接散写裸 CUTE 指令 |
| 5.5 | 用人工 golden + memverify 验证 layer 输出 | layer regression |

验收：

- 至少一个 conv layer 通过 bit-exact memverify。
- layer 实现复用 tensor API。
- 完整 ResNet50 保留为后续 model-level 目标。

## Phase 6：model / fusion 前置整理

目标：

为 model-level 迁移做接口准备，但不让它阻塞 runtime / tensor / layer
correctness。

任务：

| Task | 内容 | 输出 |
|------|------|------|
| 6.1 | 定义 model API 边界 | `cutelib/model/include/` 草案 |
| 6.2 | 定义 fusion API 边界 | `cutelib/fusion/include/` 草案 |
| 6.3 | 选择一个 mini model / layer chain | model smoke case |
| 6.4 | 继续推迟 profile 和 Top-Down 分析 | correctness-first policy |

验收：

- 一个小型串联 workload 可以用 SDK API 表达。
- fusion / profile 不作为 tensor 和 layer correctness 的前置条件。

## Phase 7：cuteqemu 和 nvwa 接入

目标：

在 SDK lib 和 memverify 接口稳定之后，把 cuteqemu / nvwa 作为 provider
接入。

接入模型：

```text
manual golden ─┐
nvwa golden ───┼──> memverify golden loader
               │
Verilator trace ─┐
FPGA xDMA ───────┼──> memverify source reader
cuteqemu memory ─┘
```

任务：

| Task | 内容 | 输出 |
|------|------|------|
| 7.1 | nvwa 输出现有 manifest 格式 | nvwa golden provider |
| 7.2 | 增加 cuteqemu memory snapshot reader | memverify reader |
| 7.3 | 第一批 case 跑通新 provider | provider regression |
| 7.4 | 保留 manual golden 作为 smoke fallback | regression fallback |

验收：

- 同一个 case manifest 可以切换 manual golden 或 nvwa golden。
- 同一个 memverify compare engine 可以消费 Verilator 或 cuteqemu memory。
- 既有 manual golden cases 继续通过。

## 里程碑

| Milestone | 定义 |
|-----------|------|
| M1 | `cuteisa + manual golden + memverify` 跑通一个 matmul |
| M2 | `cutelib/runtime` 在 smoke tests 中替代 `cuteMarcoinstHelper.h` |
| M3 | `cutelib/tensor` 跑通 INT8 和一个 blockscale dtype matmul |
| M4 | SDK test runner 替代迁移 case 中复制散落的 `compare_result.py` |
| M5 | 一个 ResNet 风格 conv layer 通过 memverify |
| M6 | cuteqemu 和 nvwa 接入现有 manifest / memverify 流程 |

## 待定问题

- `golden.bin` 的精确 byte order。
- `manifest.json` 是否第一天就使用 JSON Schema 校验。
- 第一版 source reader 是直接解析完整 Verilator output，还是只读取预过滤的
  `CML_Store_trace.out`。
- 旧 `.h` golden 导入工具放在 `tools/runner` 还是 `cute-sdk/memverify`。
- 旧 `.riscv` 二进制保留多少作为历史 artifact。

## 立即下一步

1. 新增 `plans/cutetest_migration_matrix.md`。
2. 定义第一版 `manifest.json` 格式。
3. 实现一个旧 `matmul_value_*.h` 的 golden 导入工具。
4. 实现最小 CMemoryLoader store trace reader。
5. 用人工 golden 验证 `matmul_i8_128_128_128_zeroinit`。
