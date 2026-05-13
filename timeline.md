# CUTE-SDK Timeline

## Phase 0: 基础设施与抽象冻结

**状态**: `done`

HWConfig / ChipyardConfig / 版本 manifests / JSON Schema / CUTETrace catalog + codegen + decoder。

详细内容见 `plans/phase0_schema_and_abstractions.md`、`plans/phase0.5_cutetrace_implementation_plan.md`、`plans/phase0.6_cutetrace_codegen_implementation_plan.md`。

---

## Phase 0.7: cuteisa — ISA Artifacts 统一出口

**状态**: `pending`

**目标**: 从 `configs/cute_isa_versions/*.yaml` 生成 ISA 相关产物，为 cuteqemu 和 cutelib 提供稳定共同依赖。

| Task | 内容 | 状态 |
|------|------|------|
| 0.7.1 | 实现 `cuteisa/cute_isa_v1/isa.json` 生成（从 YAML 导出结构化 JSON） | `pending` |
| 0.7.2 | 实现 `cuteisa/cute_isa_v1/instruction.h` 生成（C 头文件：指令编码、funct 宏） | `pending` |
| 0.7.3 | 实现 `cuteisa/cute_isa_v1/isa_summary.md` 生成（人类可读指令集摘要） | `pending` |
| 0.7.4 | codegen 入口脚本或集成到现有 `gen_cute_trace.py` 流程 | `pending` |

**验收**:
- [ ] `instruction.h` 包含所有 YGJK + CUTE 指令的 funct/opcode 宏定义
- [ ] `isa.json` 可被 cuteqemu 直接加载解析
- [ ] cutelib 编译时 `-I cuteisa/cute_isa_v1/` 可引入 `instruction.h`

---

## Phase 1: cuteqemu — 功能级模拟器

**状态**: `pending`

**目标**: 实现支持当前 CUTE ISA 全部指令的功能级模拟器，为 cutelib 开发提供快速迭代环境，为 memverify 提供模拟内存快照。

| Task | 内容 | 状态 |
|------|------|------|
| 1.1 | 加载 `cuteisa/cute_isa_v1/isa.json`，解析指令编码和语义 | `pending` |
| 1.2 | 实现 tensor config 指令模拟（CONFIG_TENSOR_A/B/C/D、CONFIG_TENSOR_DIM） | `pending` |
| 1.3 | 实现 matmul / conv 计算模拟 | `pending` |
| 1.4 | 实现 query 指令模拟（QUERY_BUSY / QUERY_RUNTIME / QUERY_MEM_* / QUERY_FINISH） | `pending` |
| 1.5 | 实现模拟内存模型（支持 memverify 快照导出） | `pending` |
| 1.6 | CLI 入口：加载程序二进制 + 执行 + 导出内存 | `pending` |
| 1.7 | 端到端验证：用现有 matmul 测试程序的指令序列比对 Verilator 输出 | `pending` |

**验收**:
- [ ] cuteqemu 能执行一个完整 matmul 指令序列
- [ ] 输出 D tensor 与 Verilator 仿真结果 bit 级一致
- [ ] 支持导出内存快照供 memverify 消费

**前置**: Phase 0.7（cuteisa）

## Phase 2: nvwa — Golden 生成与多平台对齐

**状态**: `pending`

**目标**: 建立方便生成 RVV 算子和 AI workload 算子的多平台对齐仓库，为 cutelib 各层级提供 golden 二进制文件。

| Task | 内容 | 状态 |
|------|------|------|
| 2.1 | 定义 golden tensor 二进制格式（shape / dtype / data） | `pending` |
| 2.2 | 实现 tensor 级 golden 生成（matmul / conv 参考实现） | `pending` |
| 2.3 | 实现 layer 级 golden 生成（conv layer / FFN） | `pending` |
| 2.4 | 实现 fusion 级 golden 生成（fused FFN） | `pending` |
| 2.5 | 多平台对齐工具：golden vs cuteqemu / Verilator / FPGA 输出 | `pending` |
| 2.6 | CLI 入口：按参数生成指定 golden 文件 | `pending` |

**验收**:
- [ ] 能生成 INT8 matmul 128x128 golden，格式可被 memverify 直接消费
- [ ] 至少一个算子完成多平台对齐（参考 vs cuteqemu vs Verilator）

---

## Phase 3: memverify — 内存比对引擎

**状态**: `pending`

**目标**: 统一三条验证路径（Verilator trace / FPGA xDMA / cuteqemu）的 bit 级内存比对。

| Task | 内容 | 状态 |
|------|------|------|
| 3.1 | 定义统一的 golden tensor 加载接口 | `pending` |
| 3.2 | 实现 Verilator trace → 内存重建（消费 CUTETrace decoded events） | `pending` |
| 3.3 | 实现 FPGA xDMA → 内存读取 | `pending` |
| 3.4 | 实现 cuteqemu → 内存快照导出/加载 | `pending` |
| 3.5 | 实现 bit 级比对引擎 + mismatch 报告 | `pending` |
| 3.6 | CLI 入口：golden vs source（trace/xdma/qemu） | `pending` |

**验收**:
- [ ] Verilator trace 重建的 D tensor 与 golden bit 级一致
- [ ] Mismatch 报告包含地址、期望值、实际值、bit 差异
- [ ] 三条路径共用同一个比对引擎

---

## Phase 4: cutelib/runtime — Runtime Lib

**状态**: `pending`

**目标**: 实现基础 runtime lib，封装 CUTE/YGJK 指令，为上层库提供稳定 API。

| Task | 内容 | 状态 |
|------|------|------|
| 4.1 | 实现 `cutelib/runtime/include/cute_runtime.h` API 定义 | `pending` |
| 4.2 | 实现 init / query / wait / config 指令封装 | `pending` |
| 4.3 | 实现 build 规则（交叉编译 .riscv） | `pending` |
| 4.4 | nvwa 生成 runtime 级 golden（hello / basic query） | `pending` |
| 4.5 | memverify 验证 runtime 级正确性 | `pending` |
| 4.6 | cuteqemu 验证 runtime 级正确性 | `pending` |

**验收**:
- [ ] `cute_runtime.h` 只依赖 `cuteisa/cute_isa_v1/instruction.h`
- [ ] BaseTest（hello / query）在 Verilator 和 cuteqemu 上都能跑通
- [ ] 三条验证路径都 pass

**前置**: Phase 0.7（cuteisa）

## Phase 5: cutelib/tensor — Tensor Lib

**状态**: `pending`

**目标**: 在 runtime lib 上包装 tensor op，实现 matmul / conv 等基础张量算子。

| Task | 内容 | 状态 |
|------|------|------|
| 5.1 | 实现 `cutelib/tensor/include/cute_tensor.h`（tensor descriptor + config） | `pending` |
| 5.2 | 实现 `cutelib/tensor/include/cute_ops.h`（matmul / conv 封装） | `pending` |
| 5.3 | matmul test driver | `pending` |
| 5.4 | nvwa 生成 tensor 级 golden（多 dtype / 多 shape） | `pending` |
| 5.5 | memverify + cuteqemu 验证 tensor 级正确性 | `pending` |

**验收**:
- [ ] `cute_matmul()` 封装与现有 `issue_cute_matmul_marco_inst` 等价
- [ ] INT8 / FP16 / BF16 matmul golden 验证通过
- [ ] 测试代码量相对现有方案显著减少

---

## Phase 6: cutelib/layer — Layer Lib

**状态**: `pending`

**目标**: 在 tensor lib 上叠加 layer 语义，支持 NN 层级测试。

| Task | 内容 | 状态 |
|------|------|------|
| 6.1 | 实现 `cutelib/layer/include/cute_layer.h` | `pending` |
| 6.2 | 选择 layer 样板（ResNet conv / LLaMA FFN） | `pending` |
| 6.3 | layer test driver | `pending` |
| 6.4 | nvwa 生成 layer 级 golden | `pending` |
| 6.5 | memverify 验证 layer 输出 | `pending` |

**验收**:
- [ ] Layer op 复用 tensor op lib
- [ ] Layer 输出与 golden bit 级一致

---

## Phase 7: cutelib/fusion — Fusion Lib

**状态**: `pending`

**目标**: 引入融合语义，验证中间数据消除和最终输出对齐。

| Task | 内容 | 状态 |
|------|------|------|
| 7.1 | 定义融合语义（fused FFN / fused QKV attention） | `pending` |
| 7.2 | 实现 `cutelib/fusion/include/cute_fusion.h` | `pending` |
| 7.3 | fusion test driver | `pending` |
| 7.4 | nvwa 生成 fusion 级 golden | `pending` |
| 7.5 | 融合 vs 非融合对比（correctness + cycles） | `pending` |

**验收**:
- [ ] 融合输出与未融合 golden bit 级一致
- [ ] 性能对比数据（fused vs non-fused cycles）

---

## Phase 8: cutelib/model — Model Lib

**状态**: `pending`

**目标**: 组合下层 lib 形成模型级测试，端到端 correctness + profile。

| Task | 内容 | 状态 |
|------|------|------|
| 8.1 | 定义模型结构（LLaMA3 block / ResNet stage） | `pending` |
| 8.2 | 实现 `cutelib/model/include/cute_model.h` | `pending` |
| 8.3 | model test driver | `pending` |
| 8.4 | nvwa 生成 model 级 golden（tolerance 模式） | `pending` |

**验收**:
- [ ] 至少一个 model variant 能运行
- [ ] Model 输出与 golden 对齐

> **推迟**: 端到端 profile（Top-Down 分析）在正确性全部验证完后再做。

---

## 依赖关系

```text
Phase 0.7: cuteisa（instruction.h + isa.json）──────────────┐
                                                             │
Phase 1: cuteqemu  ←── 消费 cuteisa/isa.json ──────────────┤
Phase 2: nvwa ──────────────────────────────────────────────┤
Phase 3: memverify ─────────────────────────────────────────┤
                                                             │
Phase 4: cutelib/runtime  ←── 消费 cuteisa/instruction.h + cuteqemu + nvwa + memverify
Phase 5: cutelib/tensor   ←── 依赖 runtime
Phase 6: cutelib/layer    ←── 依赖 tensor
Phase 7: cutelib/fusion   ←── 依赖 layer
Phase 8: cutelib/model    ←── 依赖 fusion
```

Phase 0.7 最先。Phase 1-3 可并行推进（1 依赖 0.7）。Phase 4-8 严格逐级依赖。

---

## 设计文档

- **Big Plan**: `plans/bigplan.md`、`plans/cc/newbigplan.md`
- **Phase 0 Schema**: `plans/phase0_schema_and_abstractions.md`
- **Phase 0.5 Trace**: `plans/phase0.5_cutetrace_implementation_plan.md`
- **Phase 0.6 Codegen**: `plans/phase0.6_cutetrace_codegen_implementation_plan.md`
