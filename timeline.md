# CUTE-SDK Timeline

## Phase 0: 冻结 HWConfig / Test / Trace 抽象和 Schema

**状态**: `pending`

**目标**: 定义三大对象的手写入口，使框架能静态回答 target 匹配、capability、trace level 问题。

| Task | 内容 | 状态 |
|------|------|------|
| 0.1 | 创建 `cute-sdk/` 目录骨架 | `pending` |
| 0.2 | 定义 `configs/schemas/hwconfig.schema.json` | `pending` |
| 0.3 | 创建样板 `configs/hwconfigs/cute2tops_scp64_dramsim32.yaml` | `pending` |
| 0.4 | 定义 `configs/schemas/project.schema.json` | `pending` |
| 0.5 | 创建 `cute-sdk/runtime/cute_runtime/project.yaml` | `pending` |
| 0.6 | 创建 `cute-sdk/tensor_ops/matmul/project.yaml` | `pending` |
| 0.7 | 定义 `configs/schemas/trace_filter.schema.json` + `tools/trace/format_spec.md` | `pending` |
| 0.8 | 创建占位 `configs/trace_filters/*.yaml` | `pending` |
| 0.9 | 实现 `tools/runner/cute-check-config.py` | `pending` |

**验收**:
- [ ] `cute-check-config.py --hwconfig` 通过校验
- [ ] `cute-check-config.py --project` 通过校验
- [ ] `cute-check-config.py --scan` 输出 target 匹配矩阵
- [ ] Trace level 名称 F0-F5 已作为占位冻结

**详细计划**: `plans/phase0_schema_and_abstractions.md`

---

## Phase 1: BaseTest → Runtime Lib 最小闭环

**状态**: `pending`

**目标**: 用一个最小 BaseTest 跑通 runtime lib，形成第一份可复盘 artifact。

| Task | 内容 | 状态 |
|------|------|------|
| 1.1 | 实现 `cute-sdk/runtime/cute_runtime/include/cute_runtime.h` | `pending` |
| 1.2 | 实现 `cute-sdk/runtime/cute_runtime/src/cute_runtime.c` | `pending` |
| 1.3 | 实现 `cute-sdk/runtime/cute_runtime/tests/rocc_hello.c` | `pending` |
| 1.4 | 实现 `cute-sdk/runtime/cute_runtime/build_rules/Makefile` | `pending` |
| 1.5 | 实现 `tools/runner/cute-gen-headers.py` | `pending` |
| 1.6 | 实现 `tools/runner/cute-run.py` 最小 Runner | `pending` |
| 1.7 | 端到端验证：一条命令跑通 rocc_hello | `pending` |

**验收**:
- [ ] `cute_runtime.h` 只依赖 `instruction.h.generated`
- [ ] `cute-run.py --hwconfig ... --project ... --variant rocc_hello` 跑通
- [ ] Artifact 目录结构完整
- [ ] Runtime lib 有最小可用 API

**详细计划**: `plans/phase1_basetest_runtime_loop.md`

---

## Phase 2: TensorTest → Tensor Op Lib

**状态**: `pending`

**目标**: 在 runtime lib 上包装 tensor op lib，第一次真正验证数据结果。

| Task | 内容 | 状态 |
|------|------|------|
| 2.1 | 实现 `cute-sdk/include/cute_tensor.h` | `pending` |
| 2.2 | 实现 `cute-sdk/include/cute_ops.h` | `pending` |
| 2.3 | 实现 `cute-sdk/tensor_ops/matmul/src/main.c` | `pending` |
| 2.4 | 实现 `tools/verify/cute_golden.py` | `pending` |
| 2.5 | 实现 `tools/runner/cute-gen-golden.py` | `pending` |
| 2.6 | 实现 `tools/verify/cute_verify.py` | `pending` |
| 2.7 | Runner 扩展：golden 生成 + verify | `pending` |

**验收**:
- [ ] 测试代码 ~15 行（对比现有 ~60 行）
- [ ] `cute_matmul()` 封装与现有 `issue_cute_matmul_marco_inst` 等价
- [ ] INT8 matmul golden 验证通过

**详细计划**: `plans/phase2_tensor_test_op_lib.md`

---

## Phase 3: LayerTest → Layer Op Lib

**状态**: `pending`

**目标**: 在 tensor op lib 上叠加 layer 语义，开始 trace-driven 验证。

| Task | 内容 | 状态 |
|------|------|------|
| 3.1 | 选择 layer 样板（ResNet conv 或 LLaMA FFN） | `pending` |
| 3.2 | 实现 `cute-sdk/include/cute_layer.h` | `pending` |
| 3.3 | 实现 layer test driver | `pending` |
| 3.4 | 实现 `tools/trace/parser.py` + `tools/trace/func/tensor_model.py` | `pending` |
| 3.5 | 实现 trace-driven verify 路径 | `pending` |

**验收**:
- [ ] Layer op 复用 tensor op lib
- [ ] Trace parser 能解析 legacy `CMemoryLoader_Store`
- [ ] `F1_store` 能从 trace 重建 D tensor

**详细计划**: `plans/phase3_layer_test_op_lib.md`

---

## Phase 4: FuseLayerTest → Fuse Layer Op Lib

**状态**: `pending`

**目标**: 引入融合语义，验证中间数据消除和最终输出对齐。

| Task | 内容 | 状态 |
|------|------|------|
| 4.1 | 定义 LLaMA FFN 融合语义 | `pending` |
| 4.2 | 实现 `cute-sdk/fuse_layer_ops/llama_ffn_fused/` | `pending` |
| 4.3 | 实现 `tools/trace/func/fused_model.py` | `pending` |
| 4.4 | 融合 vs 非融合对比 | `pending` |

**验收**:
- [ ] 融合输出与未融合 golden 一致
- [ ] 性能对比数据（fused vs non-fused cycles）

**详细计划**: `plans/phase4_fuse_layer_test_op_lib.md`

---

## Phase 5: SOCOptTest → Opt Op Lib

**状态**: `pending`

**目标**: 结合特定 SoC 做足量优化，实现 Top-Down 性能分析。

| Task | 内容 | 状态 |
|------|------|------|
| 5.1 | 定义 opt op 边界和 target family | `pending` |
| 5.2 | 实现 `cute-sdk/opt_ops/` | `pending` |
| 5.3 | 实现 `tools/perf/cute_perf_analyzer.py` Top-Down 引擎 | `pending` |
| 5.4 | 实现 `tools/perf/cute_perf_model.py` Roofline 模型 | `pending` |
| 5.5 | 实现 `tools/runner/cute-perf.py` CLI | `pending` |

**验收**:
- [ ] Opt project 只匹配指定 HWConfig family
- [ ] Top-Down 分析能输出 Level 0-2 指标
- [ ] Roofline 模型能计算 AI 和 Attainable TOPS

**详细计划**: `plans/phase5_soc_opt_test_op_lib.md`

---

## Phase 6: ModelTest

**状态**: `pending`

**目标**: 组合下层 lib 形成模型级测试，端到端 correctness + profile。

| Task | 内容 | 状态 |
|------|------|------|
| 6.1 | 定义 LLaMA3 block 结构和 project.yaml | `pending` |
| 6.2 | 实现 model driver | `pending` |
| 6.3 | 实现 model golden（tolerance 模式） | `pending` |
| 6.4 | 端到端 profile | `pending` |
| 6.5 | 实现 `tools/runner/cute-model-report.py` | `pending` |

**验收**:
- [ ] 至少一个 model variant 能运行
- [ ] Layer breakdown 显示各层 cycle 占比
- [ ] Fused vs non-fused 对比有数据

**详细计划**: `plans/phase6_model_test.md`

---

## 总体设计文档

- **Big Plan**: `plans/bigplan.md` — HWConfig / Test / Trace 三大抽象，目录结构，分阶段路线
- **Phase 0**: `plans/phase0_schema_and_abstractions.md` — Schema 和抽象冻结
- **Phase 1**: `plans/phase1_basetest_runtime_loop.md` — Runtime Lib 最小闭环
- **Phase 2**: `plans/phase2_tensor_test_op_lib.md` — Tensor Op Lib
- **Phase 3**: `plans/phase3_layer_test_op_lib.md` — Layer Op Lib + Trace Parser
- **Phase 4**: `plans/phase4_fuse_layer_test_op_lib.md` — Fuse Layer Op Lib
- **Phase 5**: `plans/phase5_soc_opt_test_op_lib.md` — Opt Op Lib + Perf
- **Phase 6**: `plans/phase6_model_test.md` — Model Test
