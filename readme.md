# CUTE-SDK

CUTE 加速器全栈软件开发套件。涵盖功能模拟、golden 生成、内存验证、库函数开发四个核心方向，支持 CUTE 软件编写、硬件仿真测试、FPGA demo 测试三条验证路径。

## 核心对象

### cuteisa — ISA artifacts 统一出口

从 `configs/cute_isa_versions/*.yaml` 生成的 ISA 相关产物。为 cuteqemu 和 cutelib 提供稳定的共同依赖。

```text
cute-sdk/cuteisa/
└── cute_isa_v1/
    ├── instruction.h       # C 头文件（指令编码、funct、宏定义）
    ├── isa.json            # 结构化 ISA 定义（cuteqemu 消费）
    └── isa_summary.md      # 人类可读的指令集摘要
```

消费关系：

```text
configs/cute_isa_versions/cute_isa_v1.yaml（真相源）
  → codegen → cute-sdk/cuteisa/cute_isa_v1/
                  ├── cuteqemu 消费 isa.json
                  └── cutelib/runtime 消费 instruction.h（-I 编译）
```

### cuteqemu — 功能级模拟器

支持当前 CUTE ISA 全部指令的功能级模拟器。为软件开发和验证提供无需硬件的快速参考实现。

- 模拟全部 CUTE/YGJK 指令（tensor config / matmul / conv / query 等）
- 为 memverify 提供模拟内存快照
- 为 cutelib 开发提供快速迭代环境

### nvwa — Golden 生成与多平台对齐

方便生成 RVV 算子和 AI workload 算子的多平台对齐仓库。用于生成 golden 二进制文件，支撑 cutelib 各层级的正确性验证。

- 多平台算子对齐（参考实现 vs CUTE 硬件输出）
- Golden tensor 二进制文件生成
- 支持 cutelib 各层级（runtime / tensor / layer / fusion / model）的 golden 源

### memverify — 内存比对引擎

支持不同测试平台的 bit 级内存对齐验证。统一三条验证路径的比对入口。

| 验证路径 | 数据来源 | 场景 |
|----------|----------|------|
| Verilator trace | 从 compact trace 重建内存写入 | CUTE 硬件仿真测试 |
| FPGA xDMA | 从 FPGA xDMA 读取内存 | CUTE FPGA demo 测试 |
| cuteqemu | 从模拟器内存快照 | CUTE 软件编写验证 |

- Bit 级精度比对
- 统一的 golden tensor 格式
- 差异报告（mismatch 位置、期望值、实际值）

### cutelib — 分层库函数

涵盖当前 ISA 下所有可用库函数及开发全环节（golden 源、验证脚本、库包装）。逐级依赖上级库实现。

```text
model lib           模型级组合与验证
  └── fusion lib    融合层算子（减少中间读写）
      └── layer lib NN 层算子（conv / ffn / attention）
          └── tensor lib  张量算子（matmul / conv 配置与执行）
              └── runtime lib  基础 runtime（init / query / wait / config）
```

每一级包含：
- **库实现**：C 头文件 + 源文件，编译为 .riscv 运行在目标板
- **Golden 源**：nvwa 生成的参考输出
- **验证脚本**：memverify 驱动的 bit 级比对

## 三条验证路径

```text
                    ┌── cuteqemu（模拟器）
                    │     └── memverify 比对
cutelib 开发 ──────┤
                    ├── Verilator 仿真
                    │     └── trace 解码 → memverify 比对
                    │
                    └── FPGA demo
                          └── xDMA 读回 → memverify 比对
```

## 目录结构

```text
cute-sdk/
├── cuteisa/            # ISA artifacts（instruction.h / isa.json）
│   └── cute_isa_v1/
├── cuteqemu/           # 功能级模拟器
├── nvwa/               # Golden 生成与多平台对齐
├── memverify/          # 内存比对引擎
├── cutelib/            # 分层库函数
│   ├── runtime/        #   runtime lib
│   ├── tensor/         #   tensor lib
│   ├── layer/          #   layer lib
│   ├── fusion/         #   fusion lib
│   └── model/          #   model lib
├── readme.md
└── timeline.md
```

## 与 CUTE 主仓库的关系

```text
CUTE/
├── configs/            # HWConfig / ChipyardConfig / 版本 manifests
├── tools/              # Host 端工具链（codegen / check / decode）
├── trace/              # CUTETrace catalog + parser + decoder
├── chipyard/           # Chipyard 硬件子仓库
└── cute-sdk/           # ← 本仓库：软件全栈
      ├── cuteisa          消费 configs/cute_isa_versions → 生成 instruction.h + isa.json
      ├── cuteqemu         消费 cuteisa/isa.json
      ├── nvwa             为 cutelib 生成 golden
      ├── memverify        消费 trace 解码结果 + xDMA 数据 + cuteqemu 快照
      └── cutelib          消费 cuteisa/instruction.h，编译为 .riscv
```
