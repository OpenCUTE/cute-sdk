# Phase A：L1 cutelib/runtime 扩展

> 状态：**待执行**
> 前置依赖：无（L0 cuteisa 已完成）
> 产出：`cutelib/runtime/cute_runtime.h` 补齐全部 FIFO 管理原语
> 预计改动量：1 个文件修改，~40 行新增

---

## 1. 目标

当前 `cute_runtime.h` 只有 `cute_matmul()` 和 `cute_blockscale_matmul()` 两个组合封装。
Phase B（tensor 层）的 tiled pipeline 调度需要以下能力：

1. **按 task_id 等待 + 出队** — `CUTE_TASK_END(task_id)` 等价
2. **非阻塞查询** — 检查完成状态、FIFO 满等
3. **独立出队** — 不等待，直接出队队首

这些底层 wrapper 全部已在 `instruction.h` 中存在，只是在 runtime 层缺少封装。

---

## 2. 现有 API 对照

| instruction.h 中已有 | runtime 层现状 | llama3_1B.c 对应 |
|---|---|---|
| `CUTE_SEND_MACRO_INST()` | 已封装在 `cute_matmul()` 内部 | `issue_cute_marco_inst()` |
| `CUTE_QUERY_MACRO_INST_FINISH()` | **未封装**，test.c 直接调用 | `cute_marco_inst_fifo_finish_search()` |
| `CUTE_CLEAR_INST()` | **未封装** | `cute_marco_inst_fifo_dequeue()` |
| `CUTE_QUERY_MACRO_INST_FIFO_FULL()` | **未封装** | `cute_marco_inst_fifo_full_search()` |
| `CUTE_QUERY_MACRO_INST_FIFO_INFO()` | **未封装** | `cute_marco_inst_fifo_valid_search()` |
| `CUTE_QUERY_INST()` | **未封装** | `cute_marco_inst_fifo_get_finish_tail_fifoindex()` |
| `CUTE_QUERY_ACCELERATOR_BUSY()` | **未封装** | 间接使用 |

---

## 3. 改动方案

### 3.1 修改文件

`cutelib/runtime/cute_runtime.h`

### 3.2 新增内容

在现有 `cute_blockscale_matmul()` 之后，`#endif` 之前，新增以下函数。

#### 3.2.1 FIFO 状态查询（非阻塞）

```c
/* ---- FIFO 状态查询 ---- */

// 返回已完成任务的 bitmask
// bit N = 1 表示 task_id=N 的指令已完成
// 等价于 llama3_1B.c 的 cute_marco_inst_fifo_finish_search()
static inline uint64_t cute_query_finish(void) {
    return CUTE_QUERY_MACRO_INST_FINISH();
}

// FIFO 是否已满（1=满，0=未满）
// 等价于 llama3_1B.c 的 cute_marco_inst_fifo_full_search()
static inline int cute_fifo_full(void) {
    return CUTE_QUERY_MACRO_INST_FIFO_FULL() != 0;
}

// FIFO 当前占用状态（bitmask，bit N = 1 表示位置 N 有指令）
// 等价于 llama3_1B.c 的 cute_marco_inst_fifo_valid_search()
static inline uint64_t cute_fifo_info(void) {
    return CUTE_QUERY_MACRO_INST_FIFO_INFO();
}
```

#### 3.2.2 等待 + 出队（阻塞）

```c
/* ---- 等待与出队 ---- */

// 阻塞等待特定 task_id 完成，然后出队队首
// 等价于 llama3_1B.c 的 CUTE_TASK_END(task_id)
//
// 【FIFO 顺序约束】
// CUTE 宏指令 FIFO 严格先入先出。CUTE_CLEAR_INST() 无参数，永远出队队首。
// 软件必须保证：
//   1. wait+dequeue 严格按 issue 顺序调用（不能跳过中间 task）
//   2. task_id 必须是当前 FIFO 中最早进入的那条未完成指令
//   3. 先 issue 的必定先完成，不存在乱序完成
static inline void cute_wait_task(uint64_t task_id) {
    uint64_t mask = 1UL << task_id;
    while (!(CUTE_QUERY_MACRO_INST_FINISH() & mask))
        ;
    CUTE_CLEAR_INST();
}
```

#### 3.2.3 独立出队

```c
// 出队队首已完成任务（不等待）
// 同样受 FIFO 顺序约束：只能出队当前队首
// 等价于 llama3_1B.c 的 cute_marco_inst_fifo_dequeue()
static inline void cute_dequeue(void) {
    CUTE_CLEAR_INST();
}
```

#### 3.2.4 已完成尾位置查询

```c
// 返回已完成宏指令的尾编号位置
// 等价于 llama3_1B.c 的 cute_marco_inst_fifo_get_finish_tail_fifoindex()
static inline uint64_t cute_query_inst_tail(void) {
    return CUTE_QUERY_INST();
}
```

#### 3.2.5 加速器状态查询

```c
/* ---- 加速器状态 ---- */

// 对外访存读次数
static inline uint64_t cute_query_mem_read_count(void) {
    return CUTE_QUERY_MEM_READ_COUNT();
}

// 对外访存写次数
static inline uint64_t cute_query_mem_write_count(void) {
    return CUTE_QUERY_MEM_WRITE_COUNT();
}

```

### 3.3 不改动的部分

- `cute_matmul()` — 保持现有签名和实现不变
- `cute_blockscale_matmul()` — 保持不变
- `CUTE_SCP_M` / `CUTE_SCP_N` 宏 — 保持不变

---

## 4. 修改后的完整 cute_runtime.h 结构

```text
cute_runtime.h
├── #include "instruction.h"
├── CUTE_SCP_M / CUTE_SCP_N
│
├── cute_matmul()                        # 已有
├── cute_blockscale_matmul()             # 已有
│
├── /* FIFO 状态查询 */                  # Phase A 新增
│   ├── cute_query_finish()
│   ├── cute_fifo_full()
│   └── cute_fifo_info()
│
├── /* 等待与出队 */                     # Phase A 新增
│   ├── cute_wait_task(task_id)
│   ├── cute_dequeue()
│   └── cute_wait_any()
│
├── /* 位置查询 */                       # Phase A 新增
│   └── cute_query_inst_tail()
│
├── /* 加速器状态 */                     # Phase A 新增
│   ├── cute_accelerator_busy()
│   ├── cute_query_runtime()
│   ├── cute_query_mem_read_count()
│   ├── cute_query_mem_write_count()
│   └── cute_query_compute_time()
│
└── #endif
```

---

## 5. 验证

### 5.1 编译验证

三个已有的 runtime test 必须继续编译通过。新增函数全部是 `static inline`，不改变已有符号。

```bash
cd /root/opencute/CUTE/cute-sdk
mkdir -p build && cd build
cmake -S .. -DCMAKE_TOOLCHAIN_FILE=../cmake/riscv-toolchain.cmake -B .
cmake --build . -j$(nproc)
```

预期：三个 `.riscv` 产出不变。

### 5.2 现有 test.c 兼容性

当前三个 test.c 直接调用 `CUTE_QUERY_MACRO_INST_FINISH()`（通过 include `instruction.h` 间接获得）。
Phase A 不改变这个行为——test.c 仍然可以直接调 instruction.h 的 wrapper，
也可以迁移到 `cute_wait_task()` 或 `cute_wait_any()`。

**不强制修改现有 test.c**，向后兼容。

### 5.3 功能验证（无仿真环境时）

如果无 Verilator / FPGA 环境，验证手段：

1. 确认编译通过
2. 对照 `cuteMarcoinstHelper.h`，逐函数确认 funct code 一致
3. 确认 `cute_wait_task` 的 bitmask 逻辑与 `CUTE_TASK_END` 一致

对照表：

| cute_runtime.h 新增 | cuteMarcoinstHelper.h 旧 | funct | 等价？ |
|---|---|---|---|
| `cute_query_finish()` | `cute_marco_inst_fifo_finish_search()` | 6 | ✓ |
| `cute_dequeue()` | `cute_marco_inst_fifo_dequeue()` | 80 (64+16) | ✓ |
| `cute_fifo_full()` | `cute_marco_inst_fifo_full_search()` | 7 | ✓ |
| `cute_fifo_info()` | `cute_marco_inst_fifo_valid_search()` | 8 | ✓ |
| `cute_query_inst_tail()` | `cute_marco_inst_fifo_get_finish_tail_fifoindex()` | 81 (64+17) | ✓ |

### 5.4 功能验证（有仿真环境时）

复用已有的 runtime test，行为应不变。可选：改写一个 test.c 使用 `cute_wait_task()` 替代 `while (!CUTE_QUERY_MACRO_INST_FINISH());`，验证 memverify 结果一致。

---

## 6. 执行步骤

| 步骤 | 内容 | 耗时估计 |
|------|------|---------|
| 1 | 修改 `cutelib/runtime/cute_runtime.h`，新增全部 wrapper | 10 min |
| 2 | 编译验证 | 2 min |
| 3 | 对照 cuteMarcoinstHelper.h 确认 funct code | 5 min |
| 4 | （可选）改写一个 test.c 用新 API，仿真验证 | — |

完成后即可进入 Phase B（cutelib/tensor）。

---

## 7. 不在 Phase A 范围内

- 不修改 `cute_matmul()` 或 `cute_blockscale_matmul()` 的签名
- 不修改现有 test.c
- 不新增 test case
- 不新增 .c 文件（全部保持 header-only）
