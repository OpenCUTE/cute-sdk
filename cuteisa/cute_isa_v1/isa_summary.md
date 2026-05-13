# cute_isa_v1 ISA Summary

Auto-generated from `isa.json`. Do not edit manually.

当前 SDK 兼容版本使用的 CUTE/YGJK 指令 manifest。

## RoCC

- opcode: `0x0B`
- cute_internal_offset: `64`

## Instruction Groups

### ygjk
直接在 RoCC 接口层处理的 YGK/RoCC 接口指令组。

- rocc_funct_offset: `0`
- instruction_count: `8`

| Instruction | Funct | RoCC Funct | Return | Description |
|-------------|-------|------------|--------|-------------|
| `QUERY_ACCELERATOR_BUSY` | `1` | `1` | 未使用 | 未使用 |
| `QUERY_RUNTIME` | `2` | `2` | 未使用 | 未使用 |
| `QUERY_MEM_READ_COUNT` | `3` | `3` | 返回对外访存读次数 | 查询加速器对外访存读次数 |
| `QUERY_MEM_WRITE_COUNT` | `4` | `4` | 返回对外访存写次数 | 查询加速器对外访存写次数 |
| `QUERY_COMPUTE_TIME` | `5` | `5` | 未使用 | 未使用 |
| `QUERY_MACRO_INST_FINISH` | `6` | `6` | 返回宏指令队列中的完成情况 (0010 = id为1的指令已完成) | 查询 CUTE 宏指令的完成情况 |
| `QUERY_MACRO_INST_FIFO_FULL` | `7` | `7` | 返回FIFO是否已满 (1=满, 0=未满) | 查询 CUTE 宏指令队列是否已满 |
| `QUERY_MACRO_INST_FIFO_INFO` | `8` | `8` | 返回FIFO中当前指令状态 (0010 = id为1的位置已有指令) | 查询 CUTE 宏指令队列目前有多少指令 |

### cute
通过 RoCC funct 偏移量 64 转发到 CUTE 核心的内部控制指令组。

- rocc_funct_offset: `64`
- instruction_count: `12`

| Instruction | Funct | RoCC Funct | Return | Description |
|-------------|-------|------------|--------|-------------|
| `SEND_MACRO_INST` | `0` | `64` | 返回指令在FIFO中的编号 | 发送已配置的宏指令到指令FIFO |
| `CONFIG_TENSOR_A` | `1` | `65` | 未使用 | 配置A张量的基地址和步长 |
| `CONFIG_TENSOR_B` | `2` | `66` | 未使用 | 配置B张量的基地址和步长 |
| `CONFIG_TENSOR_C` | `3` | `67` | 未使用 | 配置C张量的基地址和步长 |
| `CONFIG_TENSOR_D` | `4` | `68` | 未使用 | 配置D张量的基地址和步长 |
| `CONFIG_TENSOR_DIM` | `5` | `69` | 未使用 | 配置张量维度(M,N,K)，对于卷积则是(ohow,oc,ic) |
| `CONFIG_CONV_PARAMS` | `6` | `70` | 未使用 | 配置卷积相关参数（element_type, bias_type, kernel_size等） |
| `CONFIG_SCALE_A` | `7` | `71` | 未使用 | 配置A Scale（量化参数）的基地址 |
| `CONFIG_SCALE_B` | `8` | `72` | 未使用 | 配置B Scale（量化参数）的基地址 |
| `CLEAR_INST` | `16` | `80` | 未使用 | 清除队尾的宏指令 |
| `QUERY_INST` | `17` | `81` | 返回已完成宏指令的尾编号位置 | 查询当前完成宏指令的尾编号位置 |
| `RESERVED` | `18` | `82` | 未使用 | 保留指令（空操作） |

## Enums

### ElementDataType
矩阵元素数据类型。软件通过 CONFIG_CONV_PARAMS.element_type 字段填入。同时也是 datatype.h 的唯一数据源。

| Name | Value | Description |
|------|-------|-------------|
| `DataTypeI8I8I32` | `0` | Int8 * Int8 -> Int32 |
| `DataTypeF16F16F32` | `1` | FP16 * FP16 -> FP32 |
| `DataTypeBF16BF16F32` | `2` | BF16 * BF16 -> FP32 |
| `DataTypeTF32TF32F32` | `3` | TF32 * TF32 -> FP32 |
| `DataTypeI8U8I32` | `4` | Int8 * UInt8 -> Int32 |
| `DataTypeU8I8I32` | `5` | UInt8 * Int8 -> Int32 |
| `DataTypeU8U8I32` | `6` | UInt8 * UInt8 -> Int32 |
| `DataTypeMxfp8e4m3F32` | `7` | MXFP8 E4M3 * MXFP8 E4M3 -> FP32 |
| `DataTypeMxfp8e5m2F32` | `8` | MXFP8 E5M2 * MXFP8 E5M2 -> FP32 |
| `DataTypenvfp4F32` | `9` | NVFP4 * NVFP4 -> FP32 |
| `DataTypemxfp4F32` | `10` | MXFP4 * MXFP4 -> FP32 |
| `DataTypefp8e4m3F32` | `11` | FP8 E4M3 * FP8 E4M3 -> FP32 |
| `DataTypefp8e5m2F32` | `12` | FP8 E5M2 * FP8 E5M2 -> FP32 |

### CMemoryLoaderTaskType
C 张量（bias / 结果）加载模式。软件通过 CONFIG_CONV_PARAMS.bias_type 字段填入。

| Name | Value | Description |
|------|-------|-------------|
| `TaskTypeUndef` | `0` | 未定义 |
| `TaskTypeTensorZeroLoad` | `1` | 填充零（不加载，C张量初始化为0） |
| `TaskTypeTensorRepeatRowLoad` | `2` | 重复加载一行（bias向量广播） |
| `TaskTypeTensorLoad` | `3` | 完整加载所有数据 |

## Software Data Layout

- default_alignment_bytes: `64`
- default_padding: `zero`

| Name | Applies To | Alignment | Padding | Description |
|------|------------|-----------|---------|-------------|
| `blockscale_scale` | CONFIG_SCALE_A, CONFIG_SCALE_B | `64` | `zero` | BlockScale 的 A/B scale 数据由软件负责按 64 Byte 对齐；有效 scale 数据不足 64 Byte 粒度的末尾部分必须补零。 |
| `tensor_abcd_data` | CONFIG_TENSOR_A, CONFIG_TENSOR_B, CONFIG_TENSOR_C, CONFIG_TENSOR_D | `64` | `zero` | A/B/C/D tensor 数据由软件负责按 64 Byte 对齐；有效数据不足 64 Byte 粒度的末尾部分必须补零。 |
