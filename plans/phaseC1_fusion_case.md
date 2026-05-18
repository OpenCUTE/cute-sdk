# Phase C1 fusion plans

Phase C1 拆成两个独立计划：

1. [`phaseC1_fuse_vector.md`](phaseC1_fuse_vector.md)
   - 实现纯 vector fusion primitive。
   - 放在 `cutelib/primitive`。
   - 测试放在 `tests/primitive/primitive_fuse_*`。

2. [`phaseC1_fuse_tensor_vec.md`](phaseC1_fuse_tensor_vec.md)
   - 实现 tensor tile_op + vector fusion post-op。
   - 放在 `cutelib/fusion`。
   - 测试放在 `tests/fusion/fusion_matmul_*`。
   - 一个 fusion case 目录里保留一个 `case.json`，但生成 `notile` / `nopipeline` / `pipeline` 三个 binary，共用同一份 golden。

旧的合并式 `fusion case` 计划不再作为执行入口。
