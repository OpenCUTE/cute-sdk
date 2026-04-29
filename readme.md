# CUTE-SDK

CUTE 加速器的目标板软件栈。提供 runtime lib、tensor op lib、layer op lib 等 C 语言库，通过 `project.yaml` 声明式管理测试项目和硬件配置匹配。与 host 端工具链（`tools/`）配合完成编译、仿真、golden 验证和性能分析。基于 `HWConfig / Test / Trace` 三大抽象，从 BaseTest 到 ModelTest 逐层沉淀软件库。
