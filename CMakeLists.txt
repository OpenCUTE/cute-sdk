cmake_minimum_required(VERSION 3.15)
project(cute_sdk C)

# ---- 全局 RISC-V 编译参数 ----
add_compile_options(
    -std=gnu99 -O3
    -march=rv64imafdcv -mabi=lp64d
    -specs=htif_nano.specs
    -fno-common -fno-builtin-printf -Wall
)
add_link_options(-static -specs=htif_nano.specs)

# ============================================================
# L0: cuteisa — ISA 原子指令（header-only）
# ============================================================
add_library(cuteisa INTERFACE)
target_include_directories(cuteisa INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/cuteisa/cute_isa_v1
)

# ============================================================
# L1: cutelib/runtime — 组合封装（header-only，依赖 cuteisa）
# ============================================================
add_library(cutelib_runtime INTERFACE)
target_include_directories(cutelib_runtime INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/cutelib/runtime
)
target_link_libraries(cutelib_runtime INTERFACE cuteisa)

# ============================================================
# L2: cutelib/tensor — tensor API（header-only，依赖 cutelib_runtime）
# ============================================================
add_library(cutelib_tensor INTERFACE)
target_include_directories(cutelib_tensor INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/cutelib/tensor/include
)
target_link_libraries(cutelib_tensor INTERFACE cutelib_runtime)

# ============================================================
# Tests
# ============================================================

function(add_runtime_test case_dir)
    get_filename_component(case_name ${case_dir} NAME)
    set(test_src ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/${case_dir}/test.c)
    set(data_dir  ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/${case_dir})

    if(NOT EXISTS ${test_src})
        message(WARNING "Skipping ${case_name}: test.c not found")
        return()
    endif()

    add_executable(${case_name} ${test_src})
    target_include_directories(${case_name} PRIVATE ${data_dir})
    target_link_libraries(${case_name} PRIVATE cutelib_runtime)
    set_target_properties(${case_name} PROPERTIES
        SUFFIX .riscv
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/runtime
    )
endfunction()

add_runtime_test(runtime_matmul_i8_128_128_128_zeroinit)
add_runtime_test(runtime_matmul_i8_128_128_128_zeroinit_transpose)
add_runtime_test(runtime_matmul_mxfp8e4m3_64_64_64_zeroinit)
add_runtime_test(runtime_matmul_i8_tiled_128x128_fifo)

# ============================================================
# Tensor-level tests
# ============================================================

function(add_tensor_test case_dir)
    get_filename_component(case_name ${case_dir} NAME)
    set(test_src ${CMAKE_CURRENT_SOURCE_DIR}/tests/tensor/${case_dir}/test.c)
    set(data_dir  ${CMAKE_CURRENT_SOURCE_DIR}/tests/tensor/${case_dir})

    if(NOT EXISTS ${test_src})
        message(WARNING "Skipping ${case_name}: test.c not found")
        return()
    endif()

    add_executable(${case_name} ${test_src})
    target_include_directories(${case_name} PRIVATE ${data_dir})
    target_link_libraries(${case_name} PRIVATE cutelib_tensor)
    set_target_properties(${case_name} PROPERTIES
        SUFFIX .riscv
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tensor
    )
endfunction()

add_tensor_test(tensor_matmul_i8_128_128_128_zeroinit)
add_tensor_test(tensor_matmul_i8_128_128_128_zeroinit_transpose)
add_tensor_test(tensor_matmul_mxfp8e4m3_64_64_64_zeroinit)
add_tensor_test(tensor_matmul_i8_tiled_128x128_fifo)
