cmake_minimum_required(VERSION 3.15)
project(cute_sdk C)

# ---- RISC-V 编译参数 ----
add_library(cute_riscv_baremetal_options INTERFACE)
target_compile_options(cute_riscv_baremetal_options INTERFACE
    -std=gnu99 -O3
    -march=rv64imafdcv -mabi=lp64d
    -specs=htif_nano.specs
    -fno-common -fno-builtin-printf -Wall
)
target_link_options(cute_riscv_baremetal_options INTERFACE
    -static -specs=htif_nano.specs
)

add_library(cute_riscv_vector_baremetal_options INTERFACE)
target_compile_options(cute_riscv_vector_baremetal_options INTERFACE
    -std=gnu99 -O3
    -march=rv64gcv_zvl512b_zvfh -mabi=lp64d
    -specs=htif_nano.specs
    -fno-common -fno-builtin-printf -Wall
)
target_link_options(cute_riscv_vector_baremetal_options INTERFACE
    -static -specs=htif_nano.specs
    -march=rv64gcv_zvl512b_zvfh -mabi=lp64d
)

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
# L2: cutelib/primitive — standalone vector/math primitives
# ============================================================
add_library(cutelib_primitive INTERFACE)
target_include_directories(cutelib_primitive INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/cutelib/primitive/include
)
target_link_libraries(cutelib_primitive INTERFACE cutelib_runtime)

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
    target_link_libraries(${case_name} PRIVATE cutelib_runtime cute_riscv_baremetal_options)
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
    target_link_libraries(${case_name} PRIVATE cutelib_tensor cute_riscv_baremetal_options)
    set_target_properties(${case_name} PROPERTIES
        SUFFIX .riscv
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tensor
    )
endfunction()

add_tensor_test(tensor_matmul_i8_128_128_128_zeroinit)
add_tensor_test(tensor_matmul_i8_128_128_128_zeroinit_transpose)
add_tensor_test(tensor_matmul_mxfp8e4m3_64_64_64_zeroinit)
add_tensor_test(tensor_matmul_i8_tiled_128x128_fifo)
add_tensor_test(tensor_matmul_i8_tiled_128x128_cpu_memcpy_no_pipeline)
add_tensor_test(tensor_matmul_i8_tiled_128x128_cpu_memcpy_pipeline)

# ============================================================
# Primitive tests
# ============================================================

function(add_primitive_test case_dir)
    get_filename_component(case_name ${case_dir} NAME)
    set(test_src ${CMAKE_CURRENT_SOURCE_DIR}/tests/primitive/${case_dir}/test.c)
    set(data_dir  ${CMAKE_CURRENT_SOURCE_DIR}/tests/primitive/${case_dir})

    if(NOT EXISTS ${test_src})
        message(WARNING "Skipping ${case_name}: test.c not found")
        return()
    endif()

    add_executable(${case_name} ${test_src})
    target_include_directories(${case_name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${data_dir}
    )
    target_link_libraries(${case_name} PRIVATE cutelib_primitive cute_riscv_vector_baremetal_options m)
    set_target_properties(${case_name} PROPERTIES
        SUFFIX .riscv
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/primitive
    )
endfunction()

add_primitive_test(primitive_vec_math_n256)
add_primitive_test(primitive_dequant_f32_m64_n64)
add_primitive_test(primitive_dequant_f16_m64_n64)
add_primitive_test(primitive_fuse_dequant_silu_m64_n64)
add_primitive_test(primitive_fuse_dequant_resadd_m64_n64)
add_primitive_test(primitive_fuse_dequant_bf16cvt_m64_n64)
add_primitive_test(primitive_fuse_dequant_rope_bf16cvt_m64_n64)
add_primitive_test(primitive_fuse_dequant_hadamard_m64_n128)
add_primitive_test(primitive_fuse_masked_softmax_kvscale_bf16cvt_m64_n128)
add_primitive_test(primitive_silu_m128_n128)
add_primitive_test(primitive_resadd_m64_n64)
add_primitive_test(primitive_hadamard_m64_n128)
add_primitive_test(primitive_rope_pos0_m64_head_dim64_n_head1)
add_primitive_test(primitive_rope_pos17_m64_head_dim64_n_head1)
add_primitive_test(primitive_masked_softmax_m64_n128)
add_primitive_test(primitive_smoothquant_m128_k2048)
add_primitive_test(primitive_rmsnorm_batch1_seq_len128_hidden_dim2048)
add_primitive_test(primitive_rmsnorm_scale_batch1_seq_len128_hidden_dim2048)
