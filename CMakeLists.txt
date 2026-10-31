cmake_minimum_required(VERSION 3.15)
project(cute_sdk C)

set(CUTE_TARGET_CMAKE "" CACHE FILEPATH "Generated, checked OpenCUTE target bundle")
if(CUTE_TARGET_CMAKE)
    include(${CUTE_TARGET_CMAKE})
else()
    # Legacy standalone SDK use. System/model builders always pass a bundle.
    set(CUTE_TARGET_MARCH rv64gcv_zvl512b_zvfh)
    set(CUTE_TARGET_BASE_MARCH rv64imafdcv)
    set(CUTE_TARGET_MABI lp64d)
    set(CUTE_TARGET_ISA_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/cuteisa/cute_isa_v1)
    set(CUTE_TARGET_HAS_TCM ON)
endif()
if(NOT CUTE_TARGET_BASE_MARCH)
    set(CUTE_TARGET_BASE_MARCH ${CUTE_TARGET_MARCH})
endif()

# ---- RISC-V 编译参数 ----
add_library(cute_riscv_baremetal_options INTERFACE)
target_compile_options(cute_riscv_baremetal_options INTERFACE
    -std=gnu99 -O3
    -march=${CUTE_TARGET_BASE_MARCH} -mabi=${CUTE_TARGET_MABI}
    -specs=htif_nano.specs
    -fno-common -fno-builtin-printf -Wall
)
target_link_options(cute_riscv_baremetal_options INTERFACE
    -static -specs=htif_nano.specs
    -march=${CUTE_TARGET_BASE_MARCH} -mabi=${CUTE_TARGET_MABI}
)

add_library(cute_riscv_vector_baremetal_options INTERFACE)
target_compile_options(cute_riscv_vector_baremetal_options INTERFACE
    -std=gnu99 -O3
    -march=${CUTE_TARGET_MARCH} -mabi=${CUTE_TARGET_MABI}
    -specs=htif_nano.specs
    -fno-common -fno-builtin-printf -Wall
)
target_link_options(cute_riscv_vector_baremetal_options INTERFACE
    -static -specs=htif_nano.specs
    -march=${CUTE_TARGET_MARCH} -mabi=${CUTE_TARGET_MABI}
)
if(CUTE_TARGET_CMAKE)
    target_link_options(cute_riscv_baremetal_options INTERFACE "-Wl,-T,${CUTE_TARGET_LINKER}")
    target_link_options(cute_riscv_vector_baremetal_options INTERFACE "-Wl,-T,${CUTE_TARGET_LINKER}")
    target_compile_definitions(cute_riscv_baremetal_options INTERFACE CUTE_CHECKPOINT_DRAIN_CYCLES=${CUTE_TARGET_DRAIN_CYCLES})
    target_compile_definitions(cute_riscv_vector_baremetal_options INTERFACE CUTE_CHECKPOINT_DRAIN_CYCLES=${CUTE_TARGET_DRAIN_CYCLES})
endif()

# ============================================================
# L0: cuteisa — ISA 原子指令（header-only）
# ============================================================
add_library(cuteisa INTERFACE)
target_include_directories(cuteisa INTERFACE
    ${CUTE_TARGET_ISA_INCLUDE}
    ${CUTE_TARGET_MEMORY_INCLUDE}
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
# L3: cutelib/fusion — tensor tile_op + vector primitive adapters
# ============================================================
add_library(cutelib_fusion INTERFACE)
target_include_directories(cutelib_fusion INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/cutelib/fusion/include
)
target_link_libraries(cutelib_fusion INTERFACE cutelib_tensor cutelib_primitive)

# ============================================================
# L4: cutelib/layer — layer orchestration APIs
# ============================================================
add_library(cutelib_layer INTERFACE)
target_include_directories(cutelib_layer INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/cutelib/layer/include
)
target_link_libraries(cutelib_layer INTERFACE cutelib_fusion)

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

# The system runner supplies one generated fixture directory to both platforms.
# Both are system-mode bare-metal binaries; no linux-user ABI is involved.
set(CUTE_SYSTEM_FIXTURES "" CACHE PATH "Generated system verification input headers")
if(CUTE_SYSTEM_FIXTURES)
    add_executable(system_platform_contract tests/system/platform_contract.c)
    target_include_directories(system_platform_contract PRIVATE ${CUTE_SYSTEM_FIXTURES})
    target_link_libraries(system_platform_contract PRIVATE cute_riscv_vector_baremetal_options)
    set_target_properties(system_platform_contract PROPERTIES SUFFIX .riscv
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/system)
    add_executable(system_qemu_contract tests/system/qemu_contract.c)
    target_link_libraries(system_qemu_contract PRIVATE
        cutelib_runtime cute_riscv_vector_baremetal_options)
    set_target_properties(system_qemu_contract PROPERTIES SUFFIX .riscv
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/system)
    foreach(platform IN ITEMS verilator qemu)
        add_executable(system_int8_${platform} tests/system/system_int8.c)
        target_include_directories(system_int8_${platform} PRIVATE ${CUTE_SYSTEM_FIXTURES})
        target_link_libraries(system_int8_${platform} PRIVATE
            cutelib_runtime cute_riscv_vector_baremetal_options)
        target_compile_definitions(system_int8_${platform} PRIVATE
            CUTE_SYSTEM_PLATFORM_${platform}=1)
        set_target_properties(system_int8_${platform} PROPERTIES SUFFIX .riscv
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/system)
        if(CUTE_TARGET_HAS_TCM)
        add_executable(system_tcm_${platform} tests/system/system_tcm.c)
        target_include_directories(system_tcm_${platform} PRIVATE ${CUTE_SYSTEM_FIXTURES})
        target_link_libraries(system_tcm_${platform} PRIVATE
            cutelib_runtime cute_riscv_vector_baremetal_options)
        target_compile_definitions(system_tcm_${platform} PRIVATE CUTE_SYSTEM_PLATFORM_${platform}=1)
        set_target_properties(system_tcm_${platform} PROPERTIES SUFFIX .riscv
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/system)
        endif()
    endforeach()
    foreach(fault IN ITEMS 1 2)
        add_executable(system_int8_fault${fault} tests/system/system_int8.c)
        target_include_directories(system_int8_fault${fault} PRIVATE ${CUTE_SYSTEM_FIXTURES})
        target_link_libraries(system_int8_fault${fault} PRIVATE
            cutelib_runtime cute_riscv_vector_baremetal_options)
        target_compile_definitions(system_int8_fault${fault} PRIVATE CUTE_SYSTEM_FAULT=${fault})
        set_target_properties(system_int8_fault${fault} PROPERTIES SUFFIX .riscv
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/system)
    endforeach()
endif()

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
add_primitive_test(primitive_rope_bf16_pos0_m64_head_dim64_n_head1)
add_primitive_test(primitive_rope_bf16_pos17_m64_head_dim64_n_head1)
add_primitive_test(primitive_masked_softmax_bf16_m64_n128)
add_primitive_test(primitive_smoothquant_m128_k2048)
add_primitive_test(primitive_smoothquant_stage2_m128_k2048)
add_primitive_test(primitive_rmsnorm_batch1_seq_len128_hidden_dim2048)
add_primitive_test(primitive_rmsnorm_scale_batch1_seq_len128_hidden_dim2048)

# ============================================================
# Fusion tests
# ============================================================

function(add_fusion_test_variant case_dir variant source)
    set(case_name ${case_dir}_${variant})
    set(test_src ${CMAKE_CURRENT_SOURCE_DIR}/tests/fusion/${case_dir}/${source})
    set(data_dir ${CMAKE_CURRENT_SOURCE_DIR}/tests/fusion/${case_dir})

    if(NOT EXISTS ${test_src})
        message(WARNING "Skipping ${case_name}: ${source} not found")
        return()
    endif()

    add_executable(${case_name} ${test_src})
    target_include_directories(${case_name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/fusion
        ${data_dir}
    )
    target_link_libraries(${case_name} PRIVATE
        cutelib_fusion
        cute_riscv_vector_baremetal_options
        m
    )
    set_target_properties(${case_name} PROPERTIES
        SUFFIX .riscv
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/fusion
    )
endfunction()

add_fusion_test_variant(fusion_matmul_dequant_silu notile test_notile.c)
add_fusion_test_variant(fusion_matmul_dequant_silu nopipeline test_nopipeline.c)
add_fusion_test_variant(fusion_matmul_dequant_silu pipeline test_pipeline.c)
add_fusion_test_variant(fusion_matmul_dequant_resadd notile test_notile.c)
add_fusion_test_variant(fusion_matmul_dequant_resadd nopipeline test_nopipeline.c)
add_fusion_test_variant(fusion_matmul_dequant_resadd pipeline test_pipeline.c)
add_fusion_test_variant(fusion_matmul_dequant_bf16cvt notile test_notile.c)
add_fusion_test_variant(fusion_matmul_dequant_bf16cvt nopipeline test_nopipeline.c)
add_fusion_test_variant(fusion_matmul_dequant_bf16cvt pipeline test_pipeline.c)
add_fusion_test_variant(fusion_matmul_dequant_bf16cvt_transpose notile test_notile.c)
add_fusion_test_variant(fusion_matmul_dequant_bf16cvt_transpose nopipeline test_nopipeline.c)
add_fusion_test_variant(fusion_matmul_dequant_bf16cvt_transpose pipeline test_pipeline.c)
add_fusion_test_variant(fusion_matmul_dequant_rope_bf16cvt notile test_notile.c)
add_fusion_test_variant(fusion_matmul_dequant_rope_bf16cvt nopipeline test_nopipeline.c)
add_fusion_test_variant(fusion_matmul_dequant_rope_bf16cvt pipeline test_pipeline.c)
add_fusion_test_variant(fusion_matmul_dequant_hadamard notile test_notile.c)
add_fusion_test_variant(fusion_matmul_dequant_hadamard nopipeline test_nopipeline.c)
add_fusion_test_variant(fusion_matmul_dequant_hadamard pipeline test_pipeline.c)
add_fusion_test_variant(fusion_matmul_masked_softmax_kvscale_bf16cvt notile test_notile.c)
add_fusion_test_variant(fusion_matmul_masked_softmax_kvscale_bf16cvt nopipeline test_nopipeline.c)
add_fusion_test_variant(fusion_matmul_masked_softmax_kvscale_bf16cvt pipeline test_pipeline.c)
add_fusion_test_variant(fusion_matmul_masked_softmax_kvscale_bf16cvt_m128_n128_k64 notile test_notile.c)
add_fusion_test_variant(fusion_matmul_masked_softmax_kvscale_bf16cvt_m128_n128_k64 nopipeline test_nopipeline.c)
add_fusion_test_variant(fusion_matmul_masked_softmax_kvscale_bf16cvt_m128_n128_k64 pipeline test_pipeline.c)
add_fusion_test_variant(fusion_attention_context_f16_m128_n64_k128 notile test_notile.c)
add_fusion_test_variant(fusion_attention_context_f16_m128_n64_k128 nopipeline test_nopipeline.c)
add_fusion_test_variant(fusion_attention_context_f16_m128_n64_k128 pipeline test_pipeline.c)

# ============================================================
# Layer tests
# ============================================================

function(add_layer_test case_dir)
    get_filename_component(case_name ${case_dir} NAME)
    set(test_src ${CMAKE_CURRENT_SOURCE_DIR}/tests/layer/${case_dir}/test.c)
    set(data_dir ${CMAKE_CURRENT_SOURCE_DIR}/tests/layer/${case_dir})
    set(layer_sources ${test_src})

    if(NOT EXISTS ${test_src})
        message(WARNING "Skipping ${case_name}: test.c not found")
        return()
    endif()

    set(weights_src_in ${data_dir}/weights.c.in)
    if(EXISTS ${weights_src_in})
        set(generated_layer_dir ${CMAKE_BINARY_DIR}/generated/layer)
        file(MAKE_DIRECTORY ${generated_layer_dir})
        set(weights_src ${generated_layer_dir}/${case_name}_weights.c)
        configure_file(${weights_src_in} ${weights_src} @ONLY)
        file(READ ${weights_src_in} weights_src_template)
        string(REGEX MATCHALL
            "@CMAKE_CURRENT_SOURCE_DIR@/[^\"\n]+\\.bin"
            weight_bin_refs "${weights_src_template}")
        set(weight_bin_deps)
        foreach(weight_bin_ref IN LISTS weight_bin_refs)
            string(REPLACE "@CMAKE_CURRENT_SOURCE_DIR@"
                "${CMAKE_CURRENT_SOURCE_DIR}"
                weight_bin_dep "${weight_bin_ref}")
            list(APPEND weight_bin_deps ${weight_bin_dep})
        endforeach()
        if(weight_bin_deps)
            set_source_files_properties(${weights_src} PROPERTIES
                OBJECT_DEPENDS "${weight_bin_deps}")
        endif()
        list(APPEND layer_sources ${weights_src})
    endif()

    add_executable(${case_name} ${layer_sources})
    target_include_directories(${case_name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${data_dir}
    )
    target_link_libraries(${case_name} PRIVATE
        cutelib_layer
        cute_riscv_vector_baremetal_options
        m
    )
    set_target_properties(${case_name} PROPERTIES
        SUFFIX .riscv
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/layer
    )
endfunction()

add_layer_test(llama_block_zero_weight_seq128_embed64)
add_layer_test(llama_block_zero_weight_1b_shape_seq128)
add_layer_test(llama_block_nonzero_weight_1b_shape_seq128)
add_layer_test(llama_stage_proj_q_bf16_nonzero_1b_shape_seq128)
add_layer_test(llama_stage_proj_k_bf16_nonzero_1b_shape_seq128)
add_layer_test(llama_stage_proj_v_bf16_nonzero_1b_shape_seq128)
add_layer_test(llama_stage_score_head0_bf16_nonzero_1b_shape_seq128)
add_layer_test(llama_stage_ffn_gate_nonzero_1b_shape_seq128)
add_layer_test(llama_stage_ffn_up_nonzero_1b_shape_seq128)
add_layer_test(llama_stage_ffn_down_nonzero_1b_shape_seq128)

# cutetest BERT B1-B6 / LLaMA L1-L4 migration, generated case directories.
set(CUTE_MODEL_FIXTURES "" CACHE PATH "Generated model stage fixtures")
if(CUTE_MODEL_FIXTURES)
    file(GLOB model_cases RELATIVE ${CUTE_MODEL_FIXTURES} ${CUTE_MODEL_FIXTURES}/*/model_program.h)
    foreach(model_header ${model_cases})
        get_filename_component(model_case ${model_header} DIRECTORY)
        foreach(platform qemu verilator)
            add_executable(model_${model_case}_${platform} tests/models/model.c)
            target_include_directories(model_${model_case}_${platform} PRIVATE ${CUTE_MODEL_FIXTURES}/${model_case})
            target_link_libraries(model_${model_case}_${platform} PRIVATE cutelib_primitive cute_riscv_vector_baremetal_options m)
            target_compile_options(model_${model_case}_${platform} PRIVATE -ffp-contract=off)
            target_compile_definitions(model_${model_case}_${platform} PRIVATE CUTE_SYSTEM_PLATFORM_${platform}=1)
            set_target_properties(model_${model_case}_${platform} PROPERTIES SUFFIX .riscv RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/models)
        endforeach()
    endforeach()
endif()
