set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

if(NOT DEFINED CUTE_ROOT)
    if(DEFINED ENV{CUTE_ROOT})
        set(CUTE_ROOT "$ENV{CUTE_ROOT}")
    else()
        # cute-sdk/cmake/ → cute-sdk/ → CUTE/
        get_filename_component(CUTE_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
    endif()
endif()

set(TOOLCHAIN_PREFIX "${CUTE_ROOT}/tool/riscv/bin/riscv64-unknown-elf-")

set(CMAKE_C_COMPILER   ${TOOLCHAIN_PREFIX}gcc)
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_PREFIX}g++)
set(CMAKE_OBJCOPY      ${TOOLCHAIN_PREFIX}objcopy)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
