message(STATUS "Detected SOC version: ${SOC_VERSION}")

set(ASCEND_CANN_PACKAGE_PATH "/workspace/lhb/Ascend/ascend-toolkit/latest" CACHE PATH "ASCEND CANN package installation directory")
set(RUN_MODE "npu" CACHE STRING "run mode: npu")

file(GLOB KERNEL_FILES
    *.cpp
)

message(STATUS "kernel files: ${KERNEL_FILES}")

include (npu_lib.cmake)

target_include_directories(cache_kernels
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)