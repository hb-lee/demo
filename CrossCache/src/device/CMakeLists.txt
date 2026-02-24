#***********************************************************
#***********************************************************

if(DEFINED NPU)
    message("TORCH_NPU_PATH is ${TORCH_NPU_PATH}")
    message("TORCH_PATH is ${TORCH_PATH}")
    message("CANN_PACKAGES_PATH is ${ASCEND_CANN_PACKAGE_PATH}")

    if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
        set(ASCEND_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
        set(ASCEND_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake)
        set(ASCEND_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake)
    else()
        message(FATAL_ERROR "ascendc_kernel_cmake does not exist, please check whether the cann package is installed.")
    endif()

    message("ASCEND_CMAKE_DIR: ${ASCEND_CMAKE_DIR}")
endif()

if(DEFINED NPU)
    file(GLOB DEVICE_SRCS
        "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
        "{CMAKE_CURRENT_SOURCE_DIR}/npu/*.cpp"
    )
elseif(DEFINED NPUSDK)
    file(GLOB DEVICE_SRCS
        "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
        "{CMAKE_CURRENT_SOURCE_DIR}/npu_sdk/*.cpp"
    )
else()
    file(GLOB DEVICE_SRCS
        "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
        "{CMAKE_CURRENT_SOURCE_DIR}/mem/*.cpp"
    )
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if(DEFINED NPU)
    include_directories(
        ${TORCH_PATH}/include
        ${TORCH_NPU_PATH}/include
        ${ASCEND_CANN_PACKAGE_PATH}/include
        ${ASCEND_CANN_PACKAGE_PATH}/runtime/include
    )

    link_directories(
        ${TORCH_PATH}/lib
        ${TORCH_NPU_PATH}/lib
        ${ASCEND_CANN_PACKAGE_PATH}/lib64
    )
endif()

if(DEFINED NPUSDK)
    include_directories(
        ${ASCEND_CANN_PACKAGE_PATH}/include
        ${ASCEND_CANN_PACKAGE_PATH}/runtime/include
    )

    link_directories(
        ${ASCEND_CANN_PACKAGE_PATH}/lib64
    )
endif()

#***********************************************************
#***********************************************************

set_source_files_properties(${DEVICE_SRCS}
    PROPERTIES COMPILE_FLAGS
    "-Wno-unused-function -Wno-unused-variable")

#***********************************************************
#***********************************************************

if(DEFINED NPU)
    add_subdirectory(npu/kernels)
endif()

add_library(device SHARED ${DEVICE_SRCS})

if(DEFINED NPU)
    target_link_libraries(
        device
        PUBLIC
        cache_kernels
        ascendcl
        platform
        tiling_api
        infra
        securec
    )
elseif(DEFINED NPUSDK)
    target_link_libraries(
        device
        PUBLIC
        ascendcl
        infra
        securec
    )
else()
    target_link_libraries(
        device
        PUBLIC
        infra
        securec
    )
endif()