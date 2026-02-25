#***********************************************************
#***********************************************************

set(LIBZMQ_DIR "${THIRD_PARTY}/libzmq")

include_directories(${PROJECT_SOURCE_DIR}/include/inner_h)
include_directories(${PROJECT_SOURCE_DIR}/include/cache)

include_directories(${INFRA_DIR}/include)
include_directories(${SECUREC_DIR}/include)
include_directories(${ZLOG_DIR}/include)
include_directories(${LIBZMQ_DIR}/include)

#***********************************************************
#***********************************************************

link_directories(${PROJECT_BINARY_DIR}/lib)
link_directories(${SECUREC_DIR}/lib)
link_directories(${ZLOG_DIR}/lib)
link_directories(${LIBZMQ_DIR}/lib)

#***********************************************************
#***********************************************************

if (DEFINED NPU)
    add_definitions(-D_NPU)
endif()
if (DEFINED NPUSDK)
    add_definitions(-D_NPUSDK)
endif()

add_subdirectory(cache)
add_subdirectory(device)

if (DEFINED NPU)
    add_link_options(
        "LINKER:-rpath-link,/workspace/lhb/.local/lib/python3.11/site-packages/torch.libs"
    )
endif()

add_executable(crosscache main.c)

if (DEFINED NPU)
    target_link_directories(
        crosscache
        PRIVATE
        ${PYTHON_PATH}/lib
        ${TORCH_PATH}/lib
        ${TORCH_NPU_PATH}/lib
        ${ASCEND_CANN_PACKAGE_PATH}/lib64
    )

    target_link_libraries(crosscache "-Wl,--whole-archive"
                cache
                device
                "-Wl,--no-whole-archive"
                python3.11
                dl
                c10
                torch_npu
                torch_python
                ascendcl
                platform
                tiling_api
                zmq
                infra)
elseif(DEFINED NPUSDK)
    target_link_directories(
        crosscache
        PRIVATE
        ${ASCEND_CANN_PACKAGE_PATH}/lib64
    )

    target_link_libraries(crosscache "-Wl,--whole-archive"
                cache
                device
                "-Wl,--no-whole-archive"
                ascendcl
                zmq
                infra)
else()
    target_link_libraries(crosscache "-Wl,--whole-archive"
                cache
                device
                "-Wl,--no-whole-archive"
                rt
                zmq
                infra)
endif()