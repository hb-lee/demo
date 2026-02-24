#***********************************************************
#***********************************************************

include _directories(${JSONCPP_DIR}/include)

#***********************************************************
#***********************************************************

aux_source_directory(CONFIG_SRC config.cpp)

#***********************************************************
#***********************************************************

add_library(config ${CONFIG_SRC})
if (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64")
    target_compile_definitions(config PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
else()
    target_compile_definitions(config PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
endif()
target_link_libraries(config infra
							jsoncpp)