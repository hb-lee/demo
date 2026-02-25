#!/bin/bash

source ./env.local

LOCAL_PATH="$(readlink -e "$(dirname "$0")")"
PRJ_PATH="$(readlink -e "${LOCAL_PATH}/..")"
THIRD_SUB="${PRJ_PATH}/thirdparty"
Lib_PATH="${PRJ_PATH}/lib"

ARCH=$(uname -m)

function build_jsoncpp() {
	local build_path=${THIRD_SUB}/jsoncpp/build
	mkdir ${build_path}
	mkdir -p ${Lib_PATH}/jsoncpp
	cd ${build_path}
	cmake .. -DCMAKE_INSTALL_PREFIX=jsoncpp -DBUILD_SHARED_LIBS=ON -DJSONCPP_WITH_TESTS=OFF
	make -j4
	make install
	cp -rdp ${THIRD_SUB}/jsoncpp/include ${Lib_PATH}/jsoncpp
	cp -rdp ${THIRD_SUB}/jsoncpp/build/lib ${Lib_PATH}/jsoncpp
	cd -
}

function build_zlog() {
	local zlog_dir=${THIRD_SUB}/zlog
	local build_path=${zlog_dir}/build
	mkdir ${build_path}
	mkdir -p ${Lib_PATH}/zlog
	cd ${zlog_dir}
	make PREFIX=`pwd`/build/zlog
	make PREFIX=`pwd`/build/zlog install
	cp -rdp ${build_path}/zlog/include ${Lib_PATH}/zlog
	cp -rdp ${build_path}/zlog/lib ${Lib_PATH}/zlog
	cd -
}

function build_securec() {
	local securec_dir=${THIRD_SUB}/securec
	local securec_lib=${securec_dir}/build/securec/lib
	mkdir -p ${securec_lib}
	mkdir -p ${Lib_PATH}/securec
	cd ${securec_dir}
	if [ "${ARCH}" = "aarch64" ]; then
		make -f aarch64-so.mk
	else
		make -f x86-so.mk
	fi
	cp -rdp include build/securec
	cp libsecurec.so build/securec/lib
	cp -rdp build/securec/include ${Lib_PATH}/securec
	cp -rdp build/securec/lib ${Lib_PATH}/securec
	cd -
}

function build_libzmq() {
	local libzmq_dir=${THIRD_SUB}/libzmq
	local build_path=${libzmq_dir}/build
	mkdir -p ${Lib_PATH}/libzmq
	cd ${libzmq_dir}
	./autogen.sh
	./configure --prefix=`pwd`/build
	make -j4
	make install
	cp -rdp build/include ${Lib_PATH}/libzmq
	cp -rdp build/lib ${Lib_PATH}/libzmq
	cd -
}

function prepare_thirdparty() {
	build_jsoncpp
	build_zlog
	build_securec
	build_libzmq
}

function build_program() {
	local BUILD_TYPE=$1
	
	if [ ${npu} -eq 1 ]; then
		EXTRA_FLAGS="-DPYTHON_PATH=${python_path} -DTORCH_PATH=${torch_path} -DTORCH_NPU_PATH=${torch_npu_path} \
					-DASCEND_CANN_PACKAGE_PATH=${ascend_home_path} -DSOC_VERSION=${soc_version} \
					-DPYTHON_VERSION=${python_version} -DNPU=ON"
	elif [ ${npu} -eq 2 ]; then
		EXTRA_FLAGS="-DASCEND_CANN_PACKAGE_PATH=${ascend_home_path} -DSOC_VERSION=${soc_version} \
					-DNPUSDK=ON"
	else
		EXTRA_FLAGS=""
	fi
	set -x
	if [ "${BUILD_TYPE}" == "release" ]; then
		cmake .. -DCMAKE_BUILD_TYPE=Release ${EXTRA_FLAGS}
	else
		cmake .. -DCMAKE_BUILD_TYPE=Debug ${EXTRA_FLAGS}
	fi
	set +x
	
	make clean
	make VERBOSE=1 -j
}

function main() {
	local ACTION="$1"
	local PARAMETER="$2"
	
	if [ "${ACTION}" == "prepare" ]; then
		prepare_thirdparty
	fi
	if [ "$ACTION" == "build" ]; then
		build_program $PARAMETER
	fi
	if [ "$ACTION" == "build_ext" ]; then
		(cd ../python; python3 setup.py build_ext --inplace)
	fi
}

main $@
exit $?
