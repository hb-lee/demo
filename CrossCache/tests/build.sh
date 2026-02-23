#!/bin/bash
bisheng -xasc ../src/device/npu/kernels/multi_layer_memcpy_kernel.cpp test_kernel.cpp --npu-arch=dav-2201 -ltiling_api -lplatform -lpthread -ldl -L /workspace/lhb/Ascend/ascend-toolkit/latest/lib64
