#include <stdio.h>
#include <acl/acl.h>
#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>

namespace py=pybind11;

bool ipckey_init() {
    aclError ret = aclInit(nullptr);
    if (ret == ACL_ERROR_REPEAT_INITIALIZE) return true;
    return ret == ACL_SUCCESS;
}

bool ipckey_exit() {
    aclFinalize();
    return true;
}

py::bytes ipckey_from_tensor(const torch::Tensor& t) {
    // TORCH_CHECK(t.device().is_npu(), "Tensor must be on NPU");
    int64_t ptr = reinterpret_cast<int64_t>(t.data_ptr());
    int64_t size = t.numel() * t.element_size();

    char key[65] = "name";
    aclError ret = aclrtIpcMemGetExportKey(
        reinterpret_cast<void *>(ptr), size, key, 65, 1);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "ipc export key failed, ret:%d\n", ret);
        throw std::runtime_error("aclrtIpcMemGetExportKey failed");
    }

    return py::bytes(key, 64);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ipckey_init", &ipckey_init, "init ipckey env");
    m.def("ipckey_exit", &ipckey_exit, "exit ipckey env");
    m.def("ipckey_from_tensor", &ipckey_from_tensor,
        "pack tensor ptr to ipc key");
}

