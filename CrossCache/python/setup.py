from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='ipckey_utils',
    ext_modules=[
        CppExtension(
            name='ipckey_utils',
            sources=['ipckey_utils.cpp'],
            include_dirs=['/workspace/lhb/Ascend/ascend-toolkit/latest/include',
                            '/workspace/lhb/.local/lib/python3.11/site-packages/torch_npu/include',
                            ],
            library_dirs=['/workspace/lhb/Ascend/ascend-toolkit/latest/lib64',
                '/workspace/lhb/.local/lib/python3.11/site-packages/torch_npu/lib'],
            libraries=['ascendcl', 'torch_npu'],
        )
    ],
    cmdclass={'build_ext', BuildExtension},
)

# python3 setup.py build_ext --inplace