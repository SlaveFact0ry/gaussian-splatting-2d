#
# Simple CUDA Python Extension Example
# Build with: pip install -e .
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="cuda_hello",
    packages=['cuda_hello'],
    ext_modules=[
        CUDAExtension(
            name="cuda_hello._C",  # Will be imported as: from cuda_hello import _C
            sources=[
                "ext.cpp",         # PyBind11 bindings
                "operations.cu",   # CUDA implementation
            ],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
