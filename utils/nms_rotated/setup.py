#!/usr/bin/env python
import os
import subprocess
import time
from setuptools import find_packages, setup

import torch
from torch.utils.cpp_extension import (BuildExtension, CppExtension,
                                       CUDAExtension)
def make_cuda_ext(name, module, sources, sources_cuda=[]):

    define_macros = []
    extra_compile_args = {'cxx': []}

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
        sources += sources_cuda
    else:
        print(f'Compiling {name} without CUDA')
        extension = CppExtension
        # raise EnvironmentError('CUDA is required to compile MMDetection!')

    return extension(
        name=f'{module}.{name}',
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args)

# python setup.py develop
if __name__ == '__main__':
    #write_version_py()
    setup(
        name='nms_rotated',
        ext_modules=[
            make_cuda_ext(
                name='nms_rotated_ext',
                module='',
                sources=[
                    'src/nms_rotated_cpu.cpp',
                    'src/nms_rotated_ext.cpp'
                ],
                sources_cuda=[
                    'src/nms_rotated_cuda.cu',
                    'src/poly_nms_cuda.cu',
                ]),
        ],
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False)