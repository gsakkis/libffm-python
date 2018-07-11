import sys
from setuptools import setup, Extension

USEOMP = True
USESSE = True

extra_compile_args = ['-O3', '-std=c++0x', '-march=native']
extra_link_args = []
if USEOMP:
    extra_compile_args.extend(['-fopenmp', '-DUSEOMP'])
    extra_link_args.append('-fopenmp')
if USESSE:
    extra_compile_args.append('-DUSESSE')
if sys.platform.startswith('win32'):
    extra_compile_args.append('-D_WIN32')

setup(
    name='ffm',
    version='0.1',
    description='Python wrapper for LibFFM',
    url='https://github.com/gsakkis/libffm-python',
    packages=['ffm'],
    ext_modules=[
        Extension(
            name='ffm._libffm',
            sources=['ffm/ffm-wrapper.cpp', 'libffm/timer.cpp'],
            include_dirs=['libffm'],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ],
    zip_safe=False,
    install_requires=['numpy', 'sklearn', 'click'],
    entry_points='''
        [console_scripts]
        ffm=ffm.cli:cli
    ''',
    license='BSD',
    classifiers=['License :: OSI Approved :: BSD License'],
)
