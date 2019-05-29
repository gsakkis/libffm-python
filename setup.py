import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class BuildLibFFM(build_ext):

    user_options = build_ext.user_options + [
        ('omp', None, 'use OpenMP'),
        ('sse', None, 'use SSE instructions'),
    ]
    boolean_options = build_ext.boolean_options + ['omp', 'sse']

    def initialize_options(self):
        build_ext.initialize_options(self)
        self.omp = self.sse = 0

    def build_extension(self, ext):
        ext.extra_compile_args.append('-std=c++11')
        if self.omp:
            ext.extra_compile_args.extend(['-fopenmp', '-DUSEOMP'])
            ext.extra_link_args.append('-fopenmp')
        if self.sse:
            ext.extra_compile_args.extend(['-march=native', '-DUSESSE'])
        if sys.platform.startswith('win32'):
            ext.extra_compile_args.append('-D_WIN32')
        build_ext.build_extension(self, ext)


setup(
    name='ffm',
    version='0.0.2',
    description='Python wrapper for LibFFM',
    url='https://github.com/gsakkis/libffm-python',
    packages=['ffm'],
    cmdclass={'build_ext': BuildLibFFM},
    ext_modules=[
        Extension(
            name='ffm._libffm',
            sources=['ffm/ffm-wrapper.cpp', 'libffm/timer.cpp'],
            include_dirs=['libffm'],
        )
    ],
    zip_safe=False,
    install_requires=['numpy', 'scikit-learn[alldeps]', 'click'],
    entry_points='''
        [console_scripts]
        ffm=ffm.cli:cli
    ''',
    license='BSD',
    classifiers=['License :: OSI Approved :: BSD License'],
)
