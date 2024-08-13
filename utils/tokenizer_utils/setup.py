from setuptools import setup,  Extension
from Cython.Build import cythonize

extensions = [
    Extension('decode_fast', ['decode_fast.pyx']), 
    Extension('encode_fast', ['encode_fast.pyx'])
]

setup(
    package_dir={'utils\\tokenizer_utils\\': ''},
    ext_modules=cythonize(extensions)
)