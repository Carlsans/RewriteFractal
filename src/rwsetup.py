import os

import numpy
from setuptools import setup
from Cython.Build import cythonize
from setuptools.extension import Extension


extensions = [
    Extension("*", ["src/*.pyx"],
        include_dirs=[numpy.get_include()],
        annotate=True
              ),


]
setup(
    name="RewriterEngineCython",
    ext_modules=cythonize(extensions),
)

# setup(
#     name="RewriterEngineCython",
#     ext_modules=cythonize(extensions),
#
# )


#import test
# setup(
#     #ext_modules=cythonize("/home/carl/PycharmProjects/rewriteFractals/src/RewriterEngineCython.pyx", annotate=True ),
#     ext_modules=cythonize("/home/carl/PycharmProjects/rewriteFractals/src/RewriterEngine.py", annotate=True ),
#
# )
