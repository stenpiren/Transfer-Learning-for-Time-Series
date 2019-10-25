from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

ext = Extension("dtw", sources=["dtw.pyx"], include_dirs=['.',numpy.get_include()])
setup(name="dtw", ext_modules=cythonize([ext]))
