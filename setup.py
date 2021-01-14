import distutils.core
import Cython.Build
import numpy

distutils.core.setup(
    include_dirs = [numpy.get_include()],
    ext_modules = Cython.Build.cythonize("cython_part.pyx"))