import sys
import numpy as np
import scipy
import matplotlib
import numba
import dask
import pytest

print("Python:", sys.version)
print("Executable:", sys.executable)

print("NumPy:", np.__version__)
print("SciPy:", scipy.__version__)
print("Matplotlib:", matplotlib.__version__)
print("Numba:", numba.__version__)
print("Dask:", dask.__version__)
print("PyTest:", pytest.__version__)


