
"""Top-level package for pyicon."""

__author__ = """ICON team"""
__email__ = "nils.brueggemann@uni-hamburg.de"
__version__ = "0.1.0"

# --- import pyicon basic modules
print('-----calc')
from .pyicon_calc import *
print('-----calc_xr')
from .pyicon_calc_xr import *
print('-----tb')
from .pyicon_tb import *
print('-----IconData')
from .pyicon_IconData import *
print('-----plotting')
from .pyicon_plotting import *
print('-----accessor')
from .pyicon_accessor import *
print('-----simulation')
from .pyicon_simulation import *

# --- import pyicon.view
print('-----view')
from . import view
# --- import pyicon.quickplots
print('-----quickplots')
from . import quickplots
