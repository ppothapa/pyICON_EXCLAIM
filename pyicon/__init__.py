
"""Top-level package for pyicon."""

__author__ = """ICON team"""
__email__ = "nils.brueggemann@uni-hamburg.de"
__version__ = "0.1.0"

# --- import pyicon basic modules
from .pyicon_calc import *
from .pyicon_tb import *
from .pyicon_IconData import *
from .pyicon_plotting import *

# --- import pyicon.view
from . import view
# --- import pyicon.quickplots
from . import quickplots
