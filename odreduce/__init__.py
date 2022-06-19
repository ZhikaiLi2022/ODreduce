import os

from .models import *
from .pipeline import *
from .plots import *
from .target import *
from .utils import *

__all__ = ['cli', 'pipeline', 'models', 'target', 'plots', 'utils']

__version__ = '1.0.0'

_ROOT = os.path.abspath(os.getcwd())
INFDIR = os.path.join(_ROOT, 'info')
INPDIR = os.path.join(_ROOT, 'raw')
OUTDIR = os.path.join(_ROOT, 'red')
