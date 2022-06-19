import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.interpolate import InterpolatedUnivariateSpline
from astropy.convolution import Box1DKernel, Gaussian1DKernel, convolve, convolve_fft

from odreduce import models
from odreduce import utils
from odreduce import plots

import warnings
warnings.filterwarnings("ignore")

#####################################################################
# Each star or "target" that is processed with ODreduce
# -> initialization loads in data for a single star
#    but will not execute the main module unless it
#    is in the proper ODreduce mode
#

class Target:

    def __init__(self, star, args):
        self.name = star
        self.params, self.verbose = \
            args.params, args.verbose
        self = utils.load_data(self, args)


