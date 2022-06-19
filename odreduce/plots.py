#to be continued
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from astropy.convolution import convolve, Box1DKernel

from pysyd import models
from pysyd import utils


def set_plot_params():
    """
    Sets the matplotlib parameters.

    Returns
    -------
    None

    """

    plt.style.use('dark_background')



