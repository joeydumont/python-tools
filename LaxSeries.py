# ------------------------------- Information ------------------------------- #
# Author:       Joey Dumont                    <joey.dumont@gmail.com>        #
# Created:      Aug. 14th, 2018                                               #
# Description:  Compute the Lax series at a specific order for a linearly     #
#               polarized beam.                                               #
# Dependencies: - NumPy                                                       #
#               - SciPy                                                       #
#               - matpotlib
# --------------------------------------------------------------------------- #

# --------------------------- Modules Importation --------------------------- #
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
import scipy.interpolate as interpolate
import scipy.integrate as integrate

# -------------------------------- Functions  ------------------------------- #

def ExpansionCoefficient(m,p):
    