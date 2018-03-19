# ------------------------------- Information ------------------------------- #
# Author:       Joey Dumont                    <joey.dumont@gmail.com>        #
# Created:      Mar. 15th, 2018                                               #
# Description:  Compute the discrete Fourier transform in polar Coordinates.  #
# Dependencies: - NumPy                                                       #
#               - SciPy                                                       #
# --------------------------------------------------------------------------- #

# --------------------------- Modules Importation --------------------------- #
import numpy as np
import matplotlib.pyplot as plt


def polar_fft2(samples, **kwargs):
	"""
	Computes the Fourier transform in polar coordinates, which is a Hankel
	transform and something akin a cosine transform.

	The function values in samples are assumed to be equidistant on both
	axes, though not necessarily the same sampling rate on both axes.

	kwargs can contain deltaR and deltaTheta, which gives the actual sampling
	rate of both axes. Used for normalization.
	"""
	summand   = np.zeros_like(samples, dtype=complex)
	transform = np.zeros_like(samples, dtype=complex)
	Nr        = samples.shape[0]
	Nth       = samples.shape[1]

	try:
		deltaR = kwargs['deltaR']
	except:
		deltaR = 1

	deltaTheta = 2.0*np.pi/Nth

	fr        = np.linspace(0.0, 1.0/(deltaR), Nr)
	fth       = np.linspace(-1.0/(2.0*deltaTheta), 1.0/(2*deltaTheta), Nth)

	for n in range(Nr):
		for m in range(Nth):
			# -- We compute the summand for that particular frequency.
			for i in range(Nr):
				for j in range(Nth):
					summand[i,j] = i*samples[i,j]*np.exp(-1j*i*n/Nr*np.cos(2.0*np.pi*j/Nth-m/(2.0*np.pi)))

			# -- Sum the array.
			transform[n,m] = np.sum(summand)
	transform *= deltaR**2*deltaTheta/(2.0*np.pi)
	return transform, fr, fth


def f(r,theta):
	return np.exp(-0.5*r**2)
def Hf(k):
        return np.exp(-0.5*k**2)

size_r  = 500
size_th = 10
r  = np.linspace(0.0,5.0,size_r)
th = np.linspace(0.0,2*np.pi,size_th)
samples = np.zeros((size_r,size_th))

for i in range(size_r):
	for j in range(size_th):
		samples[i,j] = f(r[i],th[j])

transform, fr, fth = polar_fft2(samples, deltaR= r[1] - r[0])

plt.figure()
plt.plot(r,samples[:,0])

plt.figure()
plt.plot(fr, np.real(transform[:,0]))
plt.plot(fr, Hf(fr))

print(transform[0,0])
plt.show()
