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
import scipy.special as sp
import scipy.interpolate as interpolate
import scipy.integrate as integrate


def polar_fft2(samples, **kwargs):
	"""
	Computes the Fourier transform in polar coordinates, which is a Hankel
	transform and something akin a cosine transform.

	The function values in samples are assumed to be equidistant on both
	axes, though not necessarily the same sampling rate on both axes.

	kwargs can contain deltaR, which gives the actual sampling
	rate of the radial axis. Used for normalization.
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

def freqSecondMoments(freq_samples, fr, fth):
	"""
	Computes the second moments of both k-vectors.
	"""
	integrand = np.zeros_like(freq_samples,dtype=float)
	for i in range(freq_samples.shape[0]):
		integrand[i,:] = fr[i]**2*np.abs(freq_samples[i,:])**2

	fr_sm = integrate.simps(integrate.simps(integrand, x=fth), x=fr)

	for j in range(freq_samples.shape[1]):
		integrand[:,j] = fth[j]**2*np.abs(freq_samples[:,j])**2

	fth_sm = integrate.simps(integrate.simps(integrand, x=fth), x=fr)

	return fr_sm, fth_sm

def GouyPhase(kr,kth,z_range,z,k):
	"""
	Computes the Gouy phase of the beam.
	"""
	if (z_range[0] > 0.0):
		print("GouyPhase: z must contain 0 for the integral to work.")
		raise

	# Interpolate the frequencies.
	kr_interp  = interpolate.interp1d(z_range,kr)
	kth_interp = interpolate.interp1d(z_range,kth)

	def GouyPhaseIntegrand(z):
		return -1.0/k**(kr_interp(z)+kth_interp(z))

	return integrate.quad(GouyPhaseIntegrand, 0, z)


def f(r,theta,z):
	#return 1.0/np.sqrt(r**2+0.1**2)
    waist_sq = 1.0+(z/2)**2
    return np.exp(-r**2/waist_sq)
	#return 1.0/(r**2+0.1**2)

def Hf(k):
	#return np.exp(-k*np.abs(0.1))/k
	#return np.exp(-0.5*k**2)
	return sp.kn(0,0.1*k)

size_r  = 50
size_th = 50
size_z  = 20

r  = np.linspace(0.0,3.0,size_r)
th = np.linspace(0.0,2*np.pi,size_th)
z  = np.linspace(-4.0,4.0,size_z)
fr_sm_z  = np.zeros((size_z))
fth_sm_z = np.zeros((size_z))

for k in range(size_z):
	samples = np.zeros((size_r,size_th))

	for i in range(size_r):
		for j in range(size_th):
			samples[i,j] = f(r[i],th[j], z[k])

	transform, fr, fth = polar_fft2(samples, deltaR= r[1] - r[0])
	fr_sm_z[k], fth_sm_z[k] = freqSecondMoments(transform, fr, fth)

gouyPhaseGaussian = np.zeros((size_z))
error             = np.zeros((size_z))

for i in range(size_z):
	gouyPhaseGaussian[i],error[i], = GouyPhase(fr_sm_z,fth_sm_z,z,z[i],1.0)

plt.figure()
plt.plot(z,gouyPhaseGaussian)
plt.plot(z,-np.arctan(z/2))
plt.show()
