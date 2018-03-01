# ------------------------------- Information ------------------------------- #
# Author:       Joey Dumont                    <joey.dumont@gmail.com>        #
# Created:      Oct. 27th, 2015                                               #
# Description:  Utility classes to analyze the output of the StrattoCalculator#
#               We provide functions that                                     #
#                   - determine the position of the focal spot,               #
#                   - plot the temporal field at that point,                  #
#                   - determine the area of the focal spot at 1/e^2,          #
#                   - determine the focal volume.                             #
# Dependencies: - NumPy                                                       #
#               - SciPy                                                       #
#               - H5Py                                                        #
#               - Matplotlib                                                  #
# --------------------------------------------------------------------------- #

# --------------------------- Modules Importation --------------------------- #
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.signal as signal
import scipy.integrate as integration
import argparse
import h5py
import time
from mpi4py import MPI
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ---------------------------- Class Definition ----------------------------- #
class Analysis3D:
  """
  We define some utility variables for convenient access to the data.
  We also provide function to compute data of physical interest, such
  as the position of the focal spot (the point of maximum intensity)
  either as a function of time, or as a function of the frequency. We
  also determine the focal spot size at the time of maximum intensity
  or as a function of frequency.
  """

  # -- CONSTANTS
  UNIT_MASS      = 9.109382914e-31
  UNIT_LENGTH    = 3.86159e-13
  UNIT_TIME      = 1.2880885e-21
  SPEED_OF_LIGHT = 299792458
  EPSILON_0      = 8.85418782e-12
  MU_0           = 4*np.pi*1.0e-7
  ALPHA          = 1.0/137.035999074
  UNIT_E_FIELD   = 1.3e18*np.sqrt(4*np.pi*ALPHA)
  UNIT_B_FIELD   = UNIT_E_FIELD/SPEED_OF_LIGHT

  def __init__(self,**kwargs):
    """
    We attach to the HDF5 objects and determine the number of frequency
    components, the number of temporal components and the size of the spatial
    mesh.
    """
    use_mpi = True
    try:
      driver = kwargs['driver']
      comm   = kwargs['comm']
    except KeyError:
      use_mpi= False
      driver = None
      comm   = None

    self.freq_file_loaded = False
    self.time_file_loaded = False

    try:
      self.field_frequency  = h5py.File(kwargs['freq_field'], 'r', driver=driver, comm=comm)
      self.freq_file_loaded = True
    except (IOError,KeyError):
      pass

    try:
      self.field_temporal    = h5py.File(kwargs['time_field'], 'r', driver=driver, comm=comm)
      self.time_field_loaded = True
    except (IOError, KeyError):
      pass

    if not freq_file_loaded and not time_file_loaded:
      raise IOError("At least one file must be loaded.")

    # -- Number of components
    if freq_file_loaded:
      self.size_freq       = len(self.field_frequency['/field'])//6
    if time_file_loaded:
      self.size_time       = len(self.field_temporal['/field'])//6

    # -- Size of mesh
    if freq_file_loaded:
      self.coord_r         = self.field_frequency['/coordinates/r']
      self.coord_theta     = self.field_frequency['/coordinates/theta']
      self.coord_z         = self.field_frequency['/coordinates/z']
      self.size_r          = self.coord_r.size
      self.size_theta      = self.coord_theta.size
      self.size_z          = self.coord_z.size

    if not freq_file_loaded and time_file_loaded:
      self.coord_r         = self.field_temporal['/coordinates/r']
      self.coord_theta     = self.field_temporal['/coordinates/theta']
      self.coord_z         = self.field_temporal['/coordinates/z']
      self.size_r          = self.coord_r.size
      self.size_theta      = self.coord_theta.size
      self.size_z          = self.coord_z.size


    self.dimensions_mesh = np.array([self.size_r, self.size_theta, self.size_z])

    # -- Cartesian meshgrid.
    self.R,  self.Th     = np.meshgrid(self.coord_r[:]*self.UNIT_LENGTH, self.coord_theta[:])
    self.Rz, self.Z      = np.meshgrid(self.coord_r[:]*self.UNIT_LENGTH, self.coord_z[:]*self.UNIT_LENGTH)
    self.X_meshgrid      = self.R*np.cos(self.Th)
    self.Y_meshgrid      = self.R*np.sin(self.Th)
    self.R_meshgrid      = self.Rz
    self.Z_meshgrid      = self.Z

    # -- Sagittal/meridional meshgrids.
    self.r_axial         = np.concatenate([-self.coord_r[:][:0:-1], self.coord_r[:]])
    self.R_axial_meshgrid,self.Z_axial_meshgrid= np.meshgrid(self.r_axial*self.UNIT_LENGTH,self.coord_z[:]*self.UNIT_LENGTH)

    # -- Temporal information
    if freq_file_loaded:
      self.omega           = self.field_frequency['/spectrum/frequency (Hz)']
      self.domega          = self.omega[1]-self.omega[0]

    if time_file_loaded:
      self.time            = self.field_temporal['time']
      self.dt              = self.time[1]-self.time[0]

  def close(self):
    """
    We close the HDF5 files that we have opened.
    """
    if freq_file_loaded:
      self.field_frequency.close()

    if time_file_loaded:
      self.field_temporal.close()

  def GetFrequencyComponent(self,comp,freq):
    """
    Returns the freq-th frequency component of the electromagnetic field.
    """
    amplitude = self.field_frequency['/field/{}-{}/amplitude'.format(comp,freq)]
    phase     = self.field_frequency['/field/{}-{}/phase'.format(comp,freq)]
    return amplitude[:]*np.exp(1j*phase[:])

  def GetTemporalComponent(self,comp,time):
    """
    Returns the time-th temporal component of the electromagnetic field.
    """
    return self.field_temporal['/field/{}-{}'.format(comp,time)]

  def FindMaximumValues(self,emFunc=None):
    """
    This finds the maximum value of a given function of the electromagnetic
    field. If none, it computes the maximum electric energy density in SI units.
    """

    maxIdx     = np.zeros((self.size_time), dtype=int)
    maxIndices = np.zeros((self.size_time), dtype=(int,3))
    maxValue   = np.zeros((self.size_time))

    if (emFunc==None):
      emFunc = self.ElectricEnergyDensity

    for i in range(self.size_time):
      if (i % 100 == 0):
        print("Analyzing temporal component {}/{}".format(i,self.size_time))

      func_value  = emFunc(self.GetTemporalComponent("Er", i),
                           self.GetTemporalComponent("Eth", i),
                           self.GetTemporalComponent("Ez", i),
                           self.GetTemporalComponent("Br", i),
                           self.GetTemporalComponent("Bth", i),
                           self.GetTemporalComponent("Bz", i))

      maxIdx[i]    = np.argmax(func_value)
      maxIndices[i]= np.unravel_index(maxIdx[i],(self.size_r,self.size_theta,self.size_z))
      maxValue[i]  = func_value[maxIndices[i][0],maxIndices[i][1],maxIndices[i][2]]

    return maxIndices, maxValue

  def FindTemporalFocalPlane(self, maxFunc=None, storeFunc=None):
    """
    We determine the position of the focal plane by the plane containing the point
    at maxFunc is highest. We then return two arrays
    containing the temporal evolution of the focal point and the temporal evolution
    of the focal plane for the functional storeFunc.
    """
    if (maxFunc==None):
      maxFunc = self.ElectricEnergyDensity
    if (storeFunc==None):
      storeFunc = self.Ez

    # -- The determine the positions of the maxima as a function of time.
    maxIndices, maxValue = self.FindMaximumValues(maxFunc)

    # -- We find the global maximum as a function of time.
    focalPointMaxIdxTime = np.argmax(maxValue)
    print("Maximum of the functional {} is {}".format(maxFunc.__name__,maxValue[focalPointMaxIdxTime]))

    # -- We build array containing the temporal evolution of the focal point
    # -- and plane.
    focalPointTime = np.zeros((self.size_time))
    focalPlaneTime = np.zeros((self.size_r,self.size_theta,self.size_time))
    for i in range(self.size_time):
      focalPointTime[i] = storeFunc(self.GetTemporalComponent("Er", i),
                                    self.GetTemporalComponent("Eth", i),
                                    self.GetTemporalComponent("Ez", i),
                                    self.GetTemporalComponent("Br", i),
                                    self.GetTemporalComponent("Bth", i),
                                    self.GetTemporalComponent("Bz", i))\
                                    [maxIndices[focalPointMaxIdxTime][0],\
                                     maxIndices[focalPointMaxIdxTime][1],\
                                     maxIndices[focalPointMaxIdxTime][2]]
      focalPlaneTime[:,:,i] = storeFunc(self.GetTemporalComponent("Er", i),
                                        self.GetTemporalComponent("Eth", i),
                                        self.GetTemporalComponent("Ez", i),
                                        self.GetTemporalComponent("Br", i),
                                        self.GetTemporalComponent("Bth", i),
                                        self.GetTemporalComponent("Bz", i))\
                                        [:,:,maxIndices[focalPointMaxIdxTime][2]]

    return maxIndices, maxValue, focalPointMaxIdxTime, focalPointTime, focalPlaneTime

  def ComputeFocalArea(self, planeInformation, threshold):
    """
    This computes the area of the beam in a given temporal plane.
    First, a contour is drawn using the contour(X,Y,Z,V) function.
    The V argument will determine the threshold for the focal area, but we will
    typically use maxValue/e^2. Then, we can use the vertices contained in the
    contour object and apply the Green's theorem to find the area. Specifically,
    ..math::
        \oint_C \left(L dx + Mdy\right) == \iint\left(\frac{\partial M}{\partial x}-\frac{\partial L}{\partial y}\right)dxdy

        with :math:`L=-y` and :math`M=x`.
    """

    def AreaFromContour(vertices):
      """
      Computes an area from the vertices of a contour.
      """
      area = 0
      x0, y0 = vertices[0]
      for [x1,y1] in vs[1:]:
        dx   = x1-x0
        dy   = y1-y0
        area += 0.5*(y0*dx-x0*dy)
        x0 = x1
        y0 = y1
      return area

    # -- Finds the maximum value of the functional.
    maxValue = np.amax(planeInformation)
    level    = [threshold*maxValue]

    # -- Draw the contour and calculate the area.
    cs      = plt.contour(self.X_meshgrid, self.Y_meshgrid,np.transpose(planeInformation), levels=level)
    contour = cs.collections[0]
    vs      = contour.get_paths()[0].vertices
    return AreaFromContour(vs)

  def ComputeBeamWaist(self, planeInformation, threshold):
    """
    This computes the beam waist in a plane of reference. The beam waist
    is defined as the distance from the maxima of the function to where
    it takes max/threshold. Typically, we will use the half-width (radius) at half
    maximum.

    To compute the beam waist on asymmetric profiles, we will compute the waist at
    different values of theta, and average the result.
    """
    # -- Find the maximum value of the functional.
    maxValue = np.amax(planeInformation)
    waist    = np.zeros((self.size_theta))

    # -- We compute the beam waist at each value of theta.
    for i in range(self.size_theta):
      for j in range(self.size_r):
        value = planeInformation[j,i]
        if (value < maxValue*threshold):
          waist[i] = self.coord_r[j]*self.UNIT_LENGTH
          break;

    return np.mean(waist)


  def TemporalDuration(self, temporalFunctional):
    """
    Computes the temporal envelope of the temporalFunctional given. This is
    done by computing the Hilbert transform of the given signal, and then
    measuring the FWHM duration.
    """
    # -- We compute the Hilbert transform.
    hilbertTransform = signal.hilbert(temporalFunctional)

    # -- We compute the squared envelope.
    envelope         = np.abs(hilbertTransform)
    envelopeSq       = np.power(envelope,2)

    # -- We compute the FWHM of the beam.
    maxEnvelope   = np.amax(envelope)
    maxEnvelopeSq = np.amax(envelopeSq)
    for i in range(envelope.size):
      if (envelope[i]>maxEnvelope/2):
        firstIndex = i
        break

    for i in reversed(range(envelope.size)):
      if (envelope[i]>maxEnvelope/2):
        lastIndex = i
        break

    envelopeIdx = self.dt*(lastIndex-firstIndex)

    for i in range(envelope.size):
      if (envelopeSq[i]>maxEnvelopeSq/2):
        firstIndex = i
        break;

    for i in reversed(range(envelope.size)):
      if (envelopeSq[i]>maxEnvelopeSq/2):
        lastIndex = i
        break;

    envelopeSqIdx = self.dt*(lastIndex-firstIndex)
    return envelopeIdx, envelopeSqIdx

  def ComputeTotalEnergyDensityTemporal(self, timeIdx):
    """
    We compute the total electromagnetic energy contained in the
    volume in which we have computed the field.
    """
    integrand = np .zeros((self.size_r, self.size_theta,self.size_z))

    Er           = self.GetTemporalComponent("Er", timeIdx)
    Eth          = self.GetTemporalComponent("Eth", timeIdx)
    Ez           = self.GetTemporalComponent("Ez", timeIdx)
    Br           = self.GetTemporalComponent("Br", timeIdx)
    Bth          = self.GetTemporalComponent("Bth", timeIdx)
    Bz           = self.GetTemporalComponent("Bz", timeIdx)

    integrand = 0.5*(Er[:]**2+Eth[:]**2+Ez[:]**2+Br[:]**2+Bth[:]**2+Bz[:]**2)

    for i in range(integrand.shape[0]):
      integrand[i,:,:] *= self.coord_r[i]

    return integration.simps(integration.simps(integration.simps(integrand, x=self.coord_z[:]), x=self.coord_theta[:]), x=self.coord_r[:])*self.UNIT_MASS*self.SPEED_OF_LIGHT**2


  def LorentzInvariantF(self,Er,Eth,Ez,Br,Bth,Bz):
    """
    Computes the Lorentz invariant 0.5*(|E|^2-|B|^2).
    """
    magneticMagnitude = np.power(Br,2)+np.power(Bth,2)+np.power(Bz,2)
    electricMagnitude = np.power(Er,2)+np.power(Eth,2)+np.power(Ez,2)

    return 0.5*(electricMagnitude-magneticMagnitude)

  def LorentzInvariantF_time(self,timeIdx):
    Er           = self.GetTemporalComponent("Er", timeIdx)[:]
    Eth          = self.GetTemporalComponent("Eth", timeIdx)[:]
    Ez           = self.GetTemporalComponent("Ez", timeIdx)[:]
    Br           = self.GetTemporalComponent("Br", timeIdx)[:]
    Bth          = self.GetTemporalComponent("Bth", timeIdx)[:]
    Bz           = self.GetTemporalComponent("Bz", timeIdx)[:]

    return self.LorentzInvariantF(Er,Eth,Ez,Br,Bth,Bz)


  def LorentzInvariantG(self,Er,Eth,Ez,Br,Bth,Bz):
    """
    Computes the Lorentz invariant (E \cdot B)/c.
    """
    return (Er[:]*Br[:]+Eth[:]*Bth[:]+Ez[:]*Bz[:])

  def LorentzInvariantG_time(self,timeIdx):
    Er           = self.GetTemporalComponent("Er", timeIdx)[:]
    Eth          = self.GetTemporalComponent("Eth", timeIdx)[:]
    Ez           = self.GetTemporalComponent("Ez", timeIdx)[:]
    Br           = self.GetTemporalComponent("Br", timeIdx)[:]
    Bth          = self.GetTemporalComponent("Bth", timeIdx)[:]
    Bz           = self.GetTemporalComponent("Bz", timeIdx)[:]

    return self.LorentzInvariantG(Er,Eth,Ez,Br,Bth,Bz)

  def LorentzInvariantE_time(self,timeIdx):
    """
    Computes the Lorentz invariant sqrt(sqrt(F^2+G^2)+F).
    """
    F = self.LorentzInvariantF_time(timeIdx)
    G = self.LorentzInvariantG_time(timeIdx)

    return np.sqrt(np.sqrt(F**2+G**2)+F)

  def LorentzInvariantH_time(self,timeIdx):
    """
    Computes the Lorentz invariant sqrt(sqrt(F^2+G^2)-F).
    """
    F = self.LorentzInvariantF_time(timeIdx)
    G = self.LorentzInvariantG_time(timeIdx)

    return np.sqrt(np.sqrt(F**2+G**2)-F)

  def ElectricEnergyDensity(self,Er,Eth,Ez,Br,Bth,Bz):
    """
    Computes the electric energy intensity in the temporal domain (in W/cm^2).
    """
    electric_intensity = 0.5*self.SPEED_OF_LIGHT                           \
                            *self.EPSILON_0                                \
                            *np.power(self.UNIT_E_FIELD,2)                 \
                            *(np.power(Er,2)+np.power(Eth,2)+np.power(Ez,2))
    return electric_intensity/1e4

  def MagneticEnergyDensity(self,Er,Eth,Ez,Br,Bth,Bz):
    """
    Computes the magnetic energy intensity in the temporal domain (in W/cm^2).
    """
    magnetic_intensity = 0.5*self.SPEED_OF_LIGHT                           \
                            /self.MU_0                                     \
                            *np.power(self.UNIT_B_FIELD,2)                 \
                            *(np.power(Br,2)+np.power(Bth,2)+np.power(Bz,2))
    return magnetic_intensity/1e4

  def ElectromagneticEnergyDensity(self,Er,Eth,Ez,Br,Bth,Bz):
    """
    Computes the total electromagnetic energy density in the temporal domain.
    """
    em_intensity = self.ElectricEnergyDensity(Er,Eth,Ez,Br,Bth,Bz)+self.MagneticEnergyDensity(Er,Eth,Ez,Br,Bth,Bz)
    return em_intensity

  def Er(self,Er,Eth,Ez,Br,Bth,Bz):
    """
    Returns the Er component of the electric field (in V/m).
    """
    return self.UNIT_E_FIELD*Er[:]

  def Eth(self,Er,Eth,Ez,Br,Bth,Bz):
    """
    Return the Eth component of the electric field (in V/m).
    """
    return self.UNIT_E_FIELD*Eth[:]

  def Ez(self,Er,Eth,Ez,Br,Bth,Bz):
    """
    Return the Ez component of the electric field (in V/m).
    """
    return self.UNIT_E_FIELD*Ez[:]

  def Br(self,Er,Eth,Ez,Br,Bth,Bz):
    """
    Return the Br component of the electric field (in T).
    """
    return self.UNIT_B_FIELD*Br[:]

  def Bth(self,Er,Eth,Ez,Br,Bth,Bz):
    """
    Return the Bth component of the electric field (in T).
    """
    return self.UNIT_B_FIELD*Bth[:]

  def Bz(self,Er,Eth,Ez,Br,Bth,Bz):
    """
    Return the Bz component of the electric field (in T).
    """
    return self.UNIT_B_FIELD*Bz[:]

  def ExCart(self,Er,Eth,Ez,Br,Bth,Bz):
    """
    Returns the component Ex.
    """
    Ex = np.zeros_like(Er)
    for i in range(self.size_theta):
      c = np.cos(self.coord_theta[i])
      s = np.sin(self.coord_theta[i])
      Ex[:,i,:] = c*Er[:,i,:]-s*Eth[:,i,:]

    return np.abs(Ex)

  def EyCart(self,Er,Eth,Ez,Br,Bth,Bz):
    """
    Returns the Ey component.
    """
    Ey = np.zeros_like(Er)
    for i in range(self.size_theta):
      c = np.cos(self.coord_theta[i])
      s = np.sin(self.coord_theta[i])
      Ey[:,i,:] = s*Er[:,i,:]+c*Eth[:,i,:]

    return np.abs(Ey)

  def EzCart(self,Er,Eth,Ez,Br,Bth,Bz):
    return np.abs(Ez)

  def BxCart(self,Er,Eth,Ez,Br,Bth,Bz):
    Bx = np.zeros_like(Er)
    for i in range(self.size_theta):
      c = np.cos(self.coord_theta[i])
      s = np.sin(self.coord_theta[i])
      Bx[:,i,:] = c*Br[:,i,:]-s*Bth[:,i,:]

    return np.abs(Bx)

  def ByCart(self,Er,Eth,Ez,Br,Bth,Bz):
    """
    Returns the Ey component.
    """
    By = np.zeros_like(Er)
    for i in range(self.size_theta):
      c = np.cos(self.coord_theta[i])
      s = np.sin(self.coord_theta[i])
      By[:,i,:] = s*Br[:,i,:]+c*Bth[:,i,:]

    return np.abs(By)

  def BzCart(self,Er,Eth,Ez,Br,Bth,Bz):
    return np.abs(Bz)

  def GetFocalPlaneInTimeCartesian(self,z_idx):
    """
    Returns the Cartesian components of the electromagnetic field in a given
    z plane, usually the focal lane, as a function of time.
    """
    ExFocalPlaneTime = np.zeros((self.size_r,self.size_theta,self.size_time))
    EyFocalPlaneTime = np.zeros((self.size_r,self.size_theta,self.size_time))
    EzFocalPlaneTime = np.zeros((self.size_r,self.size_theta,self.size_time))
    BxFocalPlaneTime = np.zeros((self.size_r,self.size_theta,self.size_time))
    ByFocalPlaneTime = np.zeros((self.size_r,self.size_theta,self.size_time))
    BzFocalPlaneTime = np.zeros((self.size_r,self.size_theta,self.size_time))

    for i in range(self.size_time):
      # -- We get the cylindrical components first.
      Er  = self.GetTemporalComponent("Er",  i)[:,:,z_idx]
      Eth = self.GetTemporalComponent("Eth", i)[:,:,z_idx]
      Ez  = self.GetTemporalComponent("Ez", i)[:,:,z_idx]
      Br  = self.GetTemporalComponent("Br",  i)[:,:,z_idx]
      Bth = self.GetTemporalComponent("Bth", i)[:,:,z_idx]
      Bz  = self.GetTemporalComponent("Bz", i)[:,:,z_idx]

      for j in range(Er.shape[1]):
        c = np.cos(self.coord_theta[j])
        s = np.sin(self.coord_theta[j])

        for k in range(Er.shape[0]):
          ExFocalPlaneTime[k,j,i] = c*Er[k,j]-s*Eth[k,j]
          EyFocalPlaneTime[k,j,i] = s*Er[k,j]+c*Eth[k,j]
          BxFocalPlaneTime[k,j,i] = c*Br[k,j]-s*Bth[k,j]
          ByFocalPlaneTime[k,j,i] = s*Br[k,j]+c*Bth[k,j]

      EzFocalPlaneTime[:,:,i] = Er[:,:]
      BzFocalPlaneTime[:,:,i] = Bz[:,:]

    return ExFocalPlaneTime, EyFocalPlaneTime, EzFocalPlaneTime, BxFocalPlaneTime, ByFocalPlaneTime, BzFocalPlaneTime

  def GetSagittalPlaneInTimeCartesian(self):
    """
    Return the Cartesian components of the electromagnetic field in a given
    x-axis plane, known as the sagittal plane, as a function of time.
    """
    # Test to figure out the proper size of the array.
    ExSagittalPlane = np.concatenate([-testAnalysis.GetTemporalComponent("Er",  0)[:,self.size_theta//2,:][::-1,:], self.GetTemporalComponent("Er",  0)[1:,0,:]])

    # Set the proper array sizes.
    ExSagittalPlane = np.zeros((ExSagittalPlane.shape[0],ExSagittalPlane.shape[1],self.size_time))
    EySagittalPlane = np.zeros((ExSagittalPlane.shape[0],ExSagittalPlane.shape[1],self.size_time))
    EzSagittalPlane = np.zeros((ExSagittalPlane.shape[0],ExSagittalPlane.shape[1],self.size_time))
    BxSagittalPlane = np.zeros((ExSagittalPlane.shape[0],ExSagittalPlane.shape[1],self.size_time))
    BySagittalPlane = np.zeros((ExSagittalPlane.shape[0],ExSagittalPlane.shape[1],self.size_time))
    BzSagittalPlane = np.zeros((ExSagittalPlane.shape[0],ExSagittalPlane.shape[1],self.size_time))

    for i in range(self.size_time):
      ExSagittalPlane[:,:,i] = np.concatenate([-self.GetTemporalComponent("Er",  i)[:,self.size_theta//2,:][::-1,:], self.GetTemporalComponent("Er",  i)[1:,0,:]])
      EySagittalPlane[:,:,i] = np.concatenate([-self.GetTemporalComponent("Eth", i)[:,self.size_theta//2,:][::-1,:], self.GetTemporalComponent("Eth", i)[1:,0,:]])
      EzSagittalPlane[:,:,i] = np.concatenate([ self.GetTemporalComponent("Ez",  i)[:,self.size_theta//2,:][::-1,:], self.GetTemporalComponent("Ez",  i)[1:,0,:]])
      BxSagittalPlane[:,:,i] = np.concatenate([-self.GetTemporalComponent("Br",  i)[:,self.size_theta//2,:][::-1,:], self.GetTemporalComponent("Br",  i)[1:,0,:]])
      BySagittalPlane[:,:,i] = np.concatenate([-self.GetTemporalComponent("Bth", i)[:,self.size_theta//2,:][::-1,:], self.GetTemporalComponent("Bth", i)[1:,0,:]])
      BzSagittalPlane[:,:,i] = np.concatenate([ self.GetTemporalComponent("Bz",   i)[:,self.size_theta//2,:][::-1,:], self.GetTemporalComponent("Bz",  i)[1:,0,:]])

    return ExSagittalPlane, EySagittalPlane, EzSagittalPlane, BxSagittalPlane,  BySagittalPlane, BzSagittalPlane

  def GetMeridionalPlaneInTimeCartesian(self):
    """
    Returns the Cartesian components of the electromagnetic field in a given
    y plane, known as the meriodional plane, as a function of time.
    """
    # Test to figure out the array size.
    ExMeridionalPlane = np.concatenate([ self.GetTemporalComponent("Eth", 0)[:,3*self.size_theta//4,:][::-1,:], -self.GetTemporalComponent("Eth", 0)[1:,self.size_theta//4,:]])

    # Set the proper array sizes.
    ExMeridionalPlane = np.zeros((ExMeridionalPlane.shape[0], ExMeridionalPlane.shape[1],self.size_time))
    EyMeridionalPlane = np.zeros((ExMeridionalPlane.shape[0], ExMeridionalPlane.shape[1],self.size_time))
    EzMeridionalPlane = np.zeros((ExMeridionalPlane.shape[0], ExMeridionalPlane.shape[1],self.size_time))
    BxMeridionalPlane = np.zeros((ExMeridionalPlane.shape[0], ExMeridionalPlane.shape[1],self.size_time))
    ByMeridionalPlane = np.zeros((ExMeridionalPlane.shape[0], ExMeridionalPlane.shape[1],self.size_time))
    BzMeridionalPlane = np.zeros((ExMeridionalPlane.shape[0], ExMeridionalPlane.shape[1],self.size_time))

    for i in range(self.size_time):
      ExMeridionalPlane[:,:,i] = np.concatenate([ self.GetTemporalComponent("Eth", i)[:,3*self.size_theta//4,:][::-1,:], -self.GetTemporalComponent("Eth", i)[1:,self.size_theta//4,:]])
      EyMeridionalPlane[:,:,i] = np.concatenate([-self.GetTemporalComponent("Er",  i)[:,3*self.size_theta//4,:][::-1,:],  self.GetTemporalComponent("Er",  i)[1:,self.size_theta//4,:]])
      EzMeridionalPlane[:,:,i] = np.concatenate([ self.GetTemporalComponent("Ez",  i)[:,3*self.size_theta//4,:][::-1,:],  self.GetTemporalComponent("Ez",  i)[1:,self.size_theta//4,:]])
      BxMeridionalPlane[:,:,i] = np.concatenate([ self.GetTemporalComponent("Bth", i)[:,3*self.size_theta//4,:][::-1,:], -self.GetTemporalComponent("Bth", i)[1:,self.size_theta//4,:]])
      ByMeridionalPlane[:,:,i] = np.concatenate([-self.GetTemporalComponent("Br",  i)[:,3*self.size_theta//4,:][::-1,:],  self.GetTemporalComponent("Br",  i)[1:,self.size_theta//4,:]])
      BzMeridionalPlane[:,:,i] = np.concatenate([ self.GetTemporalComponent("Bz",  i)[:,3*self.size_theta//4,:][::-1,:],  self.GetTemporalComponent("Bz",  i)[1:,self.size_theta//4,:]])

    return ExMeridionalPlane, EyMeridionalPlane, EzMeridionalPlane, BxMeridionalPlane, ByMeridionalPlane, BzMeridionalPlane

class AnalysisRadial:
  """
  We define some utility variables for convenient access to the data.
  We also provide function to compute data of physical interest, such
  as the position of the focal spot (the point of maximum intensity)
  either as a function of time, or as a function of the frequency. We
  also determine the focal spot size at the time of maximum intensity
  or as a function of frequency.
  """
  UNIT_MASS      = 9.109382914e-31
  UNIT_LENGTH    = 3.86159e-13
  SPEED_OF_LIGHT = 299792458
  EPSILON_0      = 8.85418782e-12
  MU_0           = 4*np.pi*1.0e-7
  ALPHA          = 1.0/137.035999074
  UNIT_E_FIELD   = 1.3e18*np.sqrt(4*np.pi*ALPHA)
  UNIT_B_FIELD   = UNIT_E_FIELD/SPEED_OF_LIGHT

  def __init__(self, **kwargs):
    """
    We attach the HDF5 objects and determine the number of frequency
    components, the number of temporal components and the size of the
    spatial mesh.
    """

    use_mpi = True
    try:
      driver = kwargs['driver']
      comm   = kwargs['comm']
    except KeyError:
      use_mpi= False
      driver = None
      comm   = None

    self.freq_file_loaded = False
    self.time_file_loaded = False

    try:
      self.field_frequency  = h5py.File(kwargs['freq_field'], 'r', driver=driver, comm=comm)
      self.freq_file_loaded = True
    except (IOError,KeyError):
      pass

    try:
      self.field_temporal    = h5py.File(kwargs['time_field'], 'r', driver=driver, comm=comm)
      self.time_field_loaded = True
    except (IOError, KeyError):
      pass

    if not freq_file_loaded and not time_file_loaded:
      raise IOError("At least one file must be loaded.")

    # -- Number of components
    if self.freq_file_loaded:
      self.size_freq       = self.field_frequency['/spectrum'].attrs.get("num_spectral_components")[0]

    if self.time_file_loaded:
      self.size_time       = len(self.field_temporal['/time'])

    # -- Size of mesh
    if self.freq_file_loaded:
      self.coord_r         = self.field_frequency['/coordinates/r']
      self.coord_z         = self.field_frequency['/coordinates/z']
      self.size_r          = self.coord_r.size
      self.size_z          = self.coord_z.size

    if self.time_file_loaded:
      self.coord_r         = self.field_temporal['/coordinates/r']
      self.coord_z         = self.field_temporal['/coordinates/z']
      self.size_r          = self.coord_r.size
      self.size_z          = self.coord_z.size

    self.dimensions_mesh = np.array([self.size_r,  self.size_z])

    # -- Temporal information
    if self.freq_file_loaded:
      self.omega           = self.field_frequency['/spectrum/frequency (Hz)']
      self.domega          = self.omega[1]-self.omega[0]

    if self.time_file_loaded:
      self.time            = self.field_temporal['time']
      self.dt              = self.time[1]-self.time[0]

  def GetFrequencyComponent(self,comp,freq):
    """
    Returns the freq-th frequency component of the electromagnetic field.
    """
    return self.field_frequency['/field/{}-{}'.format(comp,freq)]

  def GetTemporalComponent(self,comp,time):
    """
    Returns the time-th temporal component of the electromagnetic field.
    """
    return self.field_temporal['/field/{}-{}'.format(comp,time)]

  def FindMaximumValues(self,emFunc=None):
    """
    This finds the maximum value of a given function of the electromagnetic
    field. If none, it computes the maximum electric energy density in SI units.
    """

    maxIdx     = np.zeros((self.size_time), dtype=int)
    maxIndices = np.zeros((self.size_time), dtype=(int,2))
    maxValue   = np.zeros((self.size_time))

    if (emFunc==None):
      emFunc = self.ElectricEnergyDensity

    for i in range(self.size_time):
      if (i % 100 == 0):
        print("Analyzing temporal component {}/{}".format(i,self.size_time))

      func_value  = emFunc(self.GetTemporalComponent("Er", i),
                           self.GetTemporalComponent("Ez", i),
                           self.GetTemporalComponent("Bth", i))

      maxIdx[i]    = np.argmax(func_value)
      maxIndices[i]= np.unravel_index(maxIdx[i],(self.size_r,self.size_z))
      maxValue[i]  = func_value[maxIndices[i][0],maxIndices[i][1]]

    return maxIndices, maxValue

  def FindTemporalFocalPlane(self, maxFunc=None, storeFunc=None):
    """
    We determine the position of the focal plane by the plane containing the point
    at maxFunc is highest. We then return two arrays
    containing the temporal evolution of the focal point and the temporal evolution
    of the focal plane for the functional storeFunc.
    """
    if (maxFunc==None):
      maxFunc = self.ElectricEnergyDensity
    if (storeFunc==None):
      storeFunc = self.Ez

    # -- The determine the positions of the maxima as a function of time.
    maxIndices, maxValue = self.FindMaximumValues(maxFunc)

    # -- We find the global maximum as a function of time.
    focalPointMaxIdxTime = np.argmax(maxValue)
    print("Maximum of the functional {} is {}".format(maxFunc.__name__,maxValue[focalPointMaxIdxTime]))

    # -- We build array containing the temporal evolution of the focal point
    # -- and plane.
    focalPointTime = np.zeros((self.size_time))
    focalPlaneTime = np.zeros((self.size_r,self.size_time))
    for i in range(self.size_time):
      focalPointTime[i] = storeFunc(self.GetTemporalComponent("Er", i),
                                    self.GetTemporalComponent("Ez", i),
                                    self.GetTemporalComponent("Bth", i)) \
                                    [maxIndices[focalPointMaxIdxTime][0],\
                                     maxIndices[focalPointMaxIdxTime][1]]
      focalPlaneTime[:,i] = storeFunc(self.GetTemporalComponent("Er", i),
                                      self.GetTemporalComponent("Ez", i),
                                      self.GetTemporalComponent("Bth", i)) \
                                      [:,maxIndices[focalPointMaxIdxTime][1]]

    return maxIndices, maxValue, focalPointMaxIdxTime, focalPointTime, focalPlaneTime

  def ComputeFocalArea(self, radialInfo, threshold):
    """
    This computes the focal area of a given functional at a given time step.
    We will find the radial position at which the functional is maximum,
    and find its radial extension. We will then compute the area with pi*r^2.
    """
    # -- Find the maximum value of the functional.
    maxIndex = np.argmax(np.abs(radialInfo))
    maxValue = radialInfo[maxIndex]

    # -- Find the radial extension of the maximum.
    for i in range(maxIndex+1,self.size_r):
      if (np.abs(radialInfo[i]) < maxValue*threshold):
        indexPlus = i
        break;

    indexMinus = -1
    for i in reversed(range(0,maxIndex)):
      if (np.abs(radialInfo[i]) < maxValue*threshold):
        indexMinus = i
        break;

    # -- Compute the radial extension.
    width = self.coord_r[indexPlus]-self.coord_r[indexMinus] if (indexMinus != -1) else 2*self.coord_r[indexPlus]
    width *= self.UNIT_LENGTH

    return width/2, np.pi*width**2/4

  def TemporalDuration(self, temporalFunctional):
    """
    Computes the temporal envelope of the temporalFunctional given. This is
    done by computing the Hilbert transform of the given signal, and then
    measuring the FWHM duration.
    """
        # -- We compute the Hilbert transform.
    hilbertTransform = signal.hilbert(temporalFunctional)

    # -- We compute the squared envelope.
    envelope         = np.abs(hilbertTransform)
    envelopeSq       = np.power(envelope,2)

    # -- We compute the FWHM of the beam.
    maxEnvelope   = np.amax(envelope)
    maxEnvelopeSq = np.amax(envelopeSq)
    for i in range(envelope.size):
      if (envelope[i]>maxEnvelope/2):
        firstIndex = i
        break

    for i in reversed(range(envelope.size)):
      if (envelope[i]>maxEnvelope/2):
        lastIndex = i
        break

    envelopeIdx = self.dt*(lastIndex-firstIndex)

    for i in range(envelope.size):
      if (envelopeSq[i]>maxEnvelopeSq/2):
        firstIndex = i
        break;

    for i in reversed(range(envelope.size)):
      if (envelopeSq[i]>maxEnvelopeSq/2):
        lastIndex = i
        break;

    envelopeSqIdx = self.dt*(lastIndex-firstIndex)
    return envelopeIdx, envelopeSqIdx

  def ComputeTotalEnergyDensityTemporal(self, timeIdx):
    """
    We compute the total energy of the system by integrating over the volume of the
    focus. Since we suppose that all the energy is contained in this volume, the final
    value should not depend on the timeIdx, as long as the whole field is present there.
    """
    # -- We prepare the array.
    integrand  = np.zeros((self.size_r,self.size_z))

    Er         = self.GetTemporalComponent("Er", timeIdx)
    Ez         = self.GetTemporalComponent("Ez", timeIdx)
    Bth        = self.GetTemporalComponent("Bth",timeIdx)

    #integrand  = 0.5*(EPSILON_0*UNIT_E_FIELD**2*(Er[:]**2+Ez[:]**2)+MU_0**(-1)*UNIT_B_FIELD**2*Bth[:]**2)
    integrand   = 0.5*(Er[:]**2+Ez[:]**2+Bth[:]**2)
    for i in range(integrand.shape[0]):
      integrand[i,:] *= self.coord_r[i]

    return 2.0*np.pi*integration.simps(integration.simps(integrand,x=self.coord_z[:]),x=self.coord_r[:])*self.UNIT_MASS*self.SPEED_OF_LIGHT**2

  def LorentzInvariantF(self,Er,Ez,Bth):
    """
    Computes the Lorentz invariant 0.5*(|E|^2-|B|^2/c^2).
    """
    magneticMagnitude = np.power(Bth,2)
    electricMagnitude = np.power(Er,2)+np.power(Ez,2)

    return 0.5*(electricMagnitude-magneticMagnitude)

  def LorentzInvariantG(self,Er,Ez,Bth):
    """
    Computes the Lorentz invariant (E \cdot B)/c.
    """
    return 0


  def ElectricEnergyDensity(self,Er,Ez,Bth):
    """
    Computes the electric energy intensity in the temporal domain (in W/cm^2).
    """
    electric_intensity = 0.5*self.SPEED_OF_LIGHT                           \
                            *self.EPSILON_0                                \
                            *np.power(self.UNIT_E_FIELD,2)                 \
                            *(np.power(Er,2)+np.power(Ez,2))
    return electric_intensity/1e4

  def MagneticEnergyDensity(self,Er,Ez,Bth):
    """
    Computes the magnetic energy intensity in the temporal domain (in W/cm^2).
    """
    magnetic_intensity = 0.5*self.SPEED_OF_LIGHT                           \
                            /self.MU_0                                     \
                            *np.power(self.UNIT_B_FIELD,2)                 \
                            *(np.power(Bth,2))
    return magnetic_intensity/1e4

  def ElectromagneticEnergyDensity(self,Er,Ez,Bth):
    """
    Computes the total electromagnetic energy density in the temporal domain.
    """
    em_intensity = ElectricEnergyDensity(Er,Ez,Bth)+MagneticEnergyDensity(Er,Ez,Bth)
    return em_intensity

  def Er(self,Er,Ez,Bth):
    """
    Returns the Er component of the electric field (in V/m).
    """
    return self.UNIT_E_FIELD*Er[:]

  def Ez(self,Er,Ez,Bth):
    """
    Return the Ez component of the electric field (in V/m).
    """
    return self.UNIT_E_FIELD*Ez[:]

  def Bth(self,Er,Ez,Bth):
    """
    Return the Bth component of the electric field (in T).
    """
    return self.UNIT_B_FIELD*Bth[:]