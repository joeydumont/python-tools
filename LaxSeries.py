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
import matplotlib
matplotlib.use('pgf')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.special as sp
import scipy.interpolate as interpolate
import scipy.integrate as integrate
import scipy.constants as cst
import itertools
import unittest
import sympy
import vphys

pgf_with_pdflatex = vphys.default_pgf_configuration()
matplotlib.rcParams.update(pgf_with_pdflatex)

# -------------------------------- Functions -------------------------------- #
def user_mod(value, modulo):
  return value-np.abs(modulo)*np.floor(value/np.abs(modulo))

def ExpansionCoefficient(m,p):
  """
  Computes the expansion coefficient of non-paraxial terms of the
  Lax series. Its mathematical form is (Opt. Lett 28(10), 2003):
   c_p^(m) = (2m)!/[m(p-1)!(m-1)!(m+p)!].
  We compute it by first taking the logarithms, expanding the terms, and then
  computing the exponential of that.
  """
  firstNumerator   = sp.gammaln(2*m+1)
  firstDenominator = np.log(m)
  secondDenominator= sp.gammaln(p)
  thirdDenominator = sp.gammaln(m)
  fourthDenominator= sp.gammaln(m+p+1)

  result = np.exp(firstNumerator-(firstDenominator+secondDenominator+ \
                  thirdDenominator+fourthDenominator))

  return result;

def ExpansionCoefficientDirect(m,p):
  """
  Direct computation of c_p^(m) to make sure that the results agree, at least
  for small m and p.
  """
  firstNumerator   = sp.factorial(2*m,exact=True)
  firstDenominator = m
  secondDenominator= sp.factorial(p-1,exact=True)
  thirdDenominator = sp.factorial(m-1,exact=True)
  fourthDenominator= sp.factorial(m+p,exact=True)

  result = firstNumerator/(firstDenominator*secondDenominator*thirdDenominator*fourthDenominator)

  return result

# ------------------------------ SymPy Functions ---------------------------- #
def ExLax(X,Y,z,k,w_0,m_max):
  """
  Takes the series of Opt. Lett. 28(10), 2003 to compute the non-paraxial
  corrections to Ex up to order m.
  """

  # -- Symbol/initial function definition.
  x_sym, y_sym,z_sym,k_sym, w_0_sym = sympy.symbols('x_sym y_sym z_sym k_sym w_0_sym')
  z_r_sym     = k_sym*w_0_sym**2/2
  w_z_sym     = w_0_sym*sympy.sqrt(1+(z_sym/z_r_sym)**2)
  R_sym       = z_sym/(z_sym**2+z_r_sym**2)
  phi_0       = w_0_sym/w_z_sym*sympy.exp(-(x_sym**2+y_sym**2)/w_z_sym**2)    \
                *sympy.exp(-1j*k_sym*z_sym)                                   \
                *sympy.exp(1j*sympy.atan(z_sym/z_r_sym))
#                *sympy.exp(-1j*k_sym*(x_sym**2+y_sym**2)*R_sym/2)             \

  # -- Symbolic expressions for summation phi/psi = sum_m phi^(2m)/psi^(2m+1)
  phi         = phi_0
  psi         = 1j/k*sympy.diff(phi_0,x_sym)
  weginer_phi = sympy.S.Zero
  weginer_psi = sympy.S.Zero
  phi_aj      = [sympy.S.Zero]
  phi_sj      = [sympy.S.Zero]
  psi_aj      = [sympy.S.Zero]
  psi_sj      = [sympy.S.Zero]

  # -- Symbolic expressions for specific values of m, to be used in the loop.
  phi_2m      = phi
  psi_2mp1    = psi

  # -- We compute the derivatives analytically.
  derivatives = []
  for i in range(1,2*m_max+1):
    derivatives.append(sympy.diff(phi_0,z_sym,i))

  # -- We evaluate the higher-order terms.
  for m in range(1,m_max+1):
    phi_2m       = sympy.S.Zero

    for p in range(1,m+1):

      # -- We evaluate the product between the z factor and the derivative,
      # -- and add it to the symbolic expression.
      polynomial = z_sym**p*derivatives[m+p-2]
      phi_2m    += ExpansionCoefficient(m,p)*polynomial

    phi_2m   *= (1j/(2*k))**m
    psi_2mp1  = 1j/k*(sympy.diff(phi_2m,x_sym)+sympy.diff(psi_2mp1,z_sym))
    phi      += phi_2m
    psi      += psi_2mp1

    phi_aj.append()

    # -- Weginer transformation.
    numerator = sympy.S.Zero
    denominator = sympy.S.Zero

    for j in range(m+1):
      s_j = sympy.S.Zero
      for i in range(j+1):
        s_j += 
      numerator += (-1)**j*sympy.binomial(m,j)*sympy.rf(1+j,m-1)

  # -- We evaluate the magnetic field.
  Bx_sym = sympy.diff(psi, y_sym)/(1j*k)
  By_sym = (sympy.diff(phi, z_sym)-sympy.diff(psi,x_sym))/(1j*k)
  Bz_sym = -sympy.diff(phi, y_sym)/(1j*k)

  # -- We lambdify the expressions and evaluate them.
  Ex = sympy.lambdify((x_sym,y_sym,z_sym,k_sym,w_0_sym), phi)
  Ez = sympy.lambdify((x_sym,y_sym,z_sym,k_sym,w_0_sym), psi)
  Bx = sympy.lambdify((x_sym,y_sym,z_sym,k_sym,w_0_sym), Bx_sym)
  By = sympy.lambdify((x_sym,y_sym,z_sym,k_sym,w_0_sym), By_sym)
  Bz = sympy.lambdify((x_sym,y_sym,z_sym,k_sym,w_0_sym), Bz_sym)

  return Ex(X,Y,z,k,w_0), np.zeros_like(Ex(X,Y,z,k,w_0)), Ez(X,Y,z,k,w_0), Bx(X,Y,z,k,w_0), By(X,Y,z,k,w_0), Bz(X,Y,z,k,w_0)

# ----------------------------- Salamin's models ---------------------------- #
def ApplPhysB(x,y,z,k,w_0):
  """
  Implements the model in Appl. Phys. B 86, 319--326 (2007).
  """
  xi     = x/w_0
  nu     = y/w_0
  zr     = k*w_0**2/2
  zeta   = z/zr
  rho_sq = xi**2+nu**2
  f      = 1j/(zeta+1j)
  eps    = w_0/zr
  eta    = -k*z
  prefac = k*f*np.exp(-f*rho_sq)*np.exp(1j*eta)

  secondOrder = eps**2*((f*xi)**2-f**3*rho_sq**2/4)
  fourthOrder = eps**4*(f**2/8-f**3*rho_sq/4+f**4**(xi**2*rho_sq-rho_sq**2/1)\
                        +f**5*(-(xi*rho_sq)**2/4-rho_sq**3/8)+f**6*rho_sq**4/32)
  Ex          = -1j*prefac*(1+secondOrder+fourthOrder)

  secondOrder = (eps*f)**2*xi*nu
  fourthOrder = eps**4*(f**4*rho_sq-f**5*rho_sq**2/4)*xi*nu
  Ey          = -1j*prefac*(secondOrder+fourthOrder)

  firstOrder  = eps*f*xi
  thirdOrder  = eps**3*(-f**2/2+f**3*rho_sq-f**4*rho_sq**2/4)*xi
  Ez          = prefac*(firstOrder+thirdOrder)

  Bx          = np.zeros_like(Ez)

  secondOrder = eps**2*(f**2*rho_sq/2-f**3*rho_sq**2/4)
  fourthOrder = eps**4*(-f**2/8+f**3*rho_sq/4+5*f**4*rho_sq**2/16-f**5*rho_sq**3/4+f**6*rho_sq**4/32)
  By          = -1j*prefac*(1+secondOrder+fourthOrder)

  firstOrder  = eps*f*nu
  thirdOrder  = eps**3*(f**2/2+f**3*rho_sq/2-f**4*rho_sq**2/4)*nu
  Bz          = prefac*(firstOrder+thirdOrder)

  return Ex, Ey, Ez, Bx, By, Bz


def CSPSW(x,y,z,k,z_r):
  """
  Computes expressions (15-19) in Opt. Lett. 34(5), 2009.
  """
  r_sq = x**2+y**2
  Rc   = np.sqrt(r_sq+(z+1j*z_r)**2)

  Ex = -1j*k*np.exp(-1j*k*Rc)/Rc+1j/k*np.exp(-1j*k*Rc)*(1j*k/Rc**2+(1+(k*x)**2)/Rc**3-3*1j*k*x**2/Rc**4-3*x**2/Rc**5)
  Ey = -1j/k*x*y*np.exp(-1j*k*Rc)*(k**2/Rc**3-3j*k/Rc**4-3/Rc**5)
  Ez = 1j/k*x*(z+1j*z_r)*np.exp(-1j*k*Rc)*(k**2/Rc**3-3*k/Rc**4-3/Rc**5)
  Bx = np.zeros_like(Ex)
  By = (z+1j*z_r)*np.exp(-1j*k*Rc)*(1j*k/Rc**2+1/Rc**3)
  Bz = y*np.exp(-1j*k*Rc)*(1j*k/Rc**2+1/Rc**3)

  return Ex, Ey, Ez, Bx, By, Bz

# ---------------------------- Plotting Functions --------------------------- #
def DivideColorbar(ax,im):
  divider = make_axes_locatable(ax)
  cax     = divider.append_axes("right", size="5%", pad=0.1)
  cbar    = plt.colorbar(im, cax=cax)
  cbar.formatter.set_powerlimits((0,0))
  cbar.update_ticks()

def PlotAllComponents(filename,X,Y,Ex,Ey,Ez,Bx,By,Bz,levels,plot_options=None,contour_options=None):
  fig = plt.figure(figsize=(7,4))
  fig.subplots_adjust(hspace=0.3,wspace=0.5)

  ax  = plt.subplot2grid((2,3), (0,0))
  im  = plt.pcolormesh(X*1e6,Y*1e6,Ex, **plot_options)
  plt.contour(X*1e6,Y*1e6,Ex,levels, **contour_options)
  ax.set_aspect('equal')
  ax.set_ylabel(r"$y$ [\si{\micro\metre}]")
  ax.set_title(r"$E_x$")

  DivideColorbar(ax,im)

  ax  = plt.subplot2grid((2,3), (0,1))
  im  = plt.pcolormesh(X*1e6,Y*1e6,Ey, **plot_options)
  plt.contour(X*1e6,Y*1e6,Ey, levels, **contour_options)
  ax.set_aspect('equal')
  ax.set_title(r"$E_y$")

  DivideColorbar(ax,im)

  ax  = plt.subplot2grid((2,3),(0,2))
  im  = plt.pcolormesh(X*1e6,Y*1e6,Ez, **plot_options)
  plt.contour(X*1e6,Y*1e6,Ez, levels, **contour_options)
  ax.set_aspect('equal')
  ax.set_title(r"$E_z$")

  DivideColorbar(ax,im)

  ax  = plt.subplot2grid((2,3),(1,0))
  im  = plt.pcolormesh(X*1e6,Y*1e6,Bx, **plot_options)
  plt.contour(X*1e6,Y*1e6,Bx, levels, **contour_options)
  ax.set_aspect('equal')
  ax.set_ylabel(r"$y$ [\si{\micro\metre}]")
  ax.set_xlabel(r"$x$ [\si{\micro\metre}]")
  ax.set_title(r"$B_x$")

  DivideColorbar(ax,im)

  ax  = plt.subplot2grid((2,3),(1,1))
  im  = plt.pcolormesh(X*1e6,Y*1e6,By, **plot_options)
  plt.contour(X*1e6,Y*1e6,By, levels, **contour_options)
  ax.set_aspect('equal')
  ax.set_xlabel(r"$x$ [\si{\micro\metre}]")
  ax.set_title(r"$B_y$")

  DivideColorbar(ax,im)

  ax  = plt.subplot2grid((2,3),(1,2))
  im  = plt.pcolormesh(X*1e6,Y*1e6,Bz, **plot_options)
  plt.contour(X*1e6,Y*1e6,Bz, levels, **contour_options)
  ax.set_aspect('equal')
  ax.set_xlabel(r"$x$ [\si{\micro\metre}]")
  ax.set_title(r"$B_z$")

  DivideColorbar(ax,im)

  plt.savefig(filename, bbox_inches='tight', dpi=500)

# ------------------------------- Unit Testing ------------------------------ #
class TestExpansionCoefficients(unittest.TestCase):

  def test_both_functions(self):
    """
    Test that both implementations agree for small m and p.
    """
    m_l = [i for i in range(1,15)]
    p_l = [i for i in range(1,15)]

    for iter in itertools.filterfalse(lambda x : x[0] < x[1], itertools.product(m_l,p_l)):
      a = ExpansionCoefficient(*iter)
      b = ExpansionCoefficientDirect(*iter)
      print(*iter, a)
      self.assertAlmostEqual(a,b)

# ------------------------------ MAIN FUNCTION ------------------------------ #
if __name__ == "__main__":

  # -- Plot options.
  plot_options = {"cmap": "jet", "rasterized": True}
  levels=2
  contour_options = {"linestyles": '--', "linewidths": 0.5}

  # -- Substitute numerical values for fixed parameters in phi_0.
  lamb  = 800e-9
  k     = 2*np.pi/lamb
  w0    = 2.5*lamb

  z_r   = k*w0**2/2
  x_f   = np.linspace(-2.5e-6,2.5e-6,100)
  y_f   = np.linspace(-2.5e-6,2.5e-6,100)
  X, Y  = np.meshgrid(x_f,y_f)
  z     = 0

  # -- Compute the terms of the series.
  Ex_ref, Ey_ref, Ez_ref, Bx_ref, By_ref, Bz_ref = ExLax(X,Y,z,k,w0,0)
  for i in range(0,5):
    Ex, Ey, Ez, Bx, By, Bz = ExLax(X,Y,z,k,w0,i)

    Ex /= np.amax(np.abs(Ex_ref))
    Ey /= np.amax(np.abs(Ex_ref))
    Ez /= np.amax(np.abs(Ex_ref))
    Bx /= np.amax(np.abs(Ex_ref))
    By /= np.amax(np.abs(Ex_ref))
    Bz /= np.amax(np.abs(Ex_ref))

    PlotAllComponents("LaxSeries-{}.pdf".format(i),X,Y,np.abs(Ex)**2,np.abs(Ey)**2,np.abs(Ez)**2,np.abs(Bx)**2,np.abs(By)**2,np.abs(Bz)**2,levels,plot_options,contour_options)

  # -- Gouy shift of ExLax for different values of m.
  phases = []
  z_r = k*w0**2/2
  z = np.linspace(-10*z_r,10*z_r,401)
  for i in range(5):
    Ex,Ey, Ez, Bx,By,Bz = ExLax(0,0,z,k,w0,i)
    Ex_phase = np.angle(Ex*np.exp(1j*k*z))
    phases.append(Ex_phase)

  GaussianPhase = np.arctan(2*z/(k*w0**2))

  plt.figure()
  plt.plot(z*1e6,GaussianPhase)

  for i in range(5):
    plt.plot(z*1e6,phases[i])

  plt.savefig("GouyPhase-LaxSeries.pdf", bbox_inches='tight')

  # -- Salamin models.
  z = 0
  Ex, Ey, Ez, Bx, By, Bz = np.abs(ApplPhysB(X,Y,z,k,w0))

  Ex_ref = np.copy(Ex)
  Ex /= np.amax(np.abs(Ex_ref))
  Ey /= np.amax(np.abs(Ex_ref))
  Ez /= np.amax(np.abs(Ex_ref))
  Bx /= np.amax(np.abs(Ex_ref))
  By /= np.amax(np.abs(Ex_ref))
  Bz /= np.amax(np.abs(Ex_ref))

  PlotAllComponents("Salamin.pdf",X,Y,Ex**2,Ey**2,Ez**2,Bx**2,By**2,Bz**2,levels,plot_options,contour_options)

  # -- Gouy shift of ApplPhysB.
  phases = []
  z_r = k*w0**2/2
  z = np.linspace(-10*z_r,10*z_r,401)
  Ex,Ey, Ez, Bx,By,Bz = ApplPhysB(0,0,z,k,w0)
  Ex_phase = np.angle(Ex*np.exp(1j*k*z))

  GaussianPhase = np.arctan(2*z/(k*w0**2))

  plt.figure()
  plt.plot(z*1e6,GaussianPhase)
  plt.plot(z*1e6,Ex_phase+np.pi/2)
  plt.savefig("GouyPhase-ApplPhysB.pdf", bbox_inches='tight', dpi=500)

  x_f = np.linspace(-0.50*w0,0.50*w0,100)
  y_f = np.linspace(-0.50*w0,0.50*w0,100)
  X, Y = np.meshgrid(x_f,y_f)
  z   = 0
  Ex,Ey,Ez,Bx,By,Bz = CSPSW(X,Y,z,k,k*w0**2/2)

  Ex_ref = np.copy(Ex)
  Ex /= np.amax(np.abs(Ex_ref))
  Ey /= np.amax(np.abs(Ex_ref))
  Ez /= np.amax(np.abs(Ex_ref))
  Bx /= np.amax(np.abs(Ex_ref))
  By /= np.amax(np.abs(Ex_ref))
  Bz /= np.amax(np.abs(Ex_ref))

  PlotAllComponents("CSPSW.pdf", X, Y, np.abs(Ex)**2,np.abs(Ey)**2,np.abs(Ez)**2,np.abs(Bx)**2,np.abs(By)**2,np.abs(Bz)**2,levels,plot_options,contour_options)

  # -- Gouy shift of ApplPhysB.
  phases = []
  z_r = k*w0**2/2
  z = np.linspace(-10*z_r,10*z_r,401)
  Ex,Ey,Ez,Bx,By,Bz = CSPSW(0,0,z,k,z_r)
  Ex_phase = np.angle(Ex*np.exp(1j*k*np.abs(z)))
  Ex_phase[z<0] -= np.pi

  GaussianPhase = np.arctan(2*z/(k*w0**2))

  plt.figure()
  plt.plot(z*1e6,GaussianPhase)
  plt.plot(z*1e6,user_mod(Ex_phase,2*np.pi)-np.pi)
  plt.savefig("GouyPhase-CSPSW.pdf", bbox_inches='tight', dpi=500)

  # -- Run tests.
  #unittest.main()
