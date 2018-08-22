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
import scipy.constants as cst
import itertools
import unittest
import sympy

# -------------------------------- Functions -------------------------------- #
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
  phi_0       = w_0_sym/w_z_sym*sympy.exp(-(x_sym**2+y_sym**2)/w_z_sym**2)*sympy.exp(-1j*k_sym*z_sym+1j*sympy.atan(z_sym/z_r_sym))

  # -- Lambdified expressions for numerical evaluation.
  phi         = sympy.lambdify((x_sym,y_sym,z_sym,k_sym,w_0_sym), phi_0)
  psi         = sympy.lambdify((x_sym,y_sym,z_sym,k_sym,w_0_sym), psi_1)
  summand     = phi(X,Y,z,k,w_0)

  # -- We compute the derivatives analytically.
  derivatives = []
  for i in range(1,2*m_max+1):
    derivatives.append(sympy.diff(phi_0,z_sym,i))

  for m in range(1,m_max+1):
    phi_2m       = sympy.S.zero
    psi_2mp1     = sympy.S.zero

    for p in range(1,m+1):
      # -- We evaluate the product between the z factor and the derivative,
      # -- simplifty it, and then lambdify it, and add it the sum.
      polynomial = z_sym**p*derivatives[m+p-2]
      polynomial = sympy.simplify(polynomial)
      phi_2m    += ExpansionCoefficient(m,p)*polynomial

    if (m>0):
      psi_2mp1   = 1j/k*(sympy.diff(phi_2m,x)+sympy.diff(psi_2mp1))

    phi_2m *= (1j/(2*k))**m
    summand += (1j/(2*k))**m*innerSummand

  return summand, summand_z

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

  # -- Run tests.
  #unittest.main()

  # -- Substitute numerical values for fixed parameters in phi_0.
  lamb  = 800e-9
  k     = 2*np.pi*cst.c/lamb
  w0    = 0.70*lamb

  x_f = np.linspace(-2.5*w0,2.5*w0,100)
  y_f = np.linspace(-2.5*w0,2.5*w0,100)
  X, Y = np.meshgrid(x_f,y_f)
  z   = 0.0

  Ex, Ez = ExLax(X,Y,z,k,w0,3)

  plt.figure()
  plt.pcolormesh(X*1e6,Y*1e6,np.abs(Ex)**2)
  plt.contour(X*1e6,Y*1e6,np.abs(Ex)**2, linestyles='--')
  plt.gca().set_aspect('equal')

  plt.figure()
  plt.pcolormesh(X*1e6,Y*1e6,np.abs(Ez)**2)
  plt.contour(X*1e6,Y*1e6,np.abs(Ez)**2, linestyles='--')
  plt.gca().set_aspect('equal')

  Ex, Ey, Ez, Bx, By, Bz = np.abs(ApplPhysB(X,Y,z,k,w0))
  plt.figure()
  plt.pcolormesh(X*1e6,Y*1e6,Ex**2)
  plt.contour(X*1e6,Y*1e6,Ex**2, linestyles='--')
  plt.gca().set_aspect('equal')

  plt.figure()
  plt.pcolormesh(X*1e6, Y*1e6, Ez**2)
  plt.contour(X*1e6,Y*1e6, Ez**2, linestyles='--')
  plt.gca().set_aspect('equal')

  x_f = np.linspace(-0.75*w0,0.75*w0,100)
  y_f = np.linspace(-0.75*w0,0.75*w0,100)
  X, Y = np.meshgrid(x_f,y_f)
  z   = 0

  w0 = 0.70*lamb
  Ex,Ey,Ez,Bx,By,Bz = CSPSW(X,Y,z,2*np.pi/lamb,2*np.pi/lamb*w0**2/2)

  plt.figure()
  plt.pcolormesh(X*1e6,Y*1e6,np.abs(Ex/np.amax(np.abs(Ex)))**2)
  plt.contour(X*1e6,Y*1e6,np.abs(Ex)**2,linestyles='--',colors='k')
  plt.gca().set_aspect('equal')

  plt.figure()
  plt.pcolormesh(X*1e6,Y*1e6,np.abs(Ez)**2)
  plt.contour(X*1e6,Y*1e6,np.abs(Ez)**2,linestyles='--',colors='k')
  plt.gca().set_aspect('equal')
  plt.show()
