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


# ----------------------------- SymPy Declaration --------------------------- #
x, y, z,k, w_0 = sympy.symbols('x y z k w_0')
z_r     = k*w_0**2/2
w_z     = w_0*sympy.sqrt(1+(z/z_r)**2)
phi_0   = w_0/w_z*sympy.exp(-(x**2+y**2)/w_z**2)*sympy.exp(-1j*k*z+1j*sympy.atan(z/z_r))


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


if __name__ == "__main__":

  # -- Run tests.
  #unittest.main()

  # -- Substitute numerical values for fixed parameters in phi_0.
  lamb  = 800e-9

  phi_0 = phi_0.subs(k, 2*sympy.pi/lamb)
  phi_0 = phi_0.subs(w_0,lamb)

  phi = sympy.lambdify((x,y,z), phi_0)

  x_f = np.linspace(-1.5*lamb,1.5*lamb,100)
  y_f = np.linspace(-1.5*lamb,1.5*lamb,100)
  X, Y = np.meshgrid(x_f,y_f)
  z   = 0

  print(phi_0)
  print("test")
  print(phi(X,Y,z))

  plt.figure()
  plt.pcolormesh(X,Y,np.abs(phi(X,Y,z)))
  plt.show()

