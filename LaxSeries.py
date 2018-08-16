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
import itertools
import unittest

# -------------------------------- Functions  ------------------------------- #

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


# -- Tests
class TestExpansionCoefficients(unittest.TestCase):

  def test_both_functions(self):
    m_l = [i for i in range(1,15)]
    p_l = [i for i in range(1,15)]

    for iter in itertools.product(m_l,p_l):
      a = ExpansionCoefficient(*iter)
      b = ExpansionCoefficientDirect(*iter)
      print(*iter, a)
      self.assertAlmostEqual(a,b)


if __name__ == "__main__":

  # -- Run tests.
  unittest.main()


