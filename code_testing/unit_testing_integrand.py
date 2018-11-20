import sys
import os
import numpy as np
sys.path.append('src')
#os.system("cd src/cython_files && python setup.py build_ext --inplace && cython -a cython_integrand.pyx && cd ../..")
from cython_files.cython_integrand import *
from integrand_and_rk import Integrand
from numpy.testing import assert_allclose


class Test_integrands(object):
	"""
	Tests if the results are the same from cython and Python.
	"""
	w_tiled = np.random.randn(7, 1024)
	gama = np.random.randn(7) + 1j * np.random.randn(7)
	tsh = np.random.randn(7) 

	Int = Integrand(gama , tsh,
                     w_tiled,0 , cython_tick = True,
                                            timer = False)
	u0 = np.random.randn(7, 1024) + 1j *np.random.randn(7, 1024)
	NC_1 =  Int.cython_s1(u0)
	NP_1 = 	Int.python_s1(u0)
	NC_0 = 	Int.cython_s0(u0)
	NP_0 =  Int.python_s0(u0)
	
	def test_s1(self):
		assert_allclose(self.NC_1, self.NP_1)

	def test_s0(self):
		assert_allclose(self.NC_0, self.NP_0)