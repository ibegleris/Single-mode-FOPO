import numpy as np
from scipy.constants import pi
from numpy.fft import fftshift
from scipy.fftpack import fft, ifft
from six.moves import builtins
from cython_files.cython_integrand import *

from time import time
import sys
from numpy.testing import assert_allclose
import pickle as pl


#from scipy.fftpack import fftshift
import numba
from functools import lru_cache as cache
vectorize = numba.vectorize
jit = numba.jit
autojit = numba.autojit
# Pass through the @profile decorator if line profiler (kernprof) is not in use
# Thanks Paul!
try:
    builtins.profile
except AttributeError:
    def profile(func):
        return func





trgt = 'cpu'





class Integrand(object):
    """
    Serves as the intrgrand of the nonlinear term of the GNLSE. Care has been 
    taken to pick out the combination of multiplications needed for computation. 
    Additionally there is use of cython by default
    """

    def __init__(self, gama, tsh, w_tiled, s, ram, cython_tick=True, timer=False):
        self.data_minimum = ((0, 1), (0, 3), (0, 4), (0, 5), (0, 6),
                             (1, 1), (1, 3), (1, 4), (1, 5), (1, 6),
                             (2, 0), (2, 1), (2, 2), (2,
                                                      3), (2, 4), (2, 5), (2, 6),
                             (3, 3), (3, 4), (3, 5), (3, 6),
                             (4, 4), (4, 5), (4, 6), (5, 5))

        self.fwm_map = (((2, 5), (3, 11), (4, 12, 6), (5, 13, 7), (6, 14, 8, 17)),
                        ((1, 10), (2, 1), (3, 12, 2), (4, 13, 3),
                         (5, 14, 4, 17), (6, 15, 18)),
                        ((0, 5), (1, 1), (2, 6, 2), (3, 7, 3),
                         (4, 8, 4, 17), (5, 9, 18), (6, 19, 21)),
                        ((0, 11), (1, 12, 2), (2, 7, 3), (3, 14, 8, 4),
                         (4, 15, 9), (5, 16, 21), (6, 22)),
                        ((0, 12, 6), (1, 13, 3), (2, 8, 4, 17),
                         (3, 15, 9), (4, 16, 19), (5, 20), (6, 24)),
                        ((0, 13, 7), (1, 14, 4, 17), (2, 9, 18),
                         (3, 16, 21), (4, 20), (5, 23)),
                        ((0, 14, 8, 17), (1, 15, 18), (2, 19, 21), (3, 22), (4, 24)))
        self.gama = gama
        self.tsh = tsh
        self.w_tiled = w_tiled
        self.fwm_map = np.asanyarray(self.fwm_map)
        self.factors_xpm = xpm_coeff(ram.H, ram.fr)
        self.factors_fwm = fwm_coeff(ram.H, ram.fr)
        self.shape1, self.shape2 = self.w_tiled.shape


        if s == 1 and cython_tick:
            self.dAdzmm = self.cython_s1
        elif s == 0 and cython_tick:
            self.dAdzmm = self.cython_s0
        elif s == 1 and not(cython_tick):
            self.dAdzmm = self.python_s1
        elif s == 0 and not(cython_tick):
            self.dAdzmm = self.python_s0
        if timer:
            self.dAdzmm = self.timer

    #@profile
    def cython_s1(self, u0, dz):      
        return fwm_s1(u0, self.factors_xpm,
                         self.factors_fwm,dz * self.gama, self.tsh, self.w_tiled, dz, 
                         self.shape1,self.shape2)

   

    def cython_s0(self, u0, dz):
        fwm_s0 = dAdzmm
        return fwm_s0(u0, self.factors_xpm,
                         self.factors_fwm, dz * self.gama, self.tsh, self.w_tiled, dz, 
                         self.shape1,self.shape2)


    @profile
    def SPM_XPM_FWM_python(self, u0):
        # SPM-XPM
        u0_abs2 = np.abs(u0)**2
        N = np.matmul(self.factors_xpm, u0_abs2) * u0

        # FWM
        u0_multi = np.empty(
            [len(self.data_minimum), u0.shape[1]], dtype=np.complex)
        for i in range(len(self.data_minimum)):
            u0_multi[i, :] = u0[self.data_minimum[i][0], :] \
                * u0[self.data_minimum[i][1], :]

        for i in range(u0.shape[0]):
            for f in self.fwm_map[i]:
                N[i, :] += (self.factors_fwm[f[0], f[1:], np.newaxis] *
                            u0_multi[f[1:], :]).sum(axis=0) * \
                           (u0[f[0], :].conjugate())
        return N
    

    def self_step_s1(self, N):
        temp = ifft(self.w_tiled * fft(N))
        return N + self.tsh[:, np.newaxis]*temp


    @profile
    def python_s1(self, u0, dz):
        N = self.SPM_XPM_FWM_python(u0)
        N = dz * self.gama[:, np.newaxis] * self.self_step_s1(N)
        return N

   
    @profile
    def python_s0(self, u0, dz):
        N = self.SPM_XPM_FWM_python(u0)
        N = dz * self.gama[:, np.newaxis] * N
        return N


    def timer(self, u0, dz):
        """
        Times the functions of python, cython etc. 
        """
        dt1, dt2, dt3, dt4 = [], [], [], []
        NN = 1000
        for i in range(NN):
            t = time()
            N1 = self.cython_s1(u0, dz)
            dt1.append(time() - t)

            t = time()
            N2 = self.python_s1(u0, dz)
            dt2.append(time() - t)
            assert_allclose(N1, N2)
            t = time()
            N1 = self.cython_s0(u0, dz)
            dt3.append(time() - t)

            t = time()
            N2 = self.python_s0(u0, dz)
            dt4.append(time() - t)
            assert_allclose(N1, N2)
        print('cython_s1: {} +/- {}'.format(np.average(dt1), np.std(dt1)))
        print('cython_s0: {} +/- {}'.format(np.average(dt3), np.std(dt3)))
        print('python_s1: {} +/- {}'.format(np.average(dt2), np.std(dt2)))
        print('python_s0: {} +/- {}'.format(np.average(dt4), np.std(dt4)))
        
        print('s1 Cython is {} times faster than Python.'.format(np.average(dt2)/np.average(dt1)))
        print('s0 Cython is {} times faster than Python.'.format(np.average(dt4)/np.average(dt3)))
        sys.exit()
        return N


def xpm_coeff(H, fr):
    """
    XPM coefficients for the 7 band version that include Raman.
    """
    c = Factors(H, fr)
    coeff = np.zeros([7, 7], dtype=np.complex128)

    for i in range(7):
        for j in range(7):
            coeff[i, j] = c.xpm(i-j)
    return coeff


class Factors(object):
    def __init__(self, H, fr):
        self.H = H
        self.fr = fr
        self.H = np.concatenate((H[6:], H[0:6]))

    def xpm(self, i):
        if i == 0:
            return 1
        else:
            return self.H[i]*self.fr - self.fr + 2

    def f(self, i_vec):
        res = sum([self.H[i]*self.fr - self.fr + 1 for i in i_vec])
        return res


def fwm_coeff(H, fr):
    c = Factors(H, fr)
    coeff = np.zeros([7, 25], dtype=np.complex128)
    # 0
    coeff[0, 5] = c.f([1])
    coeff[0, 6] = c.f([1, 3])
    coeff[0, 7] = c.f([1, 4])
    coeff[0, 8] = c.f([1, 5])
    coeff[0, 11] = c.f([1, 2])
    coeff[0, 12] = c.f([2])
    coeff[0, 13] = c.f([2, 3])
    coeff[0, 14] = c.f([2, 4])
    coeff[0, 17] = c.f([3])
    # 1
    coeff[1, 1] = c.f([2, -1])
    coeff[1, 2] = c.f([3, -1])
    coeff[1, 3] = c.f([4, -1])
    coeff[1, 4] = c.f([5, -1])
    coeff[1, 10] = c.f([1, -1])
    coeff[1, 12] = c.f([1])
    coeff[1, 13] = c.f([1, 2])
    coeff[1, 14] = c.f([1, 3])
    coeff[1, 15] = c.f([1, 4])
    coeff[1, 17] = c.f([2])
    coeff[1, 18] = c.f([2, 3])

    # 2
    coeff[2, 1] = c.f([1, -2])
    coeff[2, 2] = c.f([2, -2])
    coeff[2, 3] = c.f([3, -2])
    coeff[2, 4] = c.f([4, -2])
    coeff[2, 5] = c.f([-1])
    coeff[2, 6] = c.f([1, -1])
    coeff[2, 7] = c.f([2, -1])
    coeff[2, 8] = c.f([3, -1])
    coeff[2, 9] = c.f([4, -1])
    coeff[2, 17] = c.f([1])
    coeff[2, 18] = c.f([1, 2])
    coeff[2, 19] = c.f([1, 3])
    coeff[2, 21] = c.f([2])

    # 3
    coeff[3, 2] = c.f([1, -3])
    coeff[3, 3] = c.f([2, -3])
    coeff[3, 4] = c.f([3, -3])
    coeff[3, 7] = c.f([1, -2])
    coeff[3, 8] = c.f([2, -2])
    coeff[3, 9] = c.f([3, -2])
    coeff[3, 11] = c.f([-1, -2])
    coeff[3, 12] = c.f([-1])
    coeff[3, 14] = c.f([1, -1])
    coeff[3, 15] = c.f([2, -1])
    coeff[3, 16] = c.f([3, -1])
    coeff[3, 21] = c.f([1])
    coeff[3, 22] = c.f([1, 2])

    # 4
    coeff[4, 3] = c.f([1, -4])
    coeff[4, 4] = c.f([2, -4])
    coeff[4, 6] = c.f([-1, -3])
    coeff[4, 8] = c.f([1, -3])
    coeff[4, 9] = c.f([2, -3])
    coeff[4, 12] = c.f([-2])
    coeff[4, 13] = c.f([-1, -2])
    coeff[4, 15] = c.f([1, -2])
    coeff[4, 16] = c.f([2, -2])
    coeff[4, 17] = c.f([-1])
    coeff[4, 19] = c.f([1, -1])
    coeff[4, 20] = c.f([2, -1])
    coeff[4, 24] = c.f([1])

    # 5
    coeff[5, 4] = c.f([1, -5])
    coeff[5, 7] = c.f([-1, -4])
    coeff[5, 9] = c.f([1, -4])
    coeff[5, 13] = c.f([-2, -3])
    coeff[5, 14] = c.f([-1, -3])
    coeff[5, 16] = c.f([1, -3])
    coeff[5, 17] = c.f([-2])
    coeff[5, 18] = c.f([-1, -2])
    coeff[5, 20] = c.f([1, -2])
    coeff[5, 21] = c.f([-1])
    coeff[5, 23] = c.f([1, -1])

    # 6
    coeff[6, 8] = c.f([-1, -5])
    coeff[6, 14] = c.f([-2, -4])
    coeff[6, 15] = c.f([-1, -4])
    coeff[6, 17] = c.f([-3])
    coeff[6, 18] = c.f([-2, -3])
    coeff[6, 19] = c.f([-1, -3])
    coeff[6, 21] = c.f([-2])
    coeff[6, 22] = c.f([-1, -2])
    coeff[6, 24] = c.f([-1])
    return coeff


@vectorize(['complex128(complex128,complex128)'], target=trgt)
def multi(x, y):
    return x*y


@vectorize(['complex128(complex128,complex128)'], target=trgt)
def add(x, y):
    return x + y


