import numpy as np
from numpy.testing import assert_allclose
import sys
import pytest
from scipy.fftpack import fft, ifft
from numpy.fft import fftshift
sys.path.append('src')
from functions import *
from integrand_and_rk import xpm_coeff,fwm_coeff, Factors
import pickle as pl
"""
Small number of tests that look in to simple tests. 
"""


def test_read_write1():
    #os.system('rm testing_data/hh51_test.hdf5')
    A = np.random.rand(10,3,5) + 1j* np.random.rand(10,3,5)
    B  = np.random.rand(10)
    C = 1
    save_variables('hh51_test','0',filepath = 'testing_data/',
                    A = A, B = B, C=C)
    A_copy, B_copy, C_copy = np.copy(A), np.copy(B), np.copy(C)
    del A,B,C
    D = read_variables('hh51_test', '0', filepath='testing_data/')

    A,B,C = D['A'], D['B'], D['C']
    os.system('rm testing_data/hh51_test.hdf5')
    assert_allclose(A,A_copy)


def test_read_write2():

    #os.system('rm testing_data/hh52_test.hdf5')
    A = np.random.rand(10,3,5) + 1j* np.random.rand(10,3,5)
    B  = np.random.rand(10)
    C = 1
    save_variables('hh52_test','0',filepath = 'testing_data/',
                    A = A, B = B, C=C)
    A_copy, B_copy, C_copy = np.copy(A), np.copy(B), np.copy(C)
    del A,B,C
    D = read_variables('hh52_test', '0', filepath='testing_data/')
    A,B,C = D['A'], D['B'], D['C']
    #locals().update(D)
    os.system('rm testing_data/hh52_test.hdf5')
    return None


def test_read_write3():


    A = np.random.rand(10,3,5) + 1j* np.random.rand(10,3,5)
    B  = np.random.rand(10)
    C = 1
    save_variables('hh53_test','0',filepath = 'testing_data/',
                    A = A, B = B, C=C)
    A_copy, B_copy, C_copy = np.copy(A), np.copy(B), np.copy(C)
    del A,B,C
    D = read_variables('hh53_test', '0', filepath='testing_data/')
    A,B,C = D['A'], D['B'], D['C']
    os.system('rm testing_data/hh53_test.hdf5')
    assert C == C_copy
    return None



    
def test_dbm2w():
    assert dbm2w(30) == 1


def test1_w2dbm():
    assert w2dbm(1) == 30


def test2_w2dbm():
    a = np.zeros(100)
    floor = np.random.rand(1)[0]
    assert_allclose(w2dbm(a,-floor), -floor*np.ones(len(a)))


def test3_w2dbm():
    with pytest.raises(ZeroDivisionError):
        w2dbm(-1)


def test_time_frequency():
    nt = 10
    dt = np.abs(np.random.rand())*10
    u1 = 10*(np.random.randn(7,2**nt) + 1j * np.random.randn(7,2**nt))
    U = fftshift(dt*fft(u1), axes = -1)
    u2 = ifft(fftshift(U, axes = -1)/dt)
    assert_allclose(u1, u2)


def test_XPM_factors():
    fr = 0
    H = np.array([0 for i in range(12)])
    test_coeff = xpm_coeff(H, fr)
    assert_allclose(test_coeff, np.ones([7,7])*2 - np.identity(7))

def test_XPM_factors_fr():
    fr = 0.18
    H = np.array([0 for i in range(12)])
    test_coeff = xpm_coeff(H, fr)
    assert_allclose(test_coeff, (np.ones([7,7])*2 - fr) - np.identity(7)*(1 -fr))

def test_XPM_factors_h():
    fr = 1
    H = np.array([i for i in range(-6,7)])
    test_coeff = xpm_coeff(H, fr)- 1 + np.identity(7)

    with open('testing_data/xpm_h.pickle', 'rb') as f:
        r = pl.load(f)
    r = np.array(r).astype(np.complex128)
    assert_allclose(r,test_coeff)



def test_FWM_factors_h():
    fr = 1
    H = np.array([i for i in range(-6,7)])
    test_coeff_fwm = fwm_coeff(H, fr)
    with open('testing_data/fwm_h.pickle', 'rb') as f:
        r = pl.load(f)
    r = np.array(r).astype(np.complex128)
    assert_allclose(r, test_coeff_fwm)

def test_FWM_factors_fr0():
    fr = 0
    H = np.array([i for i in range(-6,7)])
    test_coeff_fwm = fwm_coeff(H, fr)
    with open('testing_data/fwm_fr0.pickle', 'rb') as f:
        r = pl.load(f)
    r = np.array(r).astype(np.complex128)
    assert_allclose(r, test_coeff_fwm)