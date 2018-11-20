from numpy.testing import assert_allclose,assert_raises
import sys
sys.path.append('src')
from functions import *
from test_WDM_splice import specific_variables
from copy import deepcopy
"""
Tests function that are set for the preperation 
of the pulse propagation. 
"""
M, int_fwm, sim_wind, n2, alphadB, maxerr, ss, N, lamda_c, lamda, lams, \
            betas, fv, where, f_centrals = specific_variables(10)

def test_dF_sidebands():
    """
	Tests of the ability of dF_sidebands to find the
	sidebands expected for predetermined conditions.
    """
    lamp = 1048.17e-9
    lamda0  = 1051.85e-9
    betas =  0, 0,0, 6.756e-2 *1e-3, -1.002e-4 * 1e-3, 3.671*1e-7 * 1e-3

    F, f_p = dF_sidebands(betas, lamp,lamda0, n2, M,5,0)
  

    f_s, f_i = f_p - F, f_p + F
    lams, lami = (1e-3*c/i for i in (f_s, f_i))
    assert lams, lami == (1200.2167948665879, 930.31510086250455)



def test_noise():
    noise = Noise(int_fwm, sim_wind)
    n1 = noise.noise_func(int_fwm)
    n2 = noise.noise_func(int_fwm)
    assert_raises(AssertionError, assert_allclose, n1, n2)


class Test_loss:
    def test_loss1(a):
        loss = Loss(int_fwm, sim_wind, amax = alphadB)
        alpha_func = loss.atten_func_full(sim_wind.fv)
        assert_allclose(alpha_func, np.ones_like(alpha_func)*alphadB/4.343)
    def test_loss2(a):

        loss = Loss(int_fwm, sim_wind, amax = 2*alphadB)
        alpha_func = loss.atten_func_full(sim_wind.fv)
        maxim = np.max(alpha_func)
        assert_allclose(maxim, 2*alphadB/4.343)

    def test_loss3(a):
        loss = Loss(int_fwm, sim_wind, amax = 2*alphadB)
        alpha_func = loss.atten_func_full(sim_wind.fv)
        minim = np.min(alpha_func)
        assert minim == alphadB/4.343


def test_dispersion():

    int_fwm.alphadB = 0.1
    loss = Loss(int_fwm, sim_wind, amax = 0.1)
    alpha_func = loss.atten_func_full(sim_wind.fv)
    int_fwm.alphadB = alpha_func
    int_fwm.alpha = int_fwm.alphadB


    betas_disp = dispersion_operator(betas,lamda_c,int_fwm,sim_wind)
    #np.savetxt('testing_data/exact_dispersion.txt',betas_disp.view(np.float))


    betas_exact = np.loadtxt('testing_data/exact_dispersion.txt').view(complex)
    assert_allclose(betas_disp,betas_exact)
